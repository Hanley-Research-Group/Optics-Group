"""
Tests for tubes whose precurvature is straight / curved / straight.

Convention note: with this code base a positive body-frame curvature
along +x bends the backbone toward -y (dR/ds = R [u]x).
"""

import numpy as np
import pytest

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
    Segment,
)
from ctr_sim.kinematics.intervals import (
    backbone_segments,
    curved_intervals,
)
from ctr_sim.mechanics.forward_v2 import (
    solve_forward_kinematics_v2,
)
from ctr_sim.mechanics.sampling import (
    tip_position,
)
from ctr_sim.mechanics.torsion_v2 import (
    torsion_shooting_residual,
)
from ctr_sim.mechanics.forward import (
    solve_forward_kinematics,
)
from ctr_sim.control.jacobian import (
    numerical_position_jacobian_v2,
)
from ctr_sim.control.resolved_rate import (
    resolved_rate_step_v2,
)
from ctr_sim.control.constraints import (
    constrain_joint_step,
)

# Integration uses scipy's default ODE tolerances, so positions are
# accurate to about a micron.
POS_TOL = 5e-6

nylon = Material("Nylon", 2.7e9, 1.0e9)


def rot_z(a):
    return np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a), np.cos(a), 0.0],
        [0.0, 0.0, 1.0],
    ])


def three_piecewise_tubes():
    return [
        Tube.from_lengths(
            "outer", 0.060, 0.050, 0.030, 18.0, 4.0e-3, 3.2e-3, nylon
        ),
        Tube.from_lengths(
            "middle", 0.080, 0.045, 0.035, 22.0, 2.8e-3, 2.1e-3, nylon
        ),
        Tube.from_lengths(
            "inner", 0.100, 0.040, 0.040, 25.0, 1.8e-3, 0.8e-3, nylon
        ),
    ]


def reference_tip_equal_rotations(robot, n=200_000):
    """
    Independent reference for the tip when all tube rotations are equal
    (no torsion). Does not use backbone_segments: at every arc length
    the resultant curvature is sum(EI * kappa) / sum(EI) over the tubes
    present there, integrated as a planar curve.
    """

    betas = robot.state.insertions
    s_end = max(betas)
    ds = s_end / n
    s = (np.arange(n) + 0.5) * ds

    num = np.zeros(n)
    den = np.zeros(n)

    for tube, beta in zip(robot.tubes, betas):

        sigma = s - (beta - tube.length)
        present = (sigma >= 0) & (sigma <= tube.length)

        in_curve = (
            (sigma >= tube.length_before_curve)
            & (
                sigma
                <= tube.length_before_curve + tube.curve_length
            )
        )

        kappa = np.where(in_curve, tube.precurvature, 0.0)

        num += np.where(present, tube.EI * kappa, 0.0)
        den += np.where(present, tube.EI, 0.0)

    kappa = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)

    angle = np.cumsum(kappa * ds) - 0.5 * kappa * ds

    tip = np.array([
        0.0,
        -np.sum(np.sin(angle) * ds),
        np.sum(np.cos(angle) * ds),
    ])

    alpha = robot.state.rotations[0]

    return rot_z(alpha) @ tip


# --------------------------------------------------------------
# Tube geometry
# --------------------------------------------------------------

class TestTubeGeometry:

    def test_from_lengths_sums_to_total(self):

        tube = Tube.from_lengths(
            "t", 0.05, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
        )

        assert tube.length == pytest.approx(0.12)
        assert tube.length_before_curve == pytest.approx(0.05)
        assert tube.curve_length == pytest.approx(0.04)
        assert tube.length_after_curve == pytest.approx(0.03)

    def test_default_tube_is_fully_curved(self):

        tube = Tube("t", 0.16, 15.0, 3e-3, 2.8e-3, nylon)

        assert tube.length_before_curve == 0.0
        assert tube.curve_length == pytest.approx(0.16)
        assert tube.length_after_curve == pytest.approx(0.0)
        assert tube.has_uniform_curvature

    def test_straight_tube_is_uniform(self):

        tube = Tube("t", 0.2, 0.0, 1.6e-3, 0.0, nylon)

        assert tube.has_uniform_curvature
        assert tube.curvature_at(0.1) == 0.0

    def test_curve_only_given_start_runs_to_tip(self):

        tube = Tube(
            "t", 0.10, 12.0, 2e-3, 0.0, nylon,
            length_before_curve=0.04,
        )

        assert tube.curve_length == pytest.approx(0.06)
        assert tube.length_after_curve == pytest.approx(0.0)

    def test_piecewise_tube_is_not_uniform(self):

        tube = Tube.from_lengths(
            "t", 0.05, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
        )

        assert not tube.has_uniform_curvature

    def test_curvature_profile(self):

        tube = Tube.from_lengths(
            "t", 0.05, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
        )

        assert tube.curvature_at(0.0) == 0.0
        assert tube.curvature_at(0.049) == 0.0
        assert tube.curvature_at(0.07) == 20.0
        assert tube.curvature_at(0.091) == 0.0
        assert tube.curvature_at(0.12) == 0.0

    def test_stiffness_is_independent_of_curve_profile(self):

        a = Tube("a", 0.12, 20.0, 2e-3, 0.0, nylon)
        b = Tube.from_lengths(
            "b", 0.05, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
        )

        assert a.EI == b.EI
        assert a.GJ == b.GJ

    def test_sections_longer_than_tube_rejected(self):

        with pytest.raises(ValueError):
            Tube(
                "t", 0.10, 10.0, 2e-3, 0.0, nylon,
                length_before_curve=0.06, curve_length=0.05,
            )

    def test_negative_sections_rejected(self):

        with pytest.raises(ValueError):
            Tube(
                "t", 0.10, 10.0, 2e-3, 0.0, nylon,
                length_before_curve=-0.01,
            )

        with pytest.raises(ValueError):
            Tube(
                "t", 0.10, 10.0, 2e-3, 0.0, nylon,
                length_before_curve=0.02, curve_length=-0.01,
            )

        with pytest.raises(ValueError):
            Tube.from_lengths(
                "t", 0.02, 0.05, -0.01, 10.0, 2e-3, 0.0, nylon
            )


# --------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------

class TestSegmentation:

    def test_single_tube_splits_at_curve_boundaries(self):

        tube = Tube.from_lengths(
            "t", 0.05, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
        )
        robot = ConcentricTubeRobot(
            [tube], CTRState([0.12], [0.0])
        )

        segments = backbone_segments(robot)

        assert [
            (round(s.start, 9), round(s.end, 9)) for s in segments
        ] == [(0.0, 0.05), (0.05, 0.09), (0.09, 0.12)]

        assert [s.curvatures for s in segments] == [
            [0.0], [20.0], [0.0]
        ]

    def test_curve_that_starts_behind_the_base(self):

        tube = Tube.from_lengths(
            "t", 0.05, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
        )

        # Tube occupies s in [-0.03, 0.09]; curve in [0.02, 0.06].
        robot = ConcentricTubeRobot(
            [tube], CTRState([0.09], [0.0])
        )

        segments = backbone_segments(robot)

        assert [s.curvatures for s in segments] == [
            [0.0], [20.0], [0.0]
        ]
        assert segments[0].start == 0.0
        assert segments[1].start == pytest.approx(0.02)

        # Curve already under way at the base: free backbone starts
        # mid-curve.
        robot.state.insertions = [0.05]  # curve in [-0.02, 0.02]

        segments = backbone_segments(robot)

        assert [s.curvatures for s in segments] == [[20.0], [0.0]]

    def test_curve_entirely_behind_base_gives_straight_robot(self):

        tube = Tube.from_lengths(
            "t", 0.05, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
        )

        # beta = 0.03: tube spans s in [-0.09, 0.03], so the curve
        # occupies [-0.04, 0.00]: none of it is in the free backbone.
        robot = ConcentricTubeRobot(
            [tube], CTRState([0.03], [0.0])
        )

        segments = backbone_segments(robot)

        assert [s.curvatures for s in segments] == [[0.0]]

        bb = solve_forward_kinematics_v2(robot)
        np.testing.assert_allclose(
            tip_position(bb), [0.0, 0.0, 0.03], atol=1e-9
        )

    def test_straight_tube_adds_no_curve_boundaries(self):

        # Zero precurvature: its (meaningless) curve interval would
        # begin at s = 0.15 - 0.2 + 0.1 = 0.05, inside the free
        # backbone. It must not create a boundary there.
        straight = Tube(
            "s", 0.2, 0.0, 1.6e-3, 0.0, nylon,
            length_before_curve=0.1,
        )
        curved = Tube.from_lengths(
            "c", 0.03, 0.04, 0.03, 20.0, 3e-3, 2e-3, nylon
        )
        robot = ConcentricTubeRobot(
            [curved, straight],
            CTRState([0.10, 0.15], [0.0, 0.0]),
        )

        # "c" spans s in [0, 0.10]; its curve is [0.03, 0.07].
        (c0, c1), (s0, s1) = curved_intervals(robot)

        assert (c0, c1) == (pytest.approx(0.03), pytest.approx(0.07))
        assert s0 == s1

        starts = [
            round(s.start, 9) for s in backbone_segments(robot)
        ]

        assert starts == [0.0, 0.03, 0.07, 0.10]

    def test_coincident_boundaries_are_merged(self):

        tube = Tube.from_lengths(
            "t", 0.05, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
        )
        # curve start of the tube coincides with s = 0 exactly
        robot = ConcentricTubeRobot(
            [tube], CTRState([0.07], [0.0])
        )

        segments = backbone_segments(robot)

        assert all(s.end - s.start > 1e-9 for s in segments)

    def test_legacy_tubes_keep_original_segmentation(self):

        outer = Tube("o", 0.16, 15.0, 3e-3, 2.8e-3, nylon)
        middle = Tube("m", 0.18, 10.0, 2e-3, 1.8e-3, nylon)
        inner = Tube("i", 0.20, 0.0, 1.6e-3, 0.0, nylon)

        robot = ConcentricTubeRobot(
            [outer, middle, inner],
            CTRState([0.10, 0.12, 0.14], [0.0, 0.0, 0.0]),
        )

        segments = backbone_segments(robot)

        assert [
            (round(s.start, 9), round(s.end, 9)) for s in segments
        ] == [(0.0, 0.10), (0.10, 0.12), (0.12, 0.14)]

        assert [s.curvatures for s in segments] == [
            [15.0, 10.0, 0.0], [10.0, 0.0], [0.0]
        ]

    def test_segment_requires_curvatures_for_piecewise_tube(self):

        tube = Tube.from_lengths(
            "t", 0.05, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
        )

        with pytest.raises(ValueError):
            Segment(0.0, 0.1, [tube])

    def test_segment_curvature_length_checked(self):

        tube = Tube("t", 0.1, 5.0, 2e-3, 0.0, nylon)

        with pytest.raises(ValueError):
            Segment(0.0, 0.1, [tube], curvatures=[1.0, 2.0])

    def test_segment_defaults_to_tube_precurvature(self):

        tube = Tube("t", 0.1, 5.0, 2e-3, 0.0, nylon)

        assert Segment(0.0, 0.1, [tube]).curvatures == [5.0]


# --------------------------------------------------------------
# Forward mechanics
# --------------------------------------------------------------

class TestForwardMechanics:

    @pytest.mark.parametrize("alpha", [0.0, 0.7, -2.0])
    def test_single_tube_matches_closed_form(self, alpha):

        l1, lc, l3, kappa = 0.05, 0.04, 0.03, 20.0

        tube = Tube.from_lengths(
            "t", l1, lc, l3, kappa, 2e-3, 0.0, nylon
        )
        robot = ConcentricTubeRobot(
            [tube], CTRState([l1 + lc + l3], [alpha])
        )

        phi = kappa * lc

        expected = np.array([
            0.0,
            -((1 - np.cos(phi)) / kappa + l3 * np.sin(phi)),
            l1 + np.sin(phi) / kappa + l3 * np.cos(phi),
        ])

        tip = tip_position(solve_forward_kinematics_v2(robot))

        np.testing.assert_allclose(
            tip, rot_z(alpha) @ expected, atol=POS_TOL
        )

    def test_partially_inserted_single_tube(self):

        l1, lc, l3, kappa = 0.05, 0.04, 0.03, 20.0

        tube = Tube.from_lengths(
            "t", l1, lc, l3, kappa, 2e-3, 0.0, nylon
        )

        # beta = 0.06: tube spans s in [-0.06, 0.06]; the curve
        # occupies s in [-0.01, 0.03]; straight tail [0.03, 0.06].
        robot = ConcentricTubeRobot(
            [tube], CTRState([0.06], [0.0])
        )

        arc = 0.03
        phi = kappa * arc

        expected = np.array([
            0.0,
            -((1 - np.cos(phi)) / kappa + l3 * np.sin(phi)),
            np.sin(phi) / kappa + l3 * np.cos(phi),
        ])

        tip = tip_position(solve_forward_kinematics_v2(robot))

        np.testing.assert_allclose(tip, expected, atol=POS_TOL)

    @pytest.mark.parametrize(
        "insertions",
        [
            [0.10, 0.13, 0.17],
            [0.05, 0.12, 0.18],
            [0.14, 0.15, 0.18],
        ],
    )
    def test_three_tubes_equal_rotation_match_reference(
        self,
        insertions,
    ):

        robot = ConcentricTubeRobot(
            three_piecewise_tubes(),
            CTRState(insertions, [0.4, 0.4, 0.4]),
        )

        tip = tip_position(solve_forward_kinematics_v2(robot))

        np.testing.assert_allclose(
            tip,
            reference_tip_equal_rotations(robot),
            atol=POS_TOL,
        )

    def test_curve_length_changes_the_shape(self):

        short = Tube.from_lengths(
            "t", 0.05, 0.02, 0.05, 20.0, 2e-3, 0.0, nylon
        )
        long = Tube.from_lengths(
            "t", 0.05, 0.05, 0.02, 20.0, 2e-3, 0.0, nylon
        )

        tips = []

        for tube in (short, long):
            robot = ConcentricTubeRobot(
                [tube], CTRState([0.12], [0.0])
            )
            tips.append(
                tip_position(solve_forward_kinematics_v2(robot))
            )

        assert abs(tips[0][1]) < abs(tips[1][1])

    def test_legacy_uniform_tube_equals_explicit_full_curve(self):

        legacy = Tube("t", 0.12, 20.0, 2e-3, 0.0, nylon)
        explicit = Tube.from_lengths(
            "t", 0.0, 0.12, 0.0, 20.0, 2e-3, 0.0, nylon
        )

        tips = []

        for tube in (legacy, explicit):
            robot = ConcentricTubeRobot(
                [tube], CTRState([0.10], [0.3])
            )
            tips.append(
                tip_position(solve_forward_kinematics_v2(robot))
            )

        np.testing.assert_array_equal(tips[0], tips[1])

    def test_straight_proximal_section_is_a_pure_offset(self):
        """
        Lengthening the straight section in front of the curve moves
        the curve distally without changing its shape: the tip of a
        single tube moves along z by the added length.
        """

        tips = []

        for l1 in (0.03, 0.05):
            tube = Tube.from_lengths(
                "t", l1, 0.04, 0.03, 20.0, 2e-3, 0.0, nylon
            )
            robot = ConcentricTubeRobot(
                [tube], CTRState([tube.length], [0.0])
            )
            tips.append(
                tip_position(solve_forward_kinematics_v2(robot))
            )

        np.testing.assert_allclose(
            tips[1] - tips[0], [0.0, 0.0, 0.02], atol=POS_TOL
        )


# --------------------------------------------------------------
# Torsion
# --------------------------------------------------------------

class TestTorsion:

    @pytest.mark.parametrize(
        "rotations",
        [
            [0.5, -1.0, 2.0],
            [3.0, 0.0, -3.0],
            [0.0, 1.5, 0.0],
        ],
    )
    def test_free_tip_condition_is_satisfied(self, rotations):

        robot = ConcentricTubeRobot(
            three_piecewise_tubes(),
            CTRState([0.10, 0.14, 0.18], rotations),
        )

        bb = solve_forward_kinematics_v2(robot)

        residual = torsion_shooting_residual(
            bb.torsion_solution.base_theta_dot, robot
        )

        assert np.abs(residual).max() < 1e-6

    def test_torsion_is_continuous_across_curve_boundaries(self):

        robot = ConcentricTubeRobot(
            three_piecewise_tubes(),
            CTRState([0.10, 0.14, 0.18], [0.5, -1.0, 2.0]),
        )

        bb = solve_forward_kinematics_v2(robot)

        solutions = bb.torsion_solution.segments

        for left, right in zip(solutions[:-1], solutions[1:]):

            y_end = left.ivp_solution.y[:, -1]
            y_start = right.ivp_solution.y[:, 0]

            for k, tube in enumerate(right.segment.active_tubes):

                j = left.segment.active_tubes.index(tube)

                # theta and theta' both continuous for a tube that
                # continues across the boundary.
                np.testing.assert_allclose(
                    y_start[2 * k: 2 * k + 2],
                    y_end[2 * j: 2 * j + 2],
                    atol=1e-12,
                )

    def test_tube_is_rigid_in_its_distal_straight_section(self):
        """
        A tube that is straight at its tip carries no twisting
        moment there, so the free-tip condition forces it not to
        twist along the straight tail: theta is constant over it.
        """

        robot = ConcentricTubeRobot(
            three_piecewise_tubes(),
            CTRState([0.10, 0.14, 0.18], [0.5, -1.0, 2.0]),
        )

        bb = solve_forward_kinematics_v2(robot)

        inner_tail_start = 0.18 - 0.040

        for sol in bb.torsion_solution.segments:

            seg = sol.segment

            if seg.start < inner_tail_start - 1e-9:
                continue

            inner_index = seg.active_tubes.index(
                robot.tubes[2]
            )

            theta_dot = sol.ivp_solution.y[2 * inner_index + 1]

            # The shooting solve uses scipy's default ODE tolerances,
            # so theta' at the tip is zero to roughly 1e-5 rad/m.
            np.testing.assert_allclose(theta_dot, 0.0, atol=1e-4)


# --------------------------------------------------------------
# Control path
# --------------------------------------------------------------

class TestControlPath:

    def test_jacobian_is_finite_and_full_rank(self):

        robot = ConcentricTubeRobot(
            three_piecewise_tubes(),
            CTRState([0.10, 0.14, 0.18], [0.5, -1.0, 2.0]),
        )

        J = numerical_position_jacobian_v2(robot)

        assert J.shape == (3, 6)
        assert np.all(np.isfinite(J))
        assert np.linalg.matrix_rank(J) == 3

    def test_resolved_rate_step_moves_tip_as_commanded(self):

        # Interior configuration: no insertion limit is active, so the
        # constrained step equals the unconstrained one.
        robot = ConcentricTubeRobot(
            three_piecewise_tubes(),
            CTRState([0.09, 0.12, 0.15], [0.5, -1.0, 2.0]),
        )

        before = tip_position(solve_forward_kinematics_v2(robot))

        dx = np.array([0.0, 0.0, 1e-3])

        # Same sequence as the ROS simulator: resolved-rate step,
        # then insertion constraints.
        dq = constrain_joint_step(
            robot,
            resolved_rate_step_v2(robot, dx),
        )

        n = len(robot.tubes)

        robot.state.insertions = (
            np.array(robot.state.insertions) + dq[:n]
        ).tolist()
        robot.state.rotations = (
            np.array(robot.state.rotations) + dq[n:]
        ).tolist()

        robot.validate_configuration()

        after = tip_position(solve_forward_kinematics_v2(robot))

        # Linearization error only: within 10 % of the commanded step.
        assert np.linalg.norm((after - before) - dx) < (
            0.1 * np.linalg.norm(dx)
        )


# --------------------------------------------------------------
# V1 pipeline guard
# --------------------------------------------------------------

def test_v1_pipeline_rejects_piecewise_tubes():

    robot = ConcentricTubeRobot(
        three_piecewise_tubes(),
        CTRState([0.10, 0.14, 0.18], [0.0, 0.0, 0.0]),
    )

    with pytest.raises(NotImplementedError):
        solve_forward_kinematics(robot)
