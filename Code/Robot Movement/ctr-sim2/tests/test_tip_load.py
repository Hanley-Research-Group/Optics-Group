"""
Tests for forward mechanics with a force at the robot tip.

Convention: in this code base a positive body-frame curvature about x
bends the backbone toward -y, and the robot extends from the base
along +z.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)
from ctr_sim.kinematics.intervals import backbone_segments
from ctr_sim.mechanics.forward_v2 import solve_forward_kinematics_v2
from ctr_sim.mechanics.sampling import sample_backbone, tip_position
from ctr_sim.mechanics.loaded import tip_weight, STANDARD_GRAVITY
import ctr_sim.mechanics.loaded as loaded
from ctr_sim.control.jacobian import numerical_position_jacobian_v2
from ctr_sim.control.resolved_rate import resolved_rate_step_v2
from ctr_sim.control.constraints import constrain_joint_step

nylon = Material("Nylon 12", 1.3e9, 1.3e9 / 2.8)


def tubes():
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


def robot(insertions=(0.10, 0.14, 0.175), rotations=(0.5, -1.0, 2.0)):
    return ConcentricTubeRobot(
        tubes(), CTRState(list(insertions), list(rotations))
    )


def tip(r, force, guess=None):
    return tip_position(
        solve_forward_kinematics_v2(r, guess, tip_force=force)
    )


def rot_z(a):
    return np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a), np.cos(a), 0.0],
        [0.0, 0.0, 1.0],
    ])


def planar_reference_tip(r, Fy, Fz):
    """
    Independent planar elastica for all rotations zero and a load in
    the y-z plane, by scalar shooting on the base moment. Uses its own
    breakpoints from the tube geometry (not backbone_segments) and no
    rotation matrices:

        psi' = (sum EI kappa + m_x) / sum EI
        m_x' = sin(psi) Fz + cos(psi) Fy
        y'   = -sin(psi),   z' = cos(psi),   m_x(tip) = 0
    """

    L = max(r.state.insertions)
    marks = {0.0, L}

    for t, b in zip(r.tubes, r.state.insertions):
        for x in (
            b,
            b - t.length + t.length_before_curve,
            b - t.length + t.length_before_curve + t.curve_length,
        ):
            if 0 < x < L:
                marks.add(x)

    marks = sorted(marks)

    def props(s):
        num = den = 0.0
        for t, b in zip(r.tubes, r.state.insertions):
            sigma = s - (b - t.length)
            if 0 <= sigma <= t.length:
                num += t.EI * t.curvature_at(sigma)
                den += t.EI
        return num, den

    def run(m0):
        Y = np.array([0.0, m0, 0.0, 0.0])
        for a, b in zip(marks[:-1], marks[1:]):
            num, den = props(0.5 * (a + b))
            Y = solve_ivp(
                lambda s, Y: [
                    (num + Y[1]) / den,
                    np.sin(Y[0]) * Fz + np.cos(Y[0]) * Fy,
                    -np.sin(Y[0]),
                    np.cos(Y[0]),
                ],
                (a, b), Y, rtol=1e-11, atol=1e-13,
            ).y[:, -1]
        return Y

    scale = (abs(Fy) + abs(Fz)) * L
    m0 = root_scalar(
        lambda m: run(m)[1], x0=0.0, x1=0.1 * scale, xtol=1e-14
    ).root

    Y = run(m0)

    return np.array([0.0, Y[2], Y[3]])


# --------------------------------------------------------------
# tip_weight
# --------------------------------------------------------------

def test_tip_weight_magnitude_and_direction():

    F = tip_weight(0.02, (0, 0, -5))

    np.testing.assert_allclose(F, [0, 0, -0.02 * STANDARD_GRAVITY])

    F = tip_weight(0.01, (1, -1, 0), g=10.0)

    np.testing.assert_allclose(np.linalg.norm(F), 0.1)
    np.testing.assert_allclose(F / np.linalg.norm(F), [0.5**0.5, -0.5**0.5, 0])


@pytest.mark.parametrize(
    "mass, direction",
    [(0.01, (0, 0, 0)), (-0.01, (0, 0, -1)), (0.01, (0, -1))],
)
def test_tip_weight_rejects_bad_input(mass, direction):

    with pytest.raises(ValueError):
        tip_weight(mass, direction)


def test_bad_tip_force_rejected():

    with pytest.raises(ValueError):
        solve_forward_kinematics_v2(robot(), tip_force=[0.1, 0.0])


# --------------------------------------------------------------
# No load
# --------------------------------------------------------------

@pytest.mark.parametrize("force", [None, [0.0, 0.0, 0.0]])
def test_no_load_is_the_unloaded_solver(force):

    r = robot()

    np.testing.assert_array_equal(
        tip(r, force),
        tip_position(solve_forward_kinematics_v2(r)),
    )


def test_loaded_equations_with_zero_force_match_unloaded_model():
    """
    The loaded solver forced to run with F = 0 must reproduce the
    unloaded model. With equal rotations there is no torsion, so the
    unloaded solver is exact and the two agree tightly.
    """

    r = robot(rotations=(0.4, 0.4, 0.4))
    zero = np.zeros(3)

    x, J = loaded.solve_loaded_shooting(r, zero)
    bb = loaded._build_backbone(r, zero, x, J)

    np.testing.assert_allclose(
        tip_position(bb),
        tip_position(solve_forward_kinematics_v2(r)),
        atol=1e-6,
    )


# --------------------------------------------------------------
# Against independent solutions
# --------------------------------------------------------------

@pytest.mark.parametrize("beta", [0.15, 0.10])
def test_small_load_matches_cantilever_formula(beta):

    straight = Tube("s", 0.15, 0.0, 2e-3, 1.2e-3, nylon)
    r = ConcentricTubeRobot([straight], CTRState([beta], [0.0]))

    F = 1e-4
    expected = F * beta**3 / (3 * straight.EI)

    tip_x = tip(r, [F, 0.0, 0.0])[0]

    assert tip_x == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize(
    "insertions, force",
    [
        ((0.10, 0.14, 0.18), (0.0, 0.15, 0.0)),
        ((0.05, 0.12, 0.18), (0.0, -0.05, -0.15)),
    ],
)
def test_large_planar_load_matches_independent_elastica(
    insertions,
    force,
):

    r = robot(insertions, (0.0, 0.0, 0.0))

    loaded_tip = tip(r, force)

    # Large deflection: the load moves the tip by centimetres.
    assert np.linalg.norm(loaded_tip - tip(r, None)) > 0.01

    np.testing.assert_allclose(
        loaded_tip,
        planar_reference_tip(r, force[1], force[2]),
        atol=1e-8,
    )


def test_compliance_is_symmetric():
    """
    Maxwell-Betti reciprocity: for a conservative system the tip
    compliance dp/dF is symmetric. With unequal rotations the tubes
    twist, so this checks the coupling of torsion and bending under
    load in 3D.
    """

    r = robot()
    F0 = np.array([0.03, -0.08, -0.05])
    h = 2e-4

    C = np.column_stack([
        (tip(r, F0 + h * e) - tip(r, F0 - h * e)) / (2 * h)
        for e in np.eye(3)
    ])

    assert np.linalg.norm(C - C.T) < 1e-5 * np.linalg.norm(C)


def test_rotating_tubes_and_load_together_rotates_the_robot():

    F = np.array([0.03, -0.08, -0.05])
    d = 0.9

    before = tip(robot(), F)
    after = tip(robot(rotations=(0.5 + d, -1.0 + d, 2.0 + d)), rot_z(d) @ F)

    np.testing.assert_allclose(after, rot_z(d) @ before, atol=1e-9)


# --------------------------------------------------------------
# Solution object
# --------------------------------------------------------------

def test_loaded_solution_satisfies_boundary_conditions():

    r = robot()
    F = tip_weight(0.02, (0.3, -1, -0.5))

    bb = solve_forward_kinematics_v2(r, tip_force=F)

    residual, _ = loaded._integrate(
        r,
        backbone_segments(r),
        F,
        bb.torsion_solution.shooting_vector,
        dense=False,
    )

    assert np.abs(residual).max() < 1e-7


def test_loaded_backbone_works_with_sampling():

    r = robot()
    bb = solve_forward_kinematics_v2(r, tip_force=tip_weight(0.01, (0, -1, 0)))

    samples = sample_backbone(bb, ds=1e-3)

    np.testing.assert_allclose(
        samples.position[-1], tip_position(bb), atol=1e-12
    )

    for R in samples.rotation:
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)

    assert bb.torsion_solution.base_moment.shape == (3,)
    assert len(bb.torsion_solution.warm_start) == len(r.tubes) + 2


# --------------------------------------------------------------
# Warm starts
# --------------------------------------------------------------

def test_warm_start_from_torsion_solution(monkeypatch):

    F = tip_weight(0.01, (0, -1, 0))
    r = robot()

    previous = solve_forward_kinematics_v2(r, tip_force=F)

    r.state.insertions[1] += 5e-4
    r.state.rotations[0] += 0.01

    count = {"n": 0}
    integrate = loaded._integrate

    def counting(*args, **kwargs):
        count["n"] += 1
        return integrate(*args, **kwargs)

    monkeypatch.setattr(loaded, "_integrate", counting)

    warm = tip(r, F, previous.torsion_solution)
    warm_integrations = count["n"]

    count["n"] = 0
    cold = tip(r, F)
    cold_integrations = count["n"]

    np.testing.assert_allclose(warm, cold, atol=1e-9)

    assert warm_integrations < cold_integrations / 2


def test_warm_start_arrays_are_accepted_either_way():

    F = tip_weight(0.01, (0, -1, 0))
    r = robot()

    unloaded = solve_forward_kinematics_v2(r)
    loaded_bb = solve_forward_kinematics_v2(r, tip_force=F)

    # Unloaded warm start into a loaded solve, and loaded into unloaded.
    np.testing.assert_allclose(
        tip(r, F, unloaded.torsion_solution.warm_start),
        tip_position(loaded_bb),
        atol=1e-9,
    )
    # A different starting guess converges to the same solution, up to
    # round-off.
    np.testing.assert_allclose(
        tip(r, None, loaded_bb.torsion_solution.warm_start),
        tip_position(unloaded),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tip(r, None, loaded_bb.torsion_solution),
        tip_position(unloaded),
        atol=1e-12,
    )


# --------------------------------------------------------------
# Control path
# --------------------------------------------------------------

def test_jacobian_under_load_matches_central_differences():

    F = tip_weight(0.01, (0, -1, 0))
    q = np.array([0.10, 0.1405, 0.175, 0.51, -1.0, 2.0])

    def tip_q(q):
        r = robot(q[:3], q[3:])
        return tip(r, F)

    J_central = np.column_stack([
        (tip_q(q + 1e-6 * e) - tip_q(q - 1e-6 * e)) / 2e-6
        for e in np.eye(6)
    ])

    J = numerical_position_jacobian_v2(
        robot(q[:3], q[3:]),
        delta_insertion=1e-6,
        delta_rotation=1e-6,
        tip_force=F,
    )

    np.testing.assert_allclose(J, J_central, atol=1e-4)

    J_unloaded = numerical_position_jacobian_v2(robot(q[:3], q[3:]))

    # The load changes the Jacobian substantially.
    assert np.abs(J - J_unloaded).max() > 0.05


def test_resolved_rate_step_under_load_moves_tip_as_commanded():

    F = tip_weight(0.015, (0, -1, 0))
    r = robot((0.09, 0.12, 0.15))

    bb = solve_forward_kinematics_v2(r, tip_force=F)
    before = tip_position(bb)

    J = numerical_position_jacobian_v2(
        r, torsion_initial_guess=bb.torsion_solution, tip_force=F
    )

    dx = np.array([4e-4, 3e-4, 0.0])

    dq = constrain_joint_step(
        r, resolved_rate_step_v2(r, dx, jacobian=J, tip_force=F)
    )

    n = len(r.tubes)
    r.state.insertions = list(np.array(r.state.insertions) + dq[:n])
    r.state.rotations = list(np.array(r.state.rotations) + dq[n:])

    after = tip(r, F, bb.torsion_solution)

    assert np.linalg.norm((after - before) - dx) < 0.1 * np.linalg.norm(dx)
