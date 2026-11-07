import numpy as np

from ctr_sim.mechanics.forward_v2 import (
    solve_forward_kinematics_v2,
)

from ctr_sim.mechanics.sampling import (
    sample_backbone,
    tip_position,
)

from ctr_sim.mechanics.torsion_v2 import (
    solve_torsion_v2,
)



def test_forward_kinematics_v2(forward_robot):

    backbone = solve_forward_kinematics_v2(
        forward_robot,
    )

    #
    # We should have one solved segment
    # for every geometric segment.
    #
    assert len(backbone.segments) > 0

    #
    # Inspect the first solved segment.
    #
    segment = backbone.segments[0]

    assert segment.theta.shape == (
        len(forward_robot.tubes),
    )

    assert segment.curvature.shape == (3,)

    #
    # The segment should contain the dense IVP solution.
    #
    assert segment.ivp_solution is not None

    #
    # The final state should be finite.
    #
    assert np.all(np.isfinite(segment.ivp_solution.y))

    #
    # Dense output should be available.
    #
    assert segment.ivp_solution.sol is not None


def test_tip_position_matches_sampled_backbone(
    forward_robot,
):

    backbone = solve_forward_kinematics_v2(
        forward_robot,
    )

    samples = sample_backbone(
        backbone,
        ds=1e-3,
    )

    direct_tip = tip_position(
        backbone
    )

    sampled_tip = (
        samples.position[-1]
    )

    assert np.allclose(
        direct_tip,
        sampled_tip,
        atol=1e-9,
    )


def test_forward_kinematics_v2_with_torsion_initial_guess(
    forward_robot,
):

    #
    # First solve normally.
    #
    backbone_1 = solve_forward_kinematics_v2(
        forward_robot,
    )

    assert backbone_1.torsion_solution is not None

    #
    # Solve forward mechanics again using the
    # torsion solution as a warm start.
    #
    backbone_2 = solve_forward_kinematics_v2(
        forward_robot,
        torsion_initial_guess=(
            backbone_1
            .torsion_solution
            .base_theta_dot
        ),
    )

    #
    # Both solves should produce the same segment
    # curvatures.
    #
    assert len(
        backbone_1.segments
    ) == len(
        backbone_2.segments
    )

    for segment_1, segment_2 in zip(
        backbone_1.segments,
        backbone_2.segments,
    ):

        assert np.allclose(
            segment_1.curvature,
            segment_2.curvature,
            atol=1e-6,
        )
