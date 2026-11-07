import numpy as np

from ctr_sim.mechanics.torsion import (
    solve_torsion_bvp,
    evaluate_segment_torsion,
)
from ctr_sim.kinematics.intervals import backbone_segments


def test_evaluate_segment_torsion(forward_robot):

    segments = backbone_segments(forward_robot)

    bvp_solution = solve_torsion_bvp(forward_robot)

    theta = evaluate_segment_torsion(
        bvp_solution,
        segments[0],
    )

    assert theta.shape == (len(forward_robot.tubes),)

    assert np.all(np.isfinite(theta))