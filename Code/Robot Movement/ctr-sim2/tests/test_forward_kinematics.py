import numpy as np

from ctr_sim.mechanics import solve_forward_kinematics


def test_forward_kinematics(forward_robot):

    solution = solve_forward_kinematics(forward_robot)

    assert solution.tip_position.shape == (3,)

    assert solution.tip_rotation.shape == (3, 3)

    assert solution.backbone.position.shape[1] == 3

    assert solution.backbone.rotation.shape[1:] == (3, 3)

    assert np.allclose(
        solution.backbone.position[0],
        [0.0, 0.0, 0.0],
    )

    assert np.allclose(
        solution.backbone.rotation[0],
        np.eye(3),
    )