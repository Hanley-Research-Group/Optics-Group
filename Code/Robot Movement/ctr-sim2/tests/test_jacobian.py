import numpy as np

from ctr_sim.control import numerical_position_jacobian


def test_position_jacobian_shape(forward_robot):

    J = numerical_position_jacobian(forward_robot)

    assert J.shape == (3, 6)
    assert np.all(np.isfinite(J))