import numpy as np

from ctr_sim.control.jacobian import (
    numerical_position_jacobian_v2,
)

from ctr_sim.mechanics.forward_v2 import (
    solve_forward_kinematics_v2,
)


def test_position_jacobian_v2_shape(
    forward_robot,
):

    J = numerical_position_jacobian_v2(
        forward_robot,
    )

    assert J.shape == (
        3,
        2 * len(forward_robot.tubes),
    )

    assert np.all(
        np.isfinite(J)
    )

    print()
    print(np.round(J, 3))


def test_position_jacobian_v2_with_torsion_initial_guess(
    forward_robot,
):

    backbone = solve_forward_kinematics_v2(
        forward_robot,
    )

    torsion_initial_guess = (
        backbone
        .torsion_solution
        .base_theta_dot
    )

    J_cold = numerical_position_jacobian_v2(
        forward_robot,
    )

    J_warm = numerical_position_jacobian_v2(
        forward_robot,
        torsion_initial_guess=(
            torsion_initial_guess
        ),
    )

    assert np.allclose(
        J_warm,
        J_cold,
        atol=1e-6,
    )