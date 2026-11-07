from copy import deepcopy

import numpy as np

from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.mechanics import solve_forward_kinematics

from ctr_sim.mechanics.forward_v2 import (
    solve_forward_kinematics_v2,
)
from ctr_sim.mechanics.sampling import (
    tip_position,
)




def numerical_position_jacobian(
    robot: ConcentricTubeRobot,
    delta_insertion: float = 1e-4,
    delta_rotation: float = 1e-4,
) -> np.ndarray:
    """
    Compute the numerical position Jacobian using forward finite differences.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Robot configuration about which the Jacobian is evaluated.
    delta_insertion : float, optional
        Perturbation applied to tube insertions (m).
    delta_rotation : float, optional
        Perturbation applied to tube rotations (rad).

    Returns
    -------
    np.ndarray
        Position Jacobian with shape (3, 2 * n_tubes).

        The columns are ordered as

            [β₁ β₂ ... βₙ α₁ α₂ ... αₙ]
    """

    n = len(robot.tubes)

    J = np.zeros((3, 2 * n))

    # Nominal tip position
    x0 = solve_forward_kinematics(
        robot,
        ds=1e-4,
    ).tip_position


    #
    # TODO:
    # Insertion derivatives currently depend on the sampled backbone
    # representation used by the forward solver. A future segment-aware
    # mechanics implementation should improve these derivatives.
    #

    #
    # Insertion columns
    #
    for i in range(n):

        robot_plus = deepcopy(robot)

        robot_plus.state.insertions[i] += delta_insertion

        x_plus = solve_forward_kinematics(
            robot_plus,
            ds=1e-4,
        ).tip_position

        J[:, i] = (x_plus - x0) / delta_insertion

    #
    # Rotation columns
    #
    for i in range(n):

        robot_plus = deepcopy(robot)

        robot_plus.state.rotations[i] += delta_rotation

        x_plus = solve_forward_kinematics(
            robot_plus,
            ds=1e-4,
        ).tip_position

        J[:, n + i] = (x_plus - x0) / delta_rotation

    return J


def numerical_position_jacobian_v2(
    robot: ConcentricTubeRobot,
    delta_insertion: float = 1e-4,
    delta_rotation: float = 1e-4,
    torsion_initial_guess: np.ndarray | None = None,
    tip_force=None,
) -> np.ndarray:

    """
    Compute the numerical position Jacobian using finite differences.

    Insertion derivatives use forward differences when possible and
    backward differences at the maximum insertion limit.
    Rotation derivatives use forward differences.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Robot configuration about which the Jacobian is evaluated.
    delta_insertion : float, optional
        Perturbation applied to tube insertions (m).
    delta_rotation : float, optional
        Perturbation applied to tube rotations (rad).
    torsion_initial_guess : np.ndarray, optional
        Warm start for the nominal configuration, e.g.
        ``backbone.torsion_solution``.
    tip_force : array-like, shape (3,), optional
        Force (N) at the robot tip in the base frame. Must match
        the load used for the forward solves, otherwise the
        Jacobian describes a different robot.

    Returns
    -------
    np.ndarray
        Position Jacobian with shape (3, 2 * n_tubes).

        The columns are ordered as

            [β₁ β₂ ... βₙ α₁ α₂ ... αₙ]
    """

    n = len(robot.tubes)

    J = np.zeros((3, 2 * n))
  
    backbone = solve_forward_kinematics_v2(
        robot,
        torsion_initial_guess=torsion_initial_guess,
        tip_force=tip_force,
    )

    #
    # Reuse the nominal torsion solution as the
    # initial guess for all Jacobian perturbations.
    #
    nominal_torsion_guess = (
        backbone
        .torsion_solution
    )


    #
    # Evaluate the nominal tip position directly from
    # the final segment's continuous IVP solution.
    #
    x0 = tip_position(
        backbone
    )


    #
    # Insertion columns
    #
    for i in range(n):

        beta_i = robot.state.insertions[i]
        tube_length = robot.tubes[i].length

        #
        # Use a forward difference whenever the positive
        # perturbation remains physically valid.
        #
        if beta_i + delta_insertion <= tube_length:

            robot_plus = deepcopy(robot)

            robot_plus.state.insertions[i] += (
                delta_insertion
            )

            backbone = solve_forward_kinematics_v2(
                robot_plus,
                torsion_initial_guess=nominal_torsion_guess,
                tip_force=tip_force,
            )


            x_plus = tip_position(
                backbone
            )


            J[:, i] = (
                x_plus - x0
            ) / delta_insertion

        else:

            #
            # The tube is at its maximum insertion.
            #
            # A positive perturbation would move the tube's
            # proximal end beyond the robot base (s > 0),
            # so use a backward difference instead.
            #
            robot_minus = deepcopy(robot)

            robot_minus.state.insertions[i] -= (
                delta_insertion
            )

            backbone = solve_forward_kinematics_v2(
                robot_minus,
                torsion_initial_guess=nominal_torsion_guess,
                tip_force=tip_force,
            )

            x_minus = tip_position(
                backbone
            )

            J[:, i] = (
                x0 - x_minus
            ) / delta_insertion

    #
    # Rotation columns
    #
    for i in range(n):

        robot_plus = deepcopy(robot)

        robot_plus.state.rotations[i] += delta_rotation

        backbone = solve_forward_kinematics_v2(
            robot_plus,
            torsion_initial_guess=nominal_torsion_guess,
            tip_force=tip_force,
        )

        x_plus = tip_position(
            backbone
        )

        J[:, n + i] = (x_plus - x0) / delta_rotation

    #
    # Mechanics V2 computes the backbone independently of the
    # sampling resolution. The Jacobian samples the continuous
    # backbone only after the mechanics solve.
    #

    return J

