import numpy as np

from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.control.jacobian import (
    numerical_position_jacobian,
    numerical_position_jacobian_v2,
)


def resolved_rate_step(
    robot: ConcentricTubeRobot,
    dx: np.ndarray,
) -> np.ndarray:
    """
    Compute a joint-space increment that best achieves a desired
    Cartesian position increment.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Current robot configuration.

    dx : np.ndarray
        Desired Cartesian position increment with shape (3,).

    Returns
    -------
    np.ndarray
        Joint increment vector with shape (2 * n_tubes,).

        The entries are ordered as

            [Δβ₁ Δβ₂ ... Δβₙ Δα₁ Δα₂ ... Δαₙ]
    """

    J = numerical_position_jacobian(robot)

    gain = 0.1

    dq = gain * np.linalg.pinv(J) @ dx

    return dq

def resolved_rate_step_v2(
    robot: ConcentricTubeRobot,
    dx: np.ndarray,
    torsion_initial_guess: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    tip_force=None,
) -> np.ndarray:
    """
    Compute a joint-space increment that best achieves a desired
    Cartesian position increment using the Mechanics V2 pipeline.

    If a Jacobian is supplied, reuse it instead of recomputing one
    (in that case it must have been computed with the same
    tip_force).

    tip_force : array-like, shape (3,), optional
        Force (N) at the robot tip in the base frame.
    """

    if jacobian is None:

        J = numerical_position_jacobian_v2(
            robot,
            torsion_initial_guess=torsion_initial_guess,
            tip_force=tip_force,
        )
    else:
        J = jacobian

    gain = 1.0

    dq = gain * np.linalg.pinv(J) @ dx

    return dq