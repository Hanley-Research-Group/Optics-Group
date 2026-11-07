import numpy as np
from ctr_sim.robot import ConcentricTubeRobot



def constrain_beta_step(
    beta_max: float,
    current_beta: float,
    delta_beta: float,
) -> float:
    """
    Constrain a requested tube insertion increment.

    The resulting insertion is guaranteed to satisfy

        0 <= beta <= beta_max

    Parameters
    ----------
    beta_max : float
        Maximum allowed insertion, equal to the tube length.

    current_beta : float
        Current tube insertion.

    delta_beta : float
        Requested insertion increment.

    Returns
    -------
    float
        Constrained insertion increment.
    """

    min_step = -current_beta

    max_step = (
        beta_max - current_beta
    )

    if delta_beta > max_step:
        return max_step

    if delta_beta < min_step:
        return min_step

    return delta_beta

import numpy as np

from ctr_sim.robot import ConcentricTubeRobot


def constrain_joint_step(
    robot: ConcentricTubeRobot,
    dq: np.ndarray,
) -> np.ndarray:
    """
    Constrain a proposed joint-space increment.

    Insertion increments are limited so that every resulting
    insertion satisfies

        0 <= beta_i <= L_i

    Rotation increments are left unchanged.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Current robot configuration.

    dq : np.ndarray
        Proposed joint-space increment ordered as

            [delta_beta_1, ..., delta_beta_n,
             delta_alpha_1, ..., delta_alpha_n]

    Returns
    -------
    np.ndarray
        Constrained copy of the joint-space increment.
    """

    n = len(robot.tubes)

    if dq.shape != (2 * n,):
        raise ValueError(
            "dq must have shape (2 * n_tubes,)."
        )

    constrained_dq = dq.copy()

    for i in range(n):

        constrained_dq[i] = constrain_beta_step(
            beta_max=robot.tubes[i].length,
            current_beta=robot.state.insertions[i],
            delta_beta=dq[i],
        )

    return constrained_dq