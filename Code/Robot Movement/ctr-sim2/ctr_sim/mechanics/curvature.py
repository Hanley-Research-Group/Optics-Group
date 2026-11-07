import numpy as np
from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.backbone import Backbone
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from ctr_sim.segment import Segment
from ctr_sim.kinematics.intervals import backbone_segments


def resultant_curvature_sampled(
    robot: ConcentricTubeRobot,
    theta: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """
    Compute the resultant backbone curvature.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Robot configuration.

    theta : np.ndarray
        Tube torsion angles θ(s).

    s : np.ndarray
        Arc-length coordinates.

    Returns
    -------
    np.ndarray
        Body-frame curvature vector u(s).
    """

    n = len(robot.tubes)
    n_points = len(s)

    u = np.zeros((3, n_points))

    for k in range(n_points):

        s_k = s[k]

        sum_EI = 0.0

        kx = 0.0
        ky = 0.0

        for i in range(n):

            tube = robot.tubes[i]
            beta = robot.state.insertions[i]

            if s_k > beta:
                continue

            kappa = tube.precurvature
            theta_i = theta[i, k]

            kx += (tube.EI * kappa * np.cos(theta_i))
            ky += (tube.EI * kappa * np.sin(theta_i))
            sum_EI += tube.EI

        if sum_EI > 0:
            u[0, k] = kx / sum_EI
            u[1, k] = ky / sum_EI

    return u


def segment_curvature(
    robot: ConcentricTubeRobot,
    theta: np.ndarray,
    segment: Segment,
) -> np.ndarray:
    """
    Compute the resultant curvature over a single backbone segment.

    The torsion angles are assumed constant over the segment.
    """

    kx = 0.0
    ky = 0.0
    sum_EI = 0.0

    for tube in segment.active_tubes:

        #
        # Find the index of this tube in the robot.
        #
        i = robot.tubes.index(tube)

        kappa = segment.curvatures[
            segment.active_tubes.index(tube)
        ]
        theta_i = theta[i]

        kx += tube.EI * kappa * np.cos(theta_i)
        ky += tube.EI * kappa * np.sin(theta_i)

        sum_EI += tube.EI

    if sum_EI == 0.0:
        return np.zeros(3)

    return np.array([
        kx / sum_EI,
        ky / sum_EI,
        0.0,
    ])


def segment_curvature_v2(
    theta: np.ndarray,
    segment: Segment,
) -> np.ndarray:
    """
    Compute the resultant curvature over a single backbone segment
    using segment-local torsion angles.

    The entries of theta correspond directly to the tubes in
    segment.active_tubes.
    """

    if len(theta) != len(segment.active_tubes):
        raise ValueError(
            "theta must contain one torsion angle per active tube."
        )

    kx = 0.0
    ky = 0.0
    sum_EI = 0.0

    for i, tube in enumerate(
        segment.active_tubes
    ):

        # Precurvature of this tube within the segment (zero where the
        # tube is straight). Straight tubes still add to sum_EI.
        kappa = segment.curvatures[i]

        theta_i = theta[i]

        kx += (
            tube.EI
            * kappa
            * np.cos(theta_i)
        )

        ky += (
            tube.EI
            * kappa
            * np.sin(theta_i)
        )

        sum_EI += tube.EI

    curvature = np.zeros(3)

    if sum_EI > 0.0:

        curvature[0] = (
            kx / sum_EI
        )

        curvature[1] = (
            ky / sum_EI
        )

    return curvature