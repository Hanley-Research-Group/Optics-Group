"""
Backbone integration for concentric tube robots.
"""

import numpy as np
from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.backbone import Backbone
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from ctr_sim.segment import Segment
from ctr_sim.kinematics.intervals import backbone_segments
from ctr_sim.segment_solution import SegmentSolution



def resultant_curvature(
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


def pack_state(
    position: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Pack backbone state into a 12-element vector.
    """

    return np.concatenate((
        position,
        rotation.reshape(9),
    ))


def unpack_state(
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack a 12-element backbone state vector.
    """

    position = y[:3]

    rotation = y[3:].reshape((3, 3))

    return position, rotation



def integrate_backbone_sampled(
    u: np.ndarray,
    s: np.ndarray,
    robot: ConcentricTubeRobot,
) -> Backbone:

    """
    Integrate the backbone Cosserat equations.
    """
    
    # Create curvature interpolants.
    kx_interp = interp1d(
        s,
        u[0, :],
        kind="linear",
        fill_value="extrapolate",
    )

    ky_interp = interp1d(
        s,
        u[1, :],
        kind="linear",
        fill_value="extrapolate",
    )

    position0 = np.zeros(3)
    rotation0 = np.eye(3)
    y0 = pack_state(
        position0,
        rotation0,
    )

    solution = solve_ivp(
        fun=lambda s, y: backbone_ode(
            s,
            y,
            kx_interp,
            ky_interp,
        ),
        t_span=(s[0], s[-1]),
        y0=y0,
        t_eval=s,
    )

    position = solution.y[:3, :].T

    rotation = np.zeros((len(s), 3, 3))
    for i in range(len(s)):
        rotation[i] = solution.y[3:, i].reshape((3, 3))

    return Backbone(
        s=s,
        position=position,
        rotation=rotation,
    )

 
 

def backbone_ode(
    s: float,
    y: np.ndarray,
    kx_interp,
    ky_interp,
) -> np.ndarray:
    """
    Cosserat backbone differential equations.
    """

    position, rotation = unpack_state(y)
    kx = kx_interp(s)
    ky = ky_interp(s)
    u = np.array([kx, ky, 0.0])
    u_hat = np.array([
    [0.0,   -u[2],  u[1]],
    [u[2],   0.0,  -u[0]],
    [-u[1],  u[0],  0.0],
])
    dRds = rotation @ u_hat
    dpds = rotation[:, 2]

    return pack_state(dpds, dRds)
    



def integrate_segments(
    u: np.ndarray,
    s: np.ndarray,
    robot: ConcentricTubeRobot,
):
    """
    Temporary bridge implementation.

    Uses the existing sampled curvature to construct a
    segment-wise backbone solution.
    """

    segments = backbone_segments(robot)

    segment_solutions = []

    position = np.zeros(3)
    rotation = np.eye(3)

    for segment in segments:

        midpoint = 0.5 * (
            segment.start +
            segment.end
        )

        idx = np.argmin(
            np.abs(s - midpoint)
        )

        curvature = u[:, idx]

        # TODO:
        # Replace midpoint approximation with segment-wise torsion evaluation.
        theta = np.zeros(len(robot.tubes))

        ivp_solution, position1, rotation1 = integrate_segment(
            position,
            rotation,
            curvature,
            segment,
        )

        segment_solutions.append(
            SegmentSolution(
                segment=segment,
                theta=theta,
                curvature=curvature,
                ivp_solution=ivp_solution,
            )
        )

        position = position1
        rotation = rotation1

    backbone = Backbone(
        segments=segment_solutions,
    )

    return (
        backbone,
        position,
        rotation,
    )


def integrate_segment(
    position0: np.ndarray,
    rotation0: np.ndarray,
    curvature: np.ndarray,
    segment: Segment,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate a single backbone segment assuming constant curvature.
    """

    y0 = pack_state(
        position0,
        rotation0,
    )

    kx = curvature[0]
    ky = curvature[1]

    
    solution = solve_ivp(
        fun=lambda s, y: backbone_ode(
            s,
            y,
            lambda _: kx,
            lambda _: ky,
        ),
        t_span=(
            segment.start,
            segment.end,
        ),
        y0=y0,
        dense_output=True,
    )


    position1 = solution.y[:3, -1]

    rotation1 = solution.y[3:, -1].reshape((3, 3))

    return solution, position1, rotation1