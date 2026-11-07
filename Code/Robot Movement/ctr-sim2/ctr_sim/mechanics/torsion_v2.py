"""
Segment-aware torsion mechanics for concentric tube robots.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
# from scipy.optimize import least_squares
from copy import deepcopy

from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.segment import Segment
from ctr_sim.kinematics.intervals import backbone_segments

from ctr_sim.torsion_solution import (
    TorsionSolution,
)
from ctr_sim.torsion_segment_solution import (
    TorsionSegmentSolution,
)



def theta_index(i: int) -> int:
    """
    Index of theta_i in the segment torsion state vector.
    """
    return 2 * i


def theta_dot_index(i: int) -> int:
    """
    Index of dtheta_i/ds in the segment torsion state vector.
    """
    return 2 * i + 1


def segment_torsion_ode(
    s: float,
    y: np.ndarray,
    segment: Segment,
) -> np.ndarray:
    """
    Evaluate the torsion ODE within a single backbone segment.

    The active tube set and each tube's precurvature are constant
    within a segment, so no tube-termination or curve-boundary
    switching is required inside the ODE.

    Parameters
    ----------
    s : float
        Arc-length coordinate.

    y : np.ndarray
        Segment torsion state

            [theta_1, theta_1',
             theta_2, theta_2',
             ...]

        containing only the tubes active in this segment.

    segment : Segment
        Backbone segment whose active tubes define the torsion system.

    Returns
    -------
    np.ndarray
        State derivative dy/ds.
    """

    tubes = segment.active_tubes
    n = len(tubes)

    dyds = np.zeros_like(y)

    #
    # Total bending stiffness of the tubes present in this segment.
    # Straight tubes are included: they bend with the others.
    #
    sum_EI = sum(
        tube.EI
        for tube in tubes
    )

    for i in range(n):

        tube_i = tubes[i]

        theta_i = y[
            theta_index(i)
        ]

        theta_dot_i = y[
            theta_dot_index(i)
        ]

        dyds[
            theta_index(i)
        ] = theta_dot_i

        moment = 0.0

        # Precurvature of tube i within this segment. This is zero
        # where the tube is straight, even though the tube is active.
        kappa_i = segment.curvatures[i]

        for j in range(n):

            if j == i:
                continue

            tube_j = tubes[j]

            theta_j = y[
                theta_index(j)
            ]

            kappa_j = segment.curvatures[j]

            moment += (
                tube_j.EI
                * kappa_i
                * kappa_j
                * np.sin(
                    theta_i - theta_j
                )
            )

        #
        # Torsional equilibrium of tube i:
        #
        #   GJ_i theta_i'' = (EI_i / sum(EI))
        #                    * kappa_i
        #                    * sum_j EI_j kappa_j sin(theta_i - theta_j)
        #
        # The factor EI_i / sum(EI) is the share of the bending moment
        # carried by tube i; `moment` already contains kappa_i and
        # the sum over j.
        #
        dyds[
            theta_dot_index(i)
        ] = (
            tube_i.EI / sum_EI
            * moment / tube_i.GJ
        )

    return dyds


def map_torsion_state(
    y_from: np.ndarray,
    from_segment: Segment,
    to_segment: Segment,
) -> np.ndarray:
    """
    Map a torsion state from one segment to the next.

    Only tubes that remain active in the destination segment
    are carried across the interface. For each continuing tube,
    both theta and theta_dot are preserved.

    Parameters
    ----------
    y_from : np.ndarray
        Torsion state at the end of the source segment.

    from_segment : Segment
        Segment whose active tubes define y_from.

    to_segment : Segment
        Adjacent segment whose state is to be constructed.

    Returns
    -------
    np.ndarray
        Torsion state for the continuing tubes in to_segment.
    """

    y_to = np.zeros(
        2 * len(to_segment.active_tubes)
    )

    for j, tube in enumerate(
        to_segment.active_tubes
    ):

        if tube not in from_segment.active_tubes:
            raise ValueError(
                f"Tube {tube.name} is active in the destination "
                "segment but not in the source segment."
            )

        i = from_segment.active_tubes.index(
            tube
        )

        y_to[theta_index(j)] = (
            y_from[theta_index(i)]
        )

        y_to[theta_dot_index(j)] = (
            y_from[theta_dot_index(i)]
        )

    return y_to   

def torsion_shooting_residual(
    base_theta_dot: np.ndarray,
    robot: ConcentricTubeRobot,
) -> np.ndarray:
    """
    Compute free-tip torsional residuals for a trial set of
    base torsional strains.

    The torsion equations are integrated forward segment by
    segment. When a tube terminates, its distal theta_dot is
    recorded as a residual. Continuing tube states are mapped
    into the next segment.

    Parameters
    ----------
    base_theta_dot : np.ndarray
        Trial torsional strains at s = 0, one per tube.

    robot : ConcentricTubeRobot
        Robot configuration.

    Returns
    -------
    np.ndarray
        Free-tip torsional residuals, one per tube.

        For a valid torsion solution,

            residual[i] = theta_dot_i(beta_i) = 0
    """

    segments = backbone_segments(
        robot,
    )

    n_tubes = len(robot.tubes)

    if len(base_theta_dot) != n_tubes:
        raise ValueError(
            "base_theta_dot must contain one value per tube."
        )

    #
    # Construct the initial torsion state at s = 0.
    #
    # theta_i(0)     = alpha_i
    # theta_dot_i(0) = trial shooting variable
    #
    first_segment = segments[0]

    y = np.zeros(
        2 * len(first_segment.active_tubes)
    )

    for i, tube in enumerate(
        first_segment.active_tubes
    ):

        robot_index = robot.tubes.index(
            tube
        )

        y[theta_index(i)] = (
            robot.state.rotations[robot_index]
        )

        y[theta_dot_index(i)] = (
            base_theta_dot[robot_index]
        )

    #
    # One residual per physical tube.
    #
    residual = np.zeros(
        n_tubes
    )

    #
    # Integrate through the segment sequence.
    #
    for segment_index, segment in enumerate(
        segments
    ):

        solution = solve_ivp(
            fun=lambda s, y: segment_torsion_ode(
                s,
                y,
                segment,
            ),
            t_span=(
                segment.start,
                segment.end,
            ),
            y0=y,
        )

        if not solution.success:
            raise RuntimeError(
                "Segment torsion integration failed: "
                f"{solution.message}"
            )

        y_end = solution.y[:, -1]


        #
        # Determine which tubes terminate at this boundary.
        #
        if segment_index < len(segments) - 1:

            next_segment = segments[
                segment_index + 1
            ]

            terminating_tubes = [
                tube
                for tube in segment.active_tubes
                if tube not in next_segment.active_tubes
            ]

        else:

            next_segment = None

            # Every tube remaining in the final segment
            # terminates at its distal boundary.
            terminating_tubes = list(
                segment.active_tubes
            )

        #
        # Record theta_dot at each terminating tube's
        # physical distal end.
        #
        for tube in terminating_tubes:

            local_index = (
                segment.active_tubes.index(
                    tube
                )
            )

            robot_index = robot.tubes.index(
                tube
            )

            residual[robot_index] = y_end[
                theta_dot_index(
                    local_index
                )
            ]

        #
        # Carry continuing tubes into the next segment.
        #
        if next_segment is not None:

            y = map_torsion_state(
                y_end,
                segment,
                next_segment,
            )

    return residual

def solve_torsion_shooting(
    robot: ConcentricTubeRobot,
    initial_guess: np.ndarray | None = None,
):
    """
    Solve the segment-aware torsion problem using shooting.

    The unknown shooting variables are the torsional strains
    theta_dot_i(0) at the robot base. These are adjusted until
    every tube satisfies the free-tip condition

        theta_dot_i(beta_i) = 0.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Robot configuration.

    initial_guess : np.ndarray, optional
        Initial guess for the base torsional strains theta_dot_i(0).
        If omitted, zero torsional strain is used.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Root-finding result. The solved base torsional strains
        are stored in result.x.
    """

    n = len(robot.tubes)

    if initial_guess is None:

        initial_guess = np.zeros(
            n
        )

    else:

        initial_guess = np.asarray(
            initial_guess,
            dtype=float,
        )

        if initial_guess.shape != (n,):
            raise ValueError(
                "initial_guess must contain one "
                "base torsional strain per tube."
            )

    result = root(
        fun=lambda base_theta_dot:
            torsion_shooting_residual(
                base_theta_dot,
                robot,
            ),
        x0=initial_guess,
    )

    return result


def solve_torsion_shooting_continuation(
    robot: ConcentricTubeRobot,
    n_steps: int = 10,
):
    """
    Solve the segment-aware torsion problem using continuation
    in the commanded tube rotations.

    The robot rotations are gradually increased from zero to
    their requested values. The solved base torsional strains
    from each continuation step are used as the initial guess
    for the next step.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Target robot configuration.

    n_steps : int, optional
        Number of continuation steps between zero rotation and
        the requested robot rotations.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Shooting solution for the target robot configuration.
    """

    if n_steps < 1:
        raise ValueError(
            "n_steps must be at least 1."
        )

    target_rotations = np.asarray(
        robot.state.rotations,
        dtype=float,
    )

    continuation_robot = deepcopy(
        robot
    )

    base_theta_dot = np.zeros(
        len(robot.tubes)
    )

    result = None

    for step in range(
        1,
        n_steps + 1,
    ):

        fraction = (
            step / n_steps
        )

        continuation_robot.state.rotations = (
            fraction * target_rotations
        ).tolist()

        result = solve_torsion_shooting(
            continuation_robot,
            initial_guess=base_theta_dot,
        )

        if not result.success:
            raise RuntimeError(
                "Torsion continuation failed "
                f"at step {step}/{n_steps}: "
                f"{result.message}"
            )

        base_theta_dot = result.x

    return result


def integrate_torsion_segments(
    robot: ConcentricTubeRobot,
    base_theta_dot: np.ndarray,
) -> TorsionSolution:
    """
    Integrate the segment-aware torsion equations using solved
    base torsional strains.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Robot configuration.

    base_theta_dot : np.ndarray
        Solved torsional strains theta_dot_i(0).

    Returns
    -------
    TorsionSolution
        Piecewise torsion solution containing the IVP solution
        for every backbone segment.
    """

    segments = backbone_segments(
        robot,
    )

    first_segment = segments[0]

    y = np.zeros(
        2 * len(first_segment.active_tubes)
    )

    #
    # Construct the torsion state at s = 0.
    #
    for i, tube in enumerate(
        first_segment.active_tubes
    ):

        robot_index = robot.tubes.index(
            tube
        )

        y[theta_index(i)] = (
            robot.state.rotations[
                robot_index
            ]
        )

        y[theta_dot_index(i)] = (
            base_theta_dot[
                robot_index
            ]
        )

    segment_solutions = []

    #
    # Integrate each smooth torsion segment.
    #
    for segment_index, segment in enumerate(
        segments
    ):

        solution = solve_ivp(
            fun=lambda s, y: segment_torsion_ode(
                s,
                y,
                segment,
            ),
            t_span=(
                segment.start,
                segment.end,
            ),
            y0=y,
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
        )

        if not solution.success:
            raise RuntimeError(
                "Segment torsion integration failed: "
                f"{solution.message}"
            )

        segment_solutions.append(
            TorsionSegmentSolution(
                segment=segment,
                ivp_solution=solution,
            )
        )

        #
        # Map the terminal state into the next segment.
        #
        if segment_index < len(segments) - 1:

            next_segment = segments[
                segment_index + 1
            ]

            y_end = solution.y[:, -1]

            y = map_torsion_state(
                y_end,
                segment,
                next_segment,
            )

    return TorsionSolution(
        base_theta_dot=np.asarray(
            base_theta_dot,
            dtype=float,
        ),
        segments=segment_solutions,
    )

def solve_torsion_v2(
    robot: ConcentricTubeRobot,
    initial_guess: np.ndarray | None = None,
) -> TorsionSolution:
    """
    Solve the segment-aware torsion mechanics.

    A direct shooting solve is attempted first. If an initial
    guess is supplied, it is used for the base torsional strains.

    If the direct solve fails to converge, rotation continuation
    is used to reach the requested configuration.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Robot configuration.

    initial_guess : np.ndarray, optional
        Initial guess for the base torsional strains
        theta_dot_i(0).

    Returns
    -------
    TorsionSolution
        Complete piecewise torsion solution.
    """

    shooting_result = solve_torsion_shooting(
        robot,
        initial_guess=initial_guess,
    )

    if not shooting_result.success:

        shooting_result = (
            solve_torsion_shooting_continuation(
                robot,
            )
        )

    return integrate_torsion_segments(
        robot,
        shooting_result.x,
    )


def evaluate_segment_torsion_v2(
    torsion_solution: TorsionSolution,
    segment: Segment,
) -> np.ndarray:
    """
    Evaluate the segment-aware torsion solution at the midpoint
    of a backbone segment.

    Parameters
    ----------
    torsion_solution : TorsionSolution
        Piecewise torsion solution.

    segment : Segment
        Backbone segment at which torsion is evaluated.

    Returns
    -------
    np.ndarray
        Torsion angles theta_i for the tubes active in the segment.
    """

    #
    # Find the stored torsion solution corresponding
    # to this backbone segment.
    #
    segment_solution = None

    for candidate in torsion_solution.segments:

        if candidate.segment == segment:
            segment_solution = candidate
            break

    if segment_solution is None:
        raise ValueError(
            "No torsion solution found for the requested segment."
        )

    #
    # Evaluate the continuous IVP solution at the
    # segment midpoint.
    #
    s_mid = 0.5 * (
        segment.start
        + segment.end
    )

    state = segment_solution.ivp_solution.sol(
        np.array([s_mid])
    )[:, 0]

    #
    # The state is
    #
    # [theta_1, theta_1',
    #  theta_2, theta_2',
    #  ...]
    #
    # so keep only the theta entries.
    #
    return state[0::2]