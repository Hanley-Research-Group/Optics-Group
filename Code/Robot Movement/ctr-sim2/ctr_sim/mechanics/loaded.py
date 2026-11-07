"""
Forward mechanics with a force applied at the robot tip.

Model
-----
This is the standard externally loaded concentric tube robot model
(Rucker, Jones & Webster, "A Geometrically Exact Model for Externally
Loaded Concentric-Tube Continuum Robots", IEEE T-RO 2010), written in the
same zero-twist backbone frame and segment structure as the unloaded V2
solver.

With only a force F at the tip (no distributed load, no tip moment):

    internal force      n(s) = F                          (base frame)
    internal moment     m'   = -p' x F,     m(tip) = 0    (base frame)

    backbone curvature  u_xy = ( sum_i EI_i kappa_i e(theta_i)
                                 + (R^T m)_xy ) / sum_i EI_i
    tube torsion        GJ_i theta_i'' = EI_i kappa_i
                                         (u_x sin theta_i - u_y cos theta_i)
    kinematics          p' = R e_z,   R' = R [u]x,   u = (u_x, u_y, 0)

with e(theta) = (cos theta, sin theta) and the sums over the tubes present
at s. kappa_i is tube i's precurvature in the segment (zero where it is
straight). With F = 0 this reduces exactly to the unloaded V2 model.

Boundary value problem (solved by shooting):

    at s = 0     p = 0, R = I, theta_i = alpha_i
    unknowns     theta_i'(0) for each tube, and the two bending components
                 of the internal moment at the base
    residuals    theta_i' = 0 at each tube's own distal tip,
                 bending moment (R^T m)_xy = 0 at the robot tip.

The bending unknowns and residuals are divided by the total bending
stiffness so that all shooting variables are curvatures (1/m).

Not modeled: a moment at the tip (e.g. an object whose centre of mass is
offset from the tip), the self-weight of the tubes, contact, friction,
and buckling analysis (a solution is found, its stability is not
checked).
"""


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.backbone import Backbone
from ctr_sim.segment import Segment
from ctr_sim.segment_solution import SegmentSolution
from ctr_sim.torsion_solution import TorsionSolution
from ctr_sim.torsion_segment_solution import TorsionSegmentSolution
from ctr_sim.kinematics.intervals import backbone_segments
from ctr_sim.mechanics.torsion_v2 import (
    solve_torsion_v2,
    map_torsion_state,
)


STANDARD_GRAVITY = 9.80665

# ODE tolerances for the loaded solve. Tighter than scipy's defaults
# because the shooting residual is differentiated numerically.
_RTOL = 1e-8
_ATOL = 1e-10

# Layout of the loaded state vector:
#   [ p (3) | R (9, row-major) | m (3) | theta_1, theta_1', theta_2, ... ]
_P = slice(0, 3)
_R = slice(3, 12)
_M = slice(12, 15)
_N_FRAME = 15


def tip_weight(
    mass: float,
    gravity_direction,
    g: float = STANDARD_GRAVITY,
) -> np.ndarray:
    """
    Force (N) exerted on the robot tip by an attached mass.

    Parameters
    ----------
    mass : float
        Attached mass (kg).
    gravity_direction : array-like, shape (3,)
        Direction of gravity expressed in the robot base frame
        (ctr_base). The robot extends from the base along +z, so for
        example:

            robot pointing straight up    -> (0, 0, -1)
            robot pointing straight down  -> (0, 0, +1)
            robot horizontal, gravity -y  -> (0, -1, 0)

        Need not be normalized.
    g : float, optional
        Gravitational acceleration (m/s^2).

    Returns
    -------
    np.ndarray
        Tip force in the base frame, to pass as ``tip_force``.
    """

    direction = np.asarray(gravity_direction, dtype=float)

    if direction.shape != (3,):
        raise ValueError("gravity_direction must have 3 components.")

    norm = np.linalg.norm(direction)

    if norm == 0.0:
        raise ValueError("gravity_direction must be non-zero.")

    if mass < 0:
        raise ValueError("mass cannot be negative.")

    return mass * g * direction / norm


def as_tip_force(tip_force) -> np.ndarray | None:
    """
    Validate a tip force. Returns None for no load (None or all zeros),
    otherwise a float array of shape (3,).
    """

    if tip_force is None:
        return None

    force = np.asarray(tip_force, dtype=float)

    if force.shape != (3,):
        raise ValueError("tip_force must have 3 components (N).")

    if not np.all(np.isfinite(force)):
        raise ValueError("tip_force must be finite.")

    if not np.any(force):
        return None

    return force


class _StateSlice:
    """
    View of selected rows of an IVP solution, so the loaded solution can
    be consumed by code written for the unloaded one (which expects only
    the backbone rows, or only the torsion rows).
    """

    def __init__(self, solution, rows):
        self._solution = solution
        self._rows = rows
        self.t = solution.t
        self.y = solution.y[rows]
        self.success = solution.success
        self.message = solution.message
        self.status = solution.status

    def sol(self, s):
        return self._solution.sol(s)[self._rows]


def _segment_constants(segment: Segment):

    EI = np.array([tube.EI for tube in segment.active_tubes])
    GJ = np.array([tube.GJ for tube in segment.active_tubes])
    kappa = np.array(segment.curvatures, dtype=float)

    return EI, GJ, kappa, EI * kappa, EI.sum()


def _loaded_ode(segment: Segment, force: np.ndarray):
    """Right-hand side of the loaded equations within one segment."""

    EI, GJ, kappa, w, sum_EI = _segment_constants(segment)
    torsion_gain = EI * kappa / GJ

    def rhs(s, y):

        R = y[_R].reshape(3, 3)
        m = y[_M]

        theta = y[_N_FRAME::2]
        theta_dot = y[_N_FRAME + 1::2]

        c = np.cos(theta)
        sn = np.sin(theta)

        # Internal moment in the backbone frame.
        mb = R.T @ m

        ux = (w @ c + mb[0]) / sum_EI
        uy = (w @ sn + mb[1]) / sum_EI

        z = R[:, 2]

        dR = np.empty((3, 3))
        dR[:, 0] = -uy * z
        dR[:, 1] = ux * z
        dR[:, 2] = uy * R[:, 0] - ux * R[:, 1]

        dyds = np.empty_like(y)
        dyds[_P] = z
        dyds[_R] = dR.reshape(9)
        dyds[_M] = -np.cross(z, force)
        dyds[_N_FRAME::2] = theta_dot
        dyds[_N_FRAME + 1::2] = torsion_gain * (ux * sn - uy * c)

        return dyds

    return rhs


def _initial_state(
    robot: ConcentricTubeRobot,
    first_segment: Segment,
    shooting: np.ndarray,
) -> np.ndarray:

    n_tubes = len(robot.tubes)
    base_theta_dot = shooting[:n_tubes]
    mu = shooting[n_tubes:n_tubes + 2]

    k = len(first_segment.active_tubes)
    y = np.zeros(_N_FRAME + 2 * k)

    y[_R] = np.eye(3).reshape(9)

    _, GJ, _, _, sum_EI = _segment_constants(first_segment)

    torsional_moment = 0.0

    for local, tube in enumerate(first_segment.active_tubes):

        i = robot.tubes.index(tube)

        y[_N_FRAME + 2 * local] = robot.state.rotations[i]
        y[_N_FRAME + 2 * local + 1] = base_theta_dot[i]

        torsional_moment += GJ[local] * base_theta_dot[i]

    # R(0) = I, so the base-frame moment equals the backbone-frame one.
    # Its z component is the total torsional moment of the tubes.
    y[_M] = [
        mu[0] * sum_EI,
        mu[1] * sum_EI,
        torsional_moment,
    ]

    return y


def _integrate(
    robot: ConcentricTubeRobot,
    segments: list[Segment],
    force: np.ndarray,
    shooting: np.ndarray,
    dense: bool,
):
    """
    Integrate the loaded equations through every segment.

    Returns (residual, list of IVP solutions).
    """

    n_tubes = len(robot.tubes)
    residual = np.zeros(n_tubes + 2)

    y = _initial_state(robot, segments[0], shooting)

    solutions = []

    for index, segment in enumerate(segments):

        solution = solve_ivp(
            _loaded_ode(segment, force),
            (segment.start, segment.end),
            y,
            rtol=_RTOL,
            atol=_ATOL,
            dense_output=dense,
        )

        if not solution.success:
            raise RuntimeError(
                "Loaded segment integration failed: "
                f"{solution.message}"
            )

        solutions.append(solution)

        y_end = solution.y[:, -1]

        if index < len(segments) - 1:
            next_segment = segments[index + 1]
            terminating = [
                tube for tube in segment.active_tubes
                if tube not in next_segment.active_tubes
            ]
        else:
            next_segment = None
            terminating = list(segment.active_tubes)

        # Free-tip torsion: theta_i' = 0 at each tube's own tip.
        for tube in terminating:
            local = segment.active_tubes.index(tube)
            residual[robot.tubes.index(tube)] = (
                y_end[_N_FRAME + 2 * local + 1]
            )

        if next_segment is None:

            # Zero bending moment at the robot tip, as a curvature.
            R_tip = y_end[_R].reshape(3, 3)
            mb_tip = R_tip.T @ y_end[_M]
            sum_EI_tip = _segment_constants(segment)[4]

            residual[n_tubes:] = mb_tip[:2] / sum_EI_tip

        else:

            frame = y_end[:_N_FRAME]
            torsion = map_torsion_state(
                y_end[_N_FRAME:],
                segment,
                next_segment,
            )
            y = np.concatenate([frame, torsion])

    return residual, solutions


# A shooting solve is considered converged when every residual (all are
# curvatures or torsional strains, 1/m) is below this value.
_RESIDUAL_TOL = 1e-8


def _unloaded_base_theta_dot(robot, guess):
    """
    Base torsional strains of the unloaded robot, used to start loaded
    solves. Falls back to zero if the unloaded solve itself fails (it
    can, near torsional snap-through), since the loaded solve may still
    converge.
    """

    try:
        return solve_torsion_v2(robot, initial_guess=guess).base_theta_dot
    except RuntimeError:
        return np.zeros(len(robot.tubes))


def _fd_jacobian(fun, x, r):
    """Forward-difference shooting Jacobian at x (r = fun(x))."""

    J = np.empty((len(r), len(x)))

    for k in range(len(x)):

        h = 1e-5 * max(1.0, abs(x[k]))
        x_step = np.array(x, dtype=float)
        x_step[k] += h

        J[:, k] = (fun(x_step) - r) / h

    return J


# Cap on residual evaluations for one hybr attempt. A converging solve
# needs ~10-30; without a cap a hopeless attempt can take ~1200
# evaluations (several seconds) before load continuation starts.
_HYBR_MAX_EVALUATIONS = 60


def _hybr(fun, guess):
    """
    Robust solve with MINPACK hybr. Returns (x, J) on success, where J
    is a finite-difference shooting Jacobian at the solution (hybr's
    own internal estimate is too inaccurate to reuse), or None.
    """

    result = root(
        fun,
        guess,
        method="hybr",
        options={"maxfev": _HYBR_MAX_EVALUATIONS},
    )

    if not result.success:
        return None

    return result.x, _fd_jacobian(fun, result.x, fun(result.x))


def _broyden(fun, guess, jacobian, max_iterations=8):
    """
    Fast solve from a nearby solution, reusing its shooting Jacobian and
    refining it with Broyden updates. Typically 1-3 residual evaluations
    instead of the ~10 a fresh hybr solve needs. Returns (x, J) or None
    if it does not converge quickly.
    """

    x = np.array(guess, dtype=float)
    J = np.array(jacobian, dtype=float)

    r = fun(x)

    if np.max(np.abs(r)) < _RESIDUAL_TOL:
        return x, J

    r0 = np.max(np.abs(r))

    for _ in range(max_iterations):

        try:
            dx = -np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return None

        x_new = x + dx
        r_new = fun(x_new)

        if not np.all(np.isfinite(r_new)):
            return None

        J += np.outer(r_new - r - J @ dx, dx) / (dx @ dx)

        x, r = x_new, r_new

        if np.max(np.abs(r)) < _RESIDUAL_TOL:
            return x, J

        if np.max(np.abs(r)) > 1e3 * r0:
            return None

    return None


def _solve(fun, guess, jacobian):

    if jacobian is not None:

        solved = _broyden(fun, guess, jacobian)

        if solved is not None:
            return solved

    return _hybr(fun, guess)


def solve_loaded_shooting(
    robot: ConcentricTubeRobot,
    tip_force,
    initial_guess: np.ndarray | None = None,
    initial_jacobian: np.ndarray | None = None,
    continuation_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the loaded boundary value problem.

    Returns (x, J): the shooting vector
    [theta_1'(0) ... theta_n'(0), mu_x, mu_y], where mu is the bending
    moment at the base divided by the total bending stiffness there
    (1/m), and an approximation of the shooting Jacobian at x, which
    speeds up the next nearby solve.

    If ``initial_jacobian`` is given (from a nearby solution), a fast
    Broyden solve is tried first; then a fresh hybr solve from the
    initial guess; then, if that fails, the load is ramped up from zero
    in ``continuation_steps`` steps.
    """

    x, J, _ = _solve_loaded(
        robot,
        tip_force,
        initial_guess,
        initial_jacobian,
        continuation_steps,
    )

    return x, J


def _solve_loaded(
    robot,
    tip_force,
    initial_guess,
    initial_jacobian,
    continuation_steps=10,
):
    """
    As solve_loaded_shooting, but also returns the dense per-segment
    integration at the solution if the last residual evaluation was
    made there (otherwise None), so it need not be repeated.
    """

    force = np.asarray(tip_force, dtype=float)
    n_tubes = len(robot.tubes)
    segments = backbone_segments(robot)

    if initial_guess is None:
        guess = None
    else:
        guess = np.asarray(initial_guess, dtype=float)

        if guess.shape == (n_tubes,):
            guess = np.concatenate([guess, np.zeros(2)])
            initial_jacobian = None
        elif guess.shape != (n_tubes + 2,):
            raise ValueError(
                "initial_guess must have n_tubes or n_tubes + 2 entries."
            )

    if guess is None:
        guess = np.concatenate([
            _unloaded_base_theta_dot(robot, None),
            np.zeros(2),
        ])
        initial_jacobian = None

    # Keep the most recent integration: when a solve ends on a
    # residual evaluation at the solution, it is reused to build the
    # output instead of integrating once more.
    last = {}

    def residual_for(f):

        def fun(x):
            residual, solutions = _integrate(
                robot, segments, f, x, dense=True
            )
            last["x"] = np.array(x, dtype=float)
            last["force"] = f
            last["solutions"] = solutions
            return residual

        return fun

    def reusable(x):
        if (
            "x" in last
            and last["force"] is force
            and np.array_equal(last["x"], x)
        ):
            return last["solutions"]
        return None

    solved = _solve(residual_for(force), guess, initial_jacobian)

    if solved is not None:
        x, J = solved
        return x, J, reusable(x)

    # Load continuation: ramp the load up from zero, starting from the
    # unloaded configuration, with an adaptive step.
    x = np.concatenate([
        _unloaded_base_theta_dot(robot, guess[:n_tubes]),
        np.zeros(2),
    ])
    J = None

    fraction = 0.0
    step = 1.0 / continuation_steps
    min_step = step / 32

    while fraction < 1.0:

        target = min(1.0, fraction + step)

        solved = _solve(residual_for(target * force), x, J)

        if solved is None:

            step /= 2

            if step < min_step:
                raise RuntimeError(
                    "Loaded mechanics did not converge: load "
                    f"continuation stalled at {fraction:.1%} of the "
                    "tip force. The configuration may be near a "
                    "snap-through or buckling point."
                )

            continue

        x, J = solved
        fraction = target
        step = min(2 * step, 0.5)

    return x, J, reusable(x)


def solve_forward_kinematics_loaded(
    robot: ConcentricTubeRobot,
    tip_force,
    initial_guess=None,
) -> Backbone:
    """
    Forward mechanics with a force applied at the robot tip.

    ``initial_guess`` may be a TorsionSolution from a nearby solve
    (fastest: reuses its shooting Jacobian), a warm-start array, or
    None.

    Normally called through
    ``solve_forward_kinematics_v2(robot, tip_force=...)``.

    The returned Backbone has the same structure as the unloaded one, so
    ``sample_backbone`` and ``tip_position`` work unchanged. Under load
    the curvature and torsion vary within a segment; the per-segment
    ``theta`` and ``curvature`` fields hold their values at the segment
    midpoint.
    """

    force = as_tip_force(tip_force)

    if force is None:
        raise ValueError(
            "tip_force is zero; use solve_forward_kinematics_v2."
        )

    initial_jacobian = None

    if isinstance(initial_guess, TorsionSolution):
        initial_jacobian = initial_guess.shooting_jacobian
        initial_guess = initial_guess.warm_start

    shooting, jacobian, solutions = _solve_loaded(
        robot,
        force,
        initial_guess,
        initial_jacobian,
    )

    return _build_backbone(
        robot, force, shooting, jacobian, solutions
    )


def _build_backbone(
    robot,
    force,
    shooting,
    jacobian=None,
    solutions=None,
) -> Backbone:

    n_tubes = len(robot.tubes)
    segments = backbone_segments(robot)

    if solutions is None:
        _, solutions = _integrate(
            robot,
            segments,
            force,
            shooting,
            dense=True,
        )

    segment_solutions = []
    torsion_segments = []

    for segment, solution in zip(segments, solutions):

        k = len(segment.active_tubes)
        torsion_rows = slice(_N_FRAME, _N_FRAME + 2 * k)

        s_mid = 0.5 * (segment.start + segment.end)
        y_mid = solution.sol(np.array([s_mid]))[:, 0]

        theta_mid = y_mid[_N_FRAME::2]

        EI, _, _, w, sum_EI = _segment_constants(segment)
        mb = y_mid[_R].reshape(3, 3).T @ y_mid[_M]

        curvature = np.array([
            (w @ np.cos(theta_mid) + mb[0]) / sum_EI,
            (w @ np.sin(theta_mid) + mb[1]) / sum_EI,
            0.0,
        ])

        segment_solutions.append(
            SegmentSolution(
                segment=segment,
                theta=theta_mid,
                curvature=curvature,
                ivp_solution=_StateSlice(solution, slice(0, 12)),
            )
        )

        torsion_segments.append(
            TorsionSegmentSolution(
                segment=segment,
                ivp_solution=_StateSlice(solution, torsion_rows),
            )
        )

    first = segments[0]
    sum_EI_base = _segment_constants(first)[4]

    base_moment = np.array([
        shooting[n_tubes] * sum_EI_base,
        shooting[n_tubes + 1] * sum_EI_base,
        _initial_state(robot, first, shooting)[_M][2],
    ])

    torsion_solution = TorsionSolution(
        base_theta_dot=np.array(shooting[:n_tubes]),
        segments=torsion_segments,
        base_moment=base_moment,
        shooting_vector=np.array(shooting),
        shooting_jacobian=jacobian,
    )

    return Backbone(
        segments=segment_solutions,
        torsion_solution=torsion_solution,
    )
