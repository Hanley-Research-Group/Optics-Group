"""
Independent check of the torsion mechanics against direct energy
minimization.

For the unloaded robot the equilibrium torsion angles theta_i(s)
minimize the total elastic energy

    E = int sum_i  1/2 GJ_i theta_i'^2
                 + 1/2 EI_i | u - kappa_i(s) e(theta_i) |^2   ds

where e(theta) = (cos theta, sin theta), u is the backbone curvature
(free, so it takes its energy-minimizing value
 u = sum_i EI_i kappa_i e(theta_i) / sum_i EI_i), theta_i(0) = alpha_i,
and the tip is free.

This reference involves no ODE and no shooting, and does not use
backbone_segments: its mesh is built directly from the tube geometry.
It therefore checks the torsion ODE, the segment-wise shooting
solution, and the handling of straight/curved/straight tubes at once.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)
from ctr_sim.kinematics.intervals import backbone_segments
from ctr_sim.mechanics.torsion_v2 import (
    solve_torsion_v2,
    evaluate_segment_torsion_v2,
)


def _mesh(robot, per_region=40):
    """
    Nodes covering the free backbone, with a node at every point where
    a tube starts, ends, or changes curvature, so each element has
    constant tube presence and curvature.
    """

    marks = {0.0}

    for tube, beta in zip(robot.tubes, robot.state.insertions):

        proximal = beta - tube.length
        sigma_start, sigma_end = tube.curve_interval

        for point in (
            proximal,
            beta,
            proximal + sigma_start,
            proximal + sigma_end,
        ):
            if 0.0 <= point <= max(robot.state.insertions):
                marks.add(point)

    marks = sorted(marks)

    nodes = [marks[0]]

    for a, b in zip(marks[:-1], marks[1:]):

        if b - a < 1e-12:
            continue

        nodes.extend(np.linspace(a, b, per_region + 1)[1:])

    return np.array(nodes)


def minimize_energy(robot, per_region=40):
    """Return (nodes, theta[tube, node]) minimizing the elastic energy."""

    tubes = robot.tubes
    n = len(tubes)

    beta = np.array(robot.state.insertions)
    alpha = np.array(robot.state.rotations)

    EI = np.array([t.EI for t in tubes])
    GJ = np.array([t.GJ for t in tubes])

    s = _mesh(robot, per_region)
    h = np.diff(s)
    mid = 0.5 * (s[:-1] + s[1:])
    n_el = len(h)

    present = np.zeros((n, n_el))
    kappa = np.zeros((n, n_el))

    for i, (tube, b) in enumerate(zip(tubes, beta)):

        sigma = mid - (b - tube.length)

        present[i] = (sigma >= 0) & (sigma <= tube.length)

        kappa[i] = [
            tube.curvature_at(x) for x in sigma
        ]

    sum_EI = (EI[:, None] * present).sum(axis=0)
    w = (EI[:, None] * kappa) * present

    # Normalize the energy so L-BFGS tolerances are meaningful.
    scale = 1.0 / max(
        (EI * np.array([t.precurvature for t in tubes]) ** 2).max()
        * s[-1],
        1e-30,
    )

    def unpack(x):
        theta = np.zeros((n, n_el + 1))
        theta[:, 0] = alpha
        theta[:, 1:] = x.reshape(n, n_el)
        return theta

    def energy_and_gradient(x):

        theta = unpack(x)
        tm = 0.5 * (theta[:, :-1] + theta[:, 1:])
        d = (theta[:, 1:] - theta[:, :-1]) / h

        ux = (w * np.cos(tm)).sum(axis=0)
        uy = (w * np.sin(tm)).sum(axis=0)

        energy = np.sum(
            h * (
                0.5 * (GJ[:, None] * present * d**2).sum(axis=0)
                - 0.5 * (ux**2 + uy**2) / sum_EI
            )
        )

        dE_dd = GJ[:, None] * present * d
        dE_dtm = h * (
            -(ux * (-w * np.sin(tm)) + uy * (w * np.cos(tm)))
            / sum_EI
        )

        grad = np.zeros((n, n_el + 1))
        grad[:, 1:] += dE_dd + 0.5 * dE_dtm
        grad[:, :-1] += -dE_dd + 0.5 * dE_dtm

        return scale * energy, scale * grad[:, 1:].ravel()

    x0 = np.tile(alpha[:, None], (1, n_el)).ravel()

    result = minimize(
        energy_and_gradient,
        x0,
        jac=True,
        method="L-BFGS-B",
        options=dict(
            maxiter=100_000,
            maxfun=100_000,
            ftol=1e-20,
            gtol=1e-12,
        ),
    )

    assert np.abs(result.jac).max() < 1e-6, result.message

    return s, unpack(result.x)


def assert_torsion_matches_energy_minimum(robot, tol_deg):

    s, theta_ref = minimize_energy(robot)

    solution = solve_torsion_v2(robot)

    for segment in backbone_segments(robot):

        theta = evaluate_segment_torsion_v2(solution, segment)

        s_mid = 0.5 * (segment.start + segment.end)

        for k, tube in enumerate(segment.active_tubes):

            i = robot.tubes.index(tube)

            reference = np.interp(s_mid, s, theta_ref[i])

            assert np.rad2deg(
                abs(theta[k] - reference)
            ) < tol_deg, (
                f"{tube.name} in segment "
                f"[{segment.start:.4f}, {segment.end:.4f}]: "
                f"shooting {np.rad2deg(theta[k]):.3f} deg vs "
                f"energy minimum {np.rad2deg(reference):.3f} deg"
            )


nitinol = Material("Nitinol", 60e9, 23e9)
fiber = Material("Fiber", 15e9, 6.4e9)
nylon = Material("Nylon", 2.7e9, 1.0e9)


def legacy_robot():

    return ConcentricTubeRobot(
        [
            Tube("o", 0.16, 15.0, 3e-3, 2.8e-3, nitinol),
            Tube("m", 0.18, 10.0, 2e-3, 1.8e-3, nitinol),
            Tube("i", 0.20, 0.0, 1.6e-3, 0.0, fiber),
        ],
        CTRState([0.10, 0.12, 0.14], [0.0, np.deg2rad(30.0), 0.0]),
    )


def piecewise_robot(insertions, rotations):

    return ConcentricTubeRobot(
        [
            Tube.from_lengths(
                "outer", 0.060, 0.050, 0.030, 18.0, 4.0e-3, 3.2e-3, nylon
            ),
            Tube.from_lengths(
                "middle", 0.080, 0.045, 0.035, 22.0, 2.8e-3, 2.1e-3, nylon
            ),
            Tube.from_lengths(
                "inner", 0.100, 0.040, 0.040, 25.0, 1.8e-3, 0.8e-3, nylon
            ),
        ],
        CTRState(list(insertions), list(rotations)),
    )


def test_uniform_tubes_match_energy_minimum():

    assert_torsion_matches_energy_minimum(
        legacy_robot(), tol_deg=0.05
    )


@pytest.mark.parametrize(
    "insertions, rotations",
    [
        ([0.10, 0.14, 0.18], [0.5, -1.0, 2.0]),
        ([0.13, 0.15, 0.17], [2.5, 0.0, -2.5]),
        ([0.05, 0.12, 0.18], [0.0, 1.5, 0.0]),
    ],
)
def test_straight_curved_straight_tubes_match_energy_minimum(
    insertions,
    rotations,
):

    assert_torsion_matches_energy_minimum(
        piecewise_robot(insertions, rotations), tol_deg=0.1
    )
