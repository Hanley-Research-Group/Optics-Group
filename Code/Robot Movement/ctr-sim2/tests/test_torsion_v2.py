import numpy as np

from ctr_sim.kinematics.intervals import (
    backbone_segments,
)

from ctr_sim.mechanics.torsion_v2 import (
    segment_torsion_ode,
    map_torsion_state,
    torsion_shooting_residual,
    solve_torsion_shooting,
    solve_torsion_shooting_continuation,
    solve_torsion_v2,
    evaluate_segment_torsion_v2,
    theta_index,
    theta_dot_index,
)

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)


def test_segment_torsion_ode(
    forward_robot,
):

    segments = backbone_segments(
        forward_robot,
    )

    segment = segments[0]

    n = len(
        segment.active_tubes
    )

    y = np.zeros(
        2 * n
    )

    dyds = segment_torsion_ode(
        0.5 * (
            segment.start
            + segment.end
        ),
        y,
        segment,
    )

    assert dyds.shape == y.shape

    assert np.all(
        np.isfinite(dyds)
    )

def test_map_torsion_state(
    forward_robot,
):

    segments = backbone_segments(
        forward_robot,
    )

    segment0 = segments[0]
    segment1 = segments[1]

    n0 = len(
        segment0.active_tubes
    )

    #
    # Give every source state a unique value:
    #
    # [1, 2, 3, 4, ...]
    #
    y0 = np.arange(
        1,
        2 * n0 + 1,
        dtype=float,
    )

    y1 = map_torsion_state(
        y0,
        segment0,
        segment1,
    )

    #
    # The destination state should contain exactly
    # the theta and theta_dot values belonging to
    # tubes that continue into segment1.
    #
    expected = []

    for tube in segment1.active_tubes:

        i = segment0.active_tubes.index(
            tube
        )

        expected.extend([
            y0[theta_index(i)],
            y0[theta_dot_index(i)],
        ])

    expected = np.asarray(
        expected,
        dtype=float,
    )

    assert np.allclose(
        y1,
        expected,
    )

def test_torsion_shooting_residual(
    forward_robot,
):

    n = len(
        forward_robot.tubes
    )

    base_theta_dot = np.array([
        1.0,
        2.0,
        3.0,
    ])

    residual = torsion_shooting_residual(
        base_theta_dot,
        forward_robot,
    )

    assert residual.shape == (
        n,
    )

    assert np.all(
        np.isfinite(residual)
    )   

def test_solve_torsion_shooting(
    forward_robot,
):

    result = solve_torsion_shooting(
        forward_robot,
    )

    residual = torsion_shooting_residual(
        result.x,
        forward_robot,
    )

    assert result.success

    assert np.all(
        np.isfinite(result.x)
    )

    assert np.allclose(
        residual,
        0.0,
        atol=1e-6,
    )

def test_solve_torsion_v2(
    forward_robot,
):

    solution = solve_torsion_v2(
        forward_robot,
    )

    segments = backbone_segments(
        forward_robot,
    )

    assert len(solution.segments) == len(
        segments
    )

    assert solution.base_theta_dot.shape == (
        len(forward_robot.tubes),
    )

    for segment_solution in solution.segments:

        assert (
            segment_solution.ivp_solution.success
        )

        assert np.all(
            np.isfinite(
                segment_solution.ivp_solution.y
            )
        )

def test_evaluate_segment_torsion_v2(
    forward_robot,
):

    torsion_solution = solve_torsion_v2(
        forward_robot,
    )

    segments = backbone_segments(
        forward_robot,
    )

    for segment in segments:

        theta = evaluate_segment_torsion_v2(
            torsion_solution,
            segment,
        )

        assert theta.shape == (
            len(segment.active_tubes),
        )

        assert np.all(
            np.isfinite(theta)
        )

def test_torsion_shooting_continuation():

    nitinol = Material(
        name="Nitinol",
        youngs_modulus=60e9,
        shear_modulus=23e9,
    )

    fiber = Material(
        name="SilicaFiber",
        youngs_modulus=15e9,
        shear_modulus=6.4e9,
    )

    outer = Tube(
        name="OuterTube",
        length=0.16,
        precurvature=15.0,
        outer_diameter=3.0e-3,
        inner_diameter=2.8e-3,
        material=nitinol,
    )

    middle = Tube(
        name="MiddleTube",
        length=0.18,
        precurvature=10.0,
        outer_diameter=2.0e-3,
        inner_diameter=1.8e-3,
        material=nitinol,
    )

    inner = Tube(
        name="InnerTube",
        length=0.20,
        precurvature=0.0,
        outer_diameter=1.6e-3,
        inner_diameter=0.0,
        material=fiber,
    )

    robot = ConcentricTubeRobot(
        tubes=[
            outer,
            middle,
            inner,
        ],
        state=CTRState(
            insertions=[
                0.10,
                0.18,
                0.19,
            ],
            rotations=[
                np.pi / 2,
                0.0,
                0.0,
            ],
        ),
    )

    result = solve_torsion_shooting_continuation(
        robot,
        n_steps=10,
    )

    residual = torsion_shooting_residual(
        result.x,
        robot,
    )

    assert result.success

    assert np.all(
        np.isfinite(result.x)
    )

    assert np.allclose(
        residual,
        0.0,
        atol=1e-6,
    )


def test_solve_torsion_v2_with_initial_guess(
    forward_robot,
):

    solution_1 = solve_torsion_v2(
        forward_robot,
    )

    solution_2 = solve_torsion_v2(
        forward_robot,
        initial_guess=solution_1.base_theta_dot,
    )

    assert np.allclose(
        solution_2.base_theta_dot,
        solution_1.base_theta_dot,
        atol=1e-6,
    )