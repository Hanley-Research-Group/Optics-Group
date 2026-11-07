import time
from copy import deepcopy

import numpy as np

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)

from ctr_sim.kinematics.intervals import (
    backbone_segments,
)

from ctr_sim.mechanics.torsion_v2 import (
    solve_torsion_shooting,
    solve_torsion_shooting_continuation,
    solve_torsion_v2,
    evaluate_segment_torsion_v2,
)

from ctr_sim.mechanics.curvature import (
    segment_curvature_v2,
)

from ctr_sim.mechanics.integration import (
    integrate_segment,
)

from ctr_sim.mechanics.forward_v2 import (
    solve_forward_kinematics_v2,
)

from ctr_sim.mechanics.sampling import (
    sample_backbone,
)

from ctr_sim.control.jacobian import (
    numerical_position_jacobian_v2,
)


# ----------------------------------------
# Materials
# ----------------------------------------

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


# ----------------------------------------
# Robot
# ----------------------------------------

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

state = CTRState(
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
)

robot = ConcentricTubeRobot(
    tubes=[
        outer,
        middle,
        inner,
    ],
    state=state,
)


# ----------------------------------------
# Settings
# ----------------------------------------

delta_insertion = 1e-4
delta_rotation = 1e-4


def elapsed(start):
    return time.perf_counter() - start


# ========================================
# 1. Segment-aware forward breakdown
# ========================================

start = time.perf_counter()

segments = backbone_segments(
    robot,
)

segment_time = elapsed(start)


start = time.perf_counter()

torsion_solution = solve_torsion_v2(
    robot,
)

torsion_time = elapsed(start)


position = np.zeros(3)
rotation = np.eye(3)

torsion_evaluation_time = 0.0
curvature_time = 0.0
integration_time = 0.0

for segment in segments:

    start = time.perf_counter()

    theta = evaluate_segment_torsion_v2(
        torsion_solution,
        segment,
    )

    torsion_evaluation_time += elapsed(
        start
    )

    start = time.perf_counter()

    curvature = segment_curvature_v2(
        theta,
        segment,
    )

    curvature_time += elapsed(
        start
    )

    start = time.perf_counter()

    _, position, rotation = integrate_segment(
        position,
        rotation,
        curvature,
        segment,
    )

    integration_time += elapsed(
        start
    )


manual_forward_time = (
    segment_time
    + torsion_time
    + torsion_evaluation_time
    + curvature_time
    + integration_time
)


# ========================================
# 2. Torsion solver behavior
# ========================================

start = time.perf_counter()

direct_result = solve_torsion_shooting(
    robot,
)

direct_shooting_time = elapsed(
    start
)


start = time.perf_counter()

continuation_result = (
    solve_torsion_shooting_continuation(
        robot,
        n_steps=10,
    )
)

continuation_time = elapsed(
    start
)


#
# Time a torsion solve when supplied with a known
# good initial guess.
#
warm_start_guess = (
    continuation_result.x.copy()
)

start = time.perf_counter()

warm_result = solve_torsion_shooting(
    robot,
    initial_guess=warm_start_guess,
)

warm_shooting_time = elapsed(
    start
)


# ========================================
# 3. Complete forward solve
# ========================================

start = time.perf_counter()

backbone = solve_forward_kinematics_v2(
    robot,
)

forward_time = elapsed(start)


# ========================================
# 4. Warm-started complete forward solve
# ========================================

torsion_initial_guess = (
    backbone
    .torsion_solution
    .base_theta_dot
    .copy()
)

start = time.perf_counter()

warm_backbone = solve_forward_kinematics_v2(
    robot,
    torsion_initial_guess=torsion_initial_guess,
)

warm_forward_time = elapsed(start)


# ========================================
# 5. Backbone sampling
# ========================================

start = time.perf_counter()

samples = sample_backbone(
    backbone,
    ds=1e-3,
)

sampling_time = elapsed(start)


# ========================================
# 6. Numerical Jacobian
# ========================================

start = time.perf_counter()

J = numerical_position_jacobian_v2(
    robot,
)

jacobian_time = elapsed(start)


# ========================================
# 7. Warm-started Jacobian solve breakdown
# ========================================

n = len(robot.tubes)

perturbation_results = []


#
# Nominal solve.
#
start = time.perf_counter()

nominal_backbone = solve_forward_kinematics_v2(
    robot,
)

nominal_time = elapsed(start)


#
# This is the key warm-start information.
#
# Every finite-difference perturbation is close to
# the nominal configuration, so use the nominal
# torsional strains as its initial guess.
#
jacobian_torsion_guess = (
    nominal_backbone
    .torsion_solution
    .base_theta_dot
    .copy()
)

perturbation_results.append(
    (
        "nominal",
        nominal_time,
    )
)


#
# Insertion perturbations.
#
for i in range(n):

    beta_i = robot.state.insertions[i]
    tube_length = robot.tubes[i].length

    if (
        beta_i + delta_insertion
        <= tube_length
    ):

        robot_perturbed = deepcopy(
            robot
        )

        robot_perturbed.state.insertions[i] += (
            delta_insertion
        )

        difference_type = "forward"

    else:

        robot_perturbed = deepcopy(
            robot
        )

        robot_perturbed.state.insertions[i] -= (
            delta_insertion
        )

        difference_type = "backward"

    start = time.perf_counter()

    solve_forward_kinematics_v2(
        robot_perturbed,
        torsion_initial_guess=(
            jacobian_torsion_guess
        ),
    )

    solve_time = elapsed(start)

    perturbation_results.append(
        (
            f"beta_{i + 1} ({difference_type})",
            solve_time,
        )
    )


#
# Rotation perturbations.
#
for i in range(n):

    robot_perturbed = deepcopy(
        robot
    )

    robot_perturbed.state.rotations[i] += (
        delta_rotation
    )

    start = time.perf_counter()

    solve_forward_kinematics_v2(
        robot_perturbed,
        torsion_initial_guess=(
            jacobian_torsion_guess
        ),
    )

    solve_time = elapsed(start)

    perturbation_results.append(
        (
            f"alpha_{i + 1}",
            solve_time,
        )
    )


manual_jacobian_time = sum(
    result[1]
    for result in perturbation_results
)


# ========================================
# Results
# ========================================

print()
print("=" * 70)
print("Mechanics V2 Timing")
print("=" * 70)


print()
print("Forward mechanics breakdown:")

print(
    f"  Backbone segmentation:       "
    f"{segment_time:10.6f} s"
)

print(
    f"  Segment-aware torsion:       "
    f"{torsion_time:10.6f} s"
)

print(
    f"  Segment torsion evaluation:  "
    f"{torsion_evaluation_time:10.6f} s"
)

print(
    f"  Segment curvature:           "
    f"{curvature_time:10.6f} s"
)

print(
    f"  Segment integration:         "
    f"{integration_time:10.6f} s"
)

print()
print(
    f"  Manual breakdown total:      "
    f"{manual_forward_time:10.6f} s"
)


print()
print("Torsion solver behavior:")

print(
    f"  Direct shooting from zero:   "
    f"{direct_shooting_time:10.6f} s"
)

print(
    f"    success: "
    f"{direct_result.success}"
)

print(
    f"  10-step continuation:        "
    f"{continuation_time:10.6f} s"
)

print(
    f"    success: "
    f"{continuation_result.success}"
)

print(
    f"  Warm-started shooting:       "
    f"{warm_shooting_time:10.6f} s"
)

print(
    f"    success: "
    f"{warm_result.success}"
)


print()
print("Complete operations:")

print(
    f"  Cold forward solve:          "
    f"{forward_time:10.6f} s"
)

print(
    f"  Warm forward solve:          "
    f"{warm_forward_time:10.6f} s"
)

print(
    f"  sample_backbone:             "
    f"{sampling_time:10.6f} s"
)

print(
    f"  numerical Jacobian V2:       "
    f"{jacobian_time:10.6f} s"
)


print()
print("=" * 70)
print("Warm-Started Jacobian Forward-Solve Breakdown")
print("=" * 70)

for name, solve_time in perturbation_results:

    print(
        f"  {name:<28}"
        f"{solve_time:10.6f} s"
    )

print()
print(
    f"  total:                       "
    f"{manual_jacobian_time:10.6f} s"
)


print()
print("=" * 70)
print("Numerical Results")
print("=" * 70)

print()
print("Jacobian:")

print(
    np.round(
        J,
        4,
    )
)

print()
print("Tip:")

print(
    np.round(
        samples.position[-1],
        6,
    )
)

print()