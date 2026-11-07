import time

import numpy as np

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)

from ctr_sim.mechanics.forward_v2 import (
    solve_forward_kinematics_v2,
)

from ctr_sim.mechanics.sampling import (
    sample_backbone,
    tip_position,
)

from ctr_sim.control.jacobian import (
    numerical_position_jacobian_v2,
)


# ----------------------------------------
# Timing helper
# ----------------------------------------

def elapsed(start):
    return time.perf_counter() - start


def summarize(
    name,
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    print(
        f"{name:<32}"
        f"mean={np.mean(values):.6f} s   "
        f"median={np.median(values):.6f} s   "
        f"min={np.min(values):.6f} s   "
        f"max={np.max(values):.6f} s"
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
        0.145,
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

n_runs = 5


# ========================================
# 1. Establish a known-good solution
# ========================================

print()
print("=" * 78)
print("Mechanics V3 Timing")
print("=" * 78)

print()
print("Establishing nominal mechanics solution...")

start = time.perf_counter()

backbone = solve_forward_kinematics_v2(
    robot,
)

cold_forward_time = elapsed(
    start
)

torsion_initial_guess = (
    backbone
    .torsion_solution
    .base_theta_dot
    .copy()
)

print(
    f"Cold initial forward solve: "
    f"{cold_forward_time:.6f} s"
)


# ========================================
# 2. Warm forward solve
# ========================================

warm_forward_times = []

for _ in range(n_runs):

    start = time.perf_counter()

    solve_forward_kinematics_v2(
        robot,
        torsion_initial_guess=(
            torsion_initial_guess
        ),
    )

    warm_forward_times.append(
        elapsed(start)
    )


# ========================================
# 3. Backbone sampling at ds = 1e-3
# ========================================

sampling_1e3_times = []

for _ in range(n_runs):

    start = time.perf_counter()

    sample_backbone(
        backbone,
        ds=1e-3,
    )

    sampling_1e3_times.append(
        elapsed(start)
    )


# ========================================
# 4. Backbone sampling at ds = 1e-4
# ========================================

sampling_1e4_times = []

for _ in range(n_runs):

    start = time.perf_counter()

    sample_backbone(
        backbone,
        ds=1e-4,
    )

    sampling_1e4_times.append(
        elapsed(start)
    )


# ========================================
# 5. Direct tip extraction
# ========================================

tip_times = []

for _ in range(n_runs):

    start = time.perf_counter()

    direct_tip = tip_position(
        backbone
    )

    tip_times.append(
        elapsed(start)
    )


# ========================================
# 6. Cold numerical Jacobian
# ========================================

cold_jacobian_times = []

cold_J = None

for _ in range(n_runs):

    start = time.perf_counter()

    cold_J = numerical_position_jacobian_v2(
        robot,
    )

    cold_jacobian_times.append(
        elapsed(start)
    )


# ========================================
# 7. Warm numerical Jacobian
# ========================================

warm_jacobian_times = []

warm_J = None

for _ in range(n_runs):

    start = time.perf_counter()

    warm_J = numerical_position_jacobian_v2(
        robot,
        torsion_initial_guess=(
            torsion_initial_guess
        ),
    )

    warm_jacobian_times.append(
        elapsed(start)
    )


# ========================================
# 8. Numerical consistency checks
# ========================================

samples_1e3 = sample_backbone(
    backbone,
    ds=1e-3,
)

samples_1e4 = sample_backbone(
    backbone,
    ds=1e-4,
)

sampled_tip_1e3 = (
    samples_1e3.position[-1]
)

sampled_tip_1e4 = (
    samples_1e4.position[-1]
)


# ========================================
# Results
# ========================================

print()
print("=" * 78)
print("Timing Summary")
print("=" * 78)
print()

summarize(
    "Warm forward solve:",
    warm_forward_times,
)

summarize(
    "sample_backbone ds=1e-3:",
    sampling_1e3_times,
)

summarize(
    "sample_backbone ds=1e-4:",
    sampling_1e4_times,
)

summarize(
    "Direct tip_position:",
    tip_times,
)

print()

summarize(
    "Cold numerical Jacobian:",
    cold_jacobian_times,
)

summarize(
    "Warm numerical Jacobian:",
    warm_jacobian_times,
)


print()
print("=" * 78)
print("Tip Consistency")
print("=" * 78)
print()

print(
    "Direct tip:"
)

print(
    np.round(
        direct_tip,
        9,
    )
)

print()

print(
    "Sampled tip ds=1e-3:"
)

print(
    np.round(
        sampled_tip_1e3,
        9,
    )
)

print()

print(
    "Sampled tip ds=1e-4:"
)

print(
    np.round(
        sampled_tip_1e4,
        9,
    )
)

print()

print(
    "Max |direct - ds=1e-3|:",
    np.max(
        np.abs(
            direct_tip
            - sampled_tip_1e3
        )
    ),
)

print(
    "Max |direct - ds=1e-4|:",
    np.max(
        np.abs(
            direct_tip
            - sampled_tip_1e4
        )
    ),
)


print()
print("=" * 78)
print("Jacobian Consistency")
print("=" * 78)
print()

print(
    "Cold Jacobian:"
)

print(
    np.round(
        cold_J,
        6,
    )
)

print()

print(
    "Warm Jacobian:"
)

print(
    np.round(
        warm_J,
        6,
    )
)

print()

print(
    "Max |cold J - warm J|:",
    np.max(
        np.abs(
            cold_J
            - warm_J
        )
    ),
)

print()