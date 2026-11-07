import time
from copy import deepcopy

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
    tip_position,
)

from ctr_sim.control.jacobian import (
    numerical_position_jacobian_v2,
)

from ctr_sim.control.constraints import (
    constrain_joint_step,
)


# ----------------------------------------
# Timing helper
# ----------------------------------------

def elapsed(start):
    return time.perf_counter() - start


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
# Experiment settings
# ----------------------------------------

n_steps = 6

#
# Representative full-stick Cartesian command:
#
# max_speed = 0.01 m/s
# control dt = 0.2 s
#
# Therefore:
#
# dx = 0.002 m = 2 mm
#
dx = np.array([
    0.002,
    0.0,
    0.0,
])


# ========================================
# Initial mechanics solution
# ========================================

print()
print("=" * 78)
print("Jacobian Reuse Experiment")
print("=" * 78)

print()
print("Solving initial robot configuration...")

backbone = solve_forward_kinematics_v2(
    robot,
)

initial_tip = tip_position(
    backbone
)

torsion_initial_guess = (
    backbone
    .torsion_solution
    .base_theta_dot
    .copy()
)


# ========================================
# Compute initial Jacobian J0
# ========================================

print()
print("Computing initial warm-started Jacobian J0...")

start = time.perf_counter()

J_cached = numerical_position_jacobian_v2(
    robot,
    torsion_initial_guess=(
        torsion_initial_guess
    ),
)

initial_jacobian_time = elapsed(
    start
)

#
# Save the configuration at which J0 was computed.
#
n = len(robot.tubes)

q_jacobian = np.concatenate([
    robot.state.insertions.copy(),
    robot.state.rotations.copy(),
])


print(
    f"Initial Jacobian time: "
    f"{initial_jacobian_time:.6f} s"
)

print()

print(
    "Initial tip:",
    np.round(
        initial_tip,
        6,
    ),
)

print()

print("Initial Jacobian J0:")

print(
    np.round(
        J_cached,
        6,
    )
)


# ========================================
# Repeated teleoperation steps
# ========================================

print()
print("=" * 78)
print("Repeated +X Teleoperation Steps")
print("=" * 78)


for step in range(1, n_steps + 1):

    #
    # Current mechanics solution provides a good
    # torsion initial guess for the current robot.
    #
    torsion_initial_guess = (
        backbone
        .torsion_solution
        .base_theta_dot
        .copy()
    )

    current_tip = tip_position(
        backbone
    )

    #
    # Compute the true Jacobian at the current
    # configuration for comparison.
    #
    start = time.perf_counter()

    J_fresh = numerical_position_jacobian_v2(
        robot,
        torsion_initial_guess=(
            torsion_initial_guess
        ),
    )

    fresh_jacobian_time = elapsed(
        start
    )

    #
    # Measure how much the Jacobian has changed
    # relative to the cached J0.
    #
    jacobian_difference = (
        J_fresh - J_cached
    )

    jacobian_absolute_change = (
        np.linalg.norm(
            jacobian_difference
        )
    )

    cached_jacobian_norm = (
        np.linalg.norm(
            J_cached
        )
    )

    if cached_jacobian_norm > 0.0:

        jacobian_relative_change = (
            jacobian_absolute_change
            / cached_jacobian_norm
        )

    else:

        jacobian_relative_change = 0.0

    #
    # Compute the joint-space command using the
    # original cached Jacobian.
    #
    dq_cached = (
        np.linalg.pinv(
            J_cached
        )
        @ dx
    )

    dq_cached = constrain_joint_step(
        robot,
        dq_cached,
    )

    #
    # Compute the joint-space command using the
    # true current Jacobian.
    #
    dq_fresh = (
        np.linalg.pinv(
            J_fresh
        )
        @ dx
    )

    dq_fresh = constrain_joint_step(
        robot,
        dq_fresh,
    )

    #
    # Compare the resulting joint commands.
    #
    dq_difference = (
        dq_cached - dq_fresh
    )

    dq_absolute_difference = (
        np.linalg.norm(
            dq_difference
        )
    )

    dq_fresh_norm = (
        np.linalg.norm(
            dq_fresh
        )
    )

    if dq_fresh_norm > 0.0:

        dq_relative_difference = (
            dq_absolute_difference
            / dq_fresh_norm
        )

    else:

        dq_relative_difference = 0.0

    #
    # Measure how far the robot configuration has
    # moved from the configuration where J0 was
    # originally computed.
    #
    q_current = np.concatenate([
        robot.state.insertions.copy(),
        robot.state.rotations.copy(),
    ])

    q_change = (
        q_current - q_jacobian
    )

    #
    # Predict the requested Cartesian displacement
    # using each Jacobian and its associated command.
    #
    dx_predicted_cached = (
        J_fresh
        @ dq_cached
    )

    dx_predicted_fresh = (
        J_fresh
        @ dq_fresh
    )

    #
    # Print diagnostics before applying the step.
    #
    print()
    print(f"Step {step}")
    print("-" * 78)

    print(
        "Current tip:",
        np.round(
            current_tip,
            6,
        ),
    )

    print(
        f"Fresh Jacobian time:       "
        f"{fresh_jacobian_time:.6f} s"
    )

    print(
        f"Jacobian relative change:  "
        f"{100.0 * jacobian_relative_change:.3f} %"
    )

    print(
        f"dq relative difference:    "
        f"{100.0 * dq_relative_difference:.3f} %"
    )

    print(
        "Configuration change from J0:"
    )

    print(
        np.round(
            q_change,
            6,
        )
    )

    print(
        "dq cached:"
    )

    print(
        np.round(
            dq_cached,
            6,
        )
    )

    print(
        "dq fresh:"
    )

    print(
        np.round(
            dq_fresh,
            6,
        )
    )

    print(
        "Requested dx:"
    )

    print(
        np.round(
            dx,
            6,
        )
    )

    print(
        "Predicted dx using cached command:"
    )

    print(
        np.round(
            dx_predicted_cached,
            6,
        )
    )

    print(
        "Predicted dx using fresh command:"
    )

    print(
        np.round(
            dx_predicted_fresh,
            6,
        )
    )

    #
    # Advance the simulated robot using the FRESH
    # Jacobian command.
    #
    # This keeps the experiment's trajectory based
    # on the best available Jacobian. The cached
    # Jacobian is only being evaluated, not used to
    # generate the trajectory.
    #
    robot.state.insertions += (
        dq_fresh[:n]
    )

    robot.state.rotations += (
        dq_fresh[n:]
    )

    robot.validate_configuration()

    #
    # Recompute the true nonlinear mechanics state.
    #
    backbone = solve_forward_kinematics_v2(
        robot,
        torsion_initial_guess=(
            torsion_initial_guess
        ),
    )


# ========================================
# Final result
# ========================================

final_tip = tip_position(
    backbone
)

print()
print("=" * 78)
print("Final Result")
print("=" * 78)

print()

print(
    "Initial tip:",
    np.round(
        initial_tip,
        6,
    ),
)

print(
    "Final tip:",
    np.round(
        final_tip,
        6,
    ),
)

print(
    "Total tip displacement:",
    np.round(
        final_tip - initial_tip,
        6,
    ),
)

print()