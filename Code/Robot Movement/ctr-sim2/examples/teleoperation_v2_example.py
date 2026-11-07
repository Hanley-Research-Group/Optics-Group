import numpy as np
import matplotlib.pyplot as plt

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
)

from ctr_sim.control.resolved_rate import (
    resolved_rate_step_v2,
)

from ctr_sim.control.constraints import (
    constrain_joint_step,
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
    insertions=[0.10, 0.18, 0.19],
    rotations=[np.pi / 2, 0.0, 0.0],
)

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=state,
)


# ----------------------------------------
# Teleoperation settings
# ----------------------------------------

cartesian_step = 1e-3  # 1 mm per key press
sampling_ds = 1e-3


# ----------------------------------------
# Mechanics helper
# ----------------------------------------

def solve_robot(
    torsion_initial_guess=None,
):

    backbone = solve_forward_kinematics_v2(
        robot,
        torsion_initial_guess=torsion_initial_guess,
    )

    samples = sample_backbone(
        backbone,
        ds=sampling_ds,
    )

    return backbone, samples


# ----------------------------------------
# Initial solve
# ----------------------------------------

backbone, samples = solve_robot()


# ----------------------------------------
# Visualization
# ----------------------------------------

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(
    111,
    projection="3d",
)

position = samples.position

line, = ax.plot(
    position[:, 0],
    position[:, 1],
    position[:, 2],
    linewidth=3,
)

tip_marker, = ax.plot(
    [position[-1, 0]],
    [position[-1, 1]],
    [position[-1, 2]],
    marker="o",
    markersize=8,
    linestyle="None",
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title(
    "CTR Teleoperation V2\n"
    "Arrows: +/-X/Y   [ / ]: -/+Z   Esc: quit"
)

ax.set_xlim(-0.05, 0.20)
ax.set_ylim(-0.15, 0.15)
ax.set_zlim(-0.05, 0.20)

ax.set_box_aspect((1, 1, 1))

ax.grid(True)


# ----------------------------------------
# Keyboard callback
# ----------------------------------------

def on_key(event):

    global backbone

    if event.key == "right":
        dx = np.array([
            cartesian_step,
            0.0,
            0.0,
        ])

    elif event.key == "left":
        dx = np.array([
            -cartesian_step,
            0.0,
            0.0,
        ])

    elif event.key == "up":
        dx = np.array([
            0.0,
            cartesian_step,
            0.0,
        ])

    elif event.key == "down":
        dx = np.array([
            0.0,
            -cartesian_step,
            0.0,
        ])

    elif event.key == "]":
        dx = np.array([
            0.0,
            0.0,
            cartesian_step,
        ])

    elif event.key == "[":
        dx = np.array([
            0.0,
            0.0,
            -cartesian_step,
        ])

    elif event.key == "escape":
        plt.close(fig)
        return

    else:
        return

    print()
    print("Requested Cartesian step:", dx)

    torsion_initial_guess = (
        backbone
        .torsion_solution
        .base_theta_dot
    )

    #
    # Compute joint-space increment.
    #
    dq = resolved_rate_step_v2(
        robot,
        dx,
        torsion_initial_guess=torsion_initial_guess,
    )

    #
    # Enforce insertion limits before applying
    # the joint-space command.
    #
    dq = constrain_joint_step(
        robot,
        dq,
    )
    
    # print(
    #     "Applied joint step:",
    #     np.round(
    #         dq,
    #         6,
    #     ),
    # )

    n = len(robot.tubes)

    #
    # Update simulated robot state.
    #
    robot.state.insertions += dq[:n]
    robot.state.rotations += dq[n:]

    #
    # Verify that the resulting robot configuration
    # remains physically valid.
    #
    robot.validate_configuration()

    #
    # Recompute robot shape.
    #
    backbone, new_samples = solve_robot(
        torsion_initial_guess=torsion_initial_guess,
    )

    new_position = new_samples.position

    #
    # Update visualization.
    #
    line.set_data(
        new_position[:, 0],
        new_position[:, 1],
    )

    line.set_3d_properties(
        new_position[:, 2],
    )

    tip_marker.set_data(
        [new_position[-1, 0]],
        [new_position[-1, 1]],
    )

    tip_marker.set_3d_properties(
        [new_position[-1, 2]],
    )

    print(
        "Tip position:",
        np.round(
            new_position[-1],
            5,
        ),
    )

    print(
        "Insertions:",
        np.round(
            robot.state.insertions,
            5,
        ),
    )

    print(
        "Rotations:",
        np.round(
            robot.state.rotations,
            5,
        ),
    )

    fig.canvas.draw_idle()


# ----------------------------------------
# Start teleoperation
# ----------------------------------------

fig.canvas.mpl_connect(
    "key_press_event",
    on_key,
)

print()
print("CTR Teleoperation V2")
print("---------------------")
print("Right / Left : +X / -X")
print("Up / Down    : +Y / -Y")
print("] / [        : +Z / -Z")
print("Escape       : quit")
print()
print(
    "Initial tip:",
    np.round(
        samples.position[-1],
        5,
    ),
)

plt.show()