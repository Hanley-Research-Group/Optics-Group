"""
Visualization utilities for concentric tube robots.
"""

import matplotlib.pyplot as plt
import numpy as np

from ctr_sim.solution import CTRSolution

from ctr_sim.backbone_samples import BackboneSamples
from ctr_sim.robot import ConcentricTubeRobot


def set_axes_equal(ax) -> None:
    """
    Set equal scaling for all three axes.
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    radius = 0.5 * max(
        x_range,
        y_range,
        z_range,
    )

    ax.set_xlim3d(
        x_mid - radius,
        x_mid + radius,
    )

    ax.set_ylim3d(
        y_mid - radius,
        y_mid + radius,
    )

    ax.set_zlim3d(
        z_mid - radius,
        z_mid + radius,
    )


def plot_frame(
    ax,
    position,
    rotation,
    length=0.01,
):
    """
    Plot a coordinate frame.
    """

    origin = position

    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]
    z_axis = rotation[:, 2]

    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        x_axis[0],
        x_axis[1],
        x_axis[2],
        length=length,
        color="r",
        normalize=True,
    )

    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        y_axis[0],
        y_axis[1],
        y_axis[2],
        length=length,
        color="g",
        normalize=True,
    )

    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        z_axis[0],
        z_axis[1],
        z_axis[2],
        length=length,
        color="b",
        normalize=True,
    )


def plot_backbone(
    solution: CTRSolution,
    show_frames: bool = True,
) -> None:
    """
    Plot the robot backbone.
    """

    position = solution.backbone.position
    s = solution.backbone.s

    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # Backbone
    ax.plot(
        x,
        y,
        z,
        linewidth=3,
        color="tab:blue",
    )

    # Base
    ax.scatter(
        x[0],
        y[0],
        z[0],
        s=60,
        color="k",
        label="Base",
    )

    # Tube tips
    tube_colors = [
        "tab:orange",
        "tab:green",
        "tab:red",
    ]

    for i, beta in enumerate(solution.robot.state.insertions[:-1]):

        idx = np.argmin(np.abs(s - beta))

        ax.scatter(
            x[idx],
            y[idx],
            z[idx],
            s=70,
            marker="^",
            color=tube_colors[i],
            label=f"Tube {i+1} Tip",
        )

    # Robot tip
    ax.scatter(
        x[-1],
        y[-1],
        z[-1],
        s=70,
        color="tab:red",
        label="Tip",
    )

    # Coordinate frames
    if show_frames:

        plot_frame(
            ax,
            solution.backbone.position[0],
            solution.backbone.rotation[0],
            length=0.01,
        )

        plot_frame(
            ax,
            solution.tip_position,
            solution.tip_rotation,
            length=0.01,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title("CTR Backbone")

    ax.view_init(
        elev=25,
        azim=-60,
    )

    ax.grid(True)

    set_axes_equal(ax)

    ax.legend()

    plt.tight_layout()

    plt.show()


def plot_backbone_samples(
    samples: BackboneSamples,
    robot: ConcentricTubeRobot,
    show_frames: bool = True,
) -> None:
    """
    Plot a sampled backbone representation.
    """

    position = samples.position
    rotation = samples.rotation
    s = samples.s

    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # Backbone
    ax.plot(
        x,
        y,
        z,
        linewidth=3,
        color="tab:blue",
    )

    # Base
    ax.scatter(
        x[0],
        y[0],
        z[0],
        s=60,
        color="k",
        label="Base",
    )

    # Tube tips
    tube_colors = [
        "tab:orange",
        "tab:green",
        "tab:red",
    ]

    for i, beta in enumerate(robot.state.insertions[:-1]):

        idx = np.argmin(np.abs(s - beta))

        ax.scatter(
            x[idx],
            y[idx],
            z[idx],
            s=70,
            marker="^",
            color=tube_colors[i],
            label=f"Tube {i + 1} Tip",
        )

    # Robot tip
    ax.scatter(
        x[-1],
        y[-1],
        z[-1],
        s=70,
        color="tab:red",
        label="Tip",
    )

    # Coordinate frames
    if show_frames:

        plot_frame(
            ax,
            position[0],
            rotation[0],
            length=0.01,
        )

        plot_frame(
            ax,
            position[-1],
            rotation[-1],
            length=0.01,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title("CTR Backbone (V2)")

    ax.view_init(
        elev=25,
        azim=-60,
    )

    ax.grid(True)

    set_axes_equal(ax)

    ax.legend()

    plt.tight_layout()

    plt.show()