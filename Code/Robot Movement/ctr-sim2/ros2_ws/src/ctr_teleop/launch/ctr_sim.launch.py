import os

from ament_index_python.packages import (
    get_package_share_directory,
)

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():

    #
    # Locate the installed ctr_teleop package.
    #
    package_share = (
        get_package_share_directory(
            "ctr_teleop"
        )
    )

    #
    # Locate the saved RViz configuration.
    #
    rviz_config = os.path.join(
        package_share,
        "config",
        "ctr_sim.rviz",
    )

    #
    # Receive Xbox controller data from the
    # Windows joy_sender process.
    #
    joy_receiver = Node(
        package="ctr_teleop",
        executable="joy_receiver",
        name="joy_receiver",
        output="screen",
    )

    #
    # Convert Joy messages into Cartesian
    # velocity commands.
    #
    cartesian_teleop = Node(
        package="ctr_teleop",
        executable="cartesian_teleop",
        name="cartesian_teleop",
        output="screen",
    )

    #
    # Run the CTR simulation and control loop.
    #
    ctr_simulator = Node(
        package="ctr_teleop",
        executable="ctr_simulator",
        name="ctr_simulator",
        output="screen",
    )

    #
    # Start RViz using the saved CTR
    # visualization configuration.
    #
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            rviz_config,
        ],
        output="screen",
    )

    return LaunchDescription([
        joy_receiver,
        cartesian_teleop,
        ctr_simulator,
        rviz,
    ])