#!/usr/bin/env python3

import numpy as np

import rclpy

from scipy.spatial.transform import Rotation
from rclpy.node import Node
from geometry_msgs.msg import (
    Twist,
    Point,
    PoseStamped,
    TransformStamped,
)

from visualization_msgs.msg import (
    Marker,
    MarkerArray,
)


from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import (
    StaticTransformBroadcaster,
)

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

from ctr_sim.control.jacobian import (
    numerical_position_jacobian_v2,
)

from ctr_sim.control.constraints import (
    constrain_joint_step,
)


#
# Simulated robot definition.
#
# Every tube is described by the lengths of its three sections, measured
# from the proximal end to the distal tip (all in metres):
#
#   length_before_curve   straight section
#   curve_length          curved section (curvature = precurvature, 1/m)
#   length_after_curve    straight section out to the tip
#
# The tube length is the sum of the three. A tube that is curved over its
# whole length uses length_before_curve = 0 and length_after_curve = 0;
# a straight tube uses precurvature = 0.
#
# Tubes are listed from outermost to innermost, and each tube must be
# longer than the one outside it. Rebuild (colcon build) after editing.
#
# NOTE: these values reproduce the original uniformly-curved Nitinol
# example robot. Replace them with the parameters of your tubes. The
# materials available here are defined in create_robot().
#
TUBE_CONFIG = [
    {
        "name": "OuterTube",
        "length_before_curve": 0.0,
        "curve_length": 0.16,
        "length_after_curve": 0.0,
        "precurvature": 15.0,
        "outer_diameter": 3.0e-3,
        "inner_diameter": 2.8e-3,
        "material": "nitinol",
    },
    {
        "name": "MiddleTube",
        "length_before_curve": 0.0,
        "curve_length": 0.18,
        "length_after_curve": 0.0,
        "precurvature": 10.0,
        "outer_diameter": 2.0e-3,
        "inner_diameter": 1.8e-3,
        "material": "nitinol",
    },
    {
        "name": "InnerTube",
        "length_before_curve": 0.0,
        "curve_length": 0.20,
        "length_after_curve": 0.0,
        "precurvature": 0.0,
        "outer_diameter": 1.6e-3,
        "inner_diameter": 0.0,
        "material": "fiber",
    },
]

# Initial tube insertions (m) and rotations (rad), outermost first.
INITIAL_INSERTIONS = [0.10, 0.145, 0.19]
INITIAL_ROTATIONS = [np.pi / 2, 0.0, 0.0]


class CTRSimulator(Node):

    def __init__(self):

        super().__init__(
            "ctr_simulator"
        )

        #
        # Static TF broadcaser for the CTR base frame
        #
        self.static_tf_broadcaster = (
            StaticTransformBroadcaster(
                self
            )
        )
        
        #
        # Dynamic TF broadcaster for moving robot frames
        #

        self.tf_broadcaster = (
            TransformBroadcaster(
                self
            )
        )

        self.publish_base_transform()

        #
        # Control settings.
        #
        self.control_rate = 10.0

        self.dt = (
            1.0 / self.control_rate
        )

        #
        # Jacobian caching.
        #
        # Recompute the numerical Jacobian every three
        # active control steps and reuse it in between.
        #
        self.jacobian_update_interval = 3

        self.cached_jacobian = None

        self.jacobian_age = 0

        #
        # ROS state publication settings.
        #
        self.state_publish_rate = 20.0

        self.state_publish_dt = (
            1.0 / self.state_publish_rate
        )


        #
        # Command timeout.
        #
        # If velocity commands stop arriving, force
        # the commanded velocity to zero.
        #
        self.command_timeout = 0.5

        self.last_command_time = None


        #
        # Latest commanded Cartesian velocity.
        #
        self.cartesian_velocity = np.zeros(3)

        #
        # Create the simulated robot.
        #
        self.robot = self.create_robot()

        #
        # Solve the initial robot configuration.
        #
        self.backbone = (
            solve_forward_kinematics_v2(
                self.robot,
            )
        )

        #
        # Subscribe to Cartesian velocity commands.
        #
        self.subscription = (
            self.create_subscription(
                Twist,
                "/ctr/cartesian_velocity",
                self.velocity_callback,
                1,
            )
        )

        #
        # Publish the sampled CTR backbone for visualization.
        #
        self.backbone_publisher = (
            self.create_publisher(
                MarkerArray,
                "/ctr/backbone",
                1,
            )
        )

        self.trajectory_publisher = (
            self.create_publisher(
                Marker,
                "/ctr/tip_trajectory",
                1,
            )
        )

        #
        # Publish the CTR tip pose.
        #

        self.tip_pose_publisher = (
            self.create_publisher(
                PoseStamped,
                "/ctr/tip_pose",
                1,
            )
        )


        self.latest_samples = sample_backbone(
            self.backbone,
            ds=1e-3,
        )

        #
        # Tip trajectory history.
        #
        # Store the most recent tip positions for
        # visualization in RViz.
        #
        self.max_trajectory_points = 500

        self.tip_trajectory = []


        initial_tip = (
            self.latest_samples
            .position[-1]
            .copy()
        )

        self.tip_trajectory.append(
            initial_tip
        )


        #
        # Run the control loop at a fixed rate.
        #
        self.control_timer = self.create_timer(
            self.dt,
            self.control_callback,
        )

        #
        # Publish the latest simulated state independently
        # of the mechanics/control computation rate.
        #
        self.state_timer = self.create_timer(
            self.state_publish_dt,
            self.state_publish_callback,
        )

        self.get_logger().info(
            "CTR simulator started."
        )

        self.get_logger().info(
            f"Control rate: "
            f"{self.control_rate:.1f} Hz"
        )

    def publish_base_transform(self):

        transform = TransformStamped()

        transform.header.stamp = (
            self.get_clock().now().to_msg()
        )

        transform.header.frame_id = "world"

        transform.child_frame_id = "ctr_base"

        #
        # ctr_base is currently coincident with world.
        #
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0

        #
        # Identity rotation quaternion.
        #
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        self.static_tf_broadcaster.sendTransform(
            transform
        )

    def create_robot(self):

        #
        # Materials.
        #
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

        materials = {
            "nitinol": nitinol,
            "fiber": fiber,
        }

        #
        # Tubes, outermost first. Each tube is straight for
        # length_before_curve, curved (at `precurvature`) for
        # curve_length, then straight for length_after_curve, measured
        # from the tube's proximal end to its distal tip. The tube's
        # total length is the sum of the three.
        #
        tubes = [
            Tube.from_lengths(
                name=cfg["name"],
                length_before_curve=cfg["length_before_curve"],
                curve_length=cfg["curve_length"],
                length_after_curve=cfg["length_after_curve"],
                precurvature=cfg["precurvature"],
                outer_diameter=cfg["outer_diameter"],
                inner_diameter=cfg["inner_diameter"],
                material=materials[cfg["material"]],
            )
            for cfg in TUBE_CONFIG
        ]

        state = CTRState(
            insertions=list(INITIAL_INSERTIONS),
            rotations=list(INITIAL_ROTATIONS),
        )

        return ConcentricTubeRobot(
            tubes=tubes,
            state=state,
        )

    def velocity_callback(
        self,
        msg: Twist,
    ):

        #
        # Store the newest commanded Cartesian velocity.
        #
        self.cartesian_velocity = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
        ])

        self.last_command_time = (
            self.get_clock().now()
        )


    def publish_backbone(
        self,
        samples,
    ):

        marker_array = MarkerArray()

        stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        #
        # Clear all previously published segment markers.
        #
        clear_marker = Marker()

        clear_marker.header.frame_id = "ctr_base"
        clear_marker.header.stamp = stamp

        clear_marker.action = Marker.DELETEALL

        marker_array.markers.append(
            clear_marker
        )

        #
        # Distinct colors for mechanics segments.
        #
        colors = [
            (1.0, 0.2, 0.2),  # red
            (0.2, 1.0, 0.2),  # green
            (0.2, 0.4, 1.0),  # blue
            (1.0, 0.8, 0.2),  # yellow
            (1.0, 0.2, 1.0),  # magenta
            (0.2, 1.0, 1.0),  # cyan
        ]

        for i, segment_solution in enumerate(
            self.backbone.segments
        ):

            segment = (
                segment_solution.segment
            )

            #
            # Select sampled points belonging to
            # this mechanics segment.
            #
            if i == len(self.backbone.segments) - 1:

                mask = (
                    (samples.s >= segment.start)
                    & (samples.s <= segment.end)
                )

            else:

                mask = (
                    (samples.s >= segment.start)
                    & (samples.s < segment.end)
                )

            marker = Marker()

            marker.header.frame_id = "ctr_base"
            marker.header.stamp = stamp

            marker.ns = "ctr_segments"

            #
            # Unique marker ID for this segment.
            #
            marker.id = i

            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.scale.x = 0.003

            color = colors[
                i % len(colors)
            ]

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0

            for position in samples.position[mask]:

                point = Point()

                point.x = float(
                    position[0]
                )

                point.y = float(
                    position[1]
                )

                point.z = float(
                    position[2]
                )

                marker.points.append(
                    point
                )

            marker_array.markers.append(
                marker
            )

        self.backbone_publisher.publish(
            marker_array
        )

    def publish_tip_state(
        self,
        samples,
    ):

        #
        # Extract tip position and orientation from
        # the sampled mechanics solution.
        # 
        tip_position = (
            samples.position[-1]
        )

        tip_rotation = (
            samples.rotation[-1]
        )

        #
        # Convert the 3x3 rotation matrix into a
        # ROS-compatible quaternion [x, y, z, w].
        #
        quaternion = (
            Rotation
            .from_matrix(
                tip_rotation
            )
            .as_quat()
        )

        #
        # Use the same timestamp for the PoseStamped
        # and TF transform.
        #
        stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        #
        # Publish /ctr/tip_pose.
        #
        pose = PoseStamped()

        pose.header.stamp = stamp
        pose.header.frame_id = "ctr_base"

        pose.pose.position.x = float(
            tip_position[0]
        )

        pose.pose.position.y = float(
            tip_position[1]
        )

        pose.pose.position.z = float(
            tip_position[2]
        )

        pose.pose.orientation.x = float(
            quaternion[0]
        )

        pose.pose.orientation.y = float(
            quaternion[1]
        )

        pose.pose.orientation.z = float(
            quaternion[2]
        )

        pose.pose.orientation.w = float(
            quaternion[3]
        )

        self.tip_pose_publisher.publish(
            pose
        )

        #
        # Publish ctr_base -> ctr_tip.
        #
        transform = TransformStamped()

        transform.header.stamp = stamp
        transform.header.frame_id = "ctr_base"

        transform.child_frame_id = "ctr_tip"

        transform.transform.translation.x = float(
            tip_position[0]
        )

        transform.transform.translation.y = float(
            tip_position[1]
        )

        transform.transform.translation.z = float(
            tip_position[2]
        )

        transform.transform.rotation.x = float(
            quaternion[0]
        )

        transform.transform.rotation.y = float(
            quaternion[1]
        )

        transform.transform.rotation.z = float(
            quaternion[2]
        )

        transform.transform.rotation.w = float(
            quaternion[3]
        )

        self.tf_broadcaster.sendTransform(
            transform
        )


    def publish_tip_trajectory(self):

        marker = Marker()

        marker.header.frame_id = "ctr_base"

        marker.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        marker.ns = "ctr_tip_trajectory"
        marker.id = 0

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        #
        # Trajectory line width: 1.5 mm.
        #
        marker.scale.x = 0.0015

        #
        # White trajectory line.
        #
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        for position in self.tip_trajectory:

            point = Point()

            point.x = float(
                position[0]
            )

            point.y = float(
                position[1]
            )

            point.z = float(
                position[2]
            )

            marker.points.append(
                point
            )

        self.trajectory_publisher.publish(
            marker
        )


    def control_callback(self):
        
        #
        # Current ROS time for the command watchdog
        #
        now = self.get_clock().now()

        #
        # Stop motion if Cartesian commands have
        # stopped arriving. Command watchdog. 
        #
        if self.last_command_time is None:

            self.cartesian_velocity[:] = 0.0

        else:

            command_age = (
                now - self.last_command_time
            ).nanoseconds * 1e-9

            if command_age > self.command_timeout:

                self.cartesian_velocity[:] = 0.0

        #
        # If zero velocity is commanded, there is
        # no mechanics solve to perform.
        #
        if np.allclose(
            self.cartesian_velocity,
            0.0,
        ):
            return

        #
        # Convert velocity into the Cartesian
        # displacement requested for one nominal
        # control interval.
        #
        dx = (
            self.cartesian_velocity
            * self.dt
        )

        #
        # Use the current torsion solution to
        # warm-start the next mechanics solve.
        #
        torsion_initial_guess = (
            self.backbone
            .torsion_solution
            .base_theta_dot
        )

        control_start = (
            self.get_clock().now()
        )

        #
        # Refresh the numerical Jacobian when necessary.
        #
        jacobian_refreshed = False
        jacobian_time = 0.0

        if (
            self.cached_jacobian is None
            or self.jacobian_age
            >= self.jacobian_update_interval
        ):

            jacobian_start = (
                self.get_clock().now()
            )

            self.cached_jacobian = (
                numerical_position_jacobian_v2(
                    self.robot,
                    torsion_initial_guess=(
                        torsion_initial_guess
                    ),
                )
            )

            jacobian_end = (
                self.get_clock().now()
            )

            jacobian_time = (
                jacobian_end - jacobian_start
            ).nanoseconds * 1e-9
            
            self.jacobian_age = 0

            jacobian_refreshed = True


        #
        # Compute the resolved-rate joint step using
        # the current cached Jacobian.
        #
        dq = resolved_rate_step_v2(
            self.robot,
            dx,
            jacobian=(
                self.cached_jacobian
            ),
        )

        self.jacobian_age += 1


        #
        # Enforce insertion constraints.
        #
        dq = constrain_joint_step(
            self.robot,
            dq,
        )

        n = len(
            self.robot.tubes
        )

        #
        # Update simulated joint state.
        #
        self.robot.state.insertions += (
            dq[:n]
        )

        self.robot.state.rotations += (
            dq[n:]
        )

        #
        # Verify that the updated configuration
        # remains valid.
        #
        self.robot.validate_configuration()

        #
        # Recompute the robot shape using the
        # previous torsion solution as a warm start.
        #
        self.backbone = (
            solve_forward_kinematics_v2(
                self.robot,
                torsion_initial_guess=(
                    torsion_initial_guess
                ),
            )
        )

        control_end = (
            self.get_clock().now()
        )

        control_compute_time = (
            control_end - control_start
        ).nanoseconds * 1e-9


        self.latest_samples = sample_backbone(
            self.backbone,
            ds=1e-3,
        )

        tip = (
            self.latest_samples
            .position[-1]
        )

        #
        # Add the newest tip position to the
        # trajectory history.
        #
        self.tip_trajectory.append(
            tip.copy()
        )

        #
        # Limit trajectory history length.
        #
        if (
            len(self.tip_trajectory)
            > self.max_trajectory_points
        ):

            self.tip_trajectory.pop(0)

        
        self.get_logger().info(
            "Tip: "
            f"[{tip[0]:+.5f}, "
            f"{tip[1]:+.5f}, "
            f"{tip[2]:+.5f}], "
            f"control={control_compute_time:.3f}s, "
            f"jacobian={jacobian_time:.3f}s, "
            f"refresh={jacobian_refreshed}, "
            f"age={self.jacobian_age}"
        )

    def state_publish_callback(self):

        self.publish_backbone(
            self.latest_samples
        )

        self.publish_tip_state(
            self.latest_samples
        )

        self.publish_tip_trajectory()


def main(args=None):

    rclpy.init(
        args=args
    )

    node = CTRSimulator()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            
            rclpy.shutdown()


if __name__ == "__main__":
    main()