#!/usr/bin/env python3

import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


class CartesianTeleop(Node):

    def __init__(self):

        super().__init__(
            "cartesian_teleop"
        )

        #
        # Teleoperation settings.
        #
        self.deadzone = 0.10

        self.deadman_threshold = 0.50

        #
        # Maximum commanded Cartesian tip speed.
        #
        # 0.01 m/s = 10 mm/s
        #
        self.max_speed = 0.01

        #
        # Xbox mapping from joy_sender.py:
        #
        # axes[0] = left stick X
        # axes[1] = left stick Y
        # axes[2] = left trigger
        # axes[3] = right stick X
        # axes[4] = right stick Y
        # axes[5] = right trigger
        #
        self.subscription = self.create_subscription(
            Joy,
            "/joy",
            self.joy_callback,
            1,
        )

        self.publisher = self.create_publisher(
            Twist,
            "/ctr/cartesian_velocity",
            1,
        )

        self.get_logger().info(
            "CTR Cartesian teleoperation node started."
        )

        self.get_logger().info(
            "Hold right trigger to enable motion."
        )

    def apply_deadzone(
        self,
        value: float,
    ) -> float:

        if abs(value) < self.deadzone:
            return 0.0

        return value

    def joy_callback(
        self,
        msg: Joy,
    ):

        #
        # Ignore malformed Joy messages.
        #
        if len(msg.axes) < 6:
            return

        twist = Twist()

        right_trigger = msg.axes[5]

        #
        # Deadman released:
        # publish zero velocity.
        #
        if (
            right_trigger
            <= self.deadman_threshold
        ):

            self.publisher.publish(
                twist
            )

            return

        #
        # Deadman held:
        # map the sticks to Cartesian velocity.
        #
        x_input = self.apply_deadzone(
            msg.axes[0]
        )

        y_input = self.apply_deadzone(
            msg.axes[1]
        )

        z_input = self.apply_deadzone(
            msg.axes[4]
        )

        twist.linear.x = (
            self.max_speed
            * x_input
        )

        twist.linear.y = (
            self.max_speed
            * y_input
        )

        twist.linear.z = (
            self.max_speed
            * z_input
        )

        # self.get_logger().info(
        #     f"vx={twist.linear.x:+.4f}, "
        #     f"vy={twist.linear.y:+.4f}, "
        #     f"vz={twist.linear.z:+.4f}"
        # )

        self.publisher.publish(
            twist
        )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = CartesianTeleop()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()