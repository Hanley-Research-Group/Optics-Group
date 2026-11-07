#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import socket
import json

class JoyReceiver(Node):
    def __init__(self):
        super().__init__('joy_receiver')
        self.pub = self.create_publisher(Joy, '/joy', 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', 9870))
        self.sock.settimeout(0.1)

        self.timer = self.create_timer(0.02, self.timer_callback)  # 50 Hz
        self.get_logger().info('Joy receiver listening on UDP port 9870')

    def timer_callback(self):
        try:
            data, _ = self.sock.recvfrom(4096)
            msg_data = json.loads(data.decode())

            joy_msg = Joy()
            joy_msg.header.stamp = self.get_clock().now().to_msg()
            joy_msg.axes = [float(a) for a in msg_data['axes']]
            joy_msg.buttons = [int(b) for b in msg_data['buttons']]

            self.pub.publish(joy_msg)
        except socket.timeout:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = JoyReceiver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()