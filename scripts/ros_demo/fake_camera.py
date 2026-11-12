#!/usr/bin/env python3
"""Synthetic camera for the turtlesim demo (turtlesim has no camera).

Renders what a camera on the turtle would see in a 11x11 arena with one red cone and publishes it as
sensor_msgs/Image (rgb8) on /camera/image_raw and as sensor_msgs/CompressedImage (png) on /camera/image_raw/compressed.
It reuses hermes-robots' pure-Python raycaster, so it needs nothing beyond rclpy.
"""
import importlib.util
import math
import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from turtlesim.msg import Pose

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGING = os.path.join(HERE, "..", "..", "imaging.py")
spec = importlib.util.spec_from_file_location("hermes_robots_imaging", IMAGING)
imaging = importlib.util.module_from_spec(spec)
sys.modules["hermes_robots_imaging"] = imaging
spec.loader.exec_module(imaging)

CENTER = 5.5                      # turtlesim arena is 0..11 in both axes
CONE = (8.5 - CENTER, 7.5 - CENTER)   # the red cone, in arena-centred coordinates
W, H = 320, 240


class FakeCamera(Node):
    def __init__(self):
        super().__init__("fake_camera")
        self.pose = None
        self.create_subscription(Pose, "/turtle1/pose", self.on_pose, 10)
        self.raw = self.create_publisher(Image, "/camera/image_raw", 1)
        self.comp = self.create_publisher(CompressedImage, "/camera/image_raw/compressed", 1)
        self.create_timer(0.5, self.tick)
        self.get_logger().info("fake camera up: red cone at turtlesim (8.5, 7.5)")

    def on_pose(self, msg):
        self.pose = msg

    def tick(self):
        if self.pose is None:
            return
        x, y, yaw = self.pose.x - CENTER, self.pose.y - CENTER, self.pose.theta
        stamp = self.get_clock().now().to_msg()
        rgb = imaging.render_forward(x, y, yaw, 5.5, 5.5, [CONE], W, H, raw_rgb=True)
        m = Image()
        m.header.stamp, m.header.frame_id = stamp, "camera_link"
        m.width, m.height, m.encoding, m.is_bigendian, m.step = W, H, "rgb8", 0, W * 3
        m.data = rgb
        self.raw.publish(m)
        c = CompressedImage()
        c.header = m.header
        c.format = "png"
        c.data = imaging.render_forward(x, y, yaw, 5.5, 5.5, [CONE], W, H)
        self.comp.publish(c)


def main():
    rclpy.init()
    rclpy.spin(FakeCamera())


if __name__ == "__main__":
    main()
