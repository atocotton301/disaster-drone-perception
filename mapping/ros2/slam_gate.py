"""Forward only image tuples with successful, timestamp-matched RGB-D odometry."""
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from rtabmap_msgs.msg import OdomInfo
from std_msgs.msg import String
from message_filters import Subscriber, ApproximateTimeSynchronizer
from core import load_config, odom_valid, stamp_s, TrackingGate, validate_pair


class Gate(Node):
    def __init__(self):
        super().__init__('disaster_slam_gate')
        self.c = load_config(os.environ['DISASTER_CONFIG'])
        self.health = TrackingGate(self.c['stale_s'])
        self.last_stamp = -1.
        self.failure_stamp = -1.
        self.reason = 'Waiting for synchronized RGB/Depth/Info/Odometry/OdomInfo'
        inputs = [(Image, '/camera/color/image_raw', 'rgb'),
                  (Image, '/camera/aligned_depth_to_color/image_raw', 'depth'),
                  (CameraInfo, '/camera/color/camera_info', 'info'),
                  (Odometry, '/odom', 'odom'), (OdomInfo, '/odom_info', 'odom_info')]
        self.pubs = [self.create_publisher(t, '/disaster/slam/'+n, 5) for t, _, n in inputs]
        self.subs = [Subscriber(self, t, topic, qos_profile=qos_profile_sensor_data) for t, topic, _ in inputs]
        self.sync = ApproximateTimeSynchronizer(self.subs, 45, self.c['sync_slop_s'])
        self.sync.registerCallback(self.process)
        # Failure info may arrive without Odometry; do not wait for a complete tuple to flag loss.
        self.create_subscription(OdomInfo, '/odom_info', self.on_info, qos_profile_sensor_data)
        self.status_pub = self.create_publisher(String, '/disaster/tracking', 5)
        self.create_timer(.2, self.publish_status)

    def on_info(self, msg):
        if msg.lost:
            self.failure_stamp = max(self.failure_stamp, stamp_s(msg))
            self.health.observe(False, time.monotonic())
            self.reason = 'RTAB-Map visual odometry lost; no SLAM input forwarded'

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps(dict(status=self.health.status(time.monotonic()),
            reason=self.reason, last_accepted_stamp=self.last_stamp))))

    def process(self, rgb, depth, info, odom, odom_info):
        try:
            validate_pair((rgb.height, rgb.width), (depth.height, depth.width),
                [m.header.frame_id for m in (rgb, depth, info)],
                [stamp_s(m) for m in (rgb, depth, info, odom, odom_info)], self.c['sync_slop_s'])
            p, q = odom.pose.pose.position, odom.pose.pose.orientation
            good = odom_valid(odom_info.lost, odom.pose.covariance, [p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            if not good:
                raise ValueError('Odometry lost or invalid covariance/pose')
            if stamp_s(rgb) <= self.last_stamp:
                raise ValueError('Nonmonotonic timestamp: start a new session for replay/reset')
            if stamp_s(rgb) <= self.failure_stamp:
                raise ValueError('Discarding delayed tuple preceding latest odometry failure')
            self.last_stamp = stamp_s(rgb)
            for pub, msg in zip(self.pubs, (rgb, depth, info, odom, odom_info)):
                pub.publish(msg)
            self.reason = 'Timestamp-matched valid odometry; SLAM input enabled'
            self.health.observe(True, time.monotonic())
        except ValueError as e:
            self.reason = str(e)
            self.health.observe(False, time.monotonic())


def main():
    rclpy.init()
    node = Gate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
