"""TUM measured timestamp replay using a common clock offset. No ground truth."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from cv_bridge import CvBridge
from inputs import tum_frames


def main():
    rclpy.init()
    node = Node('public_rgbd_replay')
    bridge = CvBridge()
    rgb_pub = node.create_publisher(Image, '/camera/color/image_raw', 10)
    depth_pub = node.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)
    info_pub = node.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
    broadcaster = StaticTransformBroadcaster(node)
    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id, t.child_frame_id = 'camera_link', 'camera_color_optical_frame'
    # REP-103 camera body x forward,y left,z up -> optical z forward,x right,y down.
    t.transform.rotation.x, t.transform.rotation.y = -.5, .5
    t.transform.rotation.z, t.transform.rotation.w = -.5, .5
    broadcaster.sendTransform(t)
    try:
        # Let all consumers discover publishers before this very short sample starts.
        for _ in range(50):
            rclpy.spin_once(node, timeout_sec=.1)
        offset = None
        for frame in tum_frames(sys.argv[1]):
            if offset is None:
                offset = node.get_clock().now().nanoseconds/1e9-frame['stamp']
                first_stamp, start_wall = frame['stamp'], time.monotonic()
            due = start_wall + frame['stamp']-first_stamp
            while time.monotonic() < due:
                rclpy.spin_once(node, timeout_sec=max(0., min(.01, due-time.monotonic())))
            stamp = Time(nanoseconds=round((frame['stamp']+offset)*1e9)).to_msg()
            rgb = bridge.cv2_to_imgmsg(frame['rgb'], 'rgb8')
            depth = bridge.cv2_to_imgmsg(frame['depth_m'], '32FC1')
            info = CameraInfo()
            info.width, info.height = frame['rgb'].shape[1], frame['rgb'].shape[0]
            info.k = frame['K']
            info.d = frame['D']
            info.distortion_model = 'plumb_bob'
            info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
            k = info.k
            info.p = [k[0], 0., k[2], 0., 0., k[4], k[5], 0., 0., 0., 1., 0.]
            for msg in (rgb, depth, info):
                msg.header.stamp, msg.header.frame_id = stamp, 'camera_color_optical_frame'
            depth.header.stamp = Time(nanoseconds=round((frame['depth_stamp']+offset)*1e9)).to_msg()
            info_pub.publish(info)
            rgb_pub.publish(rgb)
            depth_pub.publish(depth)
            rclpy.spin_once(node, timeout_sec=0)
        node.get_logger().info('Public sample ended; no loop or fabricated trajectory. Tracking will become STALE.')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
