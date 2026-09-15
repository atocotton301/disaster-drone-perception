"""Relay recorded camera-internal static transforms, never old map/odom/mount TF."""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


def main():
    rclpy.init()
    node = Node('recorded_camera_tf')
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(TFMessage, '/tf_static', qos)
    cached = {}
    def relay(msg):
        for t in msg.transforms:
            if t.header.frame_id.startswith('camera') and t.child_frame_id.startswith('camera') and t.child_frame_id != 'camera_link':
                cached[t.child_frame_id] = t
        if cached:
            pub.publish(TFMessage(transforms=list(cached.values())))
    node.create_subscription(TFMessage, '/disaster/recorded_tf_static', relay, qos)
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
