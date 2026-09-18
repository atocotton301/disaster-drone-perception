"""Adapted from the existing aligned RGB-D detection_node.py; single camera owner."""
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from core import CLASSES, load_config, stamp_s, depth_meters, validate_pair, detection_rows


class Detector(Node):
    def __init__(self):
        super().__init__('disaster_detector')
        self.c = load_config(os.environ['DISASTER_CONFIG'])
        self.bridge = CvBridge()
        self.model = None
        self.status = 'NO_MODEL: configure local best.pt path'
        if self.c['weights']:
            try:
                if not Path(self.c['weights']).is_file():
                    raise ValueError('Model file missing: ' + self.c['weights'])
                from ultralytics import YOLO
                import torch
                torch.set_num_threads(2)
                self.model = YOLO(self.c['weights'])
                names = self.model.names
                if len(names) != 5 or set(names.values() if isinstance(names, dict) else names) != CLASSES:
                    raise ValueError('Model must have exactly person/fire/smoke/door/staircase')
                self.status = 'READY'
            except Exception as e:
                self.model = None
                self.status = 'MODEL_ERROR: ' + str(e)
        self.image_pub = self.create_publisher(Image, '/disaster/annotated', 2)
        self.json_pub = self.create_publisher(String, '/disaster/detections', 2)
        self.status_pub = self.create_publisher(String, '/disaster/detector_status', 2)
        self.create_timer(.5, self.heartbeat)
        self.last_input = None
        self.last = -float('inf')
        qos = QoSProfile(depth=30) if os.environ.get('DISASTER_REPLAY') else QoSProfile(depth=1,
            reliability=ReliabilityPolicy.RELIABLE if self.c.get('reliable_camera') else ReliabilityPolicy.BEST_EFFORT)
        self.subs = [Subscriber(self, typ, topic, qos_profile=qos)
                     for typ, topic in [(Image, '/camera/color/image_raw'),
                     (Image, '/camera/aligned_depth_to_color/image_raw'),
                     (CameraInfo, '/camera/color/camera_info')]]
        self.sync = ApproximateTimeSynchronizer(self.subs, 8, self.c['sync_slop_s'])
        self.sync.registerCallback(self.process)

    def heartbeat(self):
        self.status_pub.publish(String(data=json.dumps(dict(model=self.status,
            input='STALE_OR_MISSING' if self.last_input is None or time.monotonic()-self.last_input > self.c['stale_s'] else 'RGBD_OK'))))

    def process(self, rgb, depth_msg, info):
        now = time.monotonic()
        self.last_input = now
        if now-self.last < 1/self.c['detector_hz']:
            return
        self.last = now
        try:
            input_age_ms = (self.get_clock().now().nanoseconds*1e-9-stamp_s(rgb))*1000
            if not os.environ.get('DISASTER_REPLAY') and input_age_ms > 400:
                self.status = 'WAITING_FOR_FRESH_FRAME'
                return
            color = self.bridge.imgmsg_to_cv2(rgb, 'bgr8').copy()
            depth = depth_meters(self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough'), depth_msg.encoding)
            validate_pair(color.shape, depth.shape,
                [m.header.frame_id for m in (rgb, depth_msg, info)],
                [stamp_s(m) for m in (rgb, depth_msg, info)], self.c['sync_slop_s'])
            rows = []
            if self.model is not None:
                result = self.model.predict(color, imgsz=self.c.get('imgsz',640), conf=self.c['confidence'],
                    device=self.c['device'], verbose=False)[0]
                rows = detection_rows(result.boxes.data.cpu().numpy(), self.model.names, depth)
                selected = self.c.get('selected_classes')
                if selected:
                    rows = [r for r in rows if r['label'] in selected]
                self.status = 'RUNNING'
            for row in rows:
                x1, y1, x2, y2 = map(int, row['bbox'])
                distance = row['depth_m']
                label = f"{row['label']} {row['confidence']:.2f} "
                label += f'{distance:.2f}m (Z)' if distance is not None else 'range unknown'
                cv2.rectangle(color, (x1, y1), (x2, y2), (0, 185, 255), 2)
                cv2.putText(color, label, (max(0, x1), max(20, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
            image = self.bridge.cv2_to_imgmsg(color, 'bgr8')
            image.header = rgb.header
            self.image_pub.publish(image)
            self.json_pub.publish(String(data=json.dumps(dict(stamp=stamp_s(rgb),
                frame_id=rgb.header.frame_id, model_status=self.status,
                depth_delta_s=stamp_s(depth_msg)-stamp_s(rgb), detections=rows,
                processing_ms=(time.monotonic()-now)*1000, input_age_ms=input_age_ms), allow_nan=False)))
        except Exception as e:
            self.status = 'PROCESSING_ERROR: ' + str(e)
            self.get_logger().error(self.status)


def main():
    rclpy.init()
    node = Detector()
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
