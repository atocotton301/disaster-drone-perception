"""ROS subscribers, timestamped TF, and data recording for native dashboard."""
import json
import math
import os
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid, Odometry
from rtabmap_msgs.msg import MapGraph
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from PIL import Image as PILImage
from core import stamp_s, depth_meters, validate_pair, yaw, occupancy_rgb
from mapping import LocalMapper


def xyz_transform(t, xyz):
    q = t.rotation
    u = np.array([q.x, q.y, q.z])
    p = np.asarray(xyz, dtype=float)
    p = p + 2*(q.w*np.cross(u, p)+np.cross(u, np.cross(u, p)))
    return (p+np.array([t.translation.x, t.translation.y, t.translation.z])).tolist()


class Viewer(Node):
    def __init__(self, state, session, config):
        from rclpy.parameter import Parameter
        super().__init__('disaster_viewer', parameter_overrides=[Parameter('use_sim_time', value=bool(os.environ.get('DISASTER_BAG')))])
        self.s, self.out, self.c = state, Path(session), config
        self.bridge, self.mapper = CvBridge(), LocalMapper(height_band=config['local_height_band_m'])
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.camera_info = None
        self.last_frame_save = self.last_preview = self.last_map_save = -1e9
        self.events = (self.out/'detections.jsonl').open('a', encoding='utf-8', buffering=1)
        self.poses = (self.out/'poses.jsonl').open('a', encoding='utf-8', buffering=1)
        self.health_log = (self.out/'tracking.jsonl').open('a', encoding='utf-8', buffering=1)
        self.last_health = None
        self.pending_poses = []
        self.create_timer(.1, self.update_pose)
        self.create_subscription(Image, '/disaster/annotated', self.annotated, qos_profile_sensor_data)
        self.create_subscription(String, '/disaster/detections', self.detections, 10)
        self.create_subscription(String, '/disaster/tracking', self.tracking, 10)
        self.create_subscription(String, '/disaster/detector_status', self.detector_status, 10)
        self.create_subscription(Odometry, '/disaster/slam/odom', self.pose, qos_profile_sensor_data)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.grid, map_qos)
        self.create_subscription(MapGraph, '/disaster/map_graph', self.graph, 1)
        self.subs = [Subscriber(self, t, topic, qos_profile=qos_profile_sensor_data) for t, topic in [
            (Image, '/camera/color/image_raw'), (Image, '/camera/aligned_depth_to_color/image_raw'),
            (CameraInfo, '/camera/color/camera_info')]]
        self.sync = ApproximateTimeSynchronizer(self.subs, 8, self.c['sync_slop_s'])
        self.sync.registerCallback(self.rgbd)

    def rgbd(self, rgb, depth, info):
        now = time.monotonic()
        if now-self.last_preview < .2:
            return
        self.last_preview = now
        color = self.bridge.imgmsg_to_cv2(rgb, 'rgb8').copy()
        dep = depth_meters(self.bridge.imgmsg_to_cv2(depth, 'passthrough'), depth.encoding)
        validate_pair(color.shape, dep.shape, [m.header.frame_id for m in (rgb, depth, info)],
                      [stamp_s(m) for m in (rgb, depth, info)], self.c['sync_slop_s'])
        k = (info.k[0], info.k[4], info.k[2], info.k[5])
        # Undistort depth rays by remapping to the same intrinsic pinhole grid for local preview.
        K = np.array(info.k).reshape(3, 3)
        if info.distortion_model in ('plumb_bob', 'rational_polynomial') and len(info.d):
            mx, my = cv2.initUndistortRectifyMap(K, np.array(info.d), None, K,
                (info.width, info.height), cv2.CV_32FC1)
            local_depth = cv2.remap(dep, mx, my, cv2.INTER_NEAREST)
        elif not len(info.d) or not any(info.d):
            local_depth = dep
        else:
            raise ValueError('Unsupported calibration distortion: ' + info.distortion_model)
        local, _ = self.mapper.update(local_depth, k)
        colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(dep/6*255, 0, 255)), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        colored[dep == 0] = 0
        local_rgb = np.full((*local.shape, 3), (49, 61, 79), np.uint8)
        local_rgb[local > 0] = (243, 160, 65)
        with self.s.lock:
            self.camera_info = info
            self.s.times['rgb'] = now
            self.s.depth, self.s.local = colored, local_rgb
            if now-self.s.times.get('annotated', -1e9) > self.c['stale_s']:
                self.s.rgb = color
        if now-self.last_frame_save >= 1/self.c['save_frame_hz']:
            self.last_frame_save = now
            tag = f'{rgb.header.stamp.sec}_{rgb.header.stamp.nanosec:09d}'
            frames = self.out/'frames'
            frames.mkdir(exist_ok=True)
            PILImage.fromarray(color).save(frames/(tag+'_rgb.jpg'))
            np.savez_compressed(frames/(tag+'_depth.npz'), depth_m=dep, K=K,
                D=np.array(info.d), stamp=stamp_s(rgb), depth_stamp=stamp_s(depth), frame_id=rgb.header.frame_id)

    def annotated(self, msg):
        with self.s.lock:
            self.s.rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8').copy()
            self.s.times['annotated'] = time.monotonic()

    def detector_status(self, msg):
        data = json.loads(msg.data)
        with self.s.lock:
            self.s.model = data['model'] + ' / ' + data['input']
            self.s.times['model'] = time.monotonic()

    def tracking(self, msg):
        data = json.loads(msg.data)
        with self.s.lock:
            self.s.tracking, self.s.reason = data['status'], data['reason']
            self.s.times['tracking'] = time.monotonic()
            if data['status'] != 'TRACKING':
                self.s.pose = None
                self.s.markers = []
        if data['status'] != self.last_health:
            self.last_health = data['status']
            self.health_log.write(json.dumps(dict(wall_time=time.time(), **data))+'\n')

    def pose(self, msg):
        self.pending_poses.append(msg)
        self.pending_poses = self.pending_poses[-15:]

    def update_pose(self):
        with self.s.lock:
            if self.s.tracking != 'TRACKING':
                return
        # TF often arrives after the matching odometry callback; retry recent timestamps.
        for msg in reversed(self.pending_poses):
            if self.tf.can_transform('map', 'camera_link', Time.from_msg(msg.header.stamp)):
                self.resolve_pose(msg)
                self.pending_poses = [p for p in self.pending_poses if stamp_s(p) > stamp_s(msg)]
                return

    def resolve_pose(self, msg):
        try:
            tf = self.tf.lookup_transform('map', 'camera_link', Time.from_msg(msg.header.stamp))
            t, q = tf.transform.translation, tf.transform.rotation
            p = [t.x, t.y, t.z, yaw([q.x, q.y, q.z, q.w])]
            with self.s.lock:
                self.s.pose = p
                self.s.times['pose'] = time.monotonic()
            self.poses.write(json.dumps(dict(stamp=stamp_s(msg), frame='map', xyz_yaw=p))+'\n')
        except TransformException:
            with self.s.lock:
                self.s.pose = None
                self.s.reason = '해당 영상 시각의 map→camera TF 대기'

    def detections(self, msg):
        data = json.loads(msg.data)
        markers = []
        info = self.camera_info
        with self.s.lock:
            valid = self.s.tracking == 'TRACKING' and time.monotonic()-self.s.times.get('tracking', -1e9) < self.c['stale_s']
        if valid and info and info.header.frame_id == data['frame_id']:
            try:
                t = self.tf.lookup_transform('map', data['frame_id'], Time(nanoseconds=round(data['stamp']*1e9))).transform
                for row in data['detections']:
                    if row['depth_m'] is None:
                        continue
                    x1, y1, x2, y2 = row['bbox']
                    uv = np.array([[[(x1+x2)/2, (y1+y2)/2]]], np.float64)
                    if info.distortion_model not in ('plumb_bob', 'rational_polynomial') and any(info.d):
                        continue
                    ray = cv2.undistortPoints(uv, np.array(info.k).reshape(3, 3), np.array(info.d) if len(info.d) else None)[0, 0]
                    z = row['depth_m']
                    markers.append(dict(label=row['label'], xyz=xyz_transform(t, [ray[0]*z, ray[1]*z, z])))
            except TransformException:
                pass
        data['map_markers'] = markers
        data['map_marker_note'] = 'timestamp TF + ROI estimate; instantaneous, not persistent semantic map'
        self.events.write(json.dumps(data, allow_nan=False)+'\n')
        with self.s.lock:
            self.s.rows, self.s.markers = data['detections'], markers
            self.s.times['detections'] = time.monotonic()

    def grid(self, msg):
        with self.s.lock:
            if self.s.tracking != 'TRACKING' or time.monotonic()-self.s.times.get('tracking', -1e9) > self.c['stale_s']:
                return
        if msg.header.frame_id != 'map' or msg.info.resolution <= 0:
            raise ValueError('Invalid occupancy map frame/resolution')
        q, p = msg.info.origin.orientation, msg.info.origin.position
        grid = np.array(msg.data, np.int8).reshape(msg.info.height, msg.info.width)
        meta = dict(width=msg.info.width, height=msg.info.height, resolution=msg.info.resolution,
                    origin=[p.x, p.y, yaw([q.x, q.y, q.z, q.w])], frame_id='map', stamp=stamp_s(msg))
        with self.s.lock:
            self.s.grid, self.s.meta = grid, meta
            self.s.times['map'] = time.monotonic()
        self.save_map(grid, meta)

    def save_map(self, grid, meta):
        # Atomic individual files; raw NPZ includes matching metadata for unambiguous recovery.
        with (self.out/'map.tmp.npz').open('wb') as f:
            np.savez_compressed(f, occupancy=grid, metadata=json.dumps(meta))
        (self.out/'map.tmp.npz').replace(self.out/'map.npz')
        PILImage.fromarray(occupancy_rgb(grid)).save(self.out/'map.png')
        pgm = np.full(grid.shape, 205, np.uint8)
        pgm[(grid >= 0) & (grid < 50)] = 254
        pgm[grid >= 50] = 0
        PILImage.fromarray(np.flipud(pgm)).save(self.out/'map.pgm')
        (self.out/'map.yaml').write_text('image: map.pgm\nmode: trinary\nresolution: '+str(meta['resolution'])+
            '\norigin: '+json.dumps(meta['origin'])+'\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n', encoding='utf-8')

    def graph(self, msg):
        if msg.header.frame_id != 'map':
            return
        with self.s.lock:
            if self.s.tracking != 'TRACKING':
                return
        from types import SimpleNamespace
        points = [xyz_transform(SimpleNamespace(rotation=p.orientation, translation=p.position), self.c['mount_xyz_m'])
                  for i, p in sorted(zip(msg.poses_id, msg.poses)) if i > 0]
        with self.s.lock:
            self.s.path = points
        (self.out/'optimized_path.json').write_text(json.dumps(dict(frame_id='map',
            stamp=stamp_s(msg), keyframe_positions=points)), encoding='utf-8')

    def close(self):
        self.events.close()
        self.poses.close()
        self.health_log.close()


def run_backend(state, session, config, stop):
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = Viewer(state, session, config)
        while not stop.is_set() and rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=.1)
            except Exception as e:
                with state.lock:
                    state.error = 'ROS 처리 오류: '+str(e)
    finally:
        if node:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
