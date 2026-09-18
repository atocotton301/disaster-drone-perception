"""ROS-free data validation and map geometry shared by GUI, detector and tests."""
import json
import math
from pathlib import Path
import numpy as np
from mapping import clean_depth, box_distance

CLASSES = {'person', 'fire', 'smoke', 'door', 'staircase'}


def load_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text(encoding='utf-8-sig'))
    if c.get('tracking_origin', 'robot') not in ('robot', 'camera'):
        raise ValueError('tracking_origin must be robot or camera')
    if c.get('tracking_origin') == 'camera' and any(c['mount_xyz_m'] + c['mount_rpy_rad']):
        raise ValueError('Camera-origin tracking requires identity mount transform')
    selected = c.get('selected_classes')
    if selected is not None and (not isinstance(selected, list) or not selected or not set(selected) <= CLASSES):
        raise ValueError('selected_classes must contain known detection classes')
    for key in ('confidence', 'detector_hz', 'sync_slop_s', 'stale_s', 'save_frame_hz'):
        if not isinstance(c[key], (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Invalid config: ' + key)
    if c['confidence'] >= 1 or c['detector_hz'] > 30 or c['sync_slop_s'] > .03:
        raise ValueError('Invalid confidence, detector rate or synchronization tolerance')
    for key in ('mount_xyz_m', 'mount_rpy_rad'):
        if len(c[key]) != 3 or not np.isfinite(c[key]).all():
            raise ValueError('Invalid mounting transform: ' + key)
    band = c['local_height_band_m']
    if len(band) != 2 or not np.isfinite(band).all() or band[0] >= band[1]:
        raise ValueError('Invalid local height band')
    for key in ('weights', 'runs_dir'):
        if c[key]:
            p = Path(c[key]).expanduser()
            c[key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
    return c


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def depth_meters(raw, encoding):
    if encoding not in ('16UC1', '32FC1'):
        raise ValueError('Unsupported depth encoding ' + encoding)
    if raw.ndim != 2:
        raise ValueError('Depth must be 2D')
    depth = raw.astype(np.float32) * (.001 if encoding == '16UC1' else 1.)
    return clean_depth(depth)


def validate_pair(rgb_shape, depth_shape, frames, stamps, slop):
    if rgb_shape[:2] != depth_shape[:2]:
        raise ValueError('RGB/Depth sizes differ: alignment required')
    if not frames[0] or len(set(frames)) != 1:
        raise ValueError('RGB/Depth/CameraInfo optical frame mismatch')
    if not np.isfinite(stamps).all() or max(stamps)-min(stamps) > slop:
        raise ValueError('RGB-D timestamps exceed synchronization tolerance')


def yaw(q):
    x, y, z, w = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def grid_pixel(x, y, meta):
    """Inverse full planar OccupancyGrid origin; image top row = highest grid y."""
    ox, oy, a = meta['origin']
    dx, dy = x-ox, y-oy
    gx = (math.cos(a)*dx+math.sin(a)*dy)/meta['resolution']
    gy = (-math.sin(a)*dx+math.cos(a)*dy)/meta['resolution']
    return gx, meta['height']-1-gy


def occupancy_rgb(grid):
    out = np.full((*grid.shape, 3), (49, 61, 79), np.uint8)
    out[(grid >= 0) & (grid < 50)] = (222, 231, 233)
    out[grid >= 50] = (243, 160, 65)
    return np.flipud(out).copy()


def odom_valid(lost, covariance, pose):
    c, p = np.asarray(covariance), np.asarray(pose)
    return (not lost and c.size == 36 and np.isfinite(c).all()
            and (c[[0, 7, 14, 21, 28, 35]] >= 0).all()
            and (c[[0, 7, 14, 21, 28, 35]] < 9999).all()
            and np.isfinite(p).all() and p.size == 7
            and abs(np.linalg.norm(p[3:])-1) < .02)


class TrackingGate:
    def __init__(self, stale=1.5):
        self.stale = stale
        self.last = None
        self.good = False

    def observe(self, good, now):
        self.last, self.good = now, bool(good)

    def status(self, now):
        if self.last is None:
            return 'WAITING'
        if now-self.last > self.stale:
            return 'STALE'
        return 'TRACKING' if self.good else 'LOST'


def detection_rows(boxes, names, depth):
    rows = []
    for x1, y1, x2, y2, confidence, cls in boxes:
        if not np.isfinite([x1, y1, x2, y2, confidence, cls]).all():
            continue
        label = names[int(cls)]
        # Fire/smoke are not reliable opaque depth surfaces. Do not claim their range.
        surface = box_distance(depth, (x1, y1, x2, y2))
        ambiguous = label in ('fire', 'smoke')
        rows.append(dict(label=label, confidence=float(confidence),
                         bbox=[float(v) for v in (x1, y1, x2, y2)],
                         depth_m=None if ambiguous else surface,
                         roi_surface_depth_m=surface,
                         distance_status='BACKGROUND_SURFACE_ONLY' if ambiguous else
                         ('VALID_ROI_ESTIMATE' if surface is not None else 'INVALID_DEPTH')))
    return rows
