"""Real RGB-D inputs only. No generated imagery or guessed camera trajectory.

SDK alignment follows Intel RealSense align-depth2color.py (Apache-2.0),
Copyright(c) 2017 RealSense, Inc. All Rights Reserved.
Modified: generator, measured depth scale/calibration, stop event, no background removal.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def sdk_frames(stop, fps=15, sync_slop_s=.015):
    import pyrealsense2 as rs
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, fps)
    profile = pipeline.start(config)
    try:
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        while not stop.is_set():
            frames = align.process(pipeline.wait_for_frames(3000))
            depth, rgb = frames.get_depth_frame(), frames.get_color_frame()
            if not depth or not rgb:
                continue
            if rgb.get_frame_timestamp_domain() != depth.get_frame_timestamp_domain():
                continue
            if abs(rgb.get_timestamp()-depth.get_timestamp())/1000. > sync_slop_s:
                continue
            intr = rgb.profile.as_video_stream_profile().intrinsics
            # RS intrinsics distortion is retained; preview rectifies supported models.
            zero_distortion = all(float(c) == 0.0 for c in intr.coeffs)
            if not zero_distortion and intr.model not in (rs.distortion.none, rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
                raise ValueError('Unsupported RealSense color distortion model '+str(intr.model))
            yield dict(rgb=np.asanyarray(rgb.get_data()).copy(),
                depth_m=np.asanyarray(depth.get_data()).astype(np.float32)*scale,
                K=[intr.fx, 0., intr.ppx, 0., intr.fy, intr.ppy, 0., 0., 1.],
                D=list(intr.coeffs), stamp=rgb.get_timestamp()/1000.,
                depth_stamp=depth.get_timestamp()/1000., source='LIVE SDK D435i',
                timestamp_domain=str(rgb.get_frame_timestamp_domain()))
    finally:
        pipeline.stop()


def tum_frames(folder, limit=None, slop=.015):
    """Measured TUM timestamps, one-to-one matching, 5000 depth units/m.
    TUM recommends ROS-default intrinsics for its pre-registered depth images.
    """
    folder = Path(folder)
    metadata = json.loads((folder/'sequence.json').read_text()) if (folder/'sequence.json').is_file() else {}
    def index(name):
        rows = []
        for line in (folder/name).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            stamp, relative = line.split()[:2]
            path = (folder/relative).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError('Dataset path escapes input folder')
            rows.append((float(stamp), path))
        return sorted(rows)
    colors, depths = index('rgb.txt'), index('depth.txt')
    dts = np.array([t for t, _ in depths])
    candidates = []
    for i, (t, _) in enumerate(colors):
        j = int(np.searchsorted(dts, t))
        for k in (j-1, j):
            if 0 <= k < len(depths) and abs(dts[k]-t) <= slop:
                candidates.append((abs(dts[k]-t), i, k))
    pairs, used_rgb, used_depth = [], set(), set()
    for _, i, j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            pairs.append((i, j))
            used_rgb.add(i)
            used_depth.add(j)
    pairs.sort()
    if not pairs:
        raise ValueError('No RGB-D pairs within synchronization tolerance')
    for i, j in pairs[:limit]:
        ts, path = colors[i]
        dt, dp = depths[j]
        raw = np.array(Image.open(dp))
        rgb = np.array(Image.open(path).convert('RGB'))
        if raw.dtype != np.uint16 or raw.shape != rgb.shape[:2]:
            raise ValueError('Expected aligned uint16 depth and RGB images')
        yield dict(rgb=rgb, depth_m=raw.astype(np.float32)/metadata.get('depth_units_per_meter', 5000.),
            K=metadata.get('K', [525., 0., 319.5, 0., 525., 239.5, 0., 0., 1.]), D=metadata.get('D', [0.]*5),
            stamp=ts, depth_stamp=dt, source=metadata.get('source', 'TUM KINECT RECORDED RGB-D'),
            timestamp_domain=metadata.get('timestamp_domain', 'TUM_MEASURED_TIMESTAMPS'), filename=path.name)
