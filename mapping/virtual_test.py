"""Virtual RGB-D -> OpenCV RGB-D odometry -> accumulated occupied cells -> MP4.

Explicit simulation, not D435i validation and not RTAB-Map loop-closing SLAM.
The renderer's camera pose is used ONLY to generate input and evaluate error.
The mapper receives only RGB, depth, and intrinsics; no ground-truth trajectory.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from core import occupancy_rgb, grid_pixel
from mapping import clean_depth, project

W, H = 320, 240
K = np.array([[285., 0., 159.5], [0., 285., 119.5], [0., 0., 1.]], np.float32)


def scene():
    # xmin,ymin,zmin,xmax,ymax,zmax; physical dimensions in meters.
    return [
        ([-3.1, 0, -.1, 3.1, 8, 0], [157, 125, 90], 0),
        ([-3.1, 0, 3, 3.1, 8, 3.1], [205, 204, 198], 1),
        ([-3.1, 0, 0, -3, 8, 3], [192, 203, 201], 2),
        ([3, 0, 0, 3.1, 8, 3], [196, 197, 208], 2),
        ([-3, 7.9, 0, 3, 8, 3], [204, 207, 190], 2),
        ([-3, -.1, 0, 3, 0, 3], [197, 201, 203], 2),
        ([-2.4, 3.2, .72, -.6, 4.0, .82], [148, 93, 56], 0),
        ([-2.35, 3.25, 0, -2.2, 3.4, .72], [63, 68, 72], 1),
        ([-.8, 3.65, 0, -.65, 3.8, .72], [63, 68, 72], 1),
        ([-1.8, 3.65, .82, -1.1, 3.8, 1.35], [45, 59, 72], 2),
        ([1.25, 4.5, 0, 2.6, 5.1, 1.55], [167, 180, 190], 2),
        ([-2.6, 6.1, 0, -.8, 6.55, 1.9], [123, 93, 64], 2),
        ([.5, 6.3, 0, 1.3, 7.1, .65], [175, 122, 62], 2),
        ([-.45, 7.82, 0, .55, 7.9, 2.2], [91, 129, 136], 0),
        ([2.94, 2.6, .9, 3., 4., 2.1], [79, 138, 170], 2),
        ([-3., 4.5, 1., -2.94, 5.7, 2.1], [170, 92, 75], 2),
    ]


def camera(t):
    pos = np.array([.28*math.sin(t*math.pi*2), 1.+2.3*t, 1.3])
    angle = .25*math.sin(t*math.pi*2)
    right = [math.cos(angle), -math.sin(angle), 0.]
    down = [0., 0., -1.]
    forward = [math.sin(angle), math.cos(angle), 0.]
    return pos, np.array([right, down, forward]).T


def render(pos, rotation):
    v, u = np.mgrid[:H, :W]
    rays = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones((H, W))), -1)
    direction = rays @ rotation.T
    best = np.full((H, W), np.inf)
    index = np.full((H, W), -1, np.int16)
    shapes = scene()
    inv = np.divide(1., direction, out=np.full_like(direction, 1e12), where=np.abs(direction)>1e-12)
    for i, (bounds, _, _) in enumerate(shapes):
        low, high = np.array(bounds[:3]), np.array(bounds[3:])
        a, b = (low-pos)*inv, (high-pos)*inv
        near = np.minimum(a, b).max(axis=-1)
        far = np.maximum(a, b).min(axis=-1)
        distance = np.where(near > .01, near, far)
        valid = (far >= np.maximum(near, 0)) & (distance > .01) & (distance < best)
        best[valid], index[valid] = distance[valid], i
    safe = np.where(np.isfinite(best), best, 0)
    hit = pos + direction*safe[..., None]
    rgb = np.full((H, W, 3), 30, np.uint8)
    for i, (_, color, kind) in enumerate(shapes):
        selected = index == i
        x, y, z = hit[..., 0], hit[..., 1], hit[..., 2]
        if kind == 0:
            texture = .86 + .10*np.sin(24*x+3*np.sin(7*y))+.04*np.sin(70*y)
        else:
            tile = ((np.floor(x*6)+np.floor(y*6)+np.floor(z*6)) % 2)
            texture = .77+.20*tile+.03*np.sin(x*43+y*39+z*37)
        # Spatial texture provides actual photometric constraints to odometry.
        rgb[selected] = np.clip(np.array(color)*texture[selected, None], 0, 255).astype(np.uint8)
    depth = safe.astype(np.float32)
    depth[(depth < .2) | (depth > 6)] = 0
    return rgb, depth


class EstimatedMapper:
    def __init__(self):
        self.odometry = cv2.rgbd.RgbdICPOdometry_create(K, .2, 6., .07, .15)
        self.pose = np.eye(4)
        self.pose[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        self.pose[2, 3] = 1.3  # Declared initial camera height, not a moving ground-truth pose.
        self.previous = None
        self.grid = np.full((240, 200), -1, np.int8)
        self.meta = dict(width=200, height=240, origin=[-5., -2., 0.], resolution=.05)
        self.path = []

    def update(self, rgb, depth):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        valid = ((depth >= .2) & (depth <= 6)).astype(np.uint8)*255
        status = 'INITIALIZING'
        if self.previous is not None:
            old_gray, old_depth, old_mask = self.previous
            good, transform = self.odometry.compute(old_gray, old_depth, old_mask, gray, depth, valid)
            if not good or transform is None or not np.isfinite(transform).all() or np.linalg.norm(transform[:3, 3]) > .25:
                return 'LOST'  # Freeze pose and map. Retain prior reference for potential recovery.
            self.pose = self.pose @ np.linalg.inv(transform)
            status = 'TRACKING'
        self.previous = (gray, depth.copy(), valid)
        points = project(clean_depth(depth), [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], stride=3)
        world = points @ self.pose[:3, :3].T + self.pose[:3, 3]
        world = world[(world[:, 2] >= .15) & (world[:, 2] <= 2.2)]
        col = np.floor((world[:, 0]+5)/.05).astype(int)
        row = np.floor((world[:, 1]+2)/.05).astype(int)
        ok = (col >= 0) & (col < 200) & (row >= 0) & (row < 240)
        self.grid[row[ok], col[ok]] = 100  # Occupied-only map; other cells remain unknown.
        self.path.append(self.pose[:3, 3].tolist())
        return status


def board(rgb, depth, mapper, status, index, frames, fps):
    canvas = Image.new('RGB', (1280, 720), '#202326')
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
        small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    except OSError:
        font = small = ImageFont.load_default()
    d.text((24, 17), 'VIRTUAL RGB-D TEST  /  SIMULATION ONLY', fill='#efefef', font=font)
    d.text((24, 50), 'OpenCV RGB-D odometry + occupied-cell accumulation | No loop closure | No YOLO model', fill='#bdc5cc', font=small)
    canvas.paste(Image.fromarray(rgb).resize((640, 480)), (24, 104))
    map_im = Image.fromarray(occupancy_rgb(mapper.grid)).resize((400, 480), Image.Resampling.NEAREST)
    md = ImageDraw.Draw(map_im)
    pts = [tuple(v*2 for v in grid_pixel(p[0], p[1], mapper.meta)) for p in mapper.path]
    if len(pts) > 1:
        md.line(pts, fill='#50a8ee', width=3)
    if pts and status != 'LOST':
        x, y = pts[-1]
        md.ellipse((x-5, y-5, x+5, y+5), fill='#71e1b2')
    canvas.paste(map_im, (810, 104))
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(depth/6*255, 0, 255)), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    colored[depth == 0] = 0
    canvas.paste(Image.fromarray(colored).resize((160, 120)), (24, 593))
    d.text((205, 601), f'Frame {index+1}/{frames}   Playback {fps} fps   Tracking: {status}', fill='#efefef', font=font)
    p = mapper.pose[:3, 3]
    d.text((205, 635), f'Estimated camera: x {p[0]:+.2f} m / y {p[1]:+.2f} m', fill='#acbac8', font=small)
    d.text((810, 601), '5 cm / orange: occupied / gray: unknown', fill='#bac5cf', font=small)
    d.text((810, 635), 'Blue path is estimated from RGB-D input.', fill='#bac5cf', font=small)
    return np.array(canvas)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=180)
    p.add_argument('--fps', type=int, default=15)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dataset = a.out/'virtual_rgbd'
    (dataset/'rgb').mkdir(parents=True, exist_ok=True)
    (dataset/'depth').mkdir(exist_ok=True)
    (dataset/'sequence.json').write_text(json.dumps(dict(source='VIRTUAL ROOM / SIMULATION',
        data_kind='virtual', K=K.ravel().tolist(), D=[0.]*5, depth_units_per_meter=5000,
        timestamp_domain='SIMULATED_CLOCK')), encoding='utf-8')
    rgb_index, depth_index, rows = [], [], []
    writer = cv2.VideoWriter(str(a.out/'virtual_mapping.avi'), cv2.VideoWriter_fourcc(*'MJPG'), a.fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError('Video encoder unavailable')
    mapper = EstimatedMapper()
    try:
        for i in range(a.frames):
            pos, rotation = camera(i/max(1, a.frames-1))
            rgb, depth = render(pos, rotation)
            stamp = i/a.fps
            name = f'{i:06d}.png'
            Image.fromarray(rgb).save(dataset/'rgb'/name)
            Image.fromarray(np.uint16(np.clip(depth*5000, 0, 65535))).save(dataset/'depth'/name)
            rgb_index.append(f'{stamp:.9f} rgb/{name}\n')
            depth_index.append(f'{stamp:.9f} depth/{name}\n')
            begin = time.monotonic()
            status = mapper.update(rgb, depth)
            truth_relative = [pos[0], pos[1]-1., pos[2]]
            error = float(np.linalg.norm(mapper.pose[:3, 3]-truth_relative))
            rows.append(dict(frame=i, stamp=stamp, status=status, estimated_xyz=mapper.pose[:3, 3].tolist(),
                evaluation_only_truth_xyz=truth_relative, translation_error_m=error,
                processing_ms=(time.monotonic()-begin)*1000))
            frame = board(rgb, depth, mapper, status, i, a.frames, a.fps)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if i in (0, a.frames//2, a.frames-1):
                Image.fromarray(frame).save(a.out/f'frame_{i:04d}.png')
            if i % 30 == 0:
                print(f'{i}/{a.frames}: {status}, evaluation translation error {error:.3f}m', flush=True)
    finally:
        writer.release()
    (dataset/'rgb.txt').write_text(''.join(rgb_index))
    (dataset/'depth.txt').write_text(''.join(depth_index))
    np.savez_compressed(a.out/'estimated_map.npz', occupancy=mapper.grid, metadata=json.dumps(mapper.meta))
    report = dict(data_kind='VIRTUAL_NOT_MEASURED', backend='OpenCV RgbdICPOdometry + occupied-cell accumulation',
        loop_closure=False, yolo_model=None, ground_truth_used_by_mapper=False,
        frames=a.frames, fps=a.fps, lost_frames=sum(r['status']=='LOST' for r in rows),
        final_translation_error_m=rows[-1]['translation_error_m'],
        occupied_cells=int((mapper.grid == 100).sum()), frame_results=rows)
    (a.out/'simulation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(a.out/'virtual_mapping.avi'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart',
        str(a.out/'virtual_mapping.mp4')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    (a.out/'virtual_mapping.avi').unlink()
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    config.update(mount_verified=True, mount_xyz_m=[0., 0., 1.3], weights='', runs_dir='../runs')
    (a.out/'virtual_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Saved', a.out/'virtual_mapping.mp4')


if __name__ == '__main__':
    main()
