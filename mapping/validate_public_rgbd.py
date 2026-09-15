"""Validate measured Kinect RGB-D sequence from TUM. No YOLO/SLAM accuracy claim."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import tarfile
import numpy as np
from PIL import Image
from inputs import tum_frames
from mapping import clean_depth, project, LocalMapper, box_distance

URL = 'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=60)
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    archive = a.cache/'tum_xyz.tgz'
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('wb') as f:
            while chunk := response.read(1024*1024):
                f.write(chunk)
    folder = a.cache/'rgbd_dataset_freiburg1_xyz'
    if not (folder/'rgb.txt').is_file():
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                dest = (a.cache/member.name).resolve()
                if not dest.is_relative_to(a.cache.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe archive member')
            # All members and resolved paths were checked above (no links/devices).
            tar.extractall(a.cache)
    rows, mapper = [], LocalMapper()
    for i, frame in enumerate(tum_frames(folder, a.limit)):
        depth = clean_depth(frame['depth_m'])
        K = np.array(frame['K']).reshape(3, 3)
        k = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        grid, _ = mapper.update(depth, k)
        points = project(depth, k)
        assert len(points) > 100 and np.isfinite(points).all()
        assert (grid > 0).any()
        # Central image patch, explicitly not an object detection.
        h, w = depth.shape
        roi_depth = box_distance(depth, [w*.4, h*.4, w*.6, h*.6])
        rows.append(dict(frame=i, rgb_stamp=frame['stamp'], depth_stamp=frame['depth_stamp'],
            delta_s=frame['depth_stamp']-frame['stamp'], size=list(frame['rgb'].shape), valid_depth_pixels=int((depth > 0).sum()),
            projected_points=len(points), occupied_camera_local_cells=int((grid > 0).sum()),
            central_patch_depth_m=roi_depth))
        if i < 5:
            Image.fromarray(frame['rgb']).save(a.out/f'public_rgb_{i}.png')
            Image.fromarray(grid).save(a.out/f'camera_local_obstacles_{i}.png')
    with archive.open('rb') as f:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024*1024), b''):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    result = dict(source=URL, archive_sha256=digest, data_origin='MEASURED MICROSOFT KINECT / TUM fr1 xyz',
        license='CC BY 4.0', dataset_dir=str(folder.resolve()), frames=rows,
        passed=['measured RGB/depth decoding', '5000 units per meter conversion', 'one-to-one timestamp association <=15ms', 'invalid depth filtering',
                'intrinsic projection', '5cm camera-local occupancy', 'file saving'],
        not_tested=['D435i alignment accuracy', 'RTAB-Map execution/loop closure', 'five-class YOLO',
                    'Jetson CUDA/latency', 'physical range accuracy'],
        note='Measured RGB-D data. Camera-local grid is NOT accumulated SLAM. No ground-truth pose used.')
    (a.out/'public_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'frames'}, indent=2))
    print('Processed frames:', len(rows))


if __name__ == '__main__':
    main()
