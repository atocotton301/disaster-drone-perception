"""Open a saved Jetson session without ROS, camera, model or internet."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image


def last_json(path):
    value = None
    if path.is_file():
        with path.open(encoding='utf-8') as f:
            for line in f:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Ignore incomplete trailing record after interrupted power.
    return value


def load_session(state, path):
    root = Path(path).resolve()
    if not (root/'manifest.json').is_file():
        raise ValueError('Select a runs/<timestamp> session containing manifest.json')
    with state.lock:
        state.recorded = True
        state.source = '저장 기록 열람 · '+root.name
        state.tracking = 'RECORDED'
        state.reason = '실시간 입력 아님 · 저장된 마지막 지도/키프레임 궤적/영상'
        state.model = '저장된 탐지 기록'
        map_path = root/'map.npz'
        if map_path.is_file():
            with np.load(map_path, allow_pickle=False) as data:
                state.grid = data['occupancy'].copy()
                state.meta = json.loads(str(data['metadata']))
        path_file = root/'optimized_path.json'
        if path_file.is_file():
            state.path = json.loads(path_file.read_text())['keyframe_positions']
        pose = last_json(root/'poses.jsonl')
        if pose:
            state.pose = pose['xyz_yaw']
        health = last_json(root/'tracking.jsonl')
        if health:
            state.reason += ' · 마지막 위치추정 '+health['status']
            if health['status'] != 'TRACKING':
                state.pose = None
        frames = sorted((root/'frames').glob('*_depth.npz')) if (root/'frames').is_dir() else []
        if frames:
            frame = frames[-1]
            tag = frame.name.removesuffix('_depth.npz')
            rgb_file = next((p for p in [frame.parent/(tag+'_rgb.jpg'), frame.parent/(tag+'_rgb.png')] if p.exists()), None)
            if rgb_file:
                state.rgb = np.array(Image.open(rgb_file).convert('RGB'))
            with np.load(frame, allow_pickle=False) as data:
                depth = data['depth_m']
                colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(depth/6*255, 0, 255)), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                colored[depth == 0] = 0
                state.depth = colored
                frame_stamp = float(data['stamp']) if 'stamp' in data else None
                if 'local_obstacles' in data:
                    local = data['local_obstacles']
                    state.local = np.full((*local.shape, 3), (49, 61, 79), np.uint8)
                    state.local[local > 0] = (243, 160, 65)
            # Only overlay detections whose source timestamp matches this saved RGB.
            if frame_stamp is not None and (root/'detections.jsonl').is_file():
                with (root/'detections.jsonl').open(encoding='utf-8') as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if abs(event.get('stamp', -1)-frame_stamp) < 1e-5:
                            state.rows = event['detections']
                if state.rgb is not None:
                    for row in state.rows:
                        x1, y1, x2, y2 = map(int, row['bbox'])
                        cv2.rectangle(state.rgb, (x1, y1), (x2, y2), (255, 190, 0), 2)
                        text = f"{row['label']} {row['confidence']:.2f}"
                        cv2.putText(state.rgb, text, (max(x1, 0), max(y1-5, 20)), 0, .5, (255, 190, 0), 1)
        state.times = {key: time.monotonic() for key in ('rgb', 'map', 'model', 'tracking', 'pose', 'detections')}
        if state.grid is None:
            state.reason += ' · 이 세션에는 누적 SLAM 지도가 없습니다'
