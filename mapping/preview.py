"""Hardware SDK or public recording inspection, explicitly camera-local (not SLAM)."""
import json
import time
import queue
import threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from inputs import sdk_frames, tum_frames
from core import CLASSES, detection_rows, validate_pair
from mapping import clean_depth, LocalMapper


def run_preview(state, session, config, stop, source, dataset, limit=None):
    if source == 'tum' and dataset is None:
        raise ValueError('--dataset must point to extracted public RGB-D recording')
    stream = sdk_frames(stop) if source == 'sdk' else tum_frames(dataset, limit, config['sync_slop_s'])
    mapper, model = LocalMapper(height_band=config['local_height_band_m']), None
    model_status = 'NO_MODEL'
    if config['weights']:
        try:
            from ultralytics import YOLO
            if not Path(config['weights']).is_file():
                raise ValueError('Model file missing')
            model = YOLO(config['weights'])
            names = model.names
            if len(names) != 5 or set(names.values() if isinstance(names, dict) else names) != CLASSES:
                raise ValueError('Five disaster classes required')
            model_status = 'MODEL_READY'
        except Exception as e:
            model, model_status = None, 'MODEL_ERROR: '+str(e)
    frames_dir = Path(session)/'frames'
    frames_dir.mkdir(exist_ok=True)
    count, last_save = 0, -1e9
    latest = queue.Queue(maxsize=1)
    finished, acquisition_stop = threading.Event(), threading.Event()
    errors = []
    def acquire():
        first_stamp, first_wall = None, None
        try:
            for index, frame in enumerate(stream):
                if stop.is_set() or acquisition_stop.is_set():
                    break
                if source == 'tum':
                    if first_stamp is None:
                        first_stamp, first_wall = frame['stamp'], time.monotonic()
                    due = first_wall+(index if limit else frame['stamp']-first_stamp)
                    while time.monotonic() < due:
                        if stop.wait(min(.02, max(0., due-time.monotonic()))) or acquisition_stop.is_set():
                            return
                if model is None:
                    with state.lock:
                        state.rgb = frame['rgb']
                        state.times['rgb'] = time.monotonic()
                if latest.full():
                    try:
                        latest.get_nowait()
                    except queue.Empty:
                        pass
                latest.put_nowait(frame)
        except Exception as e:
            errors.append(e)
        finally:
            stream.close()
            finished.set()
    def processing_frames():
        while not stop.is_set():
            try:
                yield latest.get(timeout=.1)
            except queue.Empty:
                if finished.is_set():
                    break
        if errors:
            raise errors[0]
    producer = threading.Thread(target=acquire, daemon=True)
    producer.start()
    try:
        with (Path(session)/'detections.jsonl').open('w', encoding='utf-8') as f:
            for frame in processing_frames():
                if stop.is_set():
                    break
                started = time.monotonic()
                rgb, depth = frame['rgb'], clean_depth(frame['depth_m'])
                validate_pair(rgb.shape, depth.shape, ['color_optical']*3,
                              [frame['stamp'], frame['depth_stamp']], config['sync_slop_s'])
                K = np.array(frame['K']).reshape(3, 3)
                D = np.array(frame['D'])
                mx, my = cv2.initUndistortRectifyMap(K, D, None, K, (rgb.shape[1], rgb.shape[0]), cv2.CV_32FC1)
                rectified_depth = cv2.remap(depth, mx, my, cv2.INTER_NEAREST)
                local, _ = mapper.update(rectified_depth, (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
                local_rgb = np.full((*local.shape, 3), (49, 61, 79), np.uint8)
                local_rgb[local > 0] = (243, 160, 65)
                colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(depth/6*255, 0, 255)), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                colored[depth == 0] = 0
                rows, annotated = [], rgb.copy()
                if model:
                    result = model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), device=config['device'],
                        conf=config['confidence'], imgsz=640, verbose=False)[0]
                    rows = detection_rows(result.boxes.data.cpu().numpy(), model.names, depth)
                    for row in rows:
                        x1, y1, x2, y2 = map(int, row['bbox'])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 185, 0), 2)
                        text = f"{row['label']} {row['confidence']:.2f} "
                        text += 'range unknown' if row['depth_m'] is None else f"{row['depth_m']:.2f}m Z"
                        cv2.putText(annotated, text, (max(x1, 0), max(y1-4, 20)), 0, .5, (255, 210, 0), 1)
                with state.lock:
                    state.source = frame['source']+' · 카메라 기준 투영만 / SLAM 미실행'
                    if model is not None:
                        state.rgb = annotated
                    state.depth, state.local = colored, local_rgb
                    state.tracking, state.reason = 'NOT_RUNNING', '원본 시각 재생 · 깊이 처리 별도 주기 · 누적 지도는 ROS RTAB-Map 모드'
                    state.model, state.rows = model_status, rows
                    for key in ('rgb', 'model', 'tracking', 'detections'):
                        state.times[key] = time.monotonic()
                event = dict(frame=count, source=frame['source'], stamp=frame['stamp'],
                    timestamp_domain=frame['timestamp_domain'], model=model_status, detections=rows)
                f.write(json.dumps(event, allow_nan=False)+'\n')
                f.flush()
                if limit or time.monotonic()-last_save >= 1/config['save_frame_hz']:
                    last_save = time.monotonic()
                    Image.fromarray(annotated).save(frames_dir/f'{count:06d}_rgb.png')
                    np.savez_compressed(frames_dir/f'{count:06d}_depth.npz', depth_m=depth, K=K, D=D,
                        local_obstacles=local, stamp=frame['stamp'], depth_stamp=frame['depth_stamp'])
                count += 1
                stop.wait(max(0, 1/config['detector_hz']-(time.monotonic()-started)))
        with state.lock:
            state.reason = f'입력 종료 · {count}개 실제 RGB-D 프레임 처리 및 기록'
    finally:
        acquisition_stop.set()
        producer.join(timeout=4)
