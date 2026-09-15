"""Actual Tk window rendering with public recorded input; saves only this window."""
import argparse
import json
from pathlib import Path
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import ImageGrab
from core import load_config
from dashboard import State, Dashboard
from preview import run_preview


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    state, stop = State(), threading.Event()
    c = load_config(Path(__file__).resolve().parents[1]/'config.json')
    (a.out/'manifest.json').write_text(json.dumps(dict(source='tum', config=c)), encoding='utf-8')
    thread = threading.Thread(target=run_preview, args=(state, a.out, c, stop, 'tum', a.dataset, 5), daemon=True)
    thread.start()
    gui = Dashboard(state, a.out, stop)
    def capture():
        gui.root.update_idletasks()
        # Capture this application handle only, even when another app is in front.
        ImageGrab.grab(window=gui.root.winfo_id()).save(a.out/'actual_window.png')
        gui.snapshot()
    gui.root.after(2500, capture)
    gui.root.after(6200, gui.close)
    gui.run()
    thread.join(timeout=3)
    events = [json.loads(line) for line in (a.out/'detections.jsonl').read_text().splitlines()]
    assert len(events) == 5, len(events)
    assert all(e['detections'] == [] and e['model'] == 'NO_MODEL' for e in events)
    assert state.grid is None and state.pose is None and not state.path
    assert len(list((a.out/'frames').glob('*_rgb.png'))) == 5
    from review import load_session
    recorded = State()
    load_session(recorded, a.out)
    assert recorded.recorded and recorded.rgb is not None and recorded.depth is not None
    assert recorded.grid is None and not recorded.rows
    (a.out/'gui_result.json').write_text(json.dumps(dict(passed=True, real_frames=5,
        invented_detections=0, accumulated_slam=False, camera_pose=None, recorded_frames=5,
        camera_free_session_review=True)), encoding='utf-8')
    print('PASS: actual Tk GUI, 5 recorded RGB-D frames, saving, no fabricated detection/SLAM/pose, clean stop')


if __name__ == '__main__':
    main()
