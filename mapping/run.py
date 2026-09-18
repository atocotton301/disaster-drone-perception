"""One entry point. ROS owns the camera; preview modes never claim SLAM."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from core import load_config
from doctor import inspect


def main(argv=None):
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=root/'config.json')
    p.add_argument('--source', choices=['ros', 'offline', 'tum', 'sdk', 'session'], default='ros')
    p.add_argument('--session', type=Path, help='Saved runs/<timestamp> folder for review')
    p.add_argument('--latest', action='store_true', help='Stop recording cleanly and open latest data session on this Jetson')
    p.add_argument('--headless', action='store_true', help='Onboard recording without desktop display')
    p.add_argument('--dataset', type=Path, help='Extracted TUM RGB-D sequence folder')
    p.add_argument('--replay', type=Path, help='ROS replay of TUM RGB-D sequence; no hardware is opened')
    p.add_argument('--bag', type=Path, help='Replay this system recorded ROS bag sensor inputs; recompute SLAM')
    p.add_argument('--check', action='store_true')
    p.add_argument('--auto-close', type=float, default=0, help='GUI smoke test only, seconds')
    p.add_argument('--area-name', default='한성대 낙상관')
    p.add_argument('--show-result', action='store_true', help='Show the saved mission map after clean shutdown')
    a = p.parse_args(argv)
    if a.latest and a.source != 'session':
        p.error('--latest requires --source session')
    if a.replay and a.bag:
        p.error('Choose one replay input')
    if a.bag and not (a.bag/'metadata.yaml').is_file():
        p.error('--bag must select a finalized ROS2 bag folder')
    if a.source == 'session' and not a.session and not a.latest:
        from tkinter import Tk, filedialog
        chooser = Tk()
        chooser.withdraw()
        selected = filedialog.askdirectory(title='Jetson에서 저장한 runs/날짜_시각 폴더 선택')
        chooser.destroy()
        if not selected:
            return 0
        a.session = Path(selected)
    c = load_config(a.config)
    if a.latest:
        try:
            with socket.create_connection(('127.0.0.1', 18764), timeout=2) as client:
                client.sendall(b'STOP\n')
                print(client.recv(128).decode(), flush=True)
        except OSError:
            pass
        deadline = time.monotonic()+40
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(('127.0.0.1', 18764), timeout=.5):
                    time.sleep(.25)
            except OSError:
                break
        sessions = []
        for f in Path(c['runs_dir']).glob('*/manifest.json'):
            try:
                if json.loads(f.read_text())['source'] in ('ros', 'tum', 'sdk', 'virtual'):
                    sessions.append(f.parent)
            except (KeyError, json.JSONDecodeError):
                continue
        if not sessions:
            raise SystemExit('No recorded data session yet')
        a.session = sorted(sessions)[-1]
    progress_window = None
    if a.show_result and not a.headless:
        from mission import notice
        progress_window = notice('탐색 시작 준비', '카메라·탐지 모델·지도 실행 환경을 확인하고 있습니다.')
    report = inspect(c, a.source, bool(a.replay or a.bag))
    if a.check:
        if progress_window:progress_window.destroy()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return bool(report['errors'])
    if report['errors']:
        if progress_window:progress_window.destroy()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print('Environment incomplete. Use --source offline for an honest no-input viewer.')
        return 2
    lock = socket.socket()
    lock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        lock.bind(('127.0.0.1', 18764))
        lock.listen(1)
    except OSError:
        if progress_window:progress_window.destroy()
        print('Another disaster system instance is already running. Close it first.')
        return 2
    session = Path(c['runs_dir'])/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session.mkdir(parents=True)
    (session/'environment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = dict(config=c, source=a.source, replay=str(a.replay) if a.replay else None,
                    source_session=str(a.session) if a.session else None,
                    hardware_verified=False, area_name=a.area_name, start_time=datetime.datetime.now().isoformat())
    if c['weights'] and Path(c['weights']).is_file():
        with open(c['weights'], 'rb') as model_file:
            digest = hashlib.sha256()
            for block in iter(lambda: model_file.read(1024*1024), b''):
                digest.update(block)
            manifest['model_sha256'] = digest.hexdigest()
    (session/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    from dashboard import State, Dashboard
    state, stop = State(), threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    processes, files, threads = [], [], []
    def spawn(cmd, name, env):
        f = (session/(name+'.log')).open('w', encoding='utf-8')
        files.append(f)
        child = subprocess.Popen(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((child, name))
        return child
    def monitor():
        while not stop.wait(.5):
            for child, name in processes:
                if child.poll() is not None:
                    with state.lock:
                        state.error = f'{name} 종료 (code={child.returncode}). 로그: {session}'
                        state.tracking = 'PROCESS_EXIT'
                        state.pose = None
                        state.markers = []
                    return
    def backend_target(function, *args):
        try:
            function(*args)
        except Exception as e:
            with state.lock:
                state.error = '입력 처리 실패: '+str(e)
                state.tracking = 'ERROR'
    def stop_endpoint():
        lock.settimeout(.5)
        while not stop.is_set():
            try:
                client, _ = lock.accept()
                with client:
                    client.settimeout(2)
                    if client.recv(128).strip() == b'STOP':
                        client.sendall(b'Stopping; allow up to 30 seconds to finalize records.\n')
                        stop.set()
            except (socket.timeout, OSError):
                pass
    threads.append(threading.Thread(target=stop_endpoint, daemon=True))
    try:
        if a.source == 'ros':
            env = os.environ.copy()
            env.update(DISASTER_CONFIG=str(a.config.resolve()), DISASTER_SESSION=str(session),
                       DISASTER_REPLAY=str(a.replay.resolve()) if a.replay else '', DISASTER_PYTHON=sys.executable,
                       DISASTER_BAG=str(a.bag.resolve()) if a.bag else '')
            os.environ['DISASTER_BAG'] = env['DISASTER_BAG']
            os.environ['DISASTER_REPLAY'] = env['DISASTER_REPLAY']
            # One isolated ROS domain for this entry point and its children.
            env.setdefault('ROS_DOMAIN_ID', '73')
            os.environ['ROS_DOMAIN_ID'] = env['ROS_DOMAIN_ID']
            if a.replay:
                state.source = 'PUBLIC RGB-D REPLAY / RTAB-Map · 실시간 카메라 아님'
            if a.bag:
                state.source = '저장 센서 기록 재처리 / RTAB-Map'
            spawn(['ros2', 'launch', str(root/'ros2/indoor_rgbd.launch.py')], 'pipeline', env)
            if c['record_bag']:
                topics = ['/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw',
                    '/camera/color/camera_info', '/camera/imu', '/imu/data', '/tf', '/tf_static',
                    '/odom', '/odom_info', '/map', '/disaster/map_graph', '/disaster/tracking',
                    '/disaster/annotated', '/disaster/detections', '/disaster/detector_status']
                spawn(['ros2', 'bag', 'record', '-o', str(session/'rosbag'), *topics], 'recording', env)
            from ros2.viewer_backend import run_backend
            threads.append(threading.Thread(target=backend_target, args=(run_backend, state, session, c, stop), daemon=True))
            threads.append(threading.Thread(target=monitor, daemon=True))
        elif a.source in ('sdk', 'tum'):
            from preview import run_preview
            threads.append(threading.Thread(target=backend_target, args=(run_preview, state, session, c, stop, a.source, a.dataset), daemon=True))
        elif a.source == 'session':
            from review import load_session
            load_session(state, a.session)
        else:
            state.source = 'OFFLINE · 카메라 / ROS / 모델 입력 없음'
            state.reason = '실제 입력을 연결해야 영상·거리·지도 결과가 생성됩니다'
        for t in threads:
            t.start()
        if a.auto_close > 0:
            threading.Timer(a.auto_close, stop.set).start()
        if a.headless:
            print('Recording locally. Stop with Ctrl+C or python3 stop.py. Session:', session, flush=True)
            while not stop.wait(.5):
                if state.error:
                    print(state.error, flush=True)
                    stop.set()
        else:
            if progress_window:progress_window.destroy();progress_window=None
            Dashboard(state, session, stop, c['stale_s']).run()
            if a.show_result:
                from mission import notice
                progress_window=notice('탐색 종료 · 저장 중', '카메라 기록을 종료하고 지도·탐지 기록을 저장하고 있습니다.')
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        # SIGINT gives rosbag and RTAB-Map time to finalize metadata and SQLite.
        for child, name in reversed(processes):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        forced = []
        for child, name in reversed(processes):
            try:
                deadline=time.monotonic()+20
                while child.poll() is None:
                    if progress_window:progress_window.update()
                    try:child.wait(timeout=.2)
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:raise
            except subprocess.TimeoutExpired:
                forced.append(name)
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for t in threads:
            t.join(timeout=6)
        for f in files:
            f.close()
        (session/'shutdown.json').write_text(json.dumps(dict(forced_termination=forced,
            warning='Check SQLite/bag integrity after forced termination' if forced else '',
            finished=datetime.datetime.now().isoformat())), encoding='utf-8')
        lock.close()
    print('Saved session:', session)
    if a.source not in ('session','offline'):
        from mission_result import finalize_session
        try:
            finalize_session(session)
        except Exception as e:
            if progress_window:progress_window.destroy();progress_window=None
            (session/'export_error.txt').write_text(str(e),encoding='utf-8')
            print('Map export failed:',e,flush=True)
            return 3
        if a.show_result:
            if progress_window:progress_window.destroy();progress_window=None
            from mission import open_result_window
            open_result_window(session)
    return 0


if __name__ == '__main__':
    sys.exit(main())
