"""Read-only environment diagnostics. Never installs CUDA, torch or JetPack."""
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def inspect(config, source='ros', replay=False):
    report = dict(platform=platform.platform(), python=sys.executable, python_version=sys.version,
                  ros_distro=os.environ.get('ROS_DISTRO'), imports={}, packages={}, errors=[], warnings=[])
    for f in ('/etc/os-release', '/etc/nv_tegra_release', '/proc/meminfo'):
        if Path(f).is_file():
            report[f] = Path(f).read_text()[:2000]
    required = ['numpy', 'cv2', 'PIL', 'tkinter']
    if source == 'ros':
        if not replay and not config['mount_verified']:
            report['errors'].append('Mount transform unverified: measure base_link→camera_link, then set mount_verified=true in config.json')
        required += ['rclpy', 'cv_bridge', 'message_filters', 'tf2_ros', 'rtabmap_msgs.msg']
    if source == 'sdk':
        required += ['pyrealsense2']
    if config['weights']:
        required += ['torch', 'torchvision', 'ultralytics']
    for name in required:
        try:
            mod = importlib.import_module(name)
            report['imports'][name] = getattr(mod, '__version__', 'OK')
            if name == 'torch':
                report['torch_cuda'] = mod.version.cuda
                report['cuda_available'] = mod.cuda.is_available()
        except Exception as e:
            report['imports'][name] = str(e)
            # Missing model runtime is explicitly degraded; mapping can still run.
            bucket = 'warnings' if name in ('torch', 'torchvision', 'ultralytics') else 'errors'
            report[bucket].append('Cannot import '+name+': '+str(e))
    if not config['weights'] or not Path(config['weights']).is_file():
        report['warnings'].append('NO_MODEL: local trained best.pt unavailable; no detections will be invented')
    if source == 'ros':
        if platform.system() != 'Linux':
            report['errors'].append('Full RTAB-Map stack requires prepared Linux/Jetson ROS 2 environment')
        if not shutil.which('ros2'):
            report['errors'].append('ros2 command unavailable; source the installed ROS setup.bash')
        else:
            packages = ['rtabmap_odom', 'rtabmap_slam', 'rtabmap_msgs', 'tf2_ros', 'rosbag2_transport']
            if not replay:
                packages += ['realsense2_camera', 'imu_filter_madgwick']
            for package in packages:
                try:
                    p = subprocess.run(['ros2', 'pkg', 'prefix', package], capture_output=True, text=True, timeout=15)
                    report['packages'][package] = p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
                    if p.returncode:
                        report['errors'].append('Missing ROS package '+package)
                except Exception as e:
                    report['errors'].append(str(e))
    report['warnings'].append('Camera USB/IMU, mounting transform and GPU throughput require on-device acceptance checks')
    return report


if __name__ == '__main__':
    from core import load_config
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--source', choices=['ros', 'sdk', 'offline', 'tum', 'session'], default='ros')
    a = p.parse_args()
    result = inspect(load_config(a.config), a.source)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(bool(result['errors']))
