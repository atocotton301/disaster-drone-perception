"""Reuses existing launch adapted from RTAB-Map D435i color example.
Upstream original reference: 73c98f87a807dd48d3f7d67ff18d77b4c69ac56c.
BSD-3-Clause: ../third_party/RTABMAP_LICENSE.txt. No flight-control interfaces.
"""
import os
import sys
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, EmitEvent
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import load_config


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    c = load_config(os.environ['DISASTER_CONFIG'])
    session = Path(os.environ['DISASTER_SESSION'])
    replay = os.environ.get('DISASTER_REPLAY', '')
    bag = os.environ.get('DISASTER_BAG', '')
    python = os.environ.get('DISASTER_PYTHON', sys.executable)
    common = {'frame_id': 'base_link', 'approx_sync': True,
              'approx_sync_max_interval': c['sync_slop_s'], 'qos': 2,
              'topic_queue_size': 30, 'sync_queue_size': 30,
              'wait_imu_to_init': not bool(replay), 'use_sim_time': bool(bag)}
    remap = [('imu', '/imu/data'), ('rgb/image', '/camera/color/image_raw'),
             ('rgb/camera_info', '/camera/color/camera_info'),
             ('depth/image', '/camera/aligned_depth_to_color/image_raw')]
    x, y, z = c['mount_xyz_m']
    roll, pitch, yaw = c['mount_rpy_rad']
    actions = [Node(package='tf2_ros', executable='static_transform_publisher',
        arguments=['--x', str(x), '--y', str(y), '--z', str(z), '--roll', str(roll),
                   '--pitch', str(pitch), '--yaw', str(yaw),
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link'])]
    if not replay and not bag:
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')),
            launch_arguments={'camera_namespace': '', 'camera_name': 'camera',
                'enable_color': 'true', 'enable_depth': 'true', 'enable_gyro': 'true',
                'enable_accel': 'true', 'unite_imu_method': '2',
                'align_depth.enable': 'true', 'enable_sync': 'true',
                'rgb_camera.color_profile': '640x480x30',
                'depth_module.depth_profile': '640x480x30',
                'depth_module.emitter_enabled': '1'}.items()))
        actions.append(Node(package='imu_filter_madgwick', executable='imu_filter_madgwick_node',
            parameters=[{'use_mag': False, 'world_frame': 'enu', 'publish_tf': False}],
            remappings=[('imu/data_raw', '/camera/imu')], output='screen'))
    elif replay:
        actions.append(ExecuteProcess(cmd=[python, str(root/'ros2/replay_rgbd.py'), replay], output='screen'))
    else:
        actions.append(ExecuteProcess(cmd=[python, str(root/'ros2/bag_static_tf.py')], output='screen'))
        actions.append(ExecuteProcess(cmd=['ros2', 'bag', 'play', bag, '--clock', '--delay', '5',
            '--topics', '/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw',
            '/camera/color/camera_info', '/camera/imu', '/imu/data', '/tf_static',
            '--remap', '/tf_static:=/disaster/recorded_tf_static'], output='screen'))
    odom = Node(package='rtabmap_odom', executable='rgbd_odometry', name='rgbd_odometry',
        parameters=[common, {'odom_frame_id': 'odom', 'publish_tf': True,
                            'Odom/ResetCountdown': '0'}], remappings=remap, output='screen')
    gate = ExecuteProcess(cmd=[python, str(root/'ros2/slam_gate.py')], output='screen')
    detector = ExecuteProcess(cmd=[python, str(root/'ros2/detection_node.py')], output='screen')
    slam = Node(package='rtabmap_slam', executable='rtabmap', name='rtabmap',
        parameters=[common, {'subscribe_depth': True, 'subscribe_odom_info': True,
            'database_path': str(session/'rtabmap.db'), 'map_frame_id': 'map',
            'Grid/CellSize': '0.05', 'Grid/RangeMax': '6.0', 'Grid/3D': 'true',
            'Grid/RayTracing': 'true', 'Grid/MapFrameProjection': 'true',
            'Grid/Sensor': '1', 'Grid/GroundIsObstacle': 'false',
            'Grid/NormalsSegmentation': 'true',
            'Grid/MaxGroundHeight': '0.15', 'Grid/MaxObstacleHeight': '2.0',
            'Rtabmap/DetectionRate': '1.0'}],
        remappings=[('imu', '/imu/data'), ('rgb/image', '/disaster/slam/rgb'),
            ('depth/image', '/disaster/slam/depth'), ('rgb/camera_info', '/disaster/slam/info'),
            ('odom', '/disaster/slam/odom'), ('odom_info', '/disaster/slam/odom_info'),
            ('map', '/map'), ('mapGraph', '/disaster/map_graph')], output='screen')
    actions += [odom, gate, detector, slam]
    for process in (odom, gate, detector, slam):
        actions.append(RegisterEventHandler(OnProcessExit(target_action=process,
            on_exit=[EmitEvent(event=Shutdown(reason='A required processing node exited; see session log'))])))
    return LaunchDescription(actions)
