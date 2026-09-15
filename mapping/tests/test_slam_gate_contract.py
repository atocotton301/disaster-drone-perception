"""Execute real gate callback code with message fixtures; ROS transport not tested."""
import ast
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
from core import TrackingGate, odom_valid, stamp_s, validate_pair


source = Path(__file__).resolve().parents[1]/'ros2/slam_gate.py'
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Gate')
cls.bases = []
namespace = dict(time=time, TrackingGate=TrackingGate, odom_valid=odom_valid,
                 stamp_s=stamp_s, validate_pair=validate_pair)
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
Gate = namespace['Gate']


def header(sec, frame='optical'):
    return NS(stamp=NS(sec=sec, nanosec=0), frame_id=frame)


def messages(sec=1, lost=False):
    rgb = NS(header=header(sec), height=480, width=640)
    depth = NS(header=header(sec), height=480, width=640)
    info = NS(header=header(sec))
    pose = NS(position=NS(x=0., y=0., z=0.), orientation=NS(x=0., y=0., z=0., w=1.))
    odom = NS(header=header(sec, 'odom'), pose=NS(pose=pose, covariance=[0.]*36))
    odom_info = NS(header=header(sec), lost=lost)
    return rgb, depth, info, odom, odom_info


class GateContractTests(unittest.TestCase):
    def setUp(self):
        self.gate = object.__new__(Gate)
        self.gate.c = {'sync_slop_s': .015}
        self.gate.last_stamp = self.gate.failure_stamp = -1.
        self.gate.health = TrackingGate(1.5)
        self.outputs = []
        self.gate.pubs = [NS(publish=self.outputs.append) for _ in range(5)]

    def test_valid_tuple_forwarded_and_failed_tuple_blocked(self):
        self.gate.process(*messages())
        self.assertEqual(len(self.outputs), 5)
        self.gate.process(*messages(2, lost=True))
        self.assertEqual(len(self.outputs), 5)
        self.assertEqual(self.gate.health.status(time.monotonic()), 'LOST')

    def test_lost_without_odometry_blocks_delayed_success(self):
        self.gate.on_info(messages(5, True)[-1])
        self.gate.process(*messages(4))
        self.assertFalse(self.outputs)
        self.gate.process(*messages(6))
        self.assertEqual(len(self.outputs), 5)

    def test_duplicate_stamp_not_accumulated(self):
        self.gate.process(*messages(1))
        self.gate.process(*messages(1))
        self.assertEqual(len(self.outputs), 5)
