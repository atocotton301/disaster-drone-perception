import json
import tempfile
import unittest
from pathlib import Path
from core import load_config


class CameraOriginConfig(unittest.TestCase):
    def load(self, offset):
        c=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
        c.update(tracking_origin='camera', mount_verified=False,
                 mount_xyz_m=offset, mount_rpy_rad=[0,0,0])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'config.json';p.write_text(json.dumps(c))
            return load_config(p)

    def test_camera_coordinates_do_not_require_invented_robot_height(self):
        c=self.load([0,0,0])
        self.assertFalse(c['mount_verified'])
        self.assertEqual(c['mount_xyz_m'],[0,0,0])

    def test_robot_offset_cannot_be_mislabeled_as_camera_origin(self):
        with self.assertRaisesRegex(ValueError,'identity mount'):
            self.load([0,0,1])
