import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
import numpy as np
from core import (depth_meters, validate_pair, grid_pixel, occupancy_rgb,
                  odom_valid, TrackingGate, detection_rows, load_config)
from mapping import LocalMapper, box_distance, project, obstacle_grid


class ProcessingTests(unittest.TestCase):
    def test_depth_units_holes_and_bad_encoding(self):
        raw = np.full((12, 12), 2100, np.uint16)
        raw[4:7, 4:7] = 0
        d = depth_meters(raw, '16UC1')
        self.assertAlmostEqual(float(d[1, 1]), 2.1, places=5)
        self.assertEqual(d[5, 5], 0)
        f = np.full((12, 12), 2.1, np.float32)
        f[5, 5] = np.nan
        self.assertEqual(depth_meters(f, '32FC1')[5, 5], 0)
        with self.assertRaises(ValueError):
            depth_meters(raw, 'mono16')

    def test_alignment_and_sync_rejection(self):
        validate_pair((480, 640, 3), (480, 640), ['optical']*3, [1, 1.005, 1], .015)
        for shape, frames, stamps in [((240, 320), ['optical']*3, [1, 1, 1]),
            ((480, 640), ['rgb', 'depth', 'rgb'], [1, 1, 1]),
            ((480, 640), ['optical']*3, [1, 1.04, 1])]:
            with self.assertRaises(ValueError):
                validate_pair((480, 640, 3), shape, frames, stamps, .015)

    def test_lost_and_stale_stop_acceptance(self):
        gate = TrackingGate(1.5)
        self.assertEqual(gate.status(0), 'WAITING')
        gate.observe(True, 1)
        self.assertEqual(gate.status(1.1), 'TRACKING')
        gate.observe(False, 1.2)
        self.assertEqual(gate.status(1.3), 'LOST')
        self.assertEqual(gate.status(3), 'STALE')
        gate.observe(True, 4)
        self.assertEqual(gate.status(4), 'TRACKING')

    def test_odometry_invalid_covariance_quaternion(self):
        c = np.zeros(36)
        p = [0, 0, 0, 0, 0, 0, 1]
        self.assertTrue(odom_valid(False, c, p))
        self.assertFalse(odom_valid(True, c, p))
        for value in (-1, np.nan, 9999):
            c[0] = value
            self.assertFalse(odom_valid(False, c, p))
        c[0] = 0
        self.assertFalse(odom_valid(False, c, [0]*7))

    def test_map_origin_rotation_and_vertical_flip(self):
        meta = dict(origin=[10, 20, math.pi/2], resolution=.05, height=100)
        x, y = grid_pixel(10, 20.1, meta)
        self.assertAlmostEqual(x, 2)
        self.assertAlmostEqual(y, 99)
        image = occupancy_rgb(np.array([[-1, 100], [0, -1]], np.int8))
        np.testing.assert_array_equal(image[0, 0], [222, 231, 233])
        np.testing.assert_array_equal(image[1, 1], [243, 160, 65])

    def test_no_depth_and_smoke_do_not_invent_range(self):
        boxes = np.array([[1, 1, 10, 10, .9, 0], [1, 1, 10, 10, .8, 1]])
        rows = detection_rows(boxes, {0: 'person', 1: 'smoke'}, np.full((12, 12), 2., np.float32))
        self.assertEqual(rows[0]['depth_m'], 2.)
        self.assertIsNone(rows[1]['depth_m'])
        self.assertEqual(rows[1]['roi_surface_depth_m'], 2.)
        self.assertIsNone(box_distance(np.zeros((12, 12)), [1, 1, 10, 10]))
        self.assertIsNone(box_distance(np.ones((12, 12)), [-100, -100, -90, -90]))

    def test_projection_and_nonpersistent_local_map(self):
        points = project(np.full((8, 8), 2.), [100, 100, 4, 4], stride=1)
        np.testing.assert_allclose(points[36], [0, 0, 2])
        mapper = LocalMapper(min_points=1)
        first, _ = mapper.update(np.full((8, 8), 2.), [100, 100, 4, 4])
        second, _ = mapper.update(np.zeros((8, 8)), [100, 100, 4, 4])
        self.assertTrue(first.any())
        self.assertFalse(second.any())

    def test_height_band_uses_camera_up_axis(self):
        points = np.array([[0, -1., 2], [0, 1., 3.]])
        grid = obstacle_grid(points, height_band=(.9, 1.1))
        self.assertEqual(np.count_nonzero(grid), 1)
        self.assertEqual(grid[80, 80], 255)

    def test_config_paths_relative_to_config(self):
        config = load_config(Path(__file__).resolve().parents[1]/'config.json')
        self.assertTrue(Path(config['runs_dir']).is_absolute())
        self.assertEqual(config['weights'], '')
        self.assertFalse(config['mount_verified'])


if __name__ == '__main__':
    unittest.main()
