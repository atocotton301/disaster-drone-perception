import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from inputs import tum_frames


class TumInputTests(unittest.TestCase):
    def test_real_format_scale_and_unique_timestamp_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root/'rgb.png')
            Image.fromarray(np.full((8, 8), 10000, np.uint16)).save(root/'depth.png')
            (root/'rgb.txt').write_text('# fixture timestamps\n1.000 rgb.png\n1.010 rgb.png\n2.000 rgb.png\n')
            (root/'depth.txt').write_text('1.002 depth.png\n2.020 depth.png\n')
            frames = list(tum_frames(root))
            self.assertEqual(len(frames), 1)  # no duplicate depth, no >15ms pair
            self.assertEqual(frames[0]['stamp'], 1.)
            self.assertAlmostEqual(frames[0]['depth_stamp'], 1.002)
            np.testing.assert_allclose(frames[0]['depth_m'], 2.)
            self.assertEqual(frames[0]['timestamp_domain'], 'TUM_MEASURED_TIMESTAMPS')

    def test_index_cannot_escape_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'rgb.txt').write_text('1 ../private.png\n')
            with self.assertRaises(ValueError):
                list(tum_frames(root))
