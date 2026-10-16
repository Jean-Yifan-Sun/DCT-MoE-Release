import tempfile
from pathlib import Path
import unittest
from PIL import Image
import numpy as np

from dct_moe.evaluate import calculate_lpips


class EvaluationTests(unittest.TestCase):
    def test_lpips_identical_images_is_zero(self):
        try:
            import lpips
        except ImportError:
            self.skipTest('Install optional evaluation dependencies to test LPIPS')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for i in range(2):
                image = np.random.default_rng(i).integers(0, 256, (64, 64), dtype=np.uint8)
                Image.fromarray(image).save(path / f'{i}.png')
            value, count = calculate_lpips(path, path, device='cpu', batch_size=1, pairs=2)
            self.assertAlmostEqual(value, 0., places=7)
            self.assertEqual(count, 2)

    def test_lpips_rejects_insufficient_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Image.new('L', (64, 64)).save(path / '0.png')
            with self.assertRaises(ValueError):
                calculate_lpips(path, path, device='cpu', pairs=2)
