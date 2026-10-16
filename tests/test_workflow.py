"""Exercise public commands on synthetic images, including save/resume/sample."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from dct_moe.config import load_config
from dct_moe.codec import DCTCodec
from dct_moe.data import DCTImageFolder

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_configs_define_correct_paper_geometry(self):
        for name, expected in [('acdc', (144, 64)), ('echonet', (196, 48)), ('nih', (256, 64))]:
            cfg = load_config(ROOT / 'configs' / f'{name}.json')
            self.assertEqual(DCTCodec(**cfg['codec']).shape, expected)

    def test_folder_converts_rgb_to_grayscale_like_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new('RGB', (16, 16), color=(20, 40, 80))
            image.save(Path(directory) / 'rgb.png')
            codec = DCTCodec(16, 4, 4, 512)
            dataset = DCTImageFolder(directory, codec, augment=False)
            torch.testing.assert_close(dataset[0], torch.from_numpy(codec.encode(np.array(image.convert('L')))))

    def test_train_resume_sample_and_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, run, samples = root / 'images', root / 'run', root / 'samples'
            data.mkdir()
            for i in range(4):
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (16, 16), dtype=np.uint8)).save(data / f'{i}.png')
            cfg = load_config(ROOT / 'configs/acdc.json')
            cfg['codec'] = dict(resolution=16, block_size=4, low_freqs=4, bound=512.)
            cfg['model'].update(embed_dim=32, depth=2, num_heads=4)
            cfg['diffusion']['entropy'] = [1., 2., 3., 4.]
            cfg['train'].update(steps=2, batch_size=2, warmup_steps=0, save_interval=1, log_interval=1)
            config = root / 'tiny.json'
            config.write_text(json.dumps(cfg))
            def run_command(module, *args, success=True):
                result = subprocess.run([sys.executable, '-m', module, *map(str, args)],
                    cwd=ROOT, capture_output=True, text=True)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result
            run_command('dct_moe.train', '--config', config, '--data', data, '--output', run,
                        '--max-steps', '1', '--workers', '0', '--cpu')
            checkpoint = run / 'step_0000001'
            self.assertTrue((checkpoint / 'ema.pt').is_file())
            run_command('dct_moe.train', '--config', config, '--data', data, '--output', run,
                        '--resume', checkpoint, '--max-steps', '2', '--workers', '0', '--cpu')
            self.assertEqual(json.loads((run / 'step_0000002' / 'progress.json').read_text())['step'], 2)
            uninterrupted = root / 'uninterrupted'
            run_command('dct_moe.train', '--config', config, '--data', data, '--output', uninterrupted,
                        '--workers', '0', '--cpu')
            resumed = torch.load(run / 'step_0000002/ema.pt', weights_only=True)
            full = torch.load(uninterrupted / 'step_0000002/ema.pt', weights_only=True)
            for key in full:
                torch.testing.assert_close(resumed[key], full[key], rtol=0, atol=0, msg=key)

            run_command('dct_moe.sample', '--checkpoint', run / 'step_0000002', '--output', samples,
                        '--count', '3', '--batch-size', '2', '--steps', '5', '--device', 'cpu')
            self.assertEqual(len(list(samples.glob('*.png'))), 3)
            with Image.open(samples / '0000000.png') as image:
                self.assertEqual(image.size, (16, 16))
            # Prevent mixing old samples into an evaluation directory.
            run_command('dct_moe.sample', '--checkpoint', checkpoint, '--output', samples,
                        '--count', '1', '--device', 'cpu', success=False)
            # Stop a resume from silently using a changed model/representation.
            bad = copy.deepcopy(cfg)
            bad['codec']['bound'] = 500.
            config.write_text(json.dumps(bad))
            result = run_command('dct_moe.train', '--config', config, '--data', data, '--output', root / 'bad',
                        '--resume', checkpoint, '--workers', '0', '--cpu', success=False)
            self.assertIn('config', result.stderr.lower())


if __name__ == '__main__':
    unittest.main()
