"""Generate grayscale PNG images using deterministic DPM-Solver ODE sampling."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .codec import DCTCodec
from .config import load_config, build_model
from .diffusion import sample_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Release step directory, or legacy nnet_ema.pth file')
    parser.add_argument('--config', type=Path, help='Required for a legacy checkpoint file')
    parser.add_argument('--output', type=Path, required=True, help='New or empty image directory')
    parser.add_argument('--count', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--steps', type=int, help='Number of model evaluations (NFE), default 50')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config_path = args.config or args.checkpoint / 'config.json'
    config = load_config(config_path)
    if args.checkpoint.is_dir() and args.config is not None:
        if config != load_config(args.checkpoint / 'config.json'):
            raise ValueError('Sampling config differs from release checkpoint config')
    count = args.count if args.count is not None else config['sample']['count']
    batch = args.batch_size if args.batch_size is not None else config['sample']['batch_size']
    steps = args.steps if args.steps is not None else config['sample']['steps']
    if min(count, batch, steps) < 1:
        raise ValueError('count, batch-size and steps must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be empty to avoid mixing evaluation samples')
    codec = DCTCodec(**config['codec'])
    model = build_model(config).to(args.device).eval().requires_grad_(False)
    weights = args.checkpoint / 'ema.pt' if args.checkpoint.is_dir() else args.checkpoint
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for start in range(0, count, batch):
            size = min(batch, count - start)
            noise = torch.randn(size, *codec.shape, device=args.device, generator=generator)
            tokens = sample_tokens(model, noise, config['diffusion']['snr_scale'], steps).cpu().numpy()
            for offset, values in enumerate(tokens):
                Image.fromarray(codec.decode(values)).save(args.output / f'{start + offset:07d}.png')
            print(f'Generated {start + size}/{count}', flush=True)
    (args.output / 'sampling.json').write_text(json.dumps(dict(
        config=config, checkpoint=str(args.checkpoint), seed=args.seed, count=count,
        batch_size=batch, nfe=steps, device=args.device), indent=2) + '\n')


if __name__ == '__main__':
    main()
