"""Optional FID and paired LPIPS evaluation, using the legacy feature backbones."""
import argparse
import json
from pathlib import Path
import warnings

import torch
from PIL import Image

from .data import image_files


def calculate_lpips(samples, real, device='cuda', batch_size=32, pairs=2000):
    # Fail before loading weights if the requested protocol cannot be fulfilled.
    sample_files, real_files = image_files(samples), image_files(real)
    if pairs < 1 or batch_size < 1 or min(len(sample_files), len(real_files)) < pairs:
        raise ValueError(f'LPIPS requires {pairs} pairs; found {len(sample_files)} samples and {len(real_files)} real images')
    try:
        from lpips import LPIPS
        from torchvision import transforms
    except ImportError as error:
        raise ImportError('Install evaluation dependencies: pip install -e ".[eval]"') from error
    metric = LPIPS(net='alex', version='0.1').to(device).eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])
    resize = transforms.Resize((224, 224))
    total, count = 0., 0
    with torch.no_grad():
        # Keep batches with uniform spatial shape. Legacy behavior resizes both
        # images only when the paired image sizes differ.
        for start in range(0, pairs, batch_size):
            prepared = []
            for sample_path, real_path in zip(sample_files[start:min(start + batch_size, pairs)],
                                               real_files[start:min(start + batch_size, pairs)]):
                with Image.open(sample_path) as sample_image, Image.open(real_path) as real_image:
                    sample_image, real_image = sample_image.convert('RGB'), real_image.convert('RGB')
                    if sample_image.size != real_image.size:
                        sample_image, real_image = resize(sample_image), resize(real_image)
                    if min(sample_image.size) < 32:
                        raise ValueError('LPIPS AlexNet requires images at least 32 pixels in each dimension')
                    prepared.append((transform(sample_image), transform(real_image)))
            shapes = {tuple(a.shape) for a, _ in prepared}
            groups = [[pair for pair in prepared if tuple(pair[0].shape) == shape] for shape in shapes]
            for group in groups:
                sample_batch = torch.stack([a for a, _ in group]).to(device)
                real_batch = torch.stack([b for _, b in group]).to(device)
                values = metric(sample_batch, real_batch).flatten()
                total += values.double().sum().item()
                count += values.numel()
    return total / count, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--real', type=Path, required=True, help='Real image folder, or .npz statistics for FID-only')
    parser.add_argument('--metrics', nargs='+', choices=['fid', 'lpips'], default=['fid', 'lpips'])
    parser.add_argument('--lpips-pairs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('batch-size must be positive and workers nonnegative')
    result = dict(samples=str(args.samples), real=str(args.real))
    if 'lpips' in args.metrics and args.real.suffix == '.npz':
        raise ValueError('LPIPS requires real images, not FID statistics')
    if 'fid' in args.metrics:
        try:
            import pytorch_fid
        except ImportError as error:
            raise ImportError('Install evaluation dependencies: pip install -e ".[eval]"') from error
        # Use the same explicit file list for validation, statistics and counts.
        # pytorch-fid's independent glob would ignore uppercase extensions.
        import numpy as np
        from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
        from pytorch_fid.inception import InceptionV3
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(args.device)
        def statistics(path):
            if path.suffix == '.npz':
                with np.load(path) as data:
                    return data['mu'], data['sigma']
            files = image_files(path)
            if any(p.parent != path for p in files):
                raise ValueError('FID image folders must be flat; found nested images')
            if len(files) < 2:
                raise ValueError('FID requires at least two images in each folder')
            return calculate_activation_statistics([str(p) for p in files], model,
                args.batch_size, 2048, args.device, args.workers)
        real_mu, real_sigma = statistics(args.real)
        sample_mu, sample_sigma = statistics(args.samples)
        result['fid'] = float(calculate_frechet_distance(real_mu, real_sigma, sample_mu, sample_sigma))
        result['fid_backbone'] = 'pytorch-fid TensorFlow-compatible InceptionV3 pool3, 2048 dimensions'
        result['fid_sample_count'] = len(image_files(args.samples))
        if result['fid_sample_count'] != 50000:
            warnings.warn('Paper FID uses 50,000 generated images; this run uses a different count')
    if 'lpips' in args.metrics:
        value, count = calculate_lpips(args.samples, args.real, args.device, args.batch_size, args.lpips_pairs)
        result.update(lpips=value, lpips_pairs=count, lpips_backbone='alex',
                      lpips_pairing='first N lexicographically sorted paths, independently generated images')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
