"""Train DCT-MoE on prepared grayscale images, on one or more GPUs."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
import torch
from torch.utils.data import DataLoader

from .codec import DCTCodec
from .config import load_config, build_model
from .data import DCTImageFolder
from .diffusion import diffusion_loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='Prepared images at the configured resolution')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='Release step directory containing Accelerate state')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-steps', type=int, help='Stop early without changing the saved training config')
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def update_ema(destination, source, rate):
    for ema_parameter, parameter in zip(destination.parameters(), source.parameters()):
        ema_parameter.lerp_(parameter, 1 - rate)


def main():
    args = parse_args()
    if args.cpu:
        # Keep CPU distributed barriers from addressing visible CUDA devices.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = load_config(args.config)
    train = config['train']
    stop = args.max_steps if args.max_steps is not None else train['steps']
    if not 0 < stop <= train['steps'] or args.workers < 0:
        raise ValueError('max-steps must be in [1, configured steps]; workers must be nonnegative')
    accelerator = Accelerator(cpu=args.cpu, mixed_precision='no',
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=config['seed']),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),
                         InitProcessGroupKwargs(timeout=timedelta(seconds=3600))])
    set_seed(config['seed'])
    world = accelerator.num_processes
    batch = train['batch_size']
    if batch % world:
        raise ValueError('Global batch_size must be divisible by number of processes')
    if args.resume:
        previous = load_config(args.resume / 'config.json')
        if previous != config:
            raise ValueError('Resume config differs from saved config; use the original config')
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory is nonempty; use --resume or a new output directory')
    dataset = DCTImageFolder(args.data, DCTCodec(**config['codec']), seed=config['seed'])
    if len(dataset) < batch:
        raise ValueError(f'Dataset has {len(dataset)} images, smaller than global batch size {batch}')
    loader = DataLoader(dataset, batch_size=batch // world, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=accelerator.device.type == 'cuda',
                        generator=torch.Generator().manual_seed(config['seed']))
    model = build_model(config)
    ema = copy.deepcopy(model).to(accelerator.device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train['learning_rate'],
                                  weight_decay=train['weight_decay'], betas=tuple(train['betas']))
    warmup = train['warmup_steps']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(s / warmup, 1.) if warmup > 0 else 1.)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    set_seed(config['seed'], device_specific=True)
    # Register directly: scheduler advances once per global optimizer step.
    accelerator.register_for_checkpointing(scheduler, ema)
    step, epoch, batches_seen = 0, 0, 0
    if args.resume:
        progress = json.loads((args.resume / 'progress.json').read_text())
        if progress['world_size'] != world or progress['dataset_size'] != len(dataset):
            raise ValueError('Resume requires the same process count and dataset size')
        accelerator.load_state(str(args.resume / 'state'))
        step, epoch, batches_seen = progress['step'], progress['epoch'], progress['batches_seen']
        if step >= stop:
            raise ValueError('Checkpoint already reached max-steps')
    args.output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (args.output / 'data_manifest.json').write_text(json.dumps(
            [str(p.relative_to(args.data)) for p in dataset.paths], indent=2) + '\n')
    # Protect resume against a changed image list (no patient data are copied).
    import hashlib
    manifest = '\n'.join(str(p.relative_to(args.data)) for p in dataset.paths)
    manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
    if args.resume and progress.get('manifest_sha256') != manifest_hash:
        raise ValueError('Resume dataset file list differs from checkpoint')

    def save_checkpoint():
        target = args.output / f'step_{step:07d}'
        accelerator.wait_for_everyone()
        accelerator.save_state(str(target / 'state'))
        if accelerator.is_main_process:
            torch.save(ema.state_dict(), target / 'ema.pt')
            (target / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
            (target / 'progress.json').write_text(json.dumps(dict(
                step=step, epoch=epoch, batches_seen=batches_seen, world_size=world,
                dataset_size=len(dataset), manifest_sha256=manifest_hash), indent=2) + '\n')
        accelerator.wait_for_everyone()

    accelerator.print(f'{sum(p.numel() for p in model.parameters()):,} parameters; '
                      f'{len(dataset)} images; global batch {batch}; device {accelerator.device}')
    model.train()
    while step < stop:
        dataset.set_epoch(epoch)
        loader.set_epoch(epoch)
        remaining = accelerator.skip_first_batches(loader, batches_seen) if batches_seen else loader
        for clean in remaining:
            optimizer.zero_grad(set_to_none=True)
            loss, aux = diffusion_loss(model, clean, config['diffusion']['snr_scale'],
                                       config['diffusion'].get('entropy'))
            total = loss + config['diffusion']['aux_loss_weight'] * aux
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite training loss at step {step}')
            accelerator.backward(total)
            optimizer.step()
            scheduler.step()
            update_ema(ema, accelerator.unwrap_model(model), train['ema_rate'])
            step += 1
            batches_seen += 1
            if step == 1 or step % train['log_interval'] == 0:
                metrics = accelerator.reduce(torch.stack([loss.detach(), aux.detach()]), reduction='mean')
                accelerator.print(f'step={step} mse={metrics[0].item():.6f} aux={metrics[1].item():.6f} '
                                  f'lr={scheduler.get_last_lr()[0]:.6g}')
            if step % train['save_interval'] == 0 or step == stop:
                save_checkpoint()
            if step >= stop:
                break
        else:
            epoch += 1
            batches_seen = 0
    accelerator.end_training()


if __name__ == '__main__':
    main()
