"""Portable JSON configuration shared by all commands."""
import json
from pathlib import Path
import math
from .codec import DCTCodec
from .model import UViT_greyscale_MoE


def load_config(path):
    config = json.loads(Path(path).read_text())
    codec = DCTCodec(**config['codec'])
    diffusion = config['diffusion']
    entropy = diffusion.get('entropy')
    if entropy is not None and (len(entropy) != codec.low_freqs or
            any(not math.isfinite(x) or x <= 0 for x in entropy)):
        raise ValueError('config entropy must have one positive finite value per retained frequency')
    if not math.isfinite(diffusion['snr_scale']) or diffusion['snr_scale'] <= 0:
        raise ValueError('config snr_scale must be positive and finite')
    if config['train']['batch_size'] < 1 or config['train']['steps'] < 1:
        raise ValueError('config batch_size and steps must be positive')
    if config['train']['save_interval'] < 1 or config['train']['log_interval'] < 1:
        raise ValueError('config save_interval and log_interval must be positive')
    return config


def build_model(config):
    codec = DCTCodec(**config['codec'])
    return UViT_greyscale_MoE(tokens=codec.tokens, low_freqs=codec.low_freqs, **config['model'])
