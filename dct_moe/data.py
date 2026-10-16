"""Prepared grayscale image folders; no silent resizing or intensity conversion."""
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def image_files(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Image directory does not exist: {root}')
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in EXTENSIONS and not p.name.startswith('.'))
    if not paths:
        raise ValueError(f'No images found in {root}')
    return paths


class DCTImageFolder(Dataset):
    def __init__(self, root, codec, augment=True, seed=1234):
        self.paths = image_files(root)
        self.codec = codec
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            if image.mode.startswith(('I', 'F')):
                raise ValueError(f'{path}: prepare intensities as 8-bit images before training; got {image.mode}')
            image = np.array(image.convert('L'))
        if self.augment and random.Random(self.seed + self.epoch * len(self) + index).random() < .5:
            image = np.fliplr(image)
        return torch.from_numpy(self.codec.encode(image))
