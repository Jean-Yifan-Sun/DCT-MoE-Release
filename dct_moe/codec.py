"""Block DCT, zigzag truncation and spatial 2-by-2 block tokenization."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DCTCodec:
    resolution: int
    block_size: int
    low_freqs: int
    bound: float

    def __post_init__(self):
        if (not isinstance(self.resolution, int) or not isinstance(self.block_size, int)
                or self.block_size < 2 or self.block_size % 2
                or self.resolution < 2 * self.block_size
                or self.resolution % (2 * self.block_size)):
            raise ValueError('resolution must be divisible by 2 * block_size; block_size must be even')
        if not isinstance(self.low_freqs, int) or not 1 <= self.low_freqs <= self.block_size ** 2:
            raise ValueError('low_freqs must be between 1 and block_size squared')
        if not np.isfinite(self.bound) or self.bound <= 0:
            raise ValueError('bound must be finite and positive')

    @property
    def tokens(self):
        return (self.resolution // (2 * self.block_size)) ** 2

    @property
    def shape(self):
        return (self.tokens, 4 * self.low_freqs)

    @property
    def zigzag(self):
        b = self.block_size
        result = []
        for s in range(2 * b - 1):
            diagonal = [(i, s - i) for i in range(max(0, s - b + 1), min(s, b - 1) + 1)]
            if s % 2 == 0:
                diagonal.reverse()
            result.extend(i * b + j for i, j in diagonal)
        return np.array(result)

    def encode(self, image):
        image = np.asarray(image)
        if image.shape != (self.resolution, self.resolution) or image.dtype != np.uint8:
            raise ValueError(f'Expected uint8 grayscale image of shape {(self.resolution,) * 2}; got {image.shape}, {image.dtype}')
        b, n = self.block_size, self.resolution // self.block_size
        blocks = (image.astype(np.float32) - 128).reshape(n, b, n, b).transpose(0, 2, 1, 3)
        coefficients = np.stack([cv2.dct(np.ascontiguousarray(x)) for x in blocks.reshape(-1, b, b)])
        coefficients = coefficients.reshape(n, n, b * b)[..., self.zigzag[:self.low_freqs]]
        # [macro row, macro col, top-left/top-right/bottom-left/bottom-right, frequency]
        coefficients = coefficients.reshape(n // 2, 2, n // 2, 2, self.low_freqs).transpose(0, 2, 1, 3, 4)
        return np.ascontiguousarray(coefficients.reshape(self.shape) / self.bound, dtype=np.float32)

    def decode(self, tokens, as_uint8=True):
        tokens = np.asarray(tokens)
        if tokens.shape != self.shape or not np.isfinite(tokens).all():
            raise ValueError(f'Expected finite tokens of shape {self.shape}; got {tokens.shape}')
        b, n = self.block_size, self.resolution // self.block_size
        # Preserve the legacy generation clamp; truncation fills high frequencies with zero.
        values = np.clip(tokens, -2, 2).reshape(n // 2, n // 2, 2, 2, self.low_freqs)
        values = values.transpose(0, 2, 1, 3, 4).reshape(n * n, self.low_freqs)
        coefficients = np.zeros((n * n, b * b), dtype=np.float64)
        coefficients[:, self.zigzag[:self.low_freqs]] = values
        # Match legacy zero-fill/promotion before denormalization and float64 IDCT.
        coefficients *= self.bound
        blocks = np.stack([cv2.idct(np.ascontiguousarray(x)) for x in coefficients.reshape(-1, b, b)])
        image = blocks.reshape(n, n, b, b).transpose(0, 2, 1, 3).reshape(self.resolution, self.resolution) + 128
        # Legacy combine_blocks writes the reconstructed pixels into float32.
        image = image.astype(np.float32)
        return np.clip(image, 0, 255).astype(np.uint8) if as_uint8 else image
