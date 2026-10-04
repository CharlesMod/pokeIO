"""Screen preprocessing: RGB -> palette levels -> downsample -> bit-packing.

Observations travel through shared memory as packed uint8 (e.g. 4 pixels/byte
for 4-level screens, ~64x less traffic than float RGB) and are unpacked on the
GPU by the policy (``pokeio.policy.unpack_pixels``).
"""

from __future__ import annotations

import numpy as np


def bits_per_pixel(levels: int) -> int:
    """Packing width; 8 means 'not packed'."""
    for b in (1, 2, 4):
        if levels <= (1 << b):
            return b
    return 8


def to_gray(rgb: np.ndarray, channel: str = "luma") -> np.ndarray:
    if channel == "luma":
        r = rgb[:, :, 0].astype(np.uint16)
        g = rgb[:, :, 1].astype(np.uint16)
        b = rgb[:, :, 2].astype(np.uint16)
        return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, "rgb".index(channel)])


def quantize(gray: np.ndarray, levels: int) -> np.ndarray:
    """uint8 0..255 -> 0..levels-1 (uniform bins; exact for 4-shade GB output)."""
    return ((gray.astype(np.uint16) * levels) >> 8).astype(np.uint8)


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    return img[::factor, ::factor]


def pack(levels_img: np.ndarray, bpp: int) -> np.ndarray:
    """Pack (H, W) values < 2**bpp into (H, W*bpp/8) bytes, MSB first."""
    if bpp == 8:
        return levels_img
    per = 8 // bpp
    h, w = levels_img.shape
    v = levels_img.reshape(h, w // per, per).astype(np.uint8)
    shifts = np.arange(per - 1, -1, -1, dtype=np.uint8) * bpp
    return np.bitwise_or.reduce(v << shifts, axis=2).astype(np.uint8)


def unpack(packed: np.ndarray, bpp: int) -> np.ndarray:
    """Inverse of ``pack`` (numpy; used by tools and tests)."""
    if bpp == 8:
        return packed
    per = 8 // bpp
    shifts = np.arange(per - 1, -1, -1, dtype=np.uint8) * bpp
    mask = np.uint8((1 << bpp) - 1)
    out = (packed[..., None] >> shifts) & mask
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * per)


class ScreenProcessor:
    def __init__(self, height: int, width: int, downsample_factor: int, levels: int, channel: str):
        self.ds = downsample_factor
        self.levels = levels
        self.channel = channel
        self.h = -(-height // downsample_factor)
        self.w = -(-width // downsample_factor)
        self.bpp = bits_per_pixel(levels)
        if self.bpp < 8 and self.w % (8 // self.bpp):
            self.bpp = 8  # width not divisible: fall back to unpacked
        self.packed_w = self.w * self.bpp // 8 if self.bpp < 8 else self.w

    def levels_image(self, rgb: np.ndarray) -> np.ndarray:
        """(h, w) uint8 palette levels of the downsampled frame."""
        return downsample(quantize(to_gray(rgb, self.channel), self.levels), self.ds)

    def pack(self, levels_img: np.ndarray) -> np.ndarray:
        return pack(levels_img, self.bpp)
