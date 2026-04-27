from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def sample_range(rng: np.random.Generator, value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return float(default)
        if len(value) == 1:
            return float(value[0])
        low, high = float(value[0]), float(value[1])
        if low == high:
            return low
        return float(rng.uniform(min(low, high), max(low, high)))
    return float(value)


def sample_pair(rng: np.random.Generator, value: Any, default: float) -> tuple[float, float]:
    if value is None:
        return (float(default), float(default))
    if isinstance(value, Mapping):
        return (
            sample_range(rng, value.get("x"), default),
            sample_range(rng, value.get("y"), default),
        )
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (list, tuple)) for v in value):
        return (sample_range(rng, value[0], default), sample_range(rng, value[1], default))
    sampled = sample_range(rng, value, default)
    return (sampled, sampled)


def normalized_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)
    return yy, xx, rr


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 1e-6:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2.0 * sigma * sigma))
    kernel /= float(np.sum(kernel))
    return kernel.astype(np.float32)


def convolve1d_reflect(image: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.size == 1:
        return image.copy()
    pad = kernel.size // 2
    padded = np.pad(image, [(pad, pad) if i == axis else (0, 0) for i in range(image.ndim)], mode="reflect")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded).astype(np.float32)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel1d(sigma)
    out = convolve1d_reflect(image, kernel, axis=0)
    out = convolve1d_reflect(out, kernel, axis=1)
    return out.astype(np.float32)


def resize_bilinear(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    out_h, out_w = shape
    in_h, in_w = image.shape
    if (in_h, in_w) == (out_h, out_w):
        return image.copy()
    y = np.linspace(0.0, in_h - 1.0, out_h, dtype=np.float32)
    x = np.linspace(0.0, in_w - 1.0, out_w, dtype=np.float32)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)

    top = (1.0 - wx)[None, :] * image[y0[:, None], x0[None, :]] + wx[None, :] * image[y0[:, None], x1[None, :]]
    bot = (1.0 - wx)[None, :] * image[y1[:, None], x0[None, :]] + wx[None, :] * image[y1[:, None], x1[None, :]]
    return ((1.0 - wy)[:, None] * top + wy[:, None] * bot).astype(np.float32)


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    valid = (y >= 0.0) & (y <= h - 1.0) & (x >= 0.0) & (x <= w - 1.0)
    yc = np.clip(y, 0.0, h - 1.0)
    xc = np.clip(x, 0.0, w - 1.0)
    y0 = np.floor(yc).astype(np.int64)
    x0 = np.floor(xc).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (yc - y0).astype(np.float32)
    wx = (xc - x0).astype(np.float32)
    out = (
        (1.0 - wy) * (1.0 - wx) * image[y0, x0]
        + (1.0 - wy) * wx * image[y0, x1]
        + wy * (1.0 - wx) * image[y1, x0]
        + wy * wx * image[y1, x1]
    )
    return np.where(valid, out, float(fill_value)).astype(np.float32)


def warp_affine(
    image: np.ndarray,
    *,
    shift_y: float = 0.0,
    shift_x: float = 0.0,
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    fill_value: float = 0.0,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cy = 0.5 * (h - 1.0)
    cx = 0.5 * (w - 1.0)
    theta = np.deg2rad(float(rotation_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    inv_scale = 1.0 / max(float(scale), 1e-6)
    y = yy - cy - float(shift_y)
    x = xx - cx - float(shift_x)
    src_y = inv_scale * (c * y + s * x) + cy
    src_x = inv_scale * (-s * y + c * x) + cx
    return bilinear_sample(image, src_y, src_x, fill_value=fill_value)


def sanitize_channels(channels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value, dtype=np.float32) for name, value in channels.items()}


__all__ = [
    "bilinear_sample",
    "convolve1d_reflect",
    "gaussian_blur",
    "gaussian_kernel1d",
    "normalized_grid",
    "resize_bilinear",
    "sample_pair",
    "sample_range",
    "sanitize_channels",
    "warp_affine",
]
