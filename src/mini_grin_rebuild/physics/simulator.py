from __future__ import annotations

from typing import Dict

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.data.virtual_objects import VirtualObject


def phase_from_height(cfg: SimulationConfig, height: np.ndarray) -> np.ndarray:
    scale = (2.0 * np.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air)
    return scale * height


def simulate_capture(
    cfg: SimulationConfig,
    obj: VirtualObject,
    *,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    """
    NumPy forward model used for dataset generation / quick baselines.

    Notes:
    - The underlying measurement model is kept consistent with the existing project:
      intensity responses are squared phase gradients (with optional additive Gaussian noise).
    - For auditability, callers may pass an explicit `rng` to control noise deterministically.
      If omitted, noise uses a fresh default generator (non-deterministic, same as legacy behavior).
    """
    phase_map = phase_from_height(cfg, obj.height_map)
    grad_y, grad_x = np.gradient(phase_map, cfg.dx)
    intensity_x = np.clip(grad_x**2, a_min=0.0, a_max=None)
    intensity_y = np.clip(grad_y**2, a_min=0.0, a_max=None)
    if cfg.noise_level > 0:
        if rng is None:
            rng = np.random.default_rng()
        intensity_x = intensity_x + rng.normal(0.0, cfg.noise_level, intensity_x.shape)
        intensity_y = intensity_y + rng.normal(0.0, cfg.noise_level, intensity_y.shape)
    return {"I_x": intensity_x, "I_y": intensity_y}


__all__ = ["simulate_capture", "phase_from_height"]

