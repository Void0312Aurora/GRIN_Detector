from __future__ import annotations

from typing import Any

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.simulation.types import Capture


def phase_from_height(cfg: SimulationConfig, height: np.ndarray) -> np.ndarray:
    scale = (2.0 * np.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air)
    return scale * height


class IdealGradientEngine:
    """Legacy ideal model: intensity is squared phase-gradient plus optional Gaussian noise."""

    name = "ideal_gradient"
    version = "1"

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg

    def simulate_capture(
        self,
        height: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Capture:
        phase_map = phase_from_height(self.cfg, np.asarray(height))
        grad_y, grad_x = np.gradient(phase_map, self.cfg.dx)
        intensity_x = np.clip(grad_x**2, a_min=0.0, a_max=None)
        intensity_y = np.clip(grad_y**2, a_min=0.0, a_max=None)
        if self.cfg.noise_level > 0:
            if rng is None:
                rng = np.random.default_rng()
            intensity_x = intensity_x + rng.normal(0.0, self.cfg.noise_level, intensity_x.shape)
            intensity_y = intensity_y + rng.normal(0.0, self.cfg.noise_level, intensity_y.shape)
        capture_meta: dict[str, Any] = dict(meta or {})
        capture_meta.update(self.meta())
        return Capture(
            channels={"I_x": intensity_x, "I_y": intensity_y},
            meta=capture_meta,
        )

    def meta(self) -> dict[str, Any]:
        return {
            "engine_name": self.name,
            "engine_version": self.version,
            "model": "squared_phase_gradient",
            "noise_model": "additive_gaussian" if self.cfg.noise_level > 0 else "none",
        }


__all__ = ["IdealGradientEngine", "phase_from_height"]
