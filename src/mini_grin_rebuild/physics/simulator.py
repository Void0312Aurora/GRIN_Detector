from __future__ import annotations

from typing import Dict

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.data.virtual_objects import VirtualObject
from mini_grin_rebuild.simulation.engines.ideal_gradient import IdealGradientEngine
from mini_grin_rebuild.simulation.engines.ideal_gradient import phase_from_height


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
    return IdealGradientEngine(cfg).simulate_capture(obj.height_map, rng=rng).to_channel_dict()


__all__ = ["simulate_capture", "phase_from_height"]
