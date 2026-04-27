from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from mini_grin_rebuild.simulation.types import Capture
from mini_grin_rebuild.simulation.types import CaptureBundle


@runtime_checkable
class SimulationEngine(Protocol):
    """Offline forward simulator used to generate training/evaluation datasets."""

    name: str
    version: str

    def simulate_capture(
        self,
        height: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Capture:
        ...

    def meta(self) -> dict[str, Any]:
        ...


@runtime_checkable
class BundleSimulationEngine(SimulationEngine, Protocol):
    """Offline simulator that samples shared session parameters for a frame bundle."""

    def simulate_bundle(
        self,
        heights: Mapping[str, np.ndarray],
        *,
        rng: np.random.Generator | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CaptureBundle:
        ...


__all__ = ["BundleSimulationEngine", "SimulationEngine"]
