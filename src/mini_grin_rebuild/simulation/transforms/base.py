from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig


@dataclass
class TransformContext:
    cfg: SimulationConfig
    rng: np.random.Generator
    shape: tuple[int, int]
    frame_name: str | None = None
    bundle_params: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CaptureTransform(Protocol):
    name: str

    def sample_bundle_params(self, context: TransformContext) -> dict[str, Any]:
        ...

    def apply(
        self,
        channels: Mapping[str, np.ndarray],
        *,
        context: TransformContext,
        params: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        ...


class BaseTransform:
    name = "base"

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled", True))

    def sample_bundle_params(self, context: TransformContext) -> dict[str, Any]:
        return {}

    def apply(
        self,
        channels: Mapping[str, np.ndarray],
        *,
        context: TransformContext,
        params: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return {name: np.asarray(value, dtype=np.float32) for name, value in channels.items()}, {}


__all__ = ["BaseTransform", "CaptureTransform", "TransformContext"]
