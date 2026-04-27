from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Capture:
    """Observation produced by an offline simulation engine."""

    channels: Mapping[str, np.ndarray]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def require(self, *names: str) -> dict[str, np.ndarray]:
        missing = [name for name in names if name not in self.channels]
        if missing:
            raise KeyError(f"Capture missing channel(s): {', '.join(missing)}")
        return {name: np.asarray(self.channels[name]) for name in names}

    def to_channel_dict(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(value) for name, value in self.channels.items()}


@dataclass(frozen=True)
class CaptureBundle:
    """Multi-frame observation produced by a shared simulation session."""

    captures: Mapping[str, Capture]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def require(self, *names: str) -> dict[str, Capture]:
        missing = [name for name in names if name not in self.captures]
        if missing:
            raise KeyError(f"CaptureBundle missing frame(s): {', '.join(missing)}")
        return {name: self.captures[name] for name in names}


__all__ = ["Capture", "CaptureBundle"]
