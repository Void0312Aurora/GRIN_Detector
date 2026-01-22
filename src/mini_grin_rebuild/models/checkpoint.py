from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class CheckpointInfo:
    in_channels: int
    predict_logvar: bool


def infer_checkpoint_info(state_dict: Dict[str, torch.Tensor]) -> CheckpointInfo:
    in_channels = None
    for k, v in state_dict.items():
        if k.endswith("encoders.0.block.0.weight") and v.ndim == 4:
            in_channels = int(v.shape[1])
            break
    if in_channels is None:
        for v in state_dict.values():
            if v.ndim == 4:
                in_channels = int(v.shape[1])
                break
    if in_channels is None:
        raise RuntimeError("Could not infer in_channels from model state_dict")
    predict_logvar = any(k.startswith("logvar_head") for k in state_dict.keys())
    return CheckpointInfo(in_channels=in_channels, predict_logvar=predict_logvar)


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


__all__ = [
    "CheckpointInfo",
    "infer_checkpoint_info",
    "load_checkpoint",
    "save_checkpoint",
]

