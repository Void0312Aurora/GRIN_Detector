from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from mini_grin_rebuild.core.configs import ExperimentConfig, TrainingConfig
from mini_grin_rebuild.evaluation.metrics import masked_abs_quantile, masked_mean_abs
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer


@dataclass(frozen=True)
class GateThresholds:
    physics_rmse_max: float | None = None
    physics_p95_abs_max: float | None = None
    edge_mean_abs_max: float | None = None
    edge_p95_abs_max: float | None = None
    outside_mean_abs_max: float | None = None
    outside_p95_abs_max: float | None = None
    logvar_mean_max: float | None = None

    @classmethod
    def from_training(cls, cfg: TrainingConfig) -> "GateThresholds":
        return cls(
            physics_rmse_max=getattr(cfg, "gate_physics_rmse_max", None),
            physics_p95_abs_max=getattr(cfg, "gate_physics_p95_abs_max", None),
            edge_mean_abs_max=getattr(cfg, "gate_edge_mean_abs_max", None),
            edge_p95_abs_max=getattr(cfg, "gate_edge_p95_abs_max", None),
            outside_mean_abs_max=getattr(cfg, "gate_outside_mean_abs_max", None),
            outside_p95_abs_max=getattr(cfg, "gate_outside_p95_abs_max", None),
            logvar_mean_max=getattr(cfg, "gate_logvar_mean_max", None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "physics_rmse_max": self.physics_rmse_max,
            "physics_p95_abs_max": self.physics_p95_abs_max,
            "edge_mean_abs_max": self.edge_mean_abs_max,
            "edge_p95_abs_max": self.edge_p95_abs_max,
            "outside_mean_abs_max": self.outside_mean_abs_max,
            "outside_p95_abs_max": self.outside_p95_abs_max,
            "logvar_mean_max": self.logvar_mean_max,
        }


def radial_masks(
    cfg: ExperimentConfig,
    *,
    h: int,
    w: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dtype = torch.float32
    lin_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    lin_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)

    aperture_r = float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0)
    if aperture_r <= 0:
        aperture_r = 1.0
    edge_start_frac = float(getattr(cfg.training, "eval_edge_band_start_frac", 0.9))
    edge_start_frac = max(0.0, min(edge_start_frac, 1.0))

    aperture = rr <= aperture_r
    outside = rr > aperture_r
    edge = (rr > (edge_start_frac * aperture_r)) & aperture
    return {"aperture": aperture, "outside": outside, "edge": edge}


def _masked_rmse(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"Expected values [H,W], got {tuple(values.shape)}")
    m = mask.to(dtype=torch.bool)
    if torch.sum(m) == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=torch.float32)
    return torch.sqrt(torch.mean(values[m] ** 2))


def physics_residual_metrics(
    cfg: ExperimentConfig,
    *,
    physics: DifferentiableGradientLayer,
    standard_height: torch.Tensor,
    defect_pred: torch.Tensor,
    diff_ts_target: dict[str, torch.Tensor],
    aperture_mask: torch.Tensor,
) -> dict[str, float]:
    """
    Compute physics-consistency residual stats inside the aperture.
    All tensors are expected on the same device.

    `standard_height`: [1,1,H,W]
    `defect_pred`: [1,1,H,W]
    `diff_ts_target`: {"I_x": [H,W], "I_y": [H,W]}
    """
    height = standard_height + defect_pred
    phys_out = physics(height)
    std_phys = physics(standard_height)
    pred_ix = (phys_out["I_x"] - std_phys["I_x"]).squeeze(0).squeeze(0)
    pred_iy = (phys_out["I_y"] - std_phys["I_y"]).squeeze(0).squeeze(0)
    tx = diff_ts_target["I_x"]
    ty = diff_ts_target["I_y"]
    if tx.ndim != 2 or ty.ndim != 2:
        raise ValueError(f"Expected diff targets [H,W], got {tuple(tx.shape)} and {tuple(ty.shape)}")

    rx = pred_ix - tx.to(pred_ix.dtype)
    ry = pred_iy - ty.to(pred_iy.dtype)
    rmag = torch.sqrt(rx**2 + ry**2 + 1e-12)
    rmse = _masked_rmse(rmag, aperture_mask)
    p95 = masked_abs_quantile(rmag, aperture_mask, q=0.95)
    return {"physics_rmse": float(rmse.detach().cpu().item()), "physics_p95_abs": float(p95.detach().cpu().item())}


def artifact_metrics(
    defect_pred_hw: torch.Tensor,
    *,
    edge_mask: torch.Tensor,
    outside_mask: torch.Tensor,
) -> dict[str, float]:
    """
    Artifact metrics that do NOT rely on ground truth.
    All tensors are expected to be [H,W] / boolean masks on the same device.
    """
    scores = torch.abs(defect_pred_hw)
    edge_mean = masked_mean_abs(scores, edge_mask)
    edge_p95 = masked_abs_quantile(scores, edge_mask, q=0.95)
    out_mean = masked_mean_abs(scores, outside_mask)
    out_p95 = masked_abs_quantile(scores, outside_mask, q=0.95)
    return {
        "edge_mean_abs": float(edge_mean.detach().cpu().item()),
        "edge_p95_abs": float(edge_p95.detach().cpu().item()),
        "outside_mean_abs": float(out_mean.detach().cpu().item()),
        "outside_p95_abs": float(out_p95.detach().cpu().item()),
    }


def logvar_metrics(logvar_hw: torch.Tensor, *, aperture_mask: torch.Tensor) -> dict[str, float]:
    if logvar_hw.ndim != 2:
        raise ValueError(f"Expected logvar [H,W], got {tuple(logvar_hw.shape)}")
    m = aperture_mask.to(dtype=torch.bool)
    if torch.sum(m) == 0:
        return {"logvar_mean": float("nan")}
    return {"logvar_mean": float(torch.mean(logvar_hw[m]).detach().cpu().item())}


def gate_decision(metrics: dict[str, float], thresholds: GateThresholds) -> dict[str, Any]:
    fails: list[str] = []

    def _check(name: str, value: float, thr: float | None, *, fail_on_nan: bool = True) -> None:
        if thr is None:
            return
        if not np.isfinite(value):
            if fail_on_nan:
                fails.append(name)
            return
        if value > float(thr):
            fails.append(name)

    _check("physics_rmse", metrics.get("physics_rmse", float("nan")), thresholds.physics_rmse_max)
    _check("physics_p95_abs", metrics.get("physics_p95_abs", float("nan")), thresholds.physics_p95_abs_max)
    _check("edge_mean_abs", metrics.get("edge_mean_abs", float("nan")), thresholds.edge_mean_abs_max)
    _check("edge_p95_abs", metrics.get("edge_p95_abs", float("nan")), thresholds.edge_p95_abs_max)
    _check("outside_mean_abs", metrics.get("outside_mean_abs", float("nan")), thresholds.outside_mean_abs_max)
    _check("outside_p95_abs", metrics.get("outside_p95_abs", float("nan")), thresholds.outside_p95_abs_max)
    _check("logvar_mean", metrics.get("logvar_mean", float("nan")), thresholds.logvar_mean_max, fail_on_nan=False)

    return {"pass": len(fails) == 0, "fails": fails}


def suggest_thresholds(
    rows: list[dict[str, float]],
    *,
    q: float = 0.99,
    include_logvar: bool = False,
) -> GateThresholds:
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1), got {q}")

    def _q(name: str) -> float | None:
        vals = [r.get(name) for r in rows if np.isfinite(float(r.get(name, float("nan"))))]
        if not vals:
            return None
        return float(np.quantile(np.asarray(vals, dtype=float), q))

    return GateThresholds(
        physics_rmse_max=_q("physics_rmse"),
        physics_p95_abs_max=_q("physics_p95_abs"),
        edge_mean_abs_max=_q("edge_mean_abs"),
        edge_p95_abs_max=_q("edge_p95_abs"),
        outside_mean_abs_max=_q("outside_mean_abs"),
        outside_p95_abs_max=_q("outside_p95_abs"),
        logvar_mean_max=_q("logvar_mean") if include_logvar else None,
    )


__all__ = [
    "GateThresholds",
    "artifact_metrics",
    "gate_decision",
    "logvar_metrics",
    "physics_residual_metrics",
    "radial_masks",
    "suggest_thresholds",
]
