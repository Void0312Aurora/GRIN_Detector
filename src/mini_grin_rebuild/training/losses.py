from __future__ import annotations

from typing import Dict, Optional

import torch

from mini_grin_rebuild.core.configs import TrainingConfig
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer


def _as_bhw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 3:
        return t
    raise ValueError(f"Expected [H,W] or [B,H,W], got {tuple(t.shape)}")


def data_weight_map(
    cfg: TrainingConfig,
    physics: DifferentiableGradientLayer,
    standard_height: torch.Tensor,
) -> torch.Tensor:
    """
    Legacy weighting: down-weight regions with large standard phase gradients.
    Returns weight map of shape [B,H,W].
    """
    if standard_height.ndim == 2:
        std = standard_height.unsqueeze(0).unsqueeze(0)
    elif standard_height.ndim == 3:
        std = standard_height.unsqueeze(1)
    else:
        std = standard_height
    std_phase = physics._phase(std).squeeze(1)  # [B,H,W]
    grad_y, grad_x = physics._gradient(std_phase)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-12)
    w = 1.0 / (1.0 + cfg.data_grad_alpha * grad_mag**2)
    return torch.clamp(w, min=cfg.data_weight_min)


def diff_loss(
    cfg: TrainingConfig,
    physics: DifferentiableGradientLayer,
    *,
    standard_height: torch.Tensor,
    defect: torch.Tensor,
    diff_ts_target: Dict[str, torch.Tensor],
    logvar: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Data term (always present): match ΔI(test-standard) produced by physics(standard+defect).
    """
    if defect.ndim == 2:
        defect = defect.unsqueeze(0).unsqueeze(0)
    elif defect.ndim == 3:
        defect = defect.unsqueeze(1)
    if standard_height.ndim == 2:
        standard_height = standard_height.unsqueeze(0).unsqueeze(0)
    elif standard_height.ndim == 3:
        standard_height = standard_height.unsqueeze(1)

    height = standard_height + defect
    physics_out = physics(height)
    std_phys = physics(standard_height)

    delta_ix = physics_out["I_x"] - std_phys["I_x"]  # [B,1,H,W]
    delta_iy = physics_out["I_y"] - std_phys["I_y"]  # [B,1,H,W]
    preds = torch.cat([delta_ix, delta_iy], dim=1)   # [B,2,H,W]

    tx = _as_bhw(diff_ts_target["I_x"])
    ty = _as_bhw(diff_ts_target["I_y"])
    target = torch.stack([tx, ty], dim=1)  # [B,2,H,W]

    w = data_weight_map(cfg, physics, standard_height).unsqueeze(1)  # [B,1,H,W]
    residual = preds - target

    use_logvar = cfg.predict_logvar and (logvar is not None)
    if not use_logvar:
        return torch.mean(w * residual**2)

    if logvar.ndim == 3:
        logvar = logvar.unsqueeze(1)
    logvar = torch.clamp(logvar, min=cfg.logvar_min, max=cfg.logvar_max)
    logvar = logvar.expand_as(residual)
    return torch.mean(w * (torch.exp(-logvar) * residual**2 + logvar))


def sr_diff_loss(
    cfg: TrainingConfig,
    physics: DifferentiableGradientLayer,
    *,
    standard_height: torch.Tensor,
    reference_height: torch.Tensor,
    diff_sr_target: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Legacy SR-diff term: match ΔI(standard-reference) for both channels.
    """
    if standard_height.ndim == 2:
        standard_height = standard_height.unsqueeze(0).unsqueeze(0)
    elif standard_height.ndim == 3:
        standard_height = standard_height.unsqueeze(1)
    if reference_height.ndim == 2:
        reference_height = reference_height.unsqueeze(0).unsqueeze(0)
    elif reference_height.ndim == 3:
        reference_height = reference_height.unsqueeze(1)

    std_phys = physics(standard_height)
    ref_phys = physics(reference_height)

    pred_x = (std_phys["I_x"] - ref_phys["I_x"]).squeeze(1)  # [B,H,W]
    pred_y = (std_phys["I_y"] - ref_phys["I_y"]).squeeze(1)  # [B,H,W]
    tx = _as_bhw(diff_sr_target["I_x"])
    ty = _as_bhw(diff_sr_target["I_y"])

    # Legacy dataset trainer normalizes SR residuals by target magnitude to avoid dominance.
    target_sr = torch.stack([tx, ty], dim=1)  # [B,2,H,W]
    norm = torch.mean(torch.abs(target_sr)) + 1e-6
    txn = tx / norm
    tyn = ty / norm
    loss_x = torch.mean(torch.abs(pred_x / norm - txn))
    loss_y = torch.mean(torch.abs(pred_y / norm - tyn))
    return 0.5 * (loss_x + loss_y)


def curl_loss(defect: torch.Tensor, physics: DifferentiableGradientLayer) -> torch.Tensor:
    if defect.ndim == 2:
        defect = defect.unsqueeze(0).unsqueeze(0)
    elif defect.ndim == 3:
        defect = defect.unsqueeze(1)
    grad_y, grad_x = physics._gradient(defect.squeeze(1))
    grad_xy = grad_x[:, 1:, :] - grad_x[:, :-1, :]
    grad_xy = torch.nn.functional.pad(grad_xy, (0, 0, 1, 0))
    grad_yx = grad_y[:, :, 1:] - grad_y[:, :, :-1]
    grad_yx = torch.nn.functional.pad(grad_yx, (1, 0, 0, 0))
    return torch.mean(torch.abs(grad_xy - grad_yx))


def sparsity_loss(defect: torch.Tensor) -> torch.Tensor:
    if defect.ndim == 2:
        defect = defect.unsqueeze(0).unsqueeze(0)
    elif defect.ndim == 3:
        defect = defect.unsqueeze(1)
    return torch.mean(torch.abs(defect))


def edge_suppress_loss(defect: torch.Tensor, defect_roi_radius: float) -> torch.Tensor:
    if defect.ndim == 2:
        defect = defect.unsqueeze(0).unsqueeze(0)
    elif defect.ndim == 3:
        defect = defect.unsqueeze(1)
    defect_field = defect.squeeze(1)  # [B,H,W]
    h, w = defect_field.shape[-2], defect_field.shape[-1]
    lin_y = torch.linspace(-1.0, 1.0, h, device=defect_field.device, dtype=defect_field.dtype)
    lin_x = torch.linspace(-1.0, 1.0, w, device=defect_field.device, dtype=defect_field.dtype)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)
    center_mask = (rr <= defect_roi_radius).to(defect_field.dtype)
    edge_mask = (1.0 - center_mask)
    return torch.mean(edge_mask * torch.abs(defect_field))


def edge_band_suppress_loss(
    defect: torch.Tensor,
    *,
    aperture_radius: float,
    edge_band_start_frac: float,
) -> torch.Tensor:
    """
    Penalize |defect| in an edge band near the aperture boundary:
      edge_start = edge_band_start_frac * aperture_radius
      mask = edge_start < r <= aperture_radius
    """
    if defect.ndim == 2:
        defect = defect.unsqueeze(0).unsqueeze(0)
    elif defect.ndim == 3:
        defect = defect.unsqueeze(1)
    defect_field = defect.squeeze(1)  # [B,H,W]
    h, w = defect_field.shape[-2], defect_field.shape[-1]
    lin_y = torch.linspace(-1.0, 1.0, h, device=defect_field.device, dtype=defect_field.dtype)
    lin_x = torch.linspace(-1.0, 1.0, w, device=defect_field.device, dtype=defect_field.dtype)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)

    ap = float(aperture_radius) if float(aperture_radius) > 0 else 1.0
    start = float(edge_band_start_frac) * ap
    start = max(0.0, min(start, ap))
    mask = ((rr > start) & (rr <= ap)).to(defect_field.dtype)
    return torch.mean(mask * torch.abs(defect_field))


def total_loss(
    cfg: TrainingConfig,
    physics: DifferentiableGradientLayer,
    *,
    standard_height: torch.Tensor,
    defect: torch.Tensor,
    diff_ts_target: Dict[str, torch.Tensor],
    reference_height: Optional[torch.Tensor] = None,
    diff_sr_target: Optional[Dict[str, torch.Tensor]] = None,
    logvar: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the full loss dict using only the legacy-enabled terms.
    """
    terms: Dict[str, torch.Tensor] = {}
    terms["diff"] = diff_loss(
        cfg,
        physics,
        standard_height=standard_height,
        defect=defect,
        diff_ts_target=diff_ts_target,
        logvar=logvar,
    )

    if cfg.sr_diff_weight > 0 and reference_height is not None and diff_sr_target is not None:
        terms["sr_diff"] = cfg.sr_diff_weight * sr_diff_loss(
            cfg,
            physics,
            standard_height=standard_height,
            reference_height=reference_height,
            diff_sr_target=diff_sr_target,
        )

    if cfg.curl_weight > 0:
        terms["curl"] = cfg.curl_weight * curl_loss(defect, physics)
    if cfg.sparsity_weight > 0:
        terms["sparsity"] = cfg.sparsity_weight * sparsity_loss(defect)
    if cfg.edge_suppress_weight > 0:
        terms["edge_suppress"] = cfg.edge_suppress_weight * edge_suppress_loss(defect, cfg.defect_roi_radius)
    if getattr(cfg, "edge_band_suppress_weight", 0.0) > 0:
        ap = float(getattr(physics.cfg, "lens_radius_fraction", 1.0) or 1.0)
        edge_start = float(getattr(cfg, "eval_edge_band_start_frac", 0.9))
        terms["edge_band_suppress"] = float(getattr(cfg, "edge_band_suppress_weight")) * edge_band_suppress_loss(
            defect,
            aperture_radius=ap,
            edge_band_start_frac=edge_start,
        )
    return terms


__all__ = [
    "curl_loss",
    "data_weight_map",
    "diff_loss",
    "edge_band_suppress_loss",
    "edge_suppress_loss",
    "sparsity_loss",
    "sr_diff_loss",
    "total_loss",
]
