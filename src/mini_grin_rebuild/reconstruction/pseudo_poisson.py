from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer


def _as_bhw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 3:
        return t
    if t.ndim == 4 and t.shape[1] == 1:
        return t.squeeze(1)
    raise ValueError(f"Expected [H,W], [B,H,W] or [B,1,H,W], got {tuple(t.shape)}")


def _phase_scale(cfg) -> float:
    return float((2.0 * math.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air))


def _poisson_integrate_from_gradients(
    *,
    grad_x: torch.Tensor,
    grad_y: torch.Tensor,
    dx: float,
    pad: int = 0,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """
    Least-squares Poisson integration:
    argmin_phi ||∂x phi - grad_x||^2 + ||∂y phi - grad_y||^2.

    Returns phi with the DC component set to zero (constant offset is not observable).
    """
    gx = _as_bhw(grad_x)
    gy = _as_bhw(grad_y)
    if gx.shape != gy.shape:
        raise ValueError(f"grad_x/grad_y shape mismatch: {tuple(gx.shape)} vs {tuple(gy.shape)}")

    b, h, w = gx.shape
    orig_h, orig_w = h, w
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}")
    pad = int(pad)
    if pad > 0:
        max_pad = max(0, min(h, w) - 1)
        if pad > max_pad:
            pad = max_pad
        gx = F.pad(gx, (pad, pad, pad, pad), mode=pad_mode)
        gy = F.pad(gy, (pad, pad, pad, pad), mode=pad_mode)
        _, h, w = gx.shape
    freq_y = torch.fft.fftfreq(h, d=dx, device=gx.device, dtype=torch.float32).view(1, h, 1)
    freq_x = torch.fft.fftfreq(w, d=dx, device=gx.device, dtype=torch.float32).view(1, 1, w)

    gx_fft = torch.fft.fft2(gx)
    gy_fft = torch.fft.fft2(gy)
    div_fft = (2j * math.pi * freq_x) * gx_fft + (2j * math.pi * freq_y) * gy_fft
    denom = (2.0 * math.pi) ** 2 * (freq_x**2 + freq_y**2)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    phi_fft = -div_fft / denom
    phi_fft[:, 0, 0] = 0
    phi = torch.fft.ifft2(phi_fft).real
    if pad > 0:
        phi = phi[:, pad : pad + orig_h, pad : pad + orig_w]
    return phi


def _edge_mean_offset(field: torch.Tensor, *, defect_roi_radius: float) -> torch.Tensor:
    f = _as_bhw(field)
    b, h, w = f.shape
    lin_y = torch.linspace(-1.0, 1.0, h, device=f.device, dtype=f.dtype)
    lin_x = torch.linspace(-1.0, 1.0, w, device=f.device, dtype=f.dtype)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)
    edge_mask = (rr > defect_roi_radius).to(f.dtype)  # [H,W]
    denom = torch.sum(edge_mask) + 1e-12
    offset = torch.sum(f * edge_mask, dim=(-2, -1)) / denom  # [B]
    return f - offset.view(b, 1, 1)


def _radial_cosine_taper(
    h: int,
    w: int,
    *,
    start_radius: float,
    end_radius: float = 1.0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not (0.0 <= start_radius <= end_radius):
        raise ValueError(f"Invalid taper radii: start={start_radius}, end={end_radius}")
    if start_radius == end_radius:
        return torch.ones((h, w), device=device, dtype=dtype)

    lin_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    lin_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)

    t = (rr - start_radius) / max(end_radius - start_radius, 1e-12)
    t = torch.clamp(t, 0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * t))


def reconstruct_defect_pseudo_poisson(
    *,
    physics: DifferentiableGradientLayer,
    standard_height: torch.Tensor,
    diff_ts: Dict[str, torch.Tensor],
    defect_roi_radius: float = 0.6,
    apply_edge_offset: bool = True,
    poisson_pad: int = 0,
    pad_mode: str = "reflect",
    apply_edge_taper: bool = False,
    taper_margin: float = 0.25,
) -> torch.Tensor:
    """
    Reconstruct defect height using the closed-form quadratic relation per axis and
    Poisson integration for integrability.

    Assumptions (synthetic / small-defect regime):
    - sign(∂φ_test) == sign(∂φ_standard) almost everywhere, so we can choose the quadratic branch.
    - constant offset is fixed by forcing the mean outside the ROI to be ~0 (optional).

    Returns defect height [B,1,H,W] (same units as `standard_height`).
    """
    std = standard_height
    if std.ndim == 2:
        std = std.unsqueeze(0).unsqueeze(0)
    elif std.ndim == 3:
        std = std.unsqueeze(1)
    if std.ndim != 4 or std.shape[1] != 1:
        raise ValueError(f"Expected standard_height [B,1,H,W], got {tuple(std.shape)}")

    diff_ix = _as_bhw(diff_ts["I_x"])
    diff_iy = _as_bhw(diff_ts["I_y"])
    if diff_ix.shape != diff_iy.shape:
        raise ValueError(f"diff_ts I_x/I_y mismatch: {tuple(diff_ix.shape)} vs {tuple(diff_iy.shape)}")

    std_phase = physics._phase(std).squeeze(1)  # [B,H,W]
    g_y, g_x = physics._gradient(std_phase)

    inside_x = torch.clamp(g_x**2 + diff_ix.to(g_x.dtype), min=0.0)
    inside_y = torch.clamp(g_y**2 + diff_iy.to(g_y.dtype), min=0.0)

    sign_x = torch.sign(g_x)
    sign_y = torch.sign(g_y)
    sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)

    a_x = sign_x * torch.sqrt(inside_x)
    a_y = sign_y * torch.sqrt(inside_y)

    d_x = a_x - g_x
    d_y = a_y - g_y

    if apply_edge_taper and taper_margin > 0:
        start = min(1.0, float(defect_roi_radius) + float(taper_margin))
        taper = _radial_cosine_taper(
            d_x.shape[-2],
            d_x.shape[-1],
            start_radius=start,
            end_radius=1.0,
            device=d_x.device,
            dtype=d_x.dtype,
        )
        d_x = d_x * taper
        d_y = d_y * taper

    delta_phi = _poisson_integrate_from_gradients(
        grad_x=d_x,
        grad_y=d_y,
        dx=float(physics.cfg.dx),
        pad=poisson_pad,
        pad_mode=pad_mode,
    )
    defect = delta_phi / _phase_scale(physics.cfg)
    if apply_edge_offset:
        defect = _edge_mean_offset(defect, defect_roi_radius=defect_roi_radius)
    return defect.unsqueeze(1)


def reconstruct_defect_oracle_poisson(
    *,
    physics: DifferentiableGradientLayer,
    standard_height: torch.Tensor,
    defect_true: torch.Tensor,
    diff_ts: Dict[str, torch.Tensor],
    defect_roi_radius: float = 0.6,
    apply_edge_offset: bool = True,
    poisson_pad: int = 0,
    pad_mode: str = "reflect",
    apply_edge_taper: bool = False,
    taper_margin: float = 0.25,
) -> torch.Tensor:
    """
    Oracle baseline for the signless inverse problem.

    This uses the *ground-truth* sign of the test phase gradients to choose the
    quadratic branch per axis, then applies the same Poisson integration as the
    pseudo-Poisson baseline.

    Notes:
    - Not physically deployable, but useful as an upper bound to quantify how much
      error comes from sign ambiguity vs. integration / discretization / noise.
    - Returns defect height [B,1,H,W].
    """
    std = standard_height
    if std.ndim == 2:
        std = std.unsqueeze(0).unsqueeze(0)
    elif std.ndim == 3:
        std = std.unsqueeze(1)
    if std.ndim != 4 or std.shape[1] != 1:
        raise ValueError(f"Expected standard_height [B,1,H,W], got {tuple(std.shape)}")

    gt = defect_true
    if gt.ndim == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
    elif gt.ndim == 3:
        gt = gt.unsqueeze(1)
    if gt.ndim != 4 or gt.shape[1] != 1:
        raise ValueError(f"Expected defect_true [B,1,H,W], got {tuple(gt.shape)}")

    diff_ix = _as_bhw(diff_ts["I_x"])
    diff_iy = _as_bhw(diff_ts["I_y"])
    if diff_ix.shape != diff_iy.shape:
        raise ValueError(f"diff_ts I_x/I_y mismatch: {tuple(diff_ix.shape)} vs {tuple(diff_iy.shape)}")

    std_phase = physics._phase(std).squeeze(1)
    g_y, g_x = physics._gradient(std_phase)

    gt_phase = physics._phase(gt).squeeze(1)
    d_y_true, d_x_true = physics._gradient(gt_phase)

    inside_x = torch.clamp(g_x**2 + diff_ix.to(g_x.dtype), min=0.0)
    inside_y = torch.clamp(g_y**2 + diff_iy.to(g_y.dtype), min=0.0)

    sign_x = torch.sign(g_x + d_x_true)
    sign_y = torch.sign(g_y + d_y_true)
    sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)

    a_x = sign_x * torch.sqrt(inside_x)
    a_y = sign_y * torch.sqrt(inside_y)

    d_x = a_x - g_x
    d_y = a_y - g_y

    if apply_edge_taper and taper_margin > 0:
        start = min(1.0, float(defect_roi_radius) + float(taper_margin))
        taper = _radial_cosine_taper(
            d_x.shape[-2],
            d_x.shape[-1],
            start_radius=start,
            end_radius=1.0,
            device=d_x.device,
            dtype=d_x.dtype,
        )
        d_x = d_x * taper
        d_y = d_y * taper

    delta_phi = _poisson_integrate_from_gradients(
        grad_x=d_x,
        grad_y=d_y,
        dx=float(physics.cfg.dx),
        pad=poisson_pad,
        pad_mode=pad_mode,
    )
    defect = delta_phi / _phase_scale(physics.cfg)
    if apply_edge_offset:
        defect = _edge_mean_offset(defect, defect_roi_radius=defect_roi_radius)
    return defect.unsqueeze(1)


__all__ = ["reconstruct_defect_oracle_poisson", "reconstruct_defect_pseudo_poisson"]
