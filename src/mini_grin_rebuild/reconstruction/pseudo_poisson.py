from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer
from mini_grin_rebuild.physics.phase import phase_scale


def _as_bhw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 3:
        return t
    if t.ndim == 4 and t.shape[1] == 1:
        return t.squeeze(1)
    raise ValueError(f"Expected [H,W], [B,H,W] or [B,1,H,W], got {tuple(t.shape)}")


def _phase_scale(cfg) -> float:
    return phase_scale(cfg)


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


def _wrap_to_pi(phase: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi


def _unwrap_phase_least_squares(
    *,
    wrapped_phase: torch.Tensor,
    dx: float,
    pad: int = 0,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """
    Least-squares phase unwrapping from a wrapped phase map.

    This uses wrapped forward differences as the local phase increments and
    reconstructs the globally consistent phase with the same Poisson solver
    already used for gradient integration elsewhere in this module.
    """
    phi = _as_bhw(wrapped_phase)
    b, h, w = phi.shape
    grad_x = torch.zeros_like(phi)
    grad_y = torch.zeros_like(phi)

    if w > 1:
        dx_wrap = _wrap_to_pi(phi[..., 1:] - phi[..., :-1]) / float(dx)
        grad_x[..., :-1] = dx_wrap
        grad_x[..., -1] = dx_wrap[..., -1]
    if h > 1:
        dy_wrap = _wrap_to_pi(phi[:, 1:, :] - phi[:, :-1, :]) / float(dx)
        grad_y[:, :-1, :] = dy_wrap
        grad_y[:, -1, :] = dy_wrap[:, -1, :]

    return _poisson_integrate_from_gradients(
        grad_x=grad_x,
        grad_y=grad_y,
        dx=float(dx),
        pad=pad,
        pad_mode=pad_mode,
    )


def _edge_increments_from_gradients(
    *,
    grad_x: torch.Tensor,
    grad_y: torch.Tensor,
    dx: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gx = _as_bhw(grad_x)
    gy = _as_bhw(grad_y)
    if gx.shape != gy.shape:
        raise ValueError(f"grad_x/grad_y shape mismatch: {tuple(gx.shape)} vs {tuple(gy.shape)}")
    inc_x = 0.5 * (gx[..., :-1] + gx[..., 1:]) * float(dx)   # [B,H,W-1]
    inc_y = 0.5 * (gy[:, :-1, :] + gy[:, 1:, :]) * float(dx)  # [B,H-1,W]
    return inc_x, inc_y


def _lift_wrapped_increment_to_reference(
    *,
    inc_wrapped: torch.Tensor,
    reference_inc: torch.Tensor,
) -> torch.Tensor:
    if inc_wrapped.shape != reference_inc.shape:
        raise ValueError(f"inc_wrapped/reference_inc shape mismatch: {tuple(inc_wrapped.shape)} vs {tuple(reference_inc.shape)}")
    two_pi = inc_wrapped.new_tensor(2.0 * math.pi)
    k = torch.round((reference_inc.to(device=inc_wrapped.device, dtype=inc_wrapped.dtype) - inc_wrapped) / two_pi)
    return inc_wrapped + two_pi * k


def _residue_cut_weights(
    *,
    inc_x_wrapped: torch.Tensor,
    inc_y_wrapped: torch.Tensor,
    cut_weight: float = 0.02,
    cut_halo: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Soft branch-cut weights derived from wrapped loop residues of the *total test*
    phase increments.

    Edges adjacent to residue cells are strongly downweighted so the LS solver can
    avoid forcing inconsistent wraps through those locations. A small halo dilates
    the cut region to reduce residue leakage.
    """
    if inc_x_wrapped.ndim != 3 or inc_y_wrapped.ndim != 3:
        raise ValueError(
            f"Expected inc_x_wrapped/inc_y_wrapped [B,H,W-1]/[B,H-1,W], got {tuple(inc_x_wrapped.shape)} and {tuple(inc_y_wrapped.shape)}"
        )
    loop = (
        inc_x_wrapped[:, :-1, :]
        + inc_y_wrapped[:, :, 1:]
        - inc_x_wrapped[:, 1:, :]
        - inc_y_wrapped[:, :, :-1]
    )
    residue = torch.abs(torch.round(loop / (2.0 * math.pi))).to(dtype=inc_x_wrapped.dtype)  # [B,H-1,W-1]
    halo = max(0, int(cut_halo))
    if halo > 0 and residue.numel() > 0:
        residue = F.max_pool2d(
            residue.unsqueeze(1),
            kernel_size=2 * halo + 1,
            stride=1,
            padding=halo,
        ).squeeze(1)

    rx = torch.zeros_like(inc_x_wrapped)
    ry = torch.zeros_like(inc_y_wrapped)
    if residue.numel() > 0:
        rx[:, :-1, :] += residue
        rx[:, 1:, :] += residue
        ry[:, :, :-1] += residue
        ry[:, :, 1:] += residue

    low = max(1e-4, min(1.0, float(cut_weight)))
    wx = torch.where(rx > 0, torch.full_like(rx, low), torch.ones_like(rx))
    wy = torch.where(ry > 0, torch.full_like(ry, low), torch.ones_like(ry))
    return wx, wy


def _weighted_phase_from_wrapped_increments(
    *,
    inc_x: torch.Tensor,
    inc_y: torch.Tensor,
    weight_x: torch.Tensor | None = None,
    weight_y: torch.Tensor | None = None,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Solve
      min_phi Σ wx (phi[i,j+1]-phi[i,j]-inc_x)^2 + Σ wy (phi[i+1,j]-phi[i,j]-inc_y)^2
    with conjugate gradient on the normal equations, constrained to zero-mean phi.
    """
    ix = inc_x
    iy = inc_y
    if ix.ndim != 3 or iy.ndim != 3:
        raise ValueError(f"Expected inc_x/inc_y [B,H,W-1]/[B,H-1,W], got {tuple(ix.shape)} and {tuple(iy.shape)}")
    if ix.shape[0] != iy.shape[0] or ix.shape[1] != iy.shape[1] + 1 or ix.shape[2] + 1 != iy.shape[2]:
        raise ValueError(f"Incompatible increment shapes: {tuple(ix.shape)} and {tuple(iy.shape)}")

    b, h, wm1 = ix.shape
    w = wm1 + 1
    if weight_x is None:
        wx = torch.ones_like(ix)
    else:
        wx = ix.new_tensor(weight_x) if not torch.is_tensor(weight_x) else weight_x.to(device=ix.device, dtype=ix.dtype)
    if weight_y is None:
        wy = torch.ones_like(iy)
    else:
        wy = iy.new_tensor(weight_y) if not torch.is_tensor(weight_y) else weight_y.to(device=iy.device, dtype=iy.dtype)

    def _project_zero_mean(phi: torch.Tensor) -> torch.Tensor:
        return phi - torch.mean(phi, dim=(-2, -1), keepdim=True)

    def _apply_operator(phi: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(phi)
        dx_phi = phi[..., 1:] - phi[..., :-1]
        flux_x = wx * dx_phi
        out[..., :-1] -= flux_x
        out[..., 1:] += flux_x

        dy_phi = phi[:, 1:, :] - phi[:, :-1, :]
        flux_y = wy * dy_phi
        out[:, :-1, :] -= flux_y
        out[:, 1:, :] += flux_y
        return _project_zero_mean(out)

    rhs = torch.zeros((b, h, w), device=ix.device, dtype=ix.dtype)
    flux_x_rhs = wx * ix
    rhs[..., :-1] -= flux_x_rhs
    rhs[..., 1:] += flux_x_rhs
    flux_y_rhs = wy * iy
    rhs[:, :-1, :] -= flux_y_rhs
    rhs[:, 1:, :] += flux_y_rhs
    rhs = _project_zero_mean(rhs)

    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rr_old = torch.sum(r * r, dim=(-2, -1), keepdim=True)
    denom_floor = torch.tensor(1e-12, device=ix.device, dtype=ix.dtype)
    norm = float(h * w) ** 0.5

    for _ in range(max(1, int(max_iter))):
        ap = _apply_operator(p)
        p_ap = torch.sum(p * ap, dim=(-2, -1), keepdim=True)
        alpha = rr_old / torch.clamp(p_ap, min=denom_floor)
        x = _project_zero_mean(x + alpha * p)
        r = _project_zero_mean(r - alpha * ap)
        rr_new = torch.sum(r * r, dim=(-2, -1), keepdim=True)
        if float(torch.sqrt(torch.max(rr_new)).detach().cpu().item() / max(norm, 1e-12)) < float(tol):
            break
        beta = rr_new / torch.clamp(rr_old, min=denom_floor)
        p = _project_zero_mean(r + beta * p)
        rr_old = rr_new

    return _project_zero_mean(x)


def _wrapped_increment_quality_weights(
    *,
    inc_x_cont: torch.Tensor,
    inc_y_cont: torch.Tensor,
    inc_x_wrapped: torch.Tensor,
    inc_y_wrapped: torch.Tensor,
    safe_frac: float = 0.8,
    mag_weight: float = 4.0,
    residue_weight: float = 3.0,
    min_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Edge-quality weights for wrapped-increment unwrapping.

    Downweight:
    - edges whose continuous candidate increment magnitude approaches/exceeds pi
    - edges adjacent to non-zero wrapped loop residues
    """
    safe = max(0.0, min(float(safe_frac), 0.999))
    min_w = max(1e-4, float(min_weight))
    abs_x = torch.abs(inc_x_cont) / math.pi
    abs_y = torch.abs(inc_y_cont) / math.pi
    risk_x = torch.clamp((abs_x - safe) / max(1.0 - safe, 1e-6), min=0.0)
    risk_y = torch.clamp((abs_y - safe) / max(1.0 - safe, 1e-6), min=0.0)
    wx = torch.exp(-float(mag_weight) * risk_x**2)
    wy = torch.exp(-float(mag_weight) * risk_y**2)

    loop = (
        inc_x_wrapped[:, :-1, :]
        + inc_y_wrapped[:, :, 1:]
        - inc_x_wrapped[:, 1:, :]
        - inc_y_wrapped[:, :, :-1]
    )
    residue = torch.abs(torch.round(loop / (2.0 * math.pi)))  # [B,H-1,W-1]
    if torch.any(residue > 0):
        rx = torch.zeros_like(inc_x_wrapped)
        ry = torch.zeros_like(inc_y_wrapped)
        rx[:, :-1, :] += residue
        rx[:, 1:, :] += residue
        ry[:, :, :-1] += residue
        ry[:, :, 1:] += residue
        wx = wx / (1.0 + float(residue_weight) * rx)
        wy = wy / (1.0 + float(residue_weight) * ry)

    return torch.clamp(wx, min=min_w, max=1.0), torch.clamp(wy, min=min_w, max=1.0)


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


def reconstruct_defect_first_order_poisson(
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
    grad_floor_ratio: float = 0.05,
    grad_floor_abs: float = 1e-6,
) -> torch.Tensor:
    """
    Reconstruct defect height using the first-order small-perturbation approximation:

        ΔI_x = 2 g_x d_x + d_x^2  ->  d_x ≈ ΔI_x / (2 g_x)
        ΔI_y = 2 g_y d_y + d_y^2  ->  d_y ≈ ΔI_y / (2 g_y)

    where g is the standard phase gradient and d is the defect phase gradient.
    The recovered defect gradients are then integrated with the same Poisson solver
    used by the pseudo-poisson baseline.
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

    std_phase = physics._phase(std).squeeze(1)
    g_y, g_x = physics._gradient(std_phase)

    gx_abs = torch.abs(g_x)
    gy_abs = torch.abs(g_y)
    gx_floor = torch.clamp(gx_abs.amax(dim=(-2, -1), keepdim=True) * float(grad_floor_ratio), min=float(grad_floor_abs))
    gy_floor = torch.clamp(gy_abs.amax(dim=(-2, -1), keepdim=True) * float(grad_floor_ratio), min=float(grad_floor_abs))

    denom_x = 2.0 * torch.where(gx_abs >= gx_floor, g_x, torch.sign(g_x) * gx_floor)
    denom_y = 2.0 * torch.where(gy_abs >= gy_floor, g_y, torch.sign(g_y) * gy_floor)
    denom_x = torch.where(denom_x == 0, 2.0 * gx_floor, denom_x)
    denom_y = torch.where(denom_y == 0, 2.0 * gy_floor, denom_y)

    d_x = diff_ix.to(g_x.dtype) / denom_x
    d_y = diff_iy.to(g_y.dtype) / denom_y

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


def reconstruct_defect_first_order_sign_quadratic_poisson(
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
    grad_floor_ratio: float = 0.05,
    grad_floor_abs: float = 1e-6,
) -> torch.Tensor:
    """
    Hybrid coarse reconstruction:

    1. Use the first-order approximation only to infer the sign of the test phase gradient.
    2. Use the quadratic magnitude relation to recover |∂phi_test|.
    3. Subtract the standard gradient to get defect phase gradients, then Poisson integrate.
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

    std_phase = physics._phase(std).squeeze(1)
    g_y, g_x = physics._gradient(std_phase)

    gx_abs = torch.abs(g_x)
    gy_abs = torch.abs(g_y)
    gx_floor = torch.clamp(gx_abs.amax(dim=(-2, -1), keepdim=True) * float(grad_floor_ratio), min=float(grad_floor_abs))
    gy_floor = torch.clamp(gy_abs.amax(dim=(-2, -1), keepdim=True) * float(grad_floor_ratio), min=float(grad_floor_abs))

    denom_x = 2.0 * torch.where(gx_abs >= gx_floor, g_x, torch.sign(g_x) * gx_floor)
    denom_y = 2.0 * torch.where(gy_abs >= gy_floor, g_y, torch.sign(g_y) * gy_floor)
    denom_x = torch.where(denom_x == 0, 2.0 * gx_floor, denom_x)
    denom_y = torch.where(denom_y == 0, 2.0 * gy_floor, denom_y)

    d_x_first = diff_ix.to(g_x.dtype) / denom_x
    d_y_first = diff_iy.to(g_y.dtype) / denom_y
    test_sign_x = torch.sign(g_x + d_x_first)
    test_sign_y = torch.sign(g_y + d_y_first)
    test_sign_x = torch.where(test_sign_x == 0, torch.where(torch.sign(g_x) == 0, torch.ones_like(g_x), torch.sign(g_x)), test_sign_x)
    test_sign_y = torch.where(test_sign_y == 0, torch.where(torch.sign(g_y) == 0, torch.ones_like(g_y), torch.sign(g_y)), test_sign_y)

    inside_x = torch.clamp(g_x**2 + diff_ix.to(g_x.dtype), min=0.0)
    inside_y = torch.clamp(g_y**2 + diff_iy.to(g_y.dtype), min=0.0)
    a_x = test_sign_x * torch.sqrt(inside_x)
    a_y = test_sign_y * torch.sqrt(inside_y)

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


def reconstruct_defect_unwrap_poisson(
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
    Coarse reconstruction with a wrapped-phase prior followed by least-squares unwrapping.

    Stage 1:
    - reuse the quadratic pseudo-poisson inverse to obtain a provisional defect phase
      map that may contain wrap slips under branch stress.

    Stage 2:
    - wrap that provisional phase into [-pi, pi] and recover a globally consistent
      phase with a least-squares unwrap based on wrapped local increments.

    The residual-learning stack can then consume this unwrapped coarse solution
    exactly like the previous pseudo-poisson prior.
    """
    coarse = reconstruct_defect_pseudo_poisson(
        physics=physics,
        standard_height=standard_height,
        diff_ts=diff_ts,
        defect_roi_radius=defect_roi_radius,
        apply_edge_offset=False,
        poisson_pad=poisson_pad,
        pad_mode=pad_mode,
        apply_edge_taper=apply_edge_taper,
        taper_margin=taper_margin,
    ).squeeze(1)

    delta_phi_wrapped = _wrap_to_pi(coarse * _phase_scale(physics.cfg))
    delta_phi = _unwrap_phase_least_squares(
        wrapped_phase=delta_phi_wrapped,
        dx=float(physics.cfg.dx),
        pad=poisson_pad,
        pad_mode=pad_mode,
    )
    defect = delta_phi / _phase_scale(physics.cfg)
    if apply_edge_offset:
        defect = _edge_mean_offset(defect, defect_roi_radius=defect_roi_radius)
    return defect.unsqueeze(1)


def reconstruct_defect_wrapped_increment_poisson(
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
    cg_max_iter: int = 200,
    cg_tol: float = 1e-5,
) -> torch.Tensor:
    """
    Scheme A:
    Build wrapped local defect phase increments directly from the quadratic local
    inverse, then solve a discrete least-squares unwrap problem on those increments.
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

    std_phase = physics._phase(std).squeeze(1)
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

    inc_x_cont, inc_y_cont = _edge_increments_from_gradients(
        grad_x=d_x,
        grad_y=d_y,
        dx=float(physics.cfg.dx),
    )
    inc_x_wrapped = _wrap_to_pi(inc_x_cont)
    inc_y_wrapped = _wrap_to_pi(inc_y_cont)
    delta_phi = _weighted_phase_from_wrapped_increments(
        inc_x=inc_x_wrapped,
        inc_y=inc_y_wrapped,
        max_iter=int(cg_max_iter),
        tol=float(cg_tol),
    )
    defect = delta_phi / _phase_scale(physics.cfg)
    if apply_edge_offset:
        defect = _edge_mean_offset(defect, defect_roi_radius=defect_roi_radius)
    return defect.unsqueeze(1)


def reconstruct_defect_quality_guided_poisson(
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
    cg_max_iter: int = 200,
    cg_tol: float = 1e-5,
    quality_safe_frac: float = 0.8,
    quality_mag_weight: float = 4.0,
    quality_residue_weight: float = 3.0,
    quality_min_weight: float = 0.05,
) -> torch.Tensor:
    """
    Scheme B:
    Same wrapped-increment LS formulation as Scheme A, but with edge-quality weights
    derived from wrap-risk magnitude and local wrapped-loop residues.
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

    std_phase = physics._phase(std).squeeze(1)
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

    inc_x_cont, inc_y_cont = _edge_increments_from_gradients(
        grad_x=d_x,
        grad_y=d_y,
        dx=float(physics.cfg.dx),
    )
    inc_x_wrapped = _wrap_to_pi(inc_x_cont)
    inc_y_wrapped = _wrap_to_pi(inc_y_cont)
    weight_x, weight_y = _wrapped_increment_quality_weights(
        inc_x_cont=inc_x_cont,
        inc_y_cont=inc_y_cont,
        inc_x_wrapped=inc_x_wrapped,
        inc_y_wrapped=inc_y_wrapped,
        safe_frac=float(quality_safe_frac),
        mag_weight=float(quality_mag_weight),
        residue_weight=float(quality_residue_weight),
        min_weight=float(quality_min_weight),
    )
    delta_phi = _weighted_phase_from_wrapped_increments(
        inc_x=inc_x_wrapped,
        inc_y=inc_y_wrapped,
        weight_x=weight_x,
        weight_y=weight_y,
        max_iter=int(cg_max_iter),
        tol=float(cg_tol),
    )
    defect = delta_phi / _phase_scale(physics.cfg)
    if apply_edge_offset:
        defect = _edge_mean_offset(defect, defect_roi_radius=defect_roi_radius)
    return defect.unsqueeze(1)


def reconstruct_defect_standard_lifted_poisson(
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
    cg_max_iter: int = 200,
    cg_tol: float = 1e-5,
) -> torch.Tensor:
    """
    Scheme C:
    Recover the *test* phase local increments, wrap them to [-pi, pi], then lift
    each edge to the nearest 2*pi branch centered at the known standard-phase edge
    increment. The defect increment is the lifted test increment minus the standard
    increment, followed by least-squares integration.

    This uses the experimentally verified fact that the *defect* increment remains
    small even when the total standard/test phase is near-wrap over most edges.
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

    std_phase = physics._phase(std).squeeze(1)
    g_y, g_x = physics._gradient(std_phase)

    inside_x = torch.clamp(g_x**2 + diff_ix.to(g_x.dtype), min=0.0)
    inside_y = torch.clamp(g_y**2 + diff_iy.to(g_y.dtype), min=0.0)
    sign_x = torch.sign(g_x)
    sign_y = torch.sign(g_y)
    sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)

    test_gx = sign_x * torch.sqrt(inside_x)
    test_gy = sign_y * torch.sqrt(inside_y)

    if apply_edge_taper and taper_margin > 0:
        d_x = test_gx - g_x
        d_y = test_gy - g_y
        start = min(1.0, float(defect_roi_radius) + float(taper_margin))
        taper = _radial_cosine_taper(
            d_x.shape[-2],
            d_x.shape[-1],
            start_radius=start,
            end_radius=1.0,
            device=d_x.device,
            dtype=d_x.dtype,
        )
        test_gx = g_x + d_x * taper
        test_gy = g_y + d_y * taper

    inc_std_x, inc_std_y = _edge_increments_from_gradients(
        grad_x=g_x,
        grad_y=g_y,
        dx=float(physics.cfg.dx),
    )
    inc_test_x_cont, inc_test_y_cont = _edge_increments_from_gradients(
        grad_x=test_gx,
        grad_y=test_gy,
        dx=float(physics.cfg.dx),
    )

    inc_test_x_lifted = _lift_wrapped_increment_to_reference(
        inc_wrapped=_wrap_to_pi(inc_test_x_cont),
        reference_inc=inc_std_x,
    )
    inc_test_y_lifted = _lift_wrapped_increment_to_reference(
        inc_wrapped=_wrap_to_pi(inc_test_y_cont),
        reference_inc=inc_std_y,
    )

    inc_def_x = inc_test_x_lifted - inc_std_x
    inc_def_y = inc_test_y_lifted - inc_std_y

    delta_phi = _weighted_phase_from_wrapped_increments(
        inc_x=inc_def_x,
        inc_y=inc_def_y,
        max_iter=int(cg_max_iter),
        tol=float(cg_tol),
    )
    defect = delta_phi / _phase_scale(physics.cfg)
    if apply_edge_offset:
        defect = _edge_mean_offset(defect, defect_roi_radius=defect_roi_radius)
    return defect.unsqueeze(1)


def reconstruct_defect_residue_cut_poisson(
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
    cg_max_iter: int = 200,
    cg_tol: float = 1e-5,
    cut_weight: float = 0.02,
    cut_halo: int = 1,
) -> torch.Tensor:
    """
    Scheme D:
    1. Recover local *test* phase increments from the quadratic inverse.
    2. Wrap them and compute plaquette residues on the total test phase field.
    3. Lift each test edge to the nearest branch around the known standard edge.
    4. Solve the defect phase with residue-derived soft branch cuts.

    Compared with scheme C, the new ingredient is that the inconsistent wrap
    locations are estimated from the total test field and then explicitly
    suppressed in the integration step.
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

    std_phase = physics._phase(std).squeeze(1)
    g_y, g_x = physics._gradient(std_phase)

    inside_x = torch.clamp(g_x**2 + diff_ix.to(g_x.dtype), min=0.0)
    inside_y = torch.clamp(g_y**2 + diff_iy.to(g_y.dtype), min=0.0)
    sign_x = torch.sign(g_x)
    sign_y = torch.sign(g_y)
    sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)

    test_gx = sign_x * torch.sqrt(inside_x)
    test_gy = sign_y * torch.sqrt(inside_y)

    if apply_edge_taper and taper_margin > 0:
        d_x = test_gx - g_x
        d_y = test_gy - g_y
        start = min(1.0, float(defect_roi_radius) + float(taper_margin))
        taper = _radial_cosine_taper(
            d_x.shape[-2],
            d_x.shape[-1],
            start_radius=start,
            end_radius=1.0,
            device=d_x.device,
            dtype=d_x.dtype,
        )
        test_gx = g_x + d_x * taper
        test_gy = g_y + d_y * taper

    inc_std_x, inc_std_y = _edge_increments_from_gradients(
        grad_x=g_x,
        grad_y=g_y,
        dx=float(physics.cfg.dx),
    )
    inc_test_x_cont, inc_test_y_cont = _edge_increments_from_gradients(
        grad_x=test_gx,
        grad_y=test_gy,
        dx=float(physics.cfg.dx),
    )
    inc_test_x_wrapped = _wrap_to_pi(inc_test_x_cont)
    inc_test_y_wrapped = _wrap_to_pi(inc_test_y_cont)
    weight_x, weight_y = _residue_cut_weights(
        inc_x_wrapped=inc_test_x_wrapped,
        inc_y_wrapped=inc_test_y_wrapped,
        cut_weight=float(cut_weight),
        cut_halo=int(cut_halo),
    )

    inc_test_x_lifted = _lift_wrapped_increment_to_reference(
        inc_wrapped=inc_test_x_wrapped,
        reference_inc=inc_std_x,
    )
    inc_test_y_lifted = _lift_wrapped_increment_to_reference(
        inc_wrapped=inc_test_y_wrapped,
        reference_inc=inc_std_y,
    )
    inc_def_x = inc_test_x_lifted - inc_std_x
    inc_def_y = inc_test_y_lifted - inc_std_y

    delta_phi = _weighted_phase_from_wrapped_increments(
        inc_x=inc_def_x,
        inc_y=inc_def_y,
        weight_x=weight_x,
        weight_y=weight_y,
        max_iter=int(cg_max_iter),
        tol=float(cg_tol),
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


__all__ = [
    "reconstruct_defect_first_order_poisson",
    "reconstruct_defect_first_order_sign_quadratic_poisson",
    "reconstruct_defect_oracle_poisson",
    "reconstruct_defect_pseudo_poisson",
    "reconstruct_defect_quality_guided_poisson",
    "reconstruct_defect_residue_cut_poisson",
    "reconstruct_defect_standard_lifted_poisson",
    "reconstruct_defect_unwrap_poisson",
    "reconstruct_defect_wrapped_increment_poisson",
    "_unwrap_phase_least_squares",
]
