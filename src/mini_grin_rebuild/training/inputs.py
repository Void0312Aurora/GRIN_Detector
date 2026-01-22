from __future__ import annotations

from typing import Dict, Optional

import torch

from mini_grin_rebuild.core.configs import TrainingConfig
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer


def append_coord_channels(
    inputs: torch.Tensor,
    *,
    aperture_radius: float = 1.0,
) -> torch.Tensor:
    """
    Append (x, y, r_norm) coordinate channels to an input tensor [B,C,H,W].

    Coordinates are defined on the current tensor grid using [-1,1] linspace.
    `r_norm` is radial distance normalized by `aperture_radius` (so r_norm==1 at the aperture boundary).
    """
    if inputs.ndim != 4:
        raise ValueError(f"Expected inputs [B,C,H,W], got {tuple(inputs.shape)}")
    b, _, h, w = inputs.shape
    device = inputs.device
    dtype = inputs.dtype

    lin_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    lin_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)
    ap = float(aperture_radius) if float(aperture_radius) > 0 else 1.0
    rrn = rr / ap

    xx = xx.unsqueeze(0).unsqueeze(0).expand(b, -1, -1, -1)
    yy = yy.unsqueeze(0).unsqueeze(0).expand(b, -1, -1, -1)
    rrn = rrn.unsqueeze(0).unsqueeze(0).expand(b, -1, -1, -1)
    return torch.cat([inputs, xx, yy, rrn], dim=1)


def build_inputs(
    train_cfg: TrainingConfig,
    diff_st: Dict[str, torch.Tensor],
    physics: DifferentiableGradientLayer,
    standard_height: torch.Tensor,
    *,
    diff_sr: Optional[Dict[str, torch.Tensor]] = None,
    intensity_inputs: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> torch.Tensor:
    """
    Assemble input channels following the legacy ordering:

    1) ΔI (test-standard): I_x, I_y
    2) ΔI (standard-reference): I_x, I_y (optional)
    3) raw intensities: (standard, reference, test) × (I_x, I_y) (optional)
    4) phase-gradient channels computed from the standard surface (optional)
    """

    def _normalize(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            return t.unsqueeze(0)
        if t.ndim == 3:
            return t
        raise ValueError(f"Unexpected tensor shape {tuple(t.shape)} for input channel")

    chans: list[torch.Tensor] = []
    chans.append(_normalize(diff_st["I_x"]))
    chans.append(_normalize(diff_st["I_y"]))

    if train_cfg.use_sr_inputs and diff_sr is not None:
        chans.append(_normalize(diff_sr["I_x"]))
        chans.append(_normalize(diff_sr["I_y"]))

    if train_cfg.use_raw_intensity_inputs and intensity_inputs is not None:
        for name in ("standard", "reference", "test"):
            if name in intensity_inputs:
                intensity = intensity_inputs[name]
                if "I_x" in intensity and "I_y" in intensity:
                    chans.append(_normalize(intensity["I_x"]))
                    chans.append(_normalize(intensity["I_y"]))

    if train_cfg.use_phase_inputs and standard_height is not None:
        std = standard_height
        if std.ndim == 2:
            std = std.unsqueeze(0).unsqueeze(0)
        elif std.ndim == 3:
            std = std.unsqueeze(1)
        phase = physics._phase(std).squeeze(1)
        grad_y, grad_x = physics._gradient(phase)
        chans.append(grad_x)
        chans.append(grad_y)

    stacked = torch.stack(chans, dim=1)
    if stacked.dim() == 4 and stacked.shape[1] == 1 and stacked.shape[0] == len(chans):
        stacked = stacked.permute(1, 0, 2, 3)
    return stacked


__all__ = ["append_coord_channels", "build_inputs"]
