from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.simulation.transforms.base import BaseTransform, TransformContext
from mini_grin_rebuild.simulation.transforms.utils import gaussian_blur, normalized_grid, sample_pair, sample_range, sanitize_channels


class OpticsTransform(BaseTransform):
    name = "optics"

    def sample_bundle_params(self, context: TransformContext) -> dict[str, Any]:
        sigma_x, sigma_y = sample_pair(context.rng, self.cfg.get("psf_sigma_px"), 0.0)
        return {
            "psf_sigma_px": {"I_x": sigma_x, "I_y": sigma_y},
            "edge_falloff_width_px": sample_range(context.rng, self.cfg.get("edge_falloff_width_px"), 0.0),
            "outside_leakage": sample_range(context.rng, self.cfg.get("outside_leakage"), 0.0),
        }

    def _edge_mask(self, context: TransformContext, width_px: float, outside_leakage: float) -> np.ndarray:
        aperture = float(getattr(context.cfg, "lens_radius_fraction", 1.0) or 1.0)
        if aperture <= 0.0:
            aperture = 1.0
        _, _, rr = normalized_grid(context.shape)
        if width_px <= 0.0:
            return np.where(rr <= aperture, 1.0, float(outside_leakage)).astype(np.float32)
        pixel_norm = 2.0 / max(max(context.shape) - 1, 1)
        width_norm = max(float(width_px) * pixel_norm, 1e-6)
        start = max(0.0, aperture - width_norm)
        taper = np.ones_like(rr, dtype=np.float32)
        band = (rr > start) & (rr <= aperture)
        t = np.clip((rr - start) / max(aperture - start, 1e-6), 0.0, 1.0)
        taper[band] = (0.5 * (1.0 + np.cos(np.pi * t[band]))).astype(np.float32)
        taper[rr > aperture] = float(outside_leakage)
        return taper.astype(np.float32)

    def apply(
        self,
        channels: Mapping[str, np.ndarray],
        *,
        context: TransformContext,
        params: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        out = sanitize_channels(channels)
        sigmas = params.get("psf_sigma_px", {})
        for name in ("I_x", "I_y"):
            if name in out:
                sigma = float(sigmas.get(name, 0.0)) if isinstance(sigmas, Mapping) else 0.0
                out[name] = gaussian_blur(out[name], sigma)
        edge_width = float(params.get("edge_falloff_width_px", 0.0))
        leakage = float(params.get("outside_leakage", 0.0))
        if edge_width > 0.0 or leakage > 0.0:
            mask = self._edge_mask(context, edge_width, leakage)
            for name in out:
                out[name] = (out[name] * mask).astype(np.float32)
        return out, {"params": dict(params)}


__all__ = ["OpticsTransform"]
