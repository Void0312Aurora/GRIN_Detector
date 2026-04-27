from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.simulation.transforms.base import BaseTransform, TransformContext
from mini_grin_rebuild.simulation.transforms.utils import sample_range, sanitize_channels


class CameraTransform(BaseTransform):
    name = "camera"

    def sample_bundle_params(self, context: TransformContext) -> dict[str, Any]:
        bit_depth = int(sample_range(context.rng, self.cfg.get("bit_depth"), 0))
        return {
            "shot_noise": bool(self.cfg.get("shot_noise", False)),
            "photon_gain": sample_range(context.rng, self.cfg.get("photon_gain"), 100.0),
            "read_noise_std": sample_range(context.rng, self.cfg.get("read_noise_std"), 0.0),
            "saturation_level": sample_range(context.rng, self.cfg.get("saturation_level"), 1e9),
            "bit_depth": bit_depth,
            "bad_pixel_fraction": sample_range(context.rng, self.cfg.get("bad_pixel_fraction"), 0.0),
            "hot_pixel_value": sample_range(context.rng, self.cfg.get("hot_pixel_value"), 1e9),
        }

    def apply(
        self,
        channels: Mapping[str, np.ndarray],
        *,
        context: TransformContext,
        params: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        out = sanitize_channels(channels)
        shot_noise = bool(params.get("shot_noise", False))
        photon_gain = max(float(params.get("photon_gain", 100.0)), 1e-6)
        read_noise_std = max(float(params.get("read_noise_std", 0.0)), 0.0)
        saturation_level = max(float(params.get("saturation_level", 1e9)), 1e-6)
        bit_depth = int(params.get("bit_depth", 0) or 0)
        bad_pixel_fraction = max(float(params.get("bad_pixel_fraction", 0.0)), 0.0)
        hot_pixel_value = float(params.get("hot_pixel_value", saturation_level))

        for name in list(out.keys()):
            image = np.clip(out[name], a_min=0.0, a_max=None).astype(np.float32)
            if shot_noise:
                lam = np.clip(image * photon_gain, 0.0, 1e7)
                image = (context.rng.poisson(lam).astype(np.float32) / photon_gain).astype(np.float32)
            if read_noise_std > 0.0:
                image = image + context.rng.normal(0.0, read_noise_std, image.shape).astype(np.float32)
            image = np.clip(image, 0.0, saturation_level).astype(np.float32)
            if bit_depth > 0:
                levels = float((2**bit_depth) - 1)
                step = saturation_level / max(levels, 1.0)
                image = (np.round(image / step) * step).astype(np.float32)
            if bad_pixel_fraction > 0.0:
                mask = context.rng.random(image.shape) < bad_pixel_fraction
                if np.any(mask):
                    dead = context.rng.random(image.shape) < 0.5
                    image = image.copy()
                    image[mask & dead] = 0.0
                    image[mask & ~dead] = hot_pixel_value
            out[name] = np.clip(image, 0.0, saturation_level).astype(np.float32)
        return out, {"params": dict(params)}


__all__ = ["CameraTransform"]
