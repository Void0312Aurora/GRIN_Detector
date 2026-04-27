from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.simulation.transforms.base import BaseTransform, TransformContext
from mini_grin_rebuild.simulation.transforms.utils import sample_pair, sample_range, sanitize_channels


class ResponseTransform(BaseTransform):
    name = "response"

    def sample_bundle_params(self, context: TransformContext) -> dict[str, Any]:
        gain_x, gain_y = sample_pair(context.rng, self.cfg.get("gain"), 1.0)
        bias_x, bias_y = sample_pair(context.rng, self.cfg.get("bias"), 0.0)
        return {
            "gain": {"I_x": gain_x, "I_y": gain_y},
            "bias": {"I_x": bias_x, "I_y": bias_y},
            "saturation": sample_range(context.rng, self.cfg.get("saturation"), 1e9),
            "cross_talk": sample_range(context.rng, self.cfg.get("cross_talk"), 0.0),
        }

    def apply(
        self,
        channels: Mapping[str, np.ndarray],
        *,
        context: TransformContext,
        params: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        out = sanitize_channels(channels)
        saturation = max(float(params.get("saturation", 1e9)), 1e-6)
        gains = params.get("gain", {})
        biases = params.get("bias", {})
        for name in ("I_x", "I_y"):
            if name in out:
                gain = float(gains.get(name, 1.0)) if isinstance(gains, Mapping) else 1.0
                bias = float(biases.get(name, 0.0)) if isinstance(biases, Mapping) else 0.0
                value = gain * out[name] / (1.0 + out[name] / saturation) + bias
                out[name] = np.clip(value, a_min=0.0, a_max=None).astype(np.float32)

        cross_talk = float(params.get("cross_talk", 0.0))
        if cross_talk != 0.0 and "I_x" in out and "I_y" in out:
            ix = out["I_x"].copy()
            iy = out["I_y"].copy()
            mix = max(0.0, min(abs(cross_talk), 0.49))
            out["I_x"] = ((1.0 - mix) * ix + mix * iy).astype(np.float32)
            out["I_y"] = ((1.0 - mix) * iy + mix * ix).astype(np.float32)
        return out, {"params": dict(params)}


__all__ = ["ResponseTransform"]
