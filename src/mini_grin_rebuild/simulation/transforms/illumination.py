from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.simulation.transforms.base import BaseTransform, TransformContext
from mini_grin_rebuild.simulation.transforms.utils import resize_bilinear, sample_pair, sample_range, sanitize_channels


class IlluminationTransform(BaseTransform):
    name = "illumination"

    def sample_bundle_params(self, context: TransformContext) -> dict[str, Any]:
        strength_x, strength_y = sample_pair(context.rng, self.cfg.get("field_strength"), 0.0)
        bias_x, bias_y = sample_pair(context.rng, self.cfg.get("bias"), 0.0)
        lowres_size = int(sample_range(context.rng, self.cfg.get("lowres_size"), 5))
        lowres_size = max(2, lowres_size)
        fields: dict[str, list[list[float]]] = {}
        for name, strength in (("I_x", strength_x), ("I_y", strength_y)):
            low = context.rng.normal(0.0, 1.0, (lowres_size, lowres_size)).astype(np.float32)
            low -= float(np.mean(low))
            std = float(np.std(low)) or 1.0
            low = low / std
            field = 1.0 + float(strength) * resize_bilinear(low, context.shape)
            fields[name] = np.clip(field, 0.05, None).astype(np.float32).tolist()
        return {
            "bias": {"I_x": bias_x, "I_y": bias_y},
            "field_strength": {"I_x": strength_x, "I_y": strength_y},
            "field": fields,
        }

    def apply(
        self,
        channels: Mapping[str, np.ndarray],
        *,
        context: TransformContext,
        params: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        out = sanitize_channels(channels)
        fields = params.get("field", {})
        biases = params.get("bias", {})
        for name in ("I_x", "I_y"):
            if name not in out:
                continue
            field = np.asarray(fields.get(name, 1.0), dtype=np.float32) if isinstance(fields, Mapping) else 1.0
            bias = float(biases.get(name, 0.0)) if isinstance(biases, Mapping) else 0.0
            out[name] = np.clip(out[name] * field + bias, a_min=0.0, a_max=None).astype(np.float32)
        meta = dict(params)
        meta.pop("field", None)
        meta["field_recorded"] = bool(fields)
        return out, {"params": meta}


__all__ = ["IlluminationTransform"]
