from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.simulation.transforms.base import BaseTransform, TransformContext
from mini_grin_rebuild.simulation.transforms.utils import sample_pair, sample_range, sanitize_channels, warp_affine


class GeometryTransform(BaseTransform):
    name = "geometry"

    def sample_bundle_params(self, context: TransformContext) -> dict[str, Any]:
        frame_names = list(self.cfg.get("frame_names", ["standard", "reference", "test"]))
        capture_shift_cfg = self.cfg.get("capture_shift_px")
        rotation_cfg = self.cfg.get("rotation_deg")
        scale_cfg = self.cfg.get("scale")
        frames: dict[str, dict[str, float]] = {}
        for frame in frame_names:
            shift_y, shift_x = sample_pair(context.rng, capture_shift_cfg, 0.0)
            frames[frame] = {
                "shift_y": shift_y,
                "shift_x": shift_x,
                "rotation_deg": sample_range(context.rng, rotation_cfg, 0.0),
                "scale": sample_range(context.rng, scale_cfg, 1.0),
            }
        channel_shift: dict[str, dict[str, float]] = {}
        for name in ("I_x", "I_y"):
            shift_y, shift_x = sample_pair(context.rng, self.cfg.get("channel_shift_px"), 0.0)
            channel_shift[name] = {"shift_y": shift_y, "shift_x": shift_x}
        return {"frames": frames, "channel_shift": channel_shift}

    def apply(
        self,
        channels: Mapping[str, np.ndarray],
        *,
        context: TransformContext,
        params: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        out = sanitize_channels(channels)
        frame_params = {}
        frames = params.get("frames", {})
        if context.frame_name and isinstance(frames, Mapping):
            frame_params = dict(frames.get(context.frame_name, {}))
        channel_shift = params.get("channel_shift", {})
        for name in list(out.keys()):
            ch = dict(channel_shift.get(name, {})) if isinstance(channel_shift, Mapping) else {}
            out[name] = warp_affine(
                out[name],
                shift_y=float(frame_params.get("shift_y", 0.0)) + float(ch.get("shift_y", 0.0)),
                shift_x=float(frame_params.get("shift_x", 0.0)) + float(ch.get("shift_x", 0.0)),
                rotation_deg=float(frame_params.get("rotation_deg", 0.0)),
                scale=float(frame_params.get("scale", 1.0)),
                fill_value=0.0,
            )
        return out, {"params": dict(params)}


__all__ = ["GeometryTransform"]
