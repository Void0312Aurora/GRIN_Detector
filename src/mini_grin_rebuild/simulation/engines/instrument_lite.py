from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.simulation.engines.ideal_gradient import phase_from_height
from mini_grin_rebuild.simulation.transforms import build_transform_pipeline
from mini_grin_rebuild.simulation.transforms.base import CaptureTransform, TransformContext
from mini_grin_rebuild.simulation.types import Capture, CaptureBundle


class InstrumentLiteEngine:
    """
    Modular instrument-like simulator.

    The ideal signal remains squared phase-gradient, but realism factors are applied as
    independent, ordered transforms: response, optics, illumination, geometry, camera.
    This gives a controllable mismatch between offline capture generation and the
    differentiable forward model used by physics losses.
    """

    name = "instrument_lite"
    version = "1"

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        raw_params = getattr(cfg, "capture_engine_params", {}) or {}
        if not isinstance(raw_params, Mapping):
            raise TypeError("simulation.capture_engine_params must be a JSON object for instrument_lite")
        self.params = dict(raw_params)
        self.transforms = build_transform_pipeline(self.params)

    def _ideal_channels(self, height: np.ndarray) -> dict[str, np.ndarray]:
        phase_map = phase_from_height(self.cfg, np.asarray(height, dtype=np.float32))
        grad_y, grad_x = np.gradient(phase_map, self.cfg.dx)
        return {
            "I_x": np.clip(grad_x**2, a_min=0.0, a_max=None).astype(np.float32),
            "I_y": np.clip(grad_y**2, a_min=0.0, a_max=None).astype(np.float32),
        }

    def _sample_transform_params(
        self,
        *,
        rng: np.random.Generator,
        shape: tuple[int, int],
    ) -> dict[str, dict[str, Any]]:
        context = TransformContext(cfg=self.cfg, rng=rng, shape=shape)
        sampled: dict[str, dict[str, Any]] = {}
        for transform in self.transforms:
            sampled[transform.name] = transform.sample_bundle_params(context)
        return sampled

    def _apply_transforms(
        self,
        channels: Mapping[str, np.ndarray],
        *,
        rng: np.random.Generator,
        shape: tuple[int, int],
        frame_name: str | None,
        sampled_params: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        out = {name: np.asarray(value, dtype=np.float32) for name, value in channels.items()}
        transform_meta: dict[str, Any] = {}
        context = TransformContext(
            cfg=self.cfg,
            rng=rng,
            shape=shape,
            frame_name=frame_name,
            bundle_params={name: dict(value) for name, value in sampled_params.items()},
        )
        for transform in self.transforms:
            params = sampled_params.get(transform.name, {})
            out, meta = transform.apply(out, context=context, params=params)
            transform_meta[transform.name] = meta
        return out, transform_meta

    def _public_params(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {"omitted_array_shape": list(value.shape)}
        if isinstance(value, Mapping):
            return {
                str(key): ({"omitted": "field"} if str(key) == "field" else self._public_params(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                return {"omitted_nested_list_len": len(value)}
            return [self._public_params(item) for item in value]
        return value

    def simulate_capture(
        self,
        height: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Capture:
        if rng is None:
            rng = np.random.default_rng()
        shape = tuple(np.asarray(height).shape[-2:])
        sampled = self._sample_transform_params(rng=rng, shape=shape)  # type: ignore[arg-type]
        channels, transform_meta = self._apply_transforms(
            self._ideal_channels(height),
            rng=rng,
            shape=shape,  # type: ignore[arg-type]
            frame_name=None,
            sampled_params=sampled,
        )
        capture_meta: dict[str, Any] = dict(meta or {})
        capture_meta.update(self.meta())
        capture_meta["sampled_transforms"] = self._public_params(transform_meta)
        return Capture(channels=channels, meta=capture_meta)

    def simulate_bundle(
        self,
        heights: Mapping[str, np.ndarray],
        *,
        rng: np.random.Generator | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CaptureBundle:
        if not heights:
            raise ValueError("simulate_bundle requires at least one height frame")
        if rng is None:
            rng = np.random.default_rng()
        first = next(iter(heights.values()))
        shape = tuple(np.asarray(first).shape[-2:])
        sampled = self._sample_transform_params(rng=rng, shape=shape)  # type: ignore[arg-type]
        captures: dict[str, Capture] = {}
        frame_metas: dict[str, Any] = {}
        for frame_name, height in heights.items():
            if tuple(np.asarray(height).shape[-2:]) != shape:
                raise ValueError("All bundle frames must share the same spatial shape")
            channels, transform_meta = self._apply_transforms(
                self._ideal_channels(height),
                rng=rng,
                shape=shape,  # type: ignore[arg-type]
                frame_name=str(frame_name),
                sampled_params=sampled,
            )
            frame_meta = {"frame_name": str(frame_name), "sampled_transforms": transform_meta}
            captures[str(frame_name)] = Capture(channels=channels, meta=frame_meta)
            frame_metas[str(frame_name)] = frame_meta
        bundle_meta: dict[str, Any] = dict(meta or {})
        bundle_meta.update(self.meta())
        bundle_meta["sampled_bundle_params"] = self._public_params({name: dict(value) for name, value in sampled.items()})
        bundle_meta["frames"] = self._public_params(frame_metas)
        return CaptureBundle(captures=captures, meta=bundle_meta)

    def meta(self) -> dict[str, Any]:
        return {
            "engine_name": self.name,
            "engine_version": self.version,
            "base_model": "squared_phase_gradient",
            "pipeline": [transform.name for transform in self.transforms],
            "params": self.params,
        }


__all__ = ["InstrumentLiteEngine"]
