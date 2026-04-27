from __future__ import annotations

from typing import Any, Mapping

from mini_grin_rebuild.simulation.transforms.base import BaseTransform, CaptureTransform, TransformContext
from mini_grin_rebuild.simulation.transforms.camera import CameraTransform
from mini_grin_rebuild.simulation.transforms.geometry import GeometryTransform
from mini_grin_rebuild.simulation.transforms.illumination import IlluminationTransform
from mini_grin_rebuild.simulation.transforms.optics import OpticsTransform
from mini_grin_rebuild.simulation.transforms.response import ResponseTransform

TRANSFORM_REGISTRY = {
    ResponseTransform.name: ResponseTransform,
    OpticsTransform.name: OpticsTransform,
    IlluminationTransform.name: IlluminationTransform,
    GeometryTransform.name: GeometryTransform,
    CameraTransform.name: CameraTransform,
}

DEFAULT_PIPELINE = ["response", "optics", "illumination", "geometry", "camera"]


def build_transform_pipeline(params: Mapping[str, Any] | None) -> list[CaptureTransform]:
    params = dict(params or {})
    names = list(params.get("pipeline", DEFAULT_PIPELINE))
    transforms: list[CaptureTransform] = []
    for name in names:
        cls = TRANSFORM_REGISTRY.get(str(name))
        if cls is None:
            raise ValueError(f"Unknown instrument_lite transform: {name!r}")
        cfg = params.get(str(name), {})
        transform = cls(cfg)
        if getattr(transform, "enabled", True):
            transforms.append(transform)
    return transforms


__all__ = [
    "BaseTransform",
    "CameraTransform",
    "CaptureTransform",
    "DEFAULT_PIPELINE",
    "GeometryTransform",
    "IlluminationTransform",
    "OpticsTransform",
    "ResponseTransform",
    "TRANSFORM_REGISTRY",
    "TransformContext",
    "build_transform_pipeline",
]
