from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.physics.simulator import phase_from_height


@dataclass(frozen=True)
class PluxMetadata:
    source_path: str
    image_size_x: int
    image_size_y: int
    raw_member: str
    fov_x_raw: float | None
    fov_y_raw: float | None
    fov_unit_assumed: str
    fov_x_um: float | None
    fov_y_um: float | None
    pixel_pitch_x_um: float | None
    pixel_pitch_y_um: float | None
    technique: str | None
    measure_type: str | None
    algorithm: str | None
    measured_fraction: float | None
    instrument_manufacturer: str | None
    instrument_model: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "image_size_x": self.image_size_x,
            "image_size_y": self.image_size_y,
            "raw_member": self.raw_member,
            "fov_x_raw": self.fov_x_raw,
            "fov_y_raw": self.fov_y_raw,
            "fov_unit_assumed": self.fov_unit_assumed,
            "fov_x_um": self.fov_x_um,
            "fov_y_um": self.fov_y_um,
            "pixel_pitch_x_um": self.pixel_pitch_x_um,
            "pixel_pitch_y_um": self.pixel_pitch_y_um,
            "technique": self.technique,
            "measure_type": self.measure_type,
            "algorithm": self.algorithm,
            "measured_fraction": self.measured_fraction,
            "instrument_manufacturer": self.instrument_manufacturer,
            "instrument_model": self.instrument_model,
        }


@dataclass(frozen=True)
class PluxTopography:
    metadata: PluxMetadata
    height_um: np.ndarray
    valid_mask: np.ndarray


def _child_text(parent: ET.Element | None, tag: str) -> str | None:
    if parent is None:
        return None
    child = parent.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _parse_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_percent(text: str | None) -> float | None:
    if text is None:
        return None
    raw = text.strip()
    if raw.endswith("%"):
        raw = raw[:-1]
    try:
        return float(raw) / 100.0
    except ValueError:
        return None


def _info_map(root: ET.Element) -> dict[str, str]:
    info = root.find("INFO")
    if info is None:
        return {}
    out: dict[str, str] = {}
    for item in info:
        if not item.tag.startswith("ITEM_"):
            continue
        name = _child_text(item, "NAME")
        value = _child_text(item, "VALUE")
        if name and value is not None:
            out[name] = value
    return out


def _fov_to_um(value: float | None, *, unit: str) -> float | None:
    if value is None:
        return None
    norm = unit.lower()
    if norm in {"um", "µm"}:
        return float(value)
    if norm == "mm":
        return float(value) * 1000.0
    raise ValueError(f"Unsupported FOV unit: {unit}")


def read_plux_metadata(path: str | Path, *, fov_unit: str = "mm") -> PluxMetadata:
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("index.xml"))

    general = root.find("GENERAL")
    layer0 = root.find("LAYER_0")
    instrument = root.find("Instrument")
    info = _info_map(root)

    image_size_x = int(_child_text(general, "IMAGE_SIZE_X") or "0")
    image_size_y = int(_child_text(general, "IMAGE_SIZE_Y") or "0")
    raw_member = _child_text(layer0, "FILENAME_Z") or "LAYER_0.raw"

    fov_x_raw = _parse_float(_child_text(general, "FOV_X"))
    fov_y_raw = _parse_float(_child_text(general, "FOV_Y"))
    fov_x_um = _fov_to_um(fov_x_raw, unit=fov_unit)
    fov_y_um = _fov_to_um(fov_y_raw, unit=fov_unit)
    pixel_pitch_x_um = None if fov_x_um is None or image_size_x <= 0 else fov_x_um / float(image_size_x)
    pixel_pitch_y_um = None if fov_y_um is None or image_size_y <= 0 else fov_y_um / float(image_size_y)

    return PluxMetadata(
        source_path=str(path),
        image_size_x=image_size_x,
        image_size_y=image_size_y,
        raw_member=raw_member,
        fov_x_raw=fov_x_raw,
        fov_y_raw=fov_y_raw,
        fov_unit_assumed=fov_unit,
        fov_x_um=fov_x_um,
        fov_y_um=fov_y_um,
        pixel_pitch_x_um=pixel_pitch_x_um,
        pixel_pitch_y_um=pixel_pitch_y_um,
        technique=info.get("Technique"),
        measure_type=info.get("Measure type"),
        algorithm=info.get("Algorithm"),
        measured_fraction=_parse_percent(info.get("Measured")),
        instrument_manufacturer=_child_text(instrument, "Manufacturer"),
        instrument_model=_child_text(instrument, "Model"),
    )


def _fill_missing_with_median(height: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.array(height, dtype=np.float32, copy=True)
    if valid_mask.all():
        return out
    median = float(np.median(out[valid_mask]))
    out[~valid_mask] = median
    return out


def _subtract_plane(height: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(height.shape, dtype=np.float64)
    design = np.column_stack(
        [
            xx[valid_mask].ravel(),
            yy[valid_mask].ravel(),
            np.ones(int(valid_mask.sum()), dtype=np.float64),
        ]
    )
    target = height[valid_mask].astype(np.float64, copy=False).ravel()
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    return (height.astype(np.float64) - plane).astype(np.float32)


def _center_height(height: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    mean = float(np.mean(height[valid_mask]))
    return (height - mean).astype(np.float32)


def load_plux_topography(
    path: str | Path,
    *,
    fov_unit: str = "mm",
    fill_missing: bool = True,
    detrend_plane: bool = True,
    center_height: bool = True,
) -> PluxTopography:
    metadata = read_plux_metadata(path, fov_unit=fov_unit)
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(metadata.raw_member)
    expected = metadata.image_size_x * metadata.image_size_y
    values = np.frombuffer(raw, dtype="<f4")
    if values.size != expected:
        raise ValueError(
            f"Unexpected sample size for {path}: expected {expected} float32 values, got {values.size}"
        )

    height = values.reshape((metadata.image_size_y, metadata.image_size_x)).astype(np.float32, copy=True)
    valid_mask = np.isfinite(height)
    if not np.any(valid_mask):
        raise ValueError(f"No finite topography pixels found in {path}")

    if fill_missing:
        height = _fill_missing_with_median(height, valid_mask)
    if detrend_plane:
        height = _subtract_plane(height, valid_mask)
    if center_height:
        height = _center_height(height, valid_mask)

    return PluxTopography(metadata=metadata, height_um=height, valid_mask=valid_mask)


def simulate_spdic_from_height(
    cfg: SimulationConfig,
    *,
    height_um: np.ndarray,
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    if pixel_pitch_x_um <= 0 or pixel_pitch_y_um <= 0:
        raise ValueError("pixel_pitch_x_um and pixel_pitch_y_um must be positive")
    if not np.isfinite(height_um).all():
        raise ValueError("height_um must be finite before SPDIC simulation")

    phase = phase_from_height(cfg, height_um)
    grad_y, grad_x = np.gradient(phase, float(pixel_pitch_y_um), float(pixel_pitch_x_um))
    intensity_x = np.clip(grad_x**2, a_min=0.0, a_max=None)
    intensity_y = np.clip(grad_y**2, a_min=0.0, a_max=None)
    if cfg.noise_level > 0:
        if rng is None:
            rng = np.random.default_rng()
        intensity_x = intensity_x + rng.normal(0.0, cfg.noise_level, intensity_x.shape)
        intensity_y = intensity_y + rng.normal(0.0, cfg.noise_level, intensity_y.shape)
    return {
        "phase": phase.astype(np.float32, copy=False),
        "grad_x": grad_x.astype(np.float32, copy=False),
        "grad_y": grad_y.astype(np.float32, copy=False),
        "I_x": intensity_x.astype(np.float32, copy=False),
        "I_y": intensity_y.astype(np.float32, copy=False),
    }


__all__ = [
    "PluxMetadata",
    "PluxTopography",
    "load_plux_topography",
    "read_plux_metadata",
    "simulate_spdic_from_height",
]
