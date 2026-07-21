from __future__ import annotations

import math
from typing import Any


def phase_scale(cfg: Any) -> float:
    """Return the height-to-phase scale in rad per configured length unit.

    ``wavelength`` and height must use the same unit (micrometres in the
    project configs).  Transmission keeps the legacy optical-path model.
    Reflection assumes a front-surface round trip through the ambient medium;
    the sample refractive index therefore does not enter the geometric phase.
    """

    wavelength = float(cfg.wavelength)
    if wavelength <= 0.0:
        raise ValueError("simulation.wavelength must be > 0")

    mode = str(getattr(cfg, "phase_mode", "transmission") or "transmission").lower()
    if mode == "transmission":
        return float((2.0 * math.pi / wavelength) * (float(cfg.n_object) - float(cfg.n_air)))
    if mode == "reflection":
        incidence_deg = float(getattr(cfg, "reflection_incidence_angle_deg", 0.0) or 0.0)
        cos_incidence = math.cos(math.radians(incidence_deg))
        if cos_incidence <= 0.0:
            raise ValueError("reflection_incidence_angle_deg must have a positive cosine")
        ambient_index = float(getattr(cfg, "n_air", 1.0) or 1.0)
        if ambient_index <= 0.0:
            raise ValueError("simulation.n_air must be > 0 for reflection phase")
        return float((4.0 * math.pi / wavelength) * ambient_index * cos_incidence)
    raise ValueError(f"Unknown phase_mode={mode!r} (expected 'transmission' or 'reflection')")


def phase_model_meta(cfg: Any) -> dict[str, float | str]:
    return {
        "phase_mode": str(getattr(cfg, "phase_mode", "transmission") or "transmission"),
        "phase_scale_rad_per_height_unit": phase_scale(cfg),
        "wavelength": float(cfg.wavelength),
        "reflection_incidence_angle_deg": float(
            getattr(cfg, "reflection_incidence_angle_deg", 0.0) or 0.0
        ),
    }


__all__ = ["phase_model_meta", "phase_scale"]
