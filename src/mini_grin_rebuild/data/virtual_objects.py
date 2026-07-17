from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig


@dataclass
class VirtualObject:
    config: SimulationConfig
    height_map: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height_map.shape


def make_grid(cfg: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    lin = np.linspace(-1.0, 1.0, cfg.grid_size)
    y, x = np.meshgrid(lin, lin, indexing="ij")
    return x, y


def spherical_bowl(cfg: SimulationConfig, radius_fraction: float = 0.7) -> np.ndarray:
    x, y = make_grid(cfg)
    r2 = x**2 + y**2
    bowl = radius_fraction**2 - r2
    bowl[bowl < 0] = 0.0
    return np.sqrt(bowl) * cfg.height_scale


def microlens_reference(cfg: SimulationConfig) -> np.ndarray:
    radius_fraction = float(getattr(cfg, "lens_radius_fraction", 1.0))
    return spherical_bowl(cfg, radius_fraction=radius_fraction)


def microlens_standard(cfg: SimulationConfig) -> np.ndarray:
    """
    Microlens standard surface built as:
      standard = reference_sphere + small aspheric residual (within one wrap).

    This matches the SRT intention: subtract a spherical reference to keep the residual
    phase within a single wrap, avoiding explicit unwrapping in downstream processing.
    """
    reference = microlens_reference(cfg)
    x, y = make_grid(cfg)
    r2 = x**2 + y**2
    radius_fraction = float(getattr(cfg, "lens_radius_fraction", 1.0))
    aperture = (r2 <= radius_fraction**2).astype(float)

    r = np.sqrt(np.clip(r2, 0.0, None))
    r_norm = np.clip(r / max(radius_fraction, 1e-6), 0.0, 1.0)
    # Smooth residual with zero slope at center; zero-mean inside aperture.
    basis = 0.6 * (r_norm**4) + 0.4 * (r_norm**6)
    basis = basis * aperture
    if np.any(aperture > 0):
        basis = basis - float(np.sum(basis) / (np.sum(aperture) + 1e-12))

    wrap_height = _wrap_height(cfg)
    frac = float(getattr(cfg, "standard_residual_wrap_frac", 0.5))
    residual_amp = float(np.clip(frac, 0.0, 2.0)) * float(wrap_height)
    residual = residual_amp * basis
    return reference + residual


def aspheric_surface(
    cfg: SimulationConfig,
    radius_fraction: float = 0.75,
    c4: float = 0.25,
    c6: float = 0.0,
) -> np.ndarray:
    x, y = make_grid(cfg)
    r = np.sqrt(x**2 + y**2)
    base = np.maximum(radius_fraction - r, 0.0)
    deform = c4 * r**4 + c6 * r**6
    return np.clip(base + deform, a_min=0.0, a_max=None) * cfg.height_scale


def defect_patch(
    cfg: SimulationConfig,
    *,
    amplitude: float = 0.2,
    center: Tuple[float, float] = (0.0, 0.0),
    sigma: float = 0.1,
    sigma_phys_um: float | None = None,
    support_k: float | None = None,
) -> np.ndarray:
    sigma = _sigma_from_phys(cfg, sigma_phys_um) if sigma_phys_um is not None else sigma
    if support_k is None:
        support_k = cfg.defect_support_k
    x, y = make_grid(cfg)
    dx = x - center[0]
    dy = y - center[1]
    gauss = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    mask = (dx**2 + dy**2) <= (support_k * sigma) ** 2
    gauss = gauss * mask.astype(float)

    wrap_height = _wrap_height(cfg)
    target_height = amplitude * cfg.height_scale
    if target_height > wrap_height and not bool(getattr(cfg, "allow_defect_wrap_exceed", False)):
        target_height = wrap_height
    return target_height * gauss


def scratch_defect(
    cfg: SimulationConfig,
    *,
    amplitude: float = 0.2,
    center: Tuple[float, float] = (0.0, 0.0),
    width_phys_um: float = 2.0,
    length_phys_um: float = 40.0,
    angle_deg: float = 0.0,
    support_k: float | None = None,
) -> np.ndarray:
    if support_k is None:
        support_k = cfg.defect_support_k

    wrap_height = _wrap_height(cfg)
    target_height = amplitude * cfg.height_scale
    if target_height > wrap_height and not bool(getattr(cfg, "allow_defect_wrap_exceed", False)):
        target_height = wrap_height

    width_norm = _sigma_from_phys(cfg, width_phys_um)
    length_norm = _sigma_from_phys(cfg, length_phys_um)

    x, y = make_grid(cfg)
    cx, cy = center
    dx = x - cx
    dy = y - cy
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u = cos_t * dx + sin_t * dy
    v = -sin_t * dx + cos_t * dy

    gauss_cross = np.exp(-(v**2) / (2.0 * width_norm**2 + 1e-12))
    gauss_along = np.exp(-(u**2) / (2.0 * length_norm**2 + 1e-12))
    scratch = target_height * gauss_cross * gauss_along

    support_mask = (v**2) <= (support_k * width_norm) ** 2
    support_mask &= (u**2) <= (support_k * length_norm) ** 2
    return scratch * support_mask.astype(float)


def build_triplet(cfg: SimulationConfig) -> Dict[str, VirtualObject]:
    scene = str(getattr(cfg, "scene", "legacy"))
    if scene == "microlens_srt":
        reference = VirtualObject(cfg, microlens_reference(cfg))
        standard = VirtualObject(cfg, microlens_standard(cfg))
    else:
        standard = VirtualObject(cfg, aspheric_surface(cfg))
        reference = VirtualObject(cfg, spherical_bowl(cfg))
    sigma_phys = 0.5 * (cfg.defect_sigma_min_um + cfg.defect_sigma_max_um)
    defect = defect_patch(
        cfg,
        amplitude=0.15,
        center=(0.2, -0.1),
        sigma_phys_um=sigma_phys,
        support_k=cfg.defect_support_k,
    )
    test_h = standard.height_map + defect
    test = VirtualObject(cfg, test_h)
    defect_obj = VirtualObject(cfg, defect)
    return {"standard": standard, "reference": reference, "test": test, "defect": defect_obj}


def _wrap_height(cfg: SimulationConfig) -> float:
    scale = (2.0 * math.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air)
    return (math.pi * cfg.wrap_safety) / scale


def _sigma_from_phys(cfg: SimulationConfig, sigma_phys_um: float) -> float:
    fov = cfg.dx * cfg.grid_size
    half = max(fov / 2.0, 1e-6)
    return float(sigma_phys_um / half)


__all__ = [
    "VirtualObject",
    "make_grid",
    "microlens_reference",
    "microlens_standard",
    "spherical_bowl",
    "aspheric_surface",
    "defect_patch",
    "scratch_defect",
    "build_triplet",
]
