from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.physics.phase import phase_model_meta
from mini_grin_rebuild.simulation.engines.ideal_gradient import phase_from_height
from mini_grin_rebuild.simulation.transforms.base import TransformContext
from mini_grin_rebuild.simulation.transforms.camera import CameraTransform
from mini_grin_rebuild.simulation.transforms.utils import gaussian_blur, sample_range
from mini_grin_rebuild.simulation.types import Capture, CaptureBundle


def _complex_propagate(
    field: np.ndarray,
    *,
    dx: float,
    wavelength: float,
    aperture_sigma_freq: float,
    numerical_aperture: float,
    aperture_softness_freq: float,
    defocus_strength: float,
    aberration_strength: float,
) -> np.ndarray:
    h, w = field.shape
    fy = np.fft.fftfreq(h, d=max(dx, 1e-12)).astype(np.float32)
    fx = np.fft.fftfreq(w, d=max(dx, 1e-12)).astype(np.float32)
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    rr2 = fxx**2 + fyy**2

    field_fft = np.fft.fft2(field)

    na = float(numerical_aperture)
    if na > 0.0:
        cutoff = na / max(float(wavelength), 1e-12)
        radial_freq = np.sqrt(rr2)
        softness = max(float(aperture_softness_freq), 0.0)
        if softness > 0.0:
            edge_arg = np.clip((radial_freq - cutoff) / softness, -60.0, 60.0)
            aperture_real = 1.0 / (1.0 + np.exp(edge_arg))
        else:
            aperture_real = (radial_freq <= cutoff).astype(np.float32)
        aperture = aperture_real.astype(np.complex64)
    else:
        sigma = max(float(aperture_sigma_freq), 1e-12)
        aperture = np.exp(-0.5 * rr2 / (sigma * sigma)).astype(np.complex64)
    phase_defocus = np.exp(-1j * np.pi * float(wavelength) * float(defocus_strength) * rr2).astype(np.complex64)

    if float(aberration_strength) != 0.0:
        theta = np.arctan2(fyy, fxx).astype(np.float32)
        rho = np.sqrt(rr2).astype(np.float32)
        phase_ab = float(aberration_strength) * (rho**2) * np.cos(2.0 * theta)
        aberration = np.exp(1j * phase_ab).astype(np.complex64)
    else:
        aberration = np.ones_like(aperture, dtype=np.complex64)

    transfer = aperture * phase_defocus * aberration
    out = np.fft.ifft2(field_fft * transfer)
    return out.astype(np.complex64)


class OpticalLeakageLiteEngine:
    """
    Minimal scalar diffraction-like engine used to probe sign leakage.

    Unlike `instrument_lite`, this engine does not start from ideal DIC intensities.
    It first propagates the complex phase field and then forms:
    - raw intensity: |U|^2
    - DIC-like channels from sheared differences on the propagated field
    """

    name = "optical_leakage_lite"
    version = "4"

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        raw_params = getattr(cfg, "capture_engine_params", {}) or {}
        if not isinstance(raw_params, Mapping):
            raise TypeError("simulation.capture_engine_params must be a JSON object for optical_leakage_lite")
        self.params = dict(raw_params)
        self.emit_raw = bool(self.params.get("emit_raw", True))
        raw_texture_cfg = self.params.get("field_texture", {}) or {}
        if not isinstance(raw_texture_cfg, Mapping):
            raise TypeError("optical_leakage_lite field_texture must be a JSON object")
        self.field_texture_cfg = dict(raw_texture_cfg)
        raw_camera_cfg = self.params.get("camera", {}) or {}
        if not isinstance(raw_camera_cfg, Mapping):
            raise TypeError("optical_leakage_lite camera must be a JSON object")
        self.camera_cfg = dict(raw_camera_cfg)
        self.camera_transform = CameraTransform(raw_camera_cfg)
        raw_ghost_cfg = self.params.get("coherent_ghost", {}) or {}
        if not isinstance(raw_ghost_cfg, Mapping):
            raise TypeError("optical_leakage_lite coherent_ghost must be a JSON object")
        self.coherent_ghost_cfg = dict(raw_ghost_cfg)
        raw_reflectance_cfg = self.params.get("reflectance", {}) or {}
        if not isinstance(raw_reflectance_cfg, Mapping):
            raise TypeError("optical_leakage_lite reflectance must be a JSON object")
        self.reflectance_cfg = dict(raw_reflectance_cfg)
        raw_dark_port_cfg = self.params.get("dark_port", {}) or {}
        if not isinstance(raw_dark_port_cfg, Mapping):
            raise TypeError("optical_leakage_lite dark_port must be a JSON object")
        self.dark_port_cfg = dict(raw_dark_port_cfg)

    def _sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        p = self.params
        return {
            "defocus_strength": sample_range(rng, p.get("defocus_strength"), 0.0),
            "aperture_sigma_freq": sample_range(rng, p.get("aperture_sigma_freq"), 0.35 / max(self.cfg.dx, 1e-12)),
            "numerical_aperture": sample_range(
                rng,
                p.get("numerical_aperture"),
                float(getattr(self.cfg, "numerical_aperture", 0.0) or 0.0),
            ),
            "aperture_softness_freq": sample_range(rng, p.get("aperture_softness_freq"), 0.0),
            "aberration_strength": sample_range(rng, p.get("aberration_strength"), 0.0),
            "raw_blur_sigma_px": sample_range(rng, p.get("raw_blur_sigma_px"), 0.0),
            "dic_blur_sigma_px": sample_range(rng, p.get("dic_blur_sigma_px"), 0.0),
            "raw_gain": sample_range(rng, p.get("raw_gain"), 1.0),
            "raw_bias": sample_range(rng, p.get("raw_bias"), 0.0),
            "shear_px": max(0.5, sample_range(rng, p.get("shear_px"), 1.0)),
        }

    def _sample_reflectance_params(self, rng: np.random.Generator) -> dict[str, Any]:
        p = self.reflectance_cfg
        rim_width = max(0.5, sample_range(rng, p.get("rim_width_px"), 4.0))
        background_rough = max(0.0, sample_range(rng, p.get("background_phase_rough_rad"), 0.0))
        return {
            "enabled": bool(p.get("enabled", bool(p))),
            "lens_amplitude": max(0.0, sample_range(rng, p.get("lens_amplitude"), 1.0)),
            "background_amplitude": max(0.0, sample_range(rng, p.get("background_amplitude"), 1.0)),
            "background_phase_rough_rad": background_rough,
            "background_texture_sigma_px": max(
                0.0, sample_range(rng, p.get("background_texture_sigma_px"), 1.5)
            ),
            "edge_softness_px": max(0.1, sample_range(rng, p.get("edge_softness_px"), 1.5)),
            # Per-lens surface micro-roughness inside the cap (polishing/molding
            # residual and contamination film); a static surface property drawn
            # once per capture and shared across coherence realizations.
            "lens_phase_rough_rad": max(0.0, sample_range(rng, p.get("lens_phase_rough_rad"), 0.0)),
            "lens_texture_sigma_px": max(0.3, sample_range(rng, p.get("lens_texture_sigma_px"), 2.0)),
            # Additive weak scattering component of the lens surface, coexisting
            # with the smooth specular return: the cap phase stays intact while
            # a rough-phase field of this amplitude is added inside the cap.
            "lens_scatter_amplitude": max(0.0, sample_range(rng, p.get("lens_scatter_amplitude"), 0.0)),
            "lens_scatter_phase_rad": max(0.0, sample_range(rng, p.get("lens_scatter_phase_rad"), 3.0)),
            # Sparse point scatterers (dust, micro-pits) on the cap: fixed
            # positions per capture, random phase per coherence realization.
            "lens_point_scatter_count": max(
                0, int(round(sample_range(rng, p.get("lens_point_scatter_count"), 0.0)))
            ),
            "lens_point_scatter_amplitude": max(
                0.0, sample_range(rng, p.get("lens_point_scatter_amplitude"), 0.5)
            ),
            "rim_amplitude": max(0.0, sample_range(rng, p.get("rim_amplitude"), 0.0)),
            # Deterministic (smooth-phase) component of the seam scattering: the
            # part of the seam ring that stays coherent and can interfere with
            # the background field, producing the observed common-mode bands.
            "rim_coherent_amplitude": max(0.0, sample_range(rng, p.get("rim_coherent_amplitude"), 0.0)),
            # Seam roughness: a chipped/glued seam is many radians rough, which
            # kills its coherent (ring-diffraction) component; when absent it
            # reuses the background roughness sample (no extra rng draw).
            "rim_phase_rough_rad": max(
                0.0,
                sample_range(rng, p.get("rim_phase_rough_rad"), background_rough),
            ),
            "rim_width_px": rim_width,
            # The seam band is asymmetric in reality: a sharp inner edge at the cap
            # boundary and a scattering skirt extending outward over the substrate.
            # Defaults fall back to the symmetric rim_width_px.
            "rim_inner_width_px": max(0.5, sample_range(rng, p.get("rim_inner_width_px"), rim_width)),
            "rim_outer_width_px": max(0.5, sample_range(rng, p.get("rim_outer_width_px"), rim_width)),
            # Number of independent rough-phase realizations whose intensities are
            # averaged: partially developed speckle from finite illumination
            # spatial coherence. 1 keeps fully coherent speckle.
            "speckle_realizations": max(1, int(round(sample_range(rng, p.get("speckle_realizations"), 1.0)))),
            # Deterministic large-scale illumination dipole across the frame.
            "illumination_tilt_strength": max(
                0.0, sample_range(rng, p.get("illumination_tilt_strength"), 0.0)
            ),
            "illumination_tilt_angle_deg": sample_range(rng, p.get("illumination_tilt_angle_deg"), 0.0),
        }

    def _sample_dark_port_params(self, rng: np.random.Generator) -> dict[str, Any]:
        p = self.dark_port_cfg
        mode = str(p.get("mode", "image_shear") or "image_shear").lower()
        if mode not in ("image_shear", "fourier_tilt"):
            raise ValueError(
                f"optical_leakage_lite dark_port.mode must be 'image_shear' or 'fourier_tilt', got {mode!r}"
            )
        return {
            "enabled": bool(p.get("enabled", bool(p))),
            "mode": mode,
            "leak_amplitude": max(0.0, sample_range(rng, p.get("leak_amplitude"), 0.0)),
            "leak_phase_rad": sample_range(rng, p.get("leak_phase_rad"), 0.0),
            "fringe_cycles_per_frame": max(
                0.0, sample_range(rng, p.get("fringe_cycles_per_frame"), 0.5)
            ),
            "fringe_phase_rad": sample_range(rng, p.get("fringe_phase_rad"), 0.0),
            "fringe_angle_deg": sample_range(rng, p.get("fringe_angle_deg"), 0.0),
        }

    def _reflectance_modifier(
        self,
        rng: np.random.Generator,
        shape: tuple[int, int],
        params: Mapping[str, Any],
        point_positions: tuple[np.ndarray, np.ndarray] | None = None,
        static_lens_fields: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray | None:
        """Amplitude/phase map separating the specular lens cap from the scattering fixture.

        Background and rim rough phases are redrawn per coherence realization
        (ergodic model of finite illumination coherence over a many-radian
        rough surface); the weak lens-surface texture is a static property and
        must be supplied via ``static_lens_fields`` so it survives averaging.
        """

        if not bool(params.get("enabled", False)):
            return None
        h, w = shape
        radius_px = float(getattr(self.cfg, "lens_radius_fraction", 1.0) or 1.0) * 0.5 * min(h, w)
        yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
        xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
        y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
        rho_px = np.sqrt(x_grid**2 + y_grid**2)
        softness = max(float(params["edge_softness_px"]), 0.1)
        edge_arg = np.clip((rho_px - radius_px) / softness, -60.0, 60.0)
        outside = 1.0 / (1.0 + np.exp(-edge_arg))
        lens_amplitude = float(params["lens_amplitude"])
        background_amplitude = float(params["background_amplitude"])
        amplitude = (1.0 - outside) * lens_amplitude + outside * background_amplitude
        # The cap-to-substrate seam behaves as a rough scattering ring, not a
        # clean phase step; represent it as an additive rough-amplitude annulus.
        rim_amplitude = float(params["rim_amplitude"])
        rim_coherent = float(params.get("rim_coherent_amplitude", 0.0))
        rough_weight = np.asarray(outside, dtype=np.float32)
        ring: np.ndarray | None = None
        if rim_amplitude > 0.0 or rim_coherent > 0.0:
            inner_width = max(float(params.get("rim_inner_width_px", params["rim_width_px"])), 0.5)
            outer_width = max(float(params.get("rim_outer_width_px", params["rim_width_px"])), 0.5)
            distance = rho_px - radius_px
            ring = np.where(
                distance < 0.0,
                np.exp(-0.5 * (distance / inner_width) ** 2),
                np.exp(-0.5 * (distance / outer_width) ** 2),
            ).astype(np.float32)
        if rim_amplitude > 0.0 and ring is not None:
            amplitude = amplitude + rim_amplitude * ring
            rough_weight = np.clip(rough_weight + ring, 0.0, 1.0)
        tilt = float(params["illumination_tilt_strength"])
        if tilt > 0.0:
            # The lens specular return sits below the camera floor in the real
            # captures, so the illumination dipole is only observable on the
            # scattering regions; keep the smooth lens amplitude untouched to
            # avoid injecting an artificial amplitude gradient into the dark port.
            theta = np.deg2rad(float(params["illumination_tilt_angle_deg"]))
            xx_norm = (x_grid / max(0.5 * (w - 1), 1.0)).astype(np.float32)
            yy_norm = (y_grid / max(0.5 * (h - 1), 1.0)).astype(np.float32)
            coordinate = float(np.cos(theta)) * xx_norm + float(np.sin(theta)) * yy_norm
            tilt_factor = np.clip(1.0 + tilt * coordinate, 0.0, None)
            amplitude = amplitude * (1.0 + rough_weight * (tilt_factor - 1.0))
        rough = float(params["background_phase_rough_rad"])
        rim_rough = float(params.get("rim_phase_rough_rad", rough))
        lens_rough = float(params["lens_phase_rough_rad"])
        ring_weight = np.clip(ring, 0.0, 1.0) if ring is not None else None
        background_weight = np.asarray(outside, dtype=np.float32)
        if ring_weight is not None:
            background_weight = np.clip(background_weight - ring_weight, 0.0, 1.0)
        lens_weight = np.clip(1.0 - rough_weight, 0.0, 1.0)
        sigma = float(params["background_texture_sigma_px"])
        phase_total: np.ndarray | None = None
        if rough > 0.0:
            phase_total = background_weight * (
                rough * self._normalized_random_field(rng, shape, sigma_x_px=sigma, sigma_y_px=sigma)
            )
        if rim_rough > 0.0 and ring_weight is not None:
            rim_phase = ring_weight * (
                rim_rough * self._normalized_random_field(rng, shape, sigma_x_px=sigma, sigma_y_px=sigma)
            )
            phase_total = rim_phase if phase_total is None else phase_total + rim_phase
        if lens_rough > 0.0 and static_lens_fields is not None:
            lens_phase = lens_weight * (lens_rough * static_lens_fields["lens_rough_basis"])
            phase_total = lens_phase if phase_total is None else phase_total + lens_phase
        if phase_total is not None:
            modifier = amplitude * np.exp(1j * phase_total)
        else:
            modifier = amplitude.astype(np.complex64)
        if rim_coherent > 0.0 and ring is not None:
            # Narrow coherent seam ring: a sharper deterministic line source at
            # the cap boundary, riding on top of the rough seam skirt.
            coherent_width = max(float(params.get("rim_inner_width_px", params["rim_width_px"])), 0.5)
            coherent_ring = np.exp(-0.5 * ((rho_px - radius_px) / coherent_width) ** 2).astype(np.float32)
            modifier = np.asarray(modifier, dtype=np.complex64) + (
                rim_coherent * coherent_ring
            ).astype(np.complex64)
        lens_scatter = float(params.get("lens_scatter_amplitude", 0.0))
        if lens_scatter > 0.0 and static_lens_fields is not None:
            scatter_phase = (
                float(params.get("lens_scatter_phase_rad", 3.0))
                * static_lens_fields["lens_scatter_basis"]
            )
            modifier = modifier + (
                lens_scatter * lens_weight * np.exp(1j * scatter_phase)
            ).astype(np.complex64)
        if point_positions is not None and point_positions[0].size:
            point_amplitude = float(params.get("lens_point_scatter_amplitude", 0.0))
            if point_amplitude > 0.0:
                modifier = np.asarray(modifier, dtype=np.complex64).copy()
                phases = rng.uniform(0.0, 2.0 * np.pi, point_positions[0].size).astype(np.float32)
                modifier[point_positions] = modifier[point_positions] + (
                    point_amplitude * np.exp(1j * phases)
                ).astype(np.complex64)
        return np.asarray(modifier, dtype=np.complex64)

    def _sample_field_texture_params(self, rng: np.random.Generator) -> dict[str, float]:
        p = self.field_texture_cfg
        correlation_sigma = max(0.0, sample_range(rng, p.get("correlation_sigma_px"), 1.5))
        return {
            "amplitude_strength": max(0.0, sample_range(rng, p.get("amplitude_strength"), 0.0)),
            "phase_strength_rad": max(0.0, sample_range(rng, p.get("phase_strength_rad"), 0.0)),
            "correlation_sigma_px": correlation_sigma,
            "correlation_sigma_x_px": max(
                0.0,
                sample_range(rng, p.get("correlation_sigma_x_px"), correlation_sigma),
            ),
            "correlation_sigma_y_px": max(
                0.0,
                sample_range(rng, p.get("correlation_sigma_y_px"), correlation_sigma),
            ),
            "correlation_angle_deg": sample_range(rng, p.get("correlation_angle_deg"), 0.0),
            "illumination_strength": max(0.0, sample_range(rng, p.get("illumination_strength"), 0.0)),
            "illumination_sigma_px": max(0.0, sample_range(rng, p.get("illumination_sigma_px"), 48.0)),
            "shared_fraction": float(np.clip(sample_range(rng, p.get("shared_fraction"), 0.0), 0.0, 1.0)),
        }

    def _sample_camera_params(self, context: TransformContext) -> dict[str, Any]:
        params = self.camera_transform.sample_bundle_params(context)
        params["post_blur_sigma_px"] = max(
            0.0,
            sample_range(context.rng, self.camera_cfg.get("post_blur_sigma_px"), 0.0),
        )
        return params

    def _sample_spatial_envelope_params(
        self,
        rng: np.random.Generator,
        raw_cfg: Any,
    ) -> dict[str, Any]:
        if raw_cfg is None:
            raw_cfg = {}
        if not isinstance(raw_cfg, Mapping):
            raise TypeError("coherent ghost spatial envelopes must be JSON objects")
        p = dict(raw_cfg)
        enabled = bool(p.get("enabled", bool(p)))
        return {
            "enabled": enabled,
            "center_x_norm": sample_range(rng, p.get("center_x_norm"), 0.0),
            "center_y_norm": sample_range(rng, p.get("center_y_norm"), 0.0),
            "radius_x_norm": max(1e-6, sample_range(rng, p.get("radius_x_norm"), 1.0)),
            "radius_y_norm": max(1e-6, sample_range(rng, p.get("radius_y_norm"), 1.0)),
            "rotation_deg": sample_range(rng, p.get("rotation_deg"), 0.0),
            "order": max(0.25, sample_range(rng, p.get("order"), 2.0)),
            "floor": float(np.clip(sample_range(rng, p.get("floor"), 0.0), 0.0, 1.0)),
        }

    def _sample_single_ghost_params(
        self,
        rng: np.random.Generator,
        raw_cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        p = dict(raw_cfg)
        return {
            "amplitude": max(0.0, sample_range(rng, p.get("amplitude"), 0.0)),
            "tilt_cycles_per_frame": max(
                0.0,
                sample_range(rng, p.get("tilt_cycles_per_frame"), 0.0),
            ),
            "tilt_angle_deg": sample_range(rng, p.get("tilt_angle_deg"), 0.0),
            "defocus_delta": sample_range(rng, p.get("defocus_delta"), 0.0),
            "phase_offset_rad": sample_range(rng, p.get("phase_offset_rad"), 0.0),
            "aberration_delta": sample_range(rng, p.get("aberration_delta"), 0.0),
            "shift_x_px": sample_range(rng, p.get("shift_x_px"), 0.0),
            "shift_y_px": sample_range(rng, p.get("shift_y_px"), 0.0),
            "source_support": self._sample_spatial_envelope_params(rng, p.get("source_support")),
            "visibility_envelope": self._sample_spatial_envelope_params(
                rng,
                p.get("visibility_envelope"),
            ),
            "source_texture": self._sample_ghost_texture_params(rng, p.get("source_texture")),
        }

    def _sample_ghost_texture_params(self, rng: np.random.Generator, raw_cfg: Any) -> dict[str, Any]:
        if raw_cfg is None:
            raw_cfg = {}
        if not isinstance(raw_cfg, Mapping):
            raise TypeError("coherent ghost source_texture must be a JSON object")
        p = dict(raw_cfg)
        correlation_sigma = max(0.0, sample_range(rng, p.get("correlation_sigma_px"), 2.0))
        return {
            "enabled": bool(p.get("enabled", bool(p))),
            "amplitude_strength": max(0.0, sample_range(rng, p.get("amplitude_strength"), 0.0)),
            "phase_strength_rad": max(0.0, sample_range(rng, p.get("phase_strength_rad"), 0.0)),
            "correlation_sigma_x_px": max(
                0.0,
                sample_range(rng, p.get("correlation_sigma_x_px"), correlation_sigma),
            ),
            "correlation_sigma_y_px": max(
                0.0,
                sample_range(rng, p.get("correlation_sigma_y_px"), correlation_sigma),
            ),
            "correlation_angle_deg": sample_range(rng, p.get("correlation_angle_deg"), 0.0),
        }

    def _sample_coherent_ghost_params(self, rng: np.random.Generator) -> dict[str, Any]:
        raw_components = self.coherent_ghost_cfg.get("components")
        if raw_components is None:
            return self._sample_single_ghost_params(rng, self.coherent_ghost_cfg)
        if not isinstance(raw_components, (list, tuple)):
            raise TypeError("optical_leakage_lite coherent_ghost.components must be a JSON array")
        components: list[dict[str, Any]] = []
        for raw_component in raw_components:
            if not isinstance(raw_component, Mapping):
                raise TypeError("each coherent ghost component must be a JSON object")
            components.append(self._sample_single_ghost_params(rng, raw_component))
        return {"components": components}

    @staticmethod
    def _normalized_random_field(
        rng: np.random.Generator,
        shape: tuple[int, int],
        *,
        sigma_x_px: float,
        sigma_y_px: float,
        angle_deg: float = 0.0,
    ) -> np.ndarray:
        field = rng.normal(0.0, 1.0, shape).astype(np.float32)
        sigma_x = max(float(sigma_x_px), 0.0)
        sigma_y = max(float(sigma_y_px), 0.0)
        if sigma_x > 0.0 or sigma_y > 0.0:
            fy = np.fft.fftfreq(shape[0]).astype(np.float32)
            fx = np.fft.fftfreq(shape[1]).astype(np.float32)
            fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
            theta = np.deg2rad(float(angle_deg))
            cos_t = float(np.cos(theta))
            sin_t = float(np.sin(theta))
            fx_rot = cos_t * fxx + sin_t * fyy
            fy_rot = -sin_t * fxx + cos_t * fyy
            transfer = np.exp(
                -2.0 * (np.pi**2) * (sigma_x**2 * fx_rot**2 + sigma_y**2 * fy_rot**2)
            ).astype(np.float32)
            field = np.fft.ifft2(np.fft.fft2(field) * transfer).real.astype(np.float32)
        field = field - float(np.mean(field))
        std = float(np.std(field))
        if std <= 1e-8:
            return np.zeros(shape, dtype=np.float32)
        return (field / std).astype(np.float32)

    def _sample_texture_basis(
        self,
        rng: np.random.Generator,
        shape: tuple[int, int],
        *,
        params: Mapping[str, float],
    ) -> dict[str, np.ndarray]:
        illumination_sigma = float(params["illumination_sigma_px"])
        return {
            "amplitude": self._normalized_random_field(
                rng,
                shape,
                sigma_x_px=float(params["correlation_sigma_x_px"]),
                sigma_y_px=float(params["correlation_sigma_y_px"]),
                angle_deg=float(params["correlation_angle_deg"]),
            ),
            "phase": self._normalized_random_field(
                rng,
                shape,
                sigma_x_px=float(params["correlation_sigma_x_px"]),
                sigma_y_px=float(params["correlation_sigma_y_px"]),
                angle_deg=float(params["correlation_angle_deg"]),
            ),
            "illumination": self._normalized_random_field(
                rng,
                shape,
                sigma_x_px=illumination_sigma,
                sigma_y_px=illumination_sigma,
            ),
        }

    @staticmethod
    def _ghost_components(ghost_params: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        raw_components = ghost_params.get("components")
        if raw_components is None:
            return [ghost_params]
        if not isinstance(raw_components, (list, tuple)):
            raise TypeError("sampled coherent ghost components must be a sequence")
        return list(raw_components)

    def _sample_ghost_source_modifiers(
        self,
        rng: np.random.Generator,
        shape: tuple[int, int],
        ghost_params: Mapping[str, Any],
    ) -> list[np.ndarray]:
        modifiers: list[np.ndarray] = []
        for component in self._ghost_components(ghost_params):
            texture = component["source_texture"]
            amplitude_strength = float(texture["amplitude_strength"])
            phase_strength = float(texture["phase_strength_rad"])
            if not bool(texture["enabled"]) or (amplitude_strength <= 0.0 and phase_strength <= 0.0):
                modifiers.append(np.ones(shape, dtype=np.complex64))
                continue
            amplitude_basis = self._normalized_random_field(
                rng,
                shape,
                sigma_x_px=float(texture["correlation_sigma_x_px"]),
                sigma_y_px=float(texture["correlation_sigma_y_px"]),
                angle_deg=float(texture["correlation_angle_deg"]),
            )
            phase_basis = self._normalized_random_field(
                rng,
                shape,
                sigma_x_px=float(texture["correlation_sigma_x_px"]),
                sigma_y_px=float(texture["correlation_sigma_y_px"]),
                angle_deg=float(texture["correlation_angle_deg"]),
            )
            amplitude = np.exp(np.clip(amplitude_strength * amplitude_basis, -2.0, 2.0)).astype(np.float32)
            amplitude = amplitude / max(float(np.mean(amplitude)), 1e-8)
            modifiers.append((amplitude * np.exp(1j * phase_strength * phase_basis)).astype(np.complex64))
        return modifiers

    def _field_modifier(
        self,
        rng: np.random.Generator,
        shape: tuple[int, int],
        *,
        params: Mapping[str, float],
        shared_basis: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        amplitude_strength = float(params["amplitude_strength"])
        phase_strength = float(params["phase_strength_rad"])
        illumination_strength = float(params["illumination_strength"])
        if amplitude_strength <= 0.0 and phase_strength <= 0.0 and illumination_strength <= 0.0:
            return np.ones(shape, dtype=np.complex64)

        independent = self._sample_texture_basis(rng, shape, params=params)
        shared_fraction = float(params["shared_fraction"]) if shared_basis is not None else 0.0
        shared_weight = float(np.sqrt(shared_fraction))
        independent_weight = float(np.sqrt(max(1.0 - shared_fraction, 0.0)))

        def _combine(name: str) -> np.ndarray:
            field = independent_weight * independent[name]
            if shared_basis is not None and shared_weight > 0.0:
                field = field + shared_weight * np.asarray(shared_basis[name], dtype=np.float32)
            return np.asarray(field, dtype=np.float32)

        amplitude_log = amplitude_strength * _combine("amplitude")
        amplitude_log = amplitude_log + illumination_strength * _combine("illumination")
        amplitude_mod = np.exp(np.clip(amplitude_log, -2.0, 2.0)).astype(np.float32)
        amplitude_mod = amplitude_mod / max(float(np.mean(amplitude_mod)), 1e-8)
        phase_mod = phase_strength * _combine("phase")
        return (amplitude_mod * np.exp(1j * phase_mod)).astype(np.complex64)

    @classmethod
    def _public_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): cls._public_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._public_value(item) for item in value]
        if isinstance(value, (bool, str)):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value)
        return float(value)

    @classmethod
    def _public_params(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        public: dict[str, Any] = {}
        for key, value in params.items():
            public[str(key)] = cls._public_value(value)
        return public

    @staticmethod
    def _spatial_envelope(shape: tuple[int, int], params: Mapping[str, Any]) -> np.ndarray:
        if not bool(params.get("enabled", False)):
            return np.ones(shape, dtype=np.float32)

        y = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
        x = np.linspace(-1.0, 1.0, shape[1], dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        x_rel = xx - float(params["center_x_norm"])
        y_rel = yy - float(params["center_y_norm"])
        theta = np.deg2rad(float(params["rotation_deg"]))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        x_rot = cos_t * x_rel + sin_t * y_rel
        y_rot = -sin_t * x_rel + cos_t * y_rel
        radius_x = max(float(params["radius_x_norm"]), 1e-6)
        radius_y = max(float(params["radius_y_norm"]), 1e-6)
        order = max(float(params["order"]), 0.25)
        distance = (np.abs(x_rot) / radius_x) ** order + (np.abs(y_rot) / radius_y) ** order
        envelope = np.exp(-np.log(2.0) * distance).astype(np.float32)
        floor = float(np.clip(params.get("floor", 0.0), 0.0, 1.0))
        return (floor + (1.0 - floor) * envelope).astype(np.float32)

    def _phase_field(self, height: np.ndarray, *, modifier: np.ndarray | None = None) -> np.ndarray:
        phase = phase_from_height(self.cfg, np.asarray(height, dtype=np.float32))
        amplitude = float(getattr(self.cfg, "amplitude", 1.0) or 1.0)
        field = amplitude * np.exp(1j * phase)
        if modifier is not None:
            field = field * np.asarray(modifier, dtype=np.complex64)
        return field.astype(np.complex64)

    def _apply_coherent_ghost(
        self,
        field: np.ndarray,
        propagated: np.ndarray,
        *,
        optics_params: Mapping[str, Any],
        ghost_params: Mapping[str, Any],
        source_modifiers: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        components = self._ghost_components(ghost_params)
        active_components = [
            (index, component)
            for index, component in enumerate(components)
            if float(component["amplitude"]) > 0.0
        ]
        if not active_components:
            return propagated
        if source_modifiers is not None and len(source_modifiers) != len(components):
            raise ValueError("source_modifiers must contain one field per coherent ghost component")

        h, w = propagated.shape
        yy = np.linspace(-0.5, 0.5, h, endpoint=False, dtype=np.float32)
        xx = np.linspace(-0.5, 0.5, w, endpoint=False, dtype=np.float32)
        y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
        combined = np.asarray(propagated, dtype=np.complex64).copy()
        all_global = True
        amplitude_power = 0.0
        for component_index, component in active_components:
            amplitude = float(component["amplitude"])
            source_support_params = component["source_support"]
            visibility_params = component["visibility_envelope"]
            source_support = self._spatial_envelope((h, w), source_support_params)
            ghost_input = np.asarray(field, dtype=np.complex64) * source_support
            if source_modifiers is not None:
                ghost_input = ghost_input * np.asarray(source_modifiers[component_index], dtype=np.complex64)
            ghost = _complex_propagate(
                ghost_input,
                dx=float(self.cfg.dx),
                wavelength=float(self.cfg.wavelength),
                aperture_sigma_freq=float(optics_params["aperture_sigma_freq"]),
                numerical_aperture=float(optics_params["numerical_aperture"]),
                aperture_softness_freq=float(optics_params["aperture_softness_freq"]),
                defocus_strength=float(optics_params["defocus_strength"])
                + float(component["defocus_delta"]),
                aberration_strength=float(optics_params["aberration_strength"])
                + float(component["aberration_delta"]),
            )
            shift_x = float(component["shift_x_px"])
            shift_y = float(component["shift_y_px"])
            if shift_x != 0.0:
                ghost = self._shift_complex_x(ghost, shift_x)
            if shift_y != 0.0:
                ghost = self._shift_complex_y(ghost, shift_y)

            theta = np.deg2rad(float(component["tilt_angle_deg"]))
            coordinate = float(np.cos(theta)) * x_grid + float(np.sin(theta)) * y_grid
            phase = (
                2.0 * np.pi * float(component["tilt_cycles_per_frame"]) * coordinate
                + float(component["phase_offset_rad"])
            )
            visibility = self._spatial_envelope((h, w), visibility_params)
            ghost = ghost * visibility * np.exp(1j * phase).astype(np.complex64)
            combined = combined + amplitude * ghost
            amplitude_power += amplitude * amplitude
            all_global = all_global and not bool(source_support_params["enabled"]) and not bool(
                visibility_params["enabled"]
            )

        if all_global:
            combined = combined / np.sqrt(1.0 + amplitude_power)
        return np.asarray(combined, dtype=np.complex64)

    def _shift_complex_x(self, field: np.ndarray, shift_px: float) -> np.ndarray:
        h, w = field.shape
        fx = np.fft.fftfreq(w).astype(np.float32)
        phase = np.exp(-2j * np.pi * fx[None, :] * float(shift_px)).astype(np.complex64)
        return np.fft.ifft(np.fft.fft(field, axis=1) * phase, axis=1).astype(np.complex64)

    def _shift_complex_y(self, field: np.ndarray, shift_px: float) -> np.ndarray:
        h, w = field.shape
        fy = np.fft.fftfreq(h).astype(np.float32)
        phase = np.exp(-2j * np.pi * fy[:, None] * float(shift_px)).astype(np.complex64)
        return np.fft.ifft(np.fft.fft(field, axis=0) * phase, axis=0).astype(np.complex64)

    def _dark_port_field(
        self,
        propagated: np.ndarray,
        *,
        axis: str,
        shear_px: float,
        dark_port_params: Mapping[str, Any],
    ) -> np.ndarray:
        """Postselected dark-port field for one differential axis.

        image_shear: the classic sheared difference plus a coherent leakage of the
        undifferentiated field (imperfect postselection extinction).
        fourier_tilt: the shear element acts in the Fourier plane, so the two
        polarisation copies differ by a tilt; the dark port then multiplies the
        image field by ``2i*sin(2*pi*q*r + phase)`` plus the same leakage term,
        where ``q`` is ``fringe_cycles_per_frame`` in frame units.
        """

        shift = self._shift_complex_x if axis == "x" else self._shift_complex_y
        mode = str(dark_port_params["mode"])
        leak = float(dark_port_params["leak_amplitude"]) * np.exp(
            1j * float(dark_port_params["leak_phase_rad"])
        )
        if mode == "image_shear":
            half = 0.5 * float(shear_px)
            field = shift(propagated, +half) - shift(propagated, -half)
        else:
            h, w = propagated.shape
            yy = np.linspace(-0.5, 0.5, h, endpoint=False, dtype=np.float32)
            xx = np.linspace(-0.5, 0.5, w, endpoint=False, dtype=np.float32)
            y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
            theta = np.deg2rad(float(dark_port_params["fringe_angle_deg"]) + (90.0 if axis == "y" else 0.0))
            coordinate = float(np.cos(theta)) * x_grid + float(np.sin(theta)) * y_grid
            argument = (
                np.pi * 2.0 * float(dark_port_params["fringe_cycles_per_frame"]) * coordinate
                + float(dark_port_params["fringe_phase_rad"])
            )
            field = propagated * (2j * np.sin(argument)).astype(np.complex64)
        if float(dark_port_params["leak_amplitude"]) > 0.0:
            field = field + leak * propagated
        return np.asarray(field, dtype=np.complex64)

    def _channel_intensities(
        self,
        propagated: np.ndarray,
        *,
        params: Mapping[str, Any],
        dark_port_params: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Pure optical intensities of one coherent realization, before any camera model."""

        shear = 0.5 * float(params["shear_px"])
        raw = np.abs(propagated) ** 2
        if dark_port_params is not None and bool(dark_port_params.get("enabled", False)):
            dic_x = (
                np.abs(
                    self._dark_port_field(
                        propagated,
                        axis="x",
                        shear_px=float(params["shear_px"]),
                        dark_port_params=dark_port_params,
                    )
                )
                ** 2
            )
            dic_y = (
                np.abs(
                    self._dark_port_field(
                        propagated,
                        axis="y",
                        shear_px=float(params["shear_px"]),
                        dark_port_params=dark_port_params,
                    )
                )
                ** 2
            )
        else:
            dic_x = np.abs(self._shift_complex_x(propagated, +shear) - self._shift_complex_x(propagated, -shear)) ** 2
            dic_y = np.abs(self._shift_complex_y(propagated, +shear) - self._shift_complex_y(propagated, -shear)) ** 2
        return {
            "raw": np.asarray(raw, dtype=np.float32),
            "dic_x": np.asarray(dic_x, dtype=np.float32),
            "dic_y": np.asarray(dic_y, dtype=np.float32),
        }

    def _finalize_channels(
        self,
        intensities: Mapping[str, np.ndarray],
        *,
        params: Mapping[str, Any],
        rng: np.random.Generator,
        camera_params: Mapping[str, Any],
        frame_name: str | None = None,
    ) -> dict[str, np.ndarray]:
        raw = np.asarray(intensities["raw"], dtype=np.float32)
        dic_x = np.asarray(intensities["dic_x"], dtype=np.float32)
        dic_y = np.asarray(intensities["dic_y"], dtype=np.float32)

        raw = float(params["raw_gain"]) * raw + float(params["raw_bias"])
        raw = np.clip(raw, a_min=0.0, a_max=None).astype(np.float32)
        dic_x = np.clip(dic_x, a_min=0.0, a_max=None).astype(np.float32)
        dic_y = np.clip(dic_y, a_min=0.0, a_max=None).astype(np.float32)

        raw_blur = float(params["raw_blur_sigma_px"])
        dic_blur = float(params["dic_blur_sigma_px"])
        if raw_blur > 0.0:
            raw = gaussian_blur(raw, raw_blur)
        if dic_blur > 0.0:
            dic_x = gaussian_blur(dic_x, dic_blur)
            dic_y = gaussian_blur(dic_y, dic_blur)

        channels = {
            "I_x": dic_x.astype(np.float32),
            "I_y": dic_y.astype(np.float32),
        }
        if self.emit_raw:
            channels["I_raw"] = raw.astype(np.float32)
        context = TransformContext(
            cfg=self.cfg,
            rng=rng,
            shape=tuple(raw.shape),
            frame_name=frame_name,
        )
        channels, _ = self.camera_transform.apply(
            channels,
            context=context,
            params=camera_params,
        )
        post_blur = float(camera_params.get("post_blur_sigma_px", 0.0) or 0.0)
        if post_blur > 0.0:
            channels = {
                name: gaussian_blur(np.asarray(value, dtype=np.float32), post_blur)
                for name, value in channels.items()
            }
        return {name: np.asarray(value, dtype=np.float32) for name, value in channels.items()}

    def _mean_optical_intensities(
        self,
        height: np.ndarray,
        *,
        modifier: np.ndarray,
        reflectance_maps: list[np.ndarray | None],
        optics_params: Mapping[str, Any],
        ghost_params: Mapping[str, Any],
        ghost_source_modifiers: list[np.ndarray],
        dark_port_params: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        """Average optical intensities over the rough-phase realizations
        (partially developed speckle); the deterministic lens contribution is
        identical in every realization."""

        accumulated: dict[str, np.ndarray] | None = None
        for reflectance in reflectance_maps:
            total_modifier = np.asarray(modifier, dtype=np.complex64)
            if reflectance is not None:
                total_modifier = (total_modifier * reflectance).astype(np.complex64)
            field = self._phase_field(height, modifier=total_modifier)
            propagated = _complex_propagate(
                field,
                dx=float(self.cfg.dx),
                wavelength=float(self.cfg.wavelength),
                aperture_sigma_freq=float(optics_params["aperture_sigma_freq"]),
                numerical_aperture=float(optics_params["numerical_aperture"]),
                aperture_softness_freq=float(optics_params["aperture_softness_freq"]),
                defocus_strength=float(optics_params["defocus_strength"]),
                aberration_strength=float(optics_params["aberration_strength"]),
            )
            propagated = self._apply_coherent_ghost(
                field,
                propagated,
                optics_params=optics_params,
                ghost_params=ghost_params,
                source_modifiers=ghost_source_modifiers,
            )
            intensities = self._channel_intensities(
                propagated,
                params=optics_params,
                dark_port_params=dark_port_params,
            )
            if accumulated is None:
                accumulated = {name: value.copy() for name, value in intensities.items()}
            else:
                for name, value in intensities.items():
                    accumulated[name] += value
        assert accumulated is not None
        count = float(max(1, len(reflectance_maps)))
        return {name: (value / count).astype(np.float32) for name, value in accumulated.items()}

    def _reflectance_realizations(
        self,
        rng: np.random.Generator,
        shape: tuple[int, int],
        params: Mapping[str, Any],
    ) -> list[np.ndarray | None]:
        if not bool(params.get("enabled", False)):
            return [None]
        count = max(1, int(params.get("speckle_realizations", 1)))

        # Dust/micro-pit positions are a fixed property of the lens under test:
        # draw them once per capture; only their speckle phase varies with the
        # coherence realization. Direct polar sampling is uniform over the
        # disc, consumes a fixed number of draws, and cannot hang for small
        # apertures (the rejection loop it replaces could).
        point_positions: tuple[np.ndarray, np.ndarray] | None = None
        point_count = int(params.get("lens_point_scatter_count", 0))
        if point_count > 0:
            h, w = shape
            fraction = getattr(self.cfg, "lens_radius_fraction", 1.0)
            fraction = 1.0 if fraction is None else float(fraction)
            radius_px = max(fraction, 0.0) * 0.5 * min(h, w)
            if radius_px >= 1.0:
                radial = 0.97 * radius_px * np.sqrt(rng.random(point_count))
                theta = 2.0 * np.pi * rng.random(point_count)
                ys = np.clip(
                    np.round(0.5 * (h - 1) + radial * np.sin(theta)).astype(np.int64), 0, h - 1
                )
                xs = np.clip(
                    np.round(0.5 * (w - 1) + radial * np.cos(theta)).astype(np.int64), 0, w - 1
                )
                point_positions = (ys, xs)

        # Static (per-capture) lens-surface texture fields shared by every
        # coherence realization: polishing/molding residual and contamination
        # scattering are fixed surface properties, so their contribution must
        # not be averaged away as K grows.
        static_lens_fields: dict[str, np.ndarray] | None = None
        lens_sigma = float(params["lens_texture_sigma_px"])
        needs_rough = float(params.get("lens_phase_rough_rad", 0.0)) > 0.0
        needs_scatter = float(params.get("lens_scatter_amplitude", 0.0)) > 0.0
        if needs_rough or needs_scatter:
            static_lens_fields = {}
            if needs_rough:
                static_lens_fields["lens_rough_basis"] = self._normalized_random_field(
                    rng, shape, sigma_x_px=lens_sigma, sigma_y_px=lens_sigma
                )
            if needs_scatter:
                static_lens_fields["lens_scatter_basis"] = self._normalized_random_field(
                    rng, shape, sigma_x_px=lens_sigma, sigma_y_px=lens_sigma
                )

        return [
            self._reflectance_modifier(
                rng,
                shape,
                params,
                point_positions=point_positions,
                static_lens_fields=static_lens_fields,
            )
            for _ in range(count)
        ]

    def simulate_capture(
        self,
        height: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        meta: dict[str, Any] | None = None,
        extra_field_modifier: np.ndarray | None = None,
    ) -> Capture:
        """`extra_field_modifier` is an optional complex amplitude/phase map applied
        to the object field, used to model scattering defects (local roughness or
        reflectance change) that a pure height perturbation cannot represent."""

        if rng is None:
            rng = np.random.default_rng()
        params = self._sample_params(rng)
        shape = tuple(np.asarray(height).shape[-2:])
        texture_params = self._sample_field_texture_params(rng)
        ghost_params = self._sample_coherent_ghost_params(rng)
        ghost_source_modifiers = self._sample_ghost_source_modifiers(rng, shape, ghost_params)
        reflectance_params = self._sample_reflectance_params(rng)
        dark_port_params = self._sample_dark_port_params(rng)
        modifier = self._field_modifier(rng, shape, params=texture_params)
        if extra_field_modifier is not None:
            modifier = (
                np.asarray(modifier, dtype=np.complex64)
                * np.asarray(extra_field_modifier, dtype=np.complex64)
            ).astype(np.complex64)
        reflectance_maps = self._reflectance_realizations(rng, shape, reflectance_params)
        camera_context = TransformContext(cfg=self.cfg, rng=rng, shape=shape)
        camera_params = self._sample_camera_params(camera_context)
        intensities = self._mean_optical_intensities(
            np.asarray(height, dtype=np.float32),
            modifier=np.asarray(modifier, dtype=np.complex64),
            reflectance_maps=reflectance_maps,
            optics_params=params,
            ghost_params=ghost_params,
            ghost_source_modifiers=ghost_source_modifiers,
            dark_port_params=dark_port_params,
        )
        channels = self._finalize_channels(
            intensities,
            params=params,
            rng=rng,
            camera_params=camera_params,
        )
        capture_meta: dict[str, Any] = dict(meta or {})
        capture_meta.update(self.meta())
        capture_meta["sampled_params"] = {
            "optics": self._public_params(params),
            "field_texture": self._public_params(texture_params),
            "coherent_ghost": self._public_params(ghost_params),
            "reflectance": self._public_params(reflectance_params),
            "dark_port": self._public_params(dark_port_params),
            "camera": self._public_params(camera_params),
        }
        return Capture(channels=channels, meta=capture_meta)

    def simulate_bundle(
        self,
        heights: Mapping[str, np.ndarray],
        *,
        rng: np.random.Generator | None = None,
        meta: dict[str, Any] | None = None,
        extra_field_modifiers: Mapping[str, np.ndarray] | None = None,
    ) -> CaptureBundle:
        if not heights:
            raise ValueError("simulate_bundle requires at least one height frame")
        if rng is None:
            rng = np.random.default_rng()
        params = self._sample_params(rng)
        first = next(iter(heights.values()))
        shape = tuple(np.asarray(first).shape[-2:])
        for height in heights.values():
            if tuple(np.asarray(height).shape[-2:]) != shape:
                raise ValueError("All bundle frames must share the same spatial shape")
        texture_params = self._sample_field_texture_params(rng)
        ghost_params = self._sample_coherent_ghost_params(rng)
        ghost_source_modifiers = self._sample_ghost_source_modifiers(rng, shape, ghost_params)
        reflectance_params = self._sample_reflectance_params(rng)
        dark_port_params = self._sample_dark_port_params(rng)
        # The reflectance rough-phase realizations are a fixture property:
        # share the same set across every frame of the bundle.
        reflectance_maps = self._reflectance_realizations(rng, shape, reflectance_params)
        texture_enabled = any(
            float(texture_params[name]) > 0.0
            for name in ("amplitude_strength", "phase_strength_rad", "illumination_strength")
        )
        shared_basis = None
        if texture_enabled and float(texture_params["shared_fraction"]) > 0.0:
            shared_basis = self._sample_texture_basis(rng, shape, params=texture_params)
        camera_context = TransformContext(cfg=self.cfg, rng=rng, shape=shape)
        camera_params = self._sample_camera_params(camera_context)
        captures: dict[str, Capture] = {}
        for frame_name, height in heights.items():
            modifier = self._field_modifier(
                rng,
                shape,
                params=texture_params,
                shared_basis=shared_basis,
            )
            extra = None if extra_field_modifiers is None else extra_field_modifiers.get(str(frame_name))
            if extra is not None:
                modifier = (
                    np.asarray(modifier, dtype=np.complex64) * np.asarray(extra, dtype=np.complex64)
                ).astype(np.complex64)
            intensities = self._mean_optical_intensities(
                np.asarray(height, dtype=np.float32),
                modifier=np.asarray(modifier, dtype=np.complex64),
                reflectance_maps=reflectance_maps,
                optics_params=params,
                ghost_params=ghost_params,
                ghost_source_modifiers=ghost_source_modifiers,
                dark_port_params=dark_port_params,
            )
            channels = self._finalize_channels(
                intensities,
                params=params,
                rng=rng,
                camera_params=camera_params,
                frame_name=str(frame_name),
            )
            captures[str(frame_name)] = Capture(
                channels=channels,
                meta={
                    "frame_name": str(frame_name),
                    "sampled_params": {
                        "optics": self._public_params(params),
                        "field_texture": self._public_params(texture_params),
                        "coherent_ghost": self._public_params(ghost_params),
                        "reflectance": self._public_params(reflectance_params),
                        "dark_port": self._public_params(dark_port_params),
                        "camera": self._public_params(camera_params),
                    },
                },
            )
        bundle_meta: dict[str, Any] = dict(meta or {})
        bundle_meta.update(self.meta())
        bundle_meta["sampled_bundle_params"] = {
            "optics": self._public_params(params),
            "field_texture": self._public_params(texture_params),
            "coherent_ghost": self._public_params(ghost_params),
            "reflectance": self._public_params(reflectance_params),
            "dark_port": self._public_params(dark_port_params),
            "camera": self._public_params(camera_params),
        }
        return CaptureBundle(captures=captures, meta=bundle_meta)

    def meta(self) -> dict[str, Any]:
        return {
            "engine_name": self.name,
            "engine_version": self.version,
            "base_model": "scalar_phase_propagation",
            "emit_raw": bool(self.emit_raw),
            "phase_model": phase_model_meta(self.cfg),
            "coherent_pupil_cutoff": (
                float(self.cfg.numerical_aperture) / float(self.cfg.wavelength)
                if float(getattr(self.cfg, "numerical_aperture", 0.0) or 0.0) > 0.0
                else None
            ),
            "params": self.params,
        }


__all__ = ["OpticalLeakageLiteEngine"]
