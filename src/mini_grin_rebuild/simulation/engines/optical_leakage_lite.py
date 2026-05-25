from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.simulation.engines.ideal_gradient import phase_from_height
from mini_grin_rebuild.simulation.transforms.utils import gaussian_blur, sample_range
from mini_grin_rebuild.simulation.types import Capture, CaptureBundle


def _complex_propagate(
    field: np.ndarray,
    *,
    dx: float,
    wavelength: float,
    aperture_sigma_freq: float,
    defocus_strength: float,
    aberration_strength: float,
) -> np.ndarray:
    h, w = field.shape
    fy = np.fft.fftfreq(h, d=max(dx, 1e-12)).astype(np.float32)
    fx = np.fft.fftfreq(w, d=max(dx, 1e-12)).astype(np.float32)
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    rr2 = fxx**2 + fyy**2

    field_fft = np.fft.fft2(field)

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
    version = "1"

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        raw_params = getattr(cfg, "capture_engine_params", {}) or {}
        if not isinstance(raw_params, Mapping):
            raise TypeError("simulation.capture_engine_params must be a JSON object for optical_leakage_lite")
        self.params = dict(raw_params)
        self.emit_raw = bool(self.params.get("emit_raw", True))

    def _sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        p = self.params
        return {
            "defocus_strength": sample_range(rng, p.get("defocus_strength"), 0.0),
            "aperture_sigma_freq": sample_range(rng, p.get("aperture_sigma_freq"), 0.35 / max(self.cfg.dx, 1e-12)),
            "aberration_strength": sample_range(rng, p.get("aberration_strength"), 0.0),
            "raw_blur_sigma_px": sample_range(rng, p.get("raw_blur_sigma_px"), 0.0),
            "dic_blur_sigma_px": sample_range(rng, p.get("dic_blur_sigma_px"), 0.0),
            "raw_gain": sample_range(rng, p.get("raw_gain"), 1.0),
            "raw_bias": sample_range(rng, p.get("raw_bias"), 0.0),
            "shear_px": max(0.5, sample_range(rng, p.get("shear_px"), 1.0)),
        }

    def _phase_field(self, height: np.ndarray) -> np.ndarray:
        phase = phase_from_height(self.cfg, np.asarray(height, dtype=np.float32))
        amplitude = float(getattr(self.cfg, "amplitude", 1.0) or 1.0)
        field = amplitude * np.exp(1j * phase)
        return field.astype(np.complex64)

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

    def _capture_from_field(self, propagated: np.ndarray, *, params: Mapping[str, Any]) -> dict[str, np.ndarray]:
        shear = 0.5 * float(params["shear_px"])
        raw = np.abs(propagated) ** 2
        dic_x = np.abs(self._shift_complex_x(propagated, +shear) - self._shift_complex_x(propagated, -shear)) ** 2
        dic_y = np.abs(self._shift_complex_y(propagated, +shear) - self._shift_complex_y(propagated, -shear)) ** 2

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
        return channels

    def simulate_capture(
        self,
        height: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Capture:
        if rng is None:
            rng = np.random.default_rng()
        params = self._sample_params(rng)
        field = self._phase_field(height)
        propagated = _complex_propagate(
            field,
            dx=float(self.cfg.dx),
            wavelength=float(self.cfg.wavelength),
            aperture_sigma_freq=float(params["aperture_sigma_freq"]),
            defocus_strength=float(params["defocus_strength"]),
            aberration_strength=float(params["aberration_strength"]),
        )
        channels = self._capture_from_field(propagated, params=params)
        capture_meta: dict[str, Any] = dict(meta or {})
        capture_meta.update(self.meta())
        capture_meta["sampled_params"] = {k: float(v) for k, v in params.items()}
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
        params = self._sample_params(rng)
        captures: dict[str, Capture] = {}
        for frame_name, height in heights.items():
            field = self._phase_field(height)
            propagated = _complex_propagate(
                field,
                dx=float(self.cfg.dx),
                wavelength=float(self.cfg.wavelength),
                aperture_sigma_freq=float(params["aperture_sigma_freq"]),
                defocus_strength=float(params["defocus_strength"]),
                aberration_strength=float(params["aberration_strength"]),
            )
            channels = self._capture_from_field(propagated, params=params)
            captures[str(frame_name)] = Capture(
                channels=channels,
                meta={
                    "frame_name": str(frame_name),
                    "sampled_params": {k: float(v) for k, v in params.items()},
                },
            )
        bundle_meta: dict[str, Any] = dict(meta or {})
        bundle_meta.update(self.meta())
        bundle_meta["sampled_bundle_params"] = {k: float(v) for k, v in params.items()}
        return CaptureBundle(captures=captures, meta=bundle_meta)

    def meta(self) -> dict[str, Any]:
        return {
            "engine_name": self.name,
            "engine_version": self.version,
            "base_model": "scalar_phase_propagation",
            "emit_raw": bool(self.emit_raw),
            "params": self.params,
        }


__all__ = ["OpticalLeakageLiteEngine"]
