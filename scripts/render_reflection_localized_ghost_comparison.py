from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_capture import _normalise_observation, _physical_radial_grid  # noqa: E402
from mini_grin_rebuild.core.configs import SimulationConfig, load_experiment_config  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_reference  # noqa: E402
from mini_grin_rebuild.simulation.engines.optical_leakage_lite import (  # noqa: E402
    OpticalLeakageLiteEngine,
    _complex_propagate,
)


def _high_pass(image: np.ndarray, sigma_px: float = 5.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return (arr - gaussian_filter(arr, sigma=float(sigma_px), mode="reflect")).astype(np.float32)


def _local_energy(signal: np.ndarray, sigma_px: float = 12.0) -> np.ndarray:
    power = gaussian_filter(np.asarray(signal, dtype=np.float32) ** 2, sigma=float(sigma_px), mode="reflect")
    return np.sqrt(np.maximum(power, 0.0)).astype(np.float32)


def _normalised_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, shape[1], dtype=np.float32)
    return np.meshgrid(y, x, indexing="ij")


def _weighted_geometry(energy: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    yy, xx = _normalised_grid(tuple(energy.shape))
    values = np.asarray(energy, dtype=np.float64).copy()
    values[~mask] = 0.0
    floor = float(np.quantile(values[mask], 0.2))
    values = np.clip(values - floor, 0.0, None)
    total = float(np.sum(values))
    if total <= 1e-12:
        return {
            "center_x_norm": 0.0,
            "center_y_norm": 0.0,
            "major_sigma_norm": 1.0,
            "minor_sigma_norm": 1.0,
            "major_angle_deg": 0.0,
        }
    center_x = float(np.sum(values * xx) / total)
    center_y = float(np.sum(values * yy) / total)
    dx = np.asarray(xx - center_x, dtype=np.float64)
    dy = np.asarray(yy - center_y, dtype=np.float64)
    covariance = np.asarray(
        [
            [np.sum(values * dx * dx) / total, np.sum(values * dx * dy) / total],
            [np.sum(values * dx * dy) / total, np.sum(values * dy * dy) / total],
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_index, minor_index = int(order[0]), int(order[1])
    major_vector = eigenvectors[:, major_index]
    return {
        "center_x_norm": center_x,
        "center_y_norm": center_y,
        "major_sigma_norm": float(np.sqrt(max(eigenvalues[major_index], 0.0))),
        "minor_sigma_norm": float(np.sqrt(max(eigenvalues[minor_index], 0.0))),
        "major_angle_deg": float(np.degrees(np.arctan2(major_vector[1], major_vector[0]))),
    }


def _ellipse_params(geometry: Mapping[str, float]) -> dict[str, float | bool]:
    return {
        "enabled": True,
        "center_x_norm": float(geometry["center_x_norm"]),
        "center_y_norm": float(geometry["center_y_norm"]),
        "radius_x_norm": float(np.clip(1.35 * geometry["major_sigma_norm"], 0.18, 0.60)),
        "radius_y_norm": float(np.clip(1.35 * geometry["minor_sigma_norm"], 0.16, 0.55)),
        "rotation_deg": float(geometry["major_angle_deg"]),
        "order": 4.0,
        "floor": 0.0,
    }


def _spectral_descriptor(
    signal: np.ndarray,
    spatial_weight: np.ndarray,
    *,
    low_cycles: float = 48.0,
    high_cycles: float = 128.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    arr = np.asarray(signal, dtype=np.float32) * np.asarray(spatial_weight, dtype=np.float32)
    arr = arr - float(np.mean(arr))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    spectrum = np.asarray(spectrum, dtype=np.float64)
    h, w = arr.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h) * h)
    fx = np.fft.fftshift(np.fft.fftfreq(w) * w)
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    radial = np.sqrt(fxx**2 + fyy**2)
    band = (radial >= float(low_cycles)) & (radial <= float(high_cycles))
    descriptor = np.zeros_like(spectrum, dtype=np.float64)
    logged = np.log1p(spectrum)
    if np.any(band):
        values = logged[band]
        descriptor[band] = (values - float(np.mean(values))) / max(float(np.std(values)), 1e-12)

    peak_power = spectrum.copy()
    peak_power[~band] = 0.0
    peaks: list[dict[str, float]] = []
    suppressed = peak_power.copy()
    for _ in range(10):
        flat_index = int(np.argmax(suppressed))
        peak_value = float(suppressed.flat[flat_index])
        if peak_value <= 0.0:
            break
        iy, ix = np.unravel_index(flat_index, suppressed.shape)
        frequency = float(radial[iy, ix])
        angle = float(np.mod(np.degrees(np.arctan2(fyy[iy, ix], fxx[iy, ix])), 180.0))
        if all(
            abs(frequency - item["cycles_per_frame"]) >= 6.0
            or min(abs(angle - item["angle_deg"]), 180.0 - abs(angle - item["angle_deg"])) >= 8.0
            for item in peaks
        ):
            peaks.append(
                {
                    "cycles_per_frame": frequency,
                    "cycles_per_pixel": frequency / float(max(h, w)),
                    "period_px": float(max(h, w)) / max(frequency, 1e-12),
                    "angle_deg": angle,
                    "relative_power": peak_value / max(float(np.max(peak_power)), 1e-12),
                }
            )
        radius = 7
        y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
        x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
        suppressed[y0:y1, x0:x1] = 0.0
        mirror_y = h - 1 - iy
        mirror_x = w - 1 - ix
        y0, y1 = max(0, mirror_y - radius), min(h, mirror_y + radius + 1)
        x0, x1 = max(0, mirror_x - radius), min(w, mirror_x + radius + 1)
        suppressed[y0:y1, x0:x1] = 0.0
        if len(peaks) >= 6:
            break
    return descriptor.astype(np.float32), band, peaks


def _correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(a[mask], dtype=np.float64)
    y = np.asarray(b[mask], dtype=np.float64)
    if x.size < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _ellipse_mask(shape: tuple[int, int], params: Mapping[str, Any], scale: float = 1.0) -> np.ndarray:
    yy, xx = _normalised_grid(shape)
    x_rel = xx - float(params["center_x_norm"])
    y_rel = yy - float(params["center_y_norm"])
    theta = np.deg2rad(float(params["rotation_deg"]))
    x_rot = float(np.cos(theta)) * x_rel + float(np.sin(theta)) * y_rel
    y_rot = -float(np.sin(theta)) * x_rel + float(np.cos(theta)) * y_rel
    radius_x = max(float(params["radius_x_norm"]) * float(scale), 1e-6)
    radius_y = max(float(params["radius_y_norm"]) * float(scale), 1e-6)
    return ((x_rot / radius_x) ** 2 + (y_rot / radius_y) ** 2) <= 1.0


def _prepare_base(cfg: SimulationConfig) -> tuple[OpticalLeakageLiteEngine, dict[str, Any], np.ndarray, np.ndarray]:
    params = dict(cfg.capture_engine_params)
    params.update(
        {
            "defocus_strength": -150.0,
            "raw_blur_sigma_px": 0.0,
            "dic_blur_sigma_px": 0.0,
            "field_texture": {},
            "camera": {},
            "coherent_ghost": {},
        }
    )
    fit_cfg = replace(cfg, capture_engine="optical_leakage_lite", capture_engine_params=params)
    engine = OpticalLeakageLiteEngine(fit_cfg)
    rng = np.random.default_rng(20260721)
    optics = engine._sample_params(rng)
    height = microlens_reference(fit_cfg)
    field = engine._phase_field(height)
    propagated = _complex_propagate(
        field,
        dx=float(fit_cfg.dx),
        wavelength=float(fit_cfg.wavelength),
        aperture_sigma_freq=float(optics["aperture_sigma_freq"]),
        numerical_aperture=float(optics["numerical_aperture"]),
        aperture_softness_freq=float(optics["aperture_softness_freq"]),
        defocus_strength=float(optics["defocus_strength"]),
        aberration_strength=float(optics["aberration_strength"]),
    )
    return engine, optics, field, propagated


def _render_component(
    engine: OpticalLeakageLiteEngine,
    optics: Mapping[str, Any],
    field: np.ndarray,
    propagated: np.ndarray,
    component: Mapping[str, Any],
    rho: np.ndarray,
    *,
    texture_seed: int = 0,
) -> np.ndarray:
    source_modifiers = engine._sample_ghost_source_modifiers(
        np.random.default_rng(int(texture_seed)),
        tuple(propagated.shape),
        component,
    )
    combined = engine._apply_coherent_ghost(
        field,
        propagated,
        optics_params=optics,
        ghost_params=component,
        source_modifiers=source_modifiers,
    )
    raw = np.abs(combined) ** 2
    raw = gaussian_filter(np.asarray(raw, dtype=np.float32), sigma=1.0, mode="reflect")
    return _normalise_observation(raw, rho)


def _candidate_component(
    envelope: Mapping[str, Any],
    *,
    amplitude: float,
    frequency: float,
    angle: float,
    defocus_delta: float,
    phase_offset: float,
) -> dict[str, Any]:
    source = dict(envelope)
    source["radius_x_norm"] = min(0.9, 1.35 * float(source["radius_x_norm"]))
    source["radius_y_norm"] = min(0.9, 1.35 * float(source["radius_y_norm"]))
    source["order"] = 2.0
    return {
        "amplitude": float(amplitude),
        "tilt_cycles_per_frame": float(frequency),
        "tilt_angle_deg": float(angle),
        "defocus_delta": float(defocus_delta),
        "phase_offset_rad": float(phase_offset),
        "aberration_delta": 0.0,
        "shift_x_px": 0.0,
        "shift_y_px": 0.0,
        "source_support": source,
        "visibility_envelope": dict(envelope),
        "source_texture": {
            "enabled": False,
            "amplitude_strength": 0.0,
            "phase_strength_rad": 0.0,
            "correlation_sigma_x_px": 1.0,
            "correlation_sigma_y_px": 1.0,
            "correlation_angle_deg": 0.0,
        },
    }


def _ghost_components(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_components = model.get("components")
    if raw_components is None:
        return [copy.deepcopy(dict(model))]
    return [copy.deepcopy(dict(component)) for component in raw_components]


def _scale_ghost_amplitudes(model: Mapping[str, Any], scale: float) -> dict[str, Any]:
    components = _ghost_components(model)
    for component in components:
        component["amplitude"] = float(
            np.clip(float(component["amplitude"]) * float(scale), 0.002, 0.25)
        )
    if model.get("components") is None:
        return components[0]
    return {"components": components}


def _fit_target(
    *,
    name: str,
    target_signal: np.ndarray,
    target_energy: np.ndarray,
    target_geometry: Mapping[str, float],
    engine: OpticalLeakageLiteEngine,
    optics: Mapping[str, Any],
    field: np.ndarray,
    propagated: np.ndarray,
    sim_clean: np.ndarray,
    rho: np.ndarray,
) -> dict[str, Any]:
    aperture_mask = rho <= 0.82
    envelope = _ellipse_params(target_geometry)
    target_weight = np.asarray(target_energy, dtype=np.float32)
    target_weight = target_weight / max(float(np.quantile(target_weight[aperture_mask], 0.99)), 1e-8)
    target_weight = np.clip(target_weight, 0.0, 1.0)
    target_descriptor, spectral_band, target_peaks = _spectral_descriptor(target_signal, target_weight)
    peak_candidates = target_peaks[:6] or [
        {"cycles_per_frame": 80.0, "angle_deg": 0.0},
        {"cycles_per_frame": 100.0, "angle_deg": 45.0},
    ]

    records: list[dict[str, Any]] = []
    base_amplitude = 0.05
    for peak in peak_candidates:
        for defocus_delta in (-80.0, -40.0, 0.0, 40.0, 80.0):
            for phase_offset in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
                component = _candidate_component(
                    envelope,
                    amplitude=base_amplitude,
                    frequency=float(peak["cycles_per_frame"]),
                    angle=float(peak["angle_deg"]),
                    defocus_delta=defocus_delta,
                    phase_offset=phase_offset,
                )
                simulated = _render_component(engine, optics, field, propagated, component, rho)
                signal = _high_pass(simulated) - _high_pass(sim_clean)
                energy = _local_energy(signal)
                descriptor, _, sim_peaks = _spectral_descriptor(signal, target_weight)
                energy_corr = _correlation(target_energy, energy, aperture_mask)
                spectrum_corr = _correlation(target_descriptor, descriptor, spectral_band)
                geometry = _weighted_geometry(energy, aperture_mask)
                center_distance = float(
                    np.hypot(
                        geometry["center_x_norm"] - target_geometry["center_x_norm"],
                        geometry["center_y_norm"] - target_geometry["center_y_norm"],
                    )
                )
                score = 0.62 * energy_corr + 0.30 * spectrum_corr - 0.08 * center_distance
                records.append(
                    {
                        "score": float(score),
                        "energy_corr": float(energy_corr),
                        "spectrum_corr": float(spectrum_corr),
                        "center_distance_norm": center_distance,
                        "component": component,
                        "simulated": simulated,
                        "signal": signal,
                        "energy": energy,
                        "geometry": geometry,
                        "spectral_peaks": sim_peaks,
                        "texture_seed": 0,
                    }
                )

    smooth_best = max(records, key=lambda record: float(record["score"]))
    texture_profiles = [
        {
            "enabled": True,
            "amplitude_strength": 0.12,
            "phase_strength_rad": 0.25,
            "correlation_sigma_x_px": 2.5,
            "correlation_sigma_y_px": 1.0,
            "correlation_angle_deg": float(smooth_best["component"]["tilt_angle_deg"]),
        },
        {
            "enabled": True,
            "amplitude_strength": 0.20,
            "phase_strength_rad": 0.50,
            "correlation_sigma_x_px": 3.5,
            "correlation_sigma_y_px": 1.2,
            "correlation_angle_deg": float(smooth_best["component"]["tilt_angle_deg"]),
        },
        {
            "enabled": True,
            "amplitude_strength": 0.28,
            "phase_strength_rad": 0.80,
            "correlation_sigma_x_px": 2.0,
            "correlation_sigma_y_px": 0.8,
            "correlation_angle_deg": float(smooth_best["component"]["tilt_angle_deg"]),
        },
        {
            "enabled": True,
            "amplitude_strength": 0.18,
            "phase_strength_rad": 1.10,
            "correlation_sigma_x_px": 1.4,
            "correlation_sigma_y_px": 1.4,
            "correlation_angle_deg": 0.0,
        },
    ]
    textured_records: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(texture_profiles):
        for seed_index in range(4):
            texture_seed = 1300 + 100 * int(name.split(".")[0]) + 10 * profile_index + seed_index
            component = copy.deepcopy(smooth_best["component"])
            component["source_texture"] = dict(profile)
            simulated = _render_component(
                engine,
                optics,
                field,
                propagated,
                component,
                rho,
                texture_seed=texture_seed,
            )
            signal = _high_pass(simulated) - _high_pass(sim_clean)
            energy = _local_energy(signal)
            descriptor, _, sim_peaks = _spectral_descriptor(signal, target_weight)
            energy_corr = _correlation(target_energy, energy, aperture_mask)
            spectrum_corr = _correlation(target_descriptor, descriptor, spectral_band)
            geometry = _weighted_geometry(energy, aperture_mask)
            center_distance = float(
                np.hypot(
                    geometry["center_x_norm"] - target_geometry["center_x_norm"],
                    geometry["center_y_norm"] - target_geometry["center_y_norm"],
                )
            )
            score = 0.48 * energy_corr + 0.47 * spectrum_corr - 0.05 * center_distance
            textured_records.append(
                {
                    "score": float(score),
                    "energy_corr": float(energy_corr),
                    "spectrum_corr": float(spectrum_corr),
                    "center_distance_norm": center_distance,
                    "component": component,
                    "simulated": simulated,
                    "signal": signal,
                    "energy": energy,
                    "geometry": geometry,
                    "spectral_peaks": sim_peaks,
                    "texture_seed": texture_seed,
                }
            )
    textured_best = max([smooth_best, *textured_records], key=lambda record: float(record["score"]))
    primary_angle = float(textured_best["component"]["tilt_angle_deg"])
    primary_frequency = float(textured_best["component"]["tilt_cycles_per_frame"])
    secondary_peaks = [
        peak
        for peak in target_peaks
        if min(
            abs(float(peak["angle_deg"]) - primary_angle),
            180.0 - abs(float(peak["angle_deg"]) - primary_angle),
        )
        >= 18.0
        or abs(float(peak["cycles_per_frame"]) - primary_frequency) >= 9.0
    ][:3]
    multi_records: list[dict[str, Any]] = []
    for peak_index, peak in enumerate(secondary_peaks):
        for amplitude_ratio in (0.18, 0.32, 0.48):
            for phase_offset in (0.0, 0.5 * np.pi):
                primary = copy.deepcopy(textured_best["component"])
                secondary_envelope = dict(envelope)
                secondary_envelope["center_x_norm"] = float(secondary_envelope["center_x_norm"]) + 0.035
                secondary_envelope["center_y_norm"] = float(secondary_envelope["center_y_norm"]) + 0.025
                secondary_envelope["radius_x_norm"] = min(
                    0.75,
                    1.15 * float(secondary_envelope["radius_x_norm"]),
                )
                secondary_envelope["radius_y_norm"] = min(
                    0.70,
                    1.15 * float(secondary_envelope["radius_y_norm"]),
                )
                secondary_envelope["order"] = 2.0
                secondary = _candidate_component(
                    secondary_envelope,
                    amplitude=float(primary["amplitude"]) * amplitude_ratio,
                    frequency=float(peak["cycles_per_frame"]),
                    angle=float(peak["angle_deg"]),
                    defocus_delta=float(primary["defocus_delta"]) + 40.0,
                    phase_offset=phase_offset,
                )
                secondary["source_texture"] = {
                    "enabled": True,
                    "amplitude_strength": 0.12,
                    "phase_strength_rad": 0.35,
                    "correlation_sigma_x_px": 2.0,
                    "correlation_sigma_y_px": 1.0,
                    "correlation_angle_deg": float(peak["angle_deg"]),
                }
                model = {"components": [primary, secondary]}
                texture_seed = 3100 + 100 * int(name.split(".")[0]) + 10 * peak_index + int(100 * amplitude_ratio) + int(phase_offset > 0.0)
                simulated = _render_component(
                    engine,
                    optics,
                    field,
                    propagated,
                    model,
                    rho,
                    texture_seed=texture_seed,
                )
                signal = _high_pass(simulated) - _high_pass(sim_clean)
                energy = _local_energy(signal)
                descriptor, _, sim_peaks = _spectral_descriptor(signal, target_weight)
                energy_corr = _correlation(target_energy, energy, aperture_mask)
                spectrum_corr = _correlation(target_descriptor, descriptor, spectral_band)
                geometry = _weighted_geometry(energy, aperture_mask)
                center_distance = float(
                    np.hypot(
                        geometry["center_x_norm"] - target_geometry["center_x_norm"],
                        geometry["center_y_norm"] - target_geometry["center_y_norm"],
                    )
                )
                score = 0.42 * energy_corr + 0.54 * spectrum_corr - 0.04 * center_distance
                multi_records.append(
                    {
                        "score": float(score),
                        "energy_corr": float(energy_corr),
                        "spectrum_corr": float(spectrum_corr),
                        "center_distance_norm": center_distance,
                        "component": model,
                        "simulated": simulated,
                        "signal": signal,
                        "energy": energy,
                        "geometry": geometry,
                        "spectral_peaks": sim_peaks,
                        "texture_seed": texture_seed,
                    }
                )
    best = max([textured_best, *multi_records], key=lambda record: float(record["score"]))
    local_mask = _ellipse_mask(tuple(target_signal.shape), envelope, scale=1.0) & aperture_mask
    target_std = float(np.std(target_signal[local_mask]))
    simulated_std = float(np.std(np.asarray(best["signal"])[local_mask]))
    amplitude_scale = target_std / max(simulated_std, 1e-8)
    calibrated_component = _scale_ghost_amplitudes(best["component"], amplitude_scale)
    final_simulated = _render_component(
        engine,
        optics,
        field,
        propagated,
        calibrated_component,
        rho,
        texture_seed=int(best["texture_seed"]),
    )
    final_signal = _high_pass(final_simulated) - _high_pass(sim_clean)
    final_energy = _local_energy(final_signal)
    final_descriptor, _, final_peaks = _spectral_descriptor(final_signal, target_weight)
    final_geometry = _weighted_geometry(final_energy, aperture_mask)
    inside = _ellipse_mask(tuple(target_signal.shape), envelope, scale=1.0) & aperture_mask
    outside = (~_ellipse_mask(tuple(target_signal.shape), envelope, scale=1.45)) & aperture_mask
    final_metrics = {
        "energy_map_correlation": _correlation(target_energy, final_energy, aperture_mask),
        "spectrum_correlation": _correlation(target_descriptor, final_descriptor, spectral_band),
        "target_local_highpass_std": target_std,
        "simulated_local_highpass_std": float(np.std(final_signal[inside])),
        "target_outside_highpass_std": float(np.std(target_signal[outside])),
        "simulated_outside_highpass_std": float(np.std(final_signal[outside])),
        "target_local_energy_fraction": float(np.sum(target_energy[inside]) / max(float(np.sum(target_energy[aperture_mask])), 1e-12)),
        "simulated_local_energy_fraction": float(np.sum(final_energy[inside]) / max(float(np.sum(final_energy[aperture_mask])), 1e-12)),
    }
    return {
        "name": name,
        "target_geometry": dict(target_geometry),
        "visibility_envelope": envelope,
        "target_spectral_peaks": target_peaks,
        "component": calibrated_component,
        "source_texture_seed": int(best["texture_seed"]),
        "search_best_score_before_amplitude_calibration": float(best["score"]),
        "metrics": final_metrics,
        "simulated": final_simulated,
        "target_signal": target_signal,
        "simulated_signal": final_signal,
        "target_energy": target_energy,
        "simulated_energy": final_energy,
        "simulated_geometry": final_geometry,
        "simulated_spectral_peaks": final_peaks,
        "target_descriptor": target_descriptor,
        "simulated_descriptor": final_descriptor,
        "spectral_band": spectral_band,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (bool, str)) or value is None:
        return value
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit localized coherent ghost propagation to real frames 13 and 14.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "reflection_microlens520_actual.json")
    parser.add_argument(
        "--empirical-npz",
        type=Path,
        default=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "empirical_reflection_differences"
        / "empirical_differences.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "reflection_localized_ghost_simulation",
    )
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.config).simulation
    rho = _physical_radial_grid(cfg)
    with np.load(args.empirical_npz) as data:
        names = [str(item) for item in data["names"].tolist()]
        registered = np.asarray(data["registered"], dtype=np.float32)
        real_rho = np.asarray(data["rho"], dtype=np.float32)
    if not np.allclose(rho, real_rho, rtol=0.0, atol=5e-3):
        rho = real_rho
    lookup = {name: registered[index] for index, name in enumerate(names)}
    high_stack = np.stack([_high_pass(image) for image in registered], axis=0)
    robust_high = np.median(high_stack, axis=0).astype(np.float32)

    engine, optics, field, propagated = _prepare_base(cfg)
    sim_clean_raw = gaussian_filter(np.abs(propagated) ** 2, sigma=1.0, mode="reflect")
    sim_clean = _normalise_observation(sim_clean_raw, rho)

    fits: dict[str, dict[str, Any]] = {}
    for target_name in ("13.bmp", "14.bmp"):
        target_signal = (_high_pass(lookup[target_name]) - robust_high).astype(np.float32)
        target_energy = _local_energy(target_signal)
        target_geometry = _weighted_geometry(target_energy, rho <= 0.82)
        fits[target_name] = _fit_target(
            name=target_name,
            target_signal=target_signal,
            target_energy=target_energy,
            target_geometry=target_geometry,
            engine=engine,
            optics=optics,
            field=field,
            propagated=propagated,
            sim_clean=sim_clean,
            rho=rho,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hp_limit = max(
        float(np.quantile(np.abs(fits[name][key][rho <= 0.82]), 0.995))
        for name in fits
        for key in ("target_signal", "simulated_signal")
    )
    energy_limit = max(
        float(np.quantile(fits[name][key][rho <= 0.82], 0.995))
        for name in fits
        for key in ("target_energy", "simulated_energy")
    )
    fig, axes = plt.subplots(4, 4, figsize=(16, 15), constrained_layout=True)
    for column_pair, target_name in enumerate(("13.bmp", "14.bmp")):
        fit = fits[target_name]
        real_column = 2 * column_pair
        sim_column = real_column + 1
        axes[0, real_column].imshow(lookup[target_name], cmap="gray", vmin=0.0, vmax=1.5)
        axes[0, real_column].set_title(f"real raw {target_name}")
        axes[0, sim_column].imshow(fit["simulated"], cmap="gray", vmin=0.0, vmax=1.5)
        axes[0, sim_column].set_title(f"localized ghost simulation for {target_name}")
        axes[1, real_column].imshow(fit["target_signal"], cmap="coolwarm", vmin=-hp_limit, vmax=hp_limit)
        axes[1, real_column].set_title("real high-pass residual")
        axes[1, sim_column].imshow(fit["simulated_signal"], cmap="coolwarm", vmin=-hp_limit, vmax=hp_limit)
        axes[1, sim_column].set_title("simulated ghost high-pass")
        axes[2, real_column].imshow(fit["target_energy"], cmap="magma", vmin=0.0, vmax=energy_limit)
        axes[2, real_column].set_title("real local stripe energy")
        axes[2, sim_column].imshow(fit["simulated_energy"], cmap="magma", vmin=0.0, vmax=energy_limit)
        axes[2, sim_column].set_title("simulated local stripe energy")
        real_spectrum = np.where(fit["spectral_band"], fit["target_descriptor"], np.nan)
        simulated_spectrum = np.where(fit["spectral_band"], fit["simulated_descriptor"], np.nan)
        axes[3, real_column].imshow(real_spectrum, cmap="viridis")
        axes[3, real_column].set_title("real localized spectrum")
        axes[3, sim_column].imshow(simulated_spectrum, cmap="viridis")
        axes[3, sim_column].set_title("simulated localized spectrum")
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(args.output_dir / "localized_ghost_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for row, target_name in enumerate(("13.bmp", "14.bmp")):
        fit = fits[target_name]
        envelope = fit["visibility_envelope"]
        inside = _ellipse_mask(tuple(rho.shape), envelope, scale=1.0)
        ys, xs = np.where(inside)
        pad = 18
        y0, y1 = max(0, int(ys.min()) - pad), min(rho.shape[0], int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(rho.shape[1], int(xs.max()) + pad + 1)
        panels = [
            ("real raw crop", lookup[target_name][y0:y1, x0:x1], "gray", 0.0, 1.5),
            ("simulated raw crop", fit["simulated"][y0:y1, x0:x1], "gray", 0.0, 1.5),
            ("real high-pass crop", fit["target_signal"][y0:y1, x0:x1], "coolwarm", -hp_limit, hp_limit),
            ("simulated high-pass crop", fit["simulated_signal"][y0:y1, x0:x1], "coolwarm", -hp_limit, hp_limit),
        ]
        for ax, (title, image, cmap, vmin, vmax) in zip(axes[row], panels):
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{target_name}: {title}")
            ax.axis("off")
    fig.savefig(args.output_dir / "localized_ghost_crops.png", dpi=200)
    plt.close(fig)

    summary_fits: dict[str, Any] = {}
    for name, fit in fits.items():
        summary_fits[name] = {
            key: fit[key]
            for key in (
                "target_geometry",
                "visibility_envelope",
                "target_spectral_peaks",
                "component",
                "source_texture_seed",
                "search_best_score_before_amplitude_calibration",
                "metrics",
                "simulated_geometry",
                "simulated_spectral_peaks",
            )
        }
    summary = {
        "scope": (
            "localized coherent ghost fit to raw registered frames; the robust median high-pass of all 24 frames "
            "is removed before fitting so gross lens shape and common camera texture do not determine the fit"
        ),
        "model": (
            "finite smooth source support before ghost propagation, followed by a detector-plane visibility envelope "
            "representing finite beam overlap or mutual coherence"
        ),
        "base_optics": _jsonable(optics),
        "raw_blur_sigma_px": 1.0,
        "fits": _jsonable(summary_fits),
        "limitations": [
            "The responsible physical surface and source coherence length are unknown.",
            "Envelope location and extent are fitted from frames 13 and 14 and are therefore in-sample.",
            "Spectral agreement does not identify a unique ghost path.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({name: fit["metrics"] for name, fit in fits.items()}, indent=2))
    print(args.output_dir / "localized_ghost_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
