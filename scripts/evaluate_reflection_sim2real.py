from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_capture import _angular_cv, _physical_radial_grid  # noqa: E402
from compare_reflection_dark_port import (  # noqa: E402
    _annulus_profile,
    _edge_metrics,
    _fit_scale,
    _image_corr,
    _load_real_crops_dn,
    _region_levels,
    _rim_dipole,
    _texture_metrics,
)
from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_reference  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402
from mini_grin_rebuild.simulation.transforms.utils import gaussian_blur  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_output(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT,
        text=text,
        stderr=subprocess.DEVNULL,
    )


def _git_status() -> list[str]:
    try:
        output = str(_git_output("status", "--porcelain=v1", "--untracked-files=all"))
    except (OSError, subprocess.CalledProcessError):
        return ["<git status unavailable>"]
    return [line for line in output.splitlines() if line]


def _git_diff() -> bytes:
    try:
        return bytes(_git_output("diff", "--binary", "HEAD", "--", text=False))
    except (OSError, subprocess.CalledProcessError):
        return b""


def _implementation_paths(configs: dict[str, Path]) -> dict[str, Path]:
    paths = [
        ROOT / "scripts" / "evaluate_reflection_sim2real.py",
        ROOT / "scripts" / "compare_reflection_capture.py",
        ROOT / "scripts" / "compare_reflection_dark_port.py",
        ROOT / "src" / "mini_grin_rebuild" / "core" / "configs.py",
        ROOT / "src" / "mini_grin_rebuild" / "data" / "virtual_objects.py",
        ROOT / "src" / "mini_grin_rebuild" / "simulation" / "factory.py",
        ROOT / "src" / "mini_grin_rebuild" / "simulation" / "engines" / "optical_leakage_lite.py",
        ROOT / "src" / "mini_grin_rebuild" / "simulation" / "transforms" / "camera.py",
        ROOT / "src" / "mini_grin_rebuild" / "simulation" / "transforms" / "utils.py",
        *configs.values(),
    ]
    output: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            key = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            key = f"external/{_sha256(resolved)[:12]}_{resolved.name}"
        output[key] = resolved
    return output


def _snapshot_implementation(output_dir: Path, paths: dict[str, Path], git_diff: bytes) -> None:
    snapshot_root = output_dir / "provenance_sources"
    for key, source in paths.items():
        destination = snapshot_root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (output_dir / "working_tree.patch").write_bytes(git_diff)


def _validate_split_manifest(manifest: dict[str, Any], available: Iterable[str]) -> None:
    blocks = manifest.get("calibration_blocks")
    temporal = manifest.get("temporal_test")
    if not isinstance(blocks, list) or not blocks or not all(isinstance(block, list) and block for block in blocks):
        raise ValueError("split manifest requires non-empty calibration_blocks")
    if not isinstance(temporal, list) or not temporal:
        raise ValueError("split manifest requires a non-empty temporal_test")
    flattened = [str(name) for block in blocks for name in block] + [str(name) for name in temporal]
    if len(flattened) != len(set(flattened)):
        raise ValueError("split manifest frame lists must be disjoint")
    available_set = set(available)
    missing = sorted(set(flattened) - available_set)
    extra = sorted(available_set - set(flattened))
    if missing or extra:
        raise ValueError(f"split manifest mismatch: missing={missing}, extra={extra}")


def _select(stack: np.ndarray, names: list[str], selected: Iterable[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    return np.stack([stack[lookup[str(name)]] for name in selected], axis=0)


def _rim_saturation_fraction(image: np.ndarray, rho: np.ndarray, *, level: float = 254.5) -> float:
    mask = (rho >= 0.94) & (rho <= 1.06)
    return float(np.mean(np.asarray(image)[mask] >= level))


def _interior_p999(image: np.ndarray, rho: np.ndarray) -> float:
    return float(np.quantile(np.asarray(image)[rho <= 0.85], 0.999))


def _angular_profile(
    image: np.ndarray,
    rho: np.ndarray,
    *,
    radial_lo: float,
    radial_hi: float,
    bins: int = 144,
    quantile: float = 0.90,
) -> np.ndarray:
    """Angular intensity profile of one radial band.

    A high radial quantile is intentional for the inner-edge diagnostic: the
    real captures contain localized bright foci along the inner edge, whereas
    an annular median hides them. The angular bins retain their spatial order
    so a complete ring cannot be confused with a four-lobe pattern.
    """

    arr = np.asarray(image, dtype=np.float32)
    if arr.shape != np.asarray(rho).shape:
        raise ValueError(f"image/rho shape mismatch: {arr.shape} != {np.asarray(rho).shape}")
    if bins < 8:
        raise ValueError("angular profile requires at least 8 bins")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("angular profile quantile must be in [0, 1]")

    h, w = arr.shape
    yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
    xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
    band = (rho >= float(radial_lo)) & (rho <= float(radial_hi))
    edges = np.linspace(0.0, 2.0 * np.pi, int(bins) + 1, dtype=np.float32)
    values: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = band & (theta >= lo) & (theta < hi)
        values.append(float(np.quantile(arr[mask], quantile)) if np.any(mask) else float("nan"))
    profile = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(profile)):
        finite = np.flatnonzero(np.isfinite(profile))
        if finite.size < 2:
            raise ValueError("angular profile contains too few populated bins")
        profile = np.interp(np.arange(profile.size), finite, profile[finite]).astype(np.float32)
    return profile


def _angular_harmonic_metrics(
    profile: np.ndarray,
    *,
    baseline: float,
    max_order: int = 8,
    prefix: str = "inner_edge",
) -> dict[str, float]:
    """Return ordered angular structure instead of only CV/dipole summaries."""

    values = np.clip(np.asarray(profile, dtype=np.float64) - float(baseline), 0.0, None)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("angular harmonic analysis expects a 1-D profile with >= 8 bins")
    mean_signal = max(float(np.mean(values)), 1e-9)
    peak_signal = max(float(np.max(values)), 1e-9)
    theta = 2.0 * np.pi * (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    output = {
        f"{prefix}_mean_signal_dn": float(np.mean(values)),
        f"{prefix}_focus_coverage_50": float(np.mean(values >= 0.5 * peak_signal)),
        f"{prefix}_peak_to_median": float(peak_signal / max(float(np.median(values)), 1e-9)),
    }
    for order in range(1, int(max_order) + 1):
        coefficient = np.mean(values * np.exp(-1j * order * theta))
        amplitude = 2.0 * abs(complex(coefficient)) / mean_signal
        peak_phase = np.mod(-np.angle(coefficient) / order, 2.0 * np.pi / order)
        output[f"{prefix}_h{order}_relative_amplitude"] = float(amplitude)
        output[f"{prefix}_h{order}_peak_phase_deg"] = float(np.rad2deg(peak_phase))
    return output


def _inner_edge_contrast_profile(
    image: np.ndarray,
    rho: np.ndarray,
    *,
    bins: int = 144,
) -> np.ndarray:
    """Local inner-edge excess, separated from the bright fixture background.

    Absolute intensity around the seam is dominated by the outside fixture and
    the illumination dipole.  Subtracting the angularly matched inner-neighbour
    band makes a full annulus distinguishable from localized edge foci.
    """

    edge = _angular_profile(
        image,
        rho,
        radial_lo=0.94,
        radial_hi=1.015,
        bins=bins,
        quantile=0.92,
    )
    inner_reference = _angular_profile(
        image,
        rho,
        radial_lo=0.84,
        radial_hi=0.91,
        bins=bins,
        quantile=0.50,
    )
    return np.clip(edge - inner_reference, 0.0, None).astype(np.float32)


def _interior_texture_metrics(image: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    """Dense interior-texture diagnostics after removing the radial background.

    The previous evaluator only used the interior median and p99.9. Sparse
    point scatterers could therefore match the tail while producing the wrong
    morphology. These metrics separate fine/medium continuous texture from
    high-kurtosis isolated points and record its spatial-frequency content.
    """

    arr = np.asarray(image, dtype=np.float32)
    radial_bins = np.linspace(0.0, 0.82, 83, dtype=np.float32)
    centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    radial_profile = np.asarray(
        [
            float(np.median(arr[(rho >= lo) & (rho < hi)]))
            if np.any((rho >= lo) & (rho < hi))
            else float("nan")
            for lo, hi in zip(radial_bins[:-1], radial_bins[1:])
        ],
        dtype=np.float32,
    )
    finite = np.flatnonzero(np.isfinite(radial_profile))
    if finite.size < 2:
        raise ValueError("interior radial profile contains too few populated bins")
    radial_profile = np.interp(np.arange(radial_profile.size), finite, radial_profile[finite])
    background = np.interp(np.clip(rho, centers[0], centers[-1]), centers, radial_profile).astype(np.float32)
    residual = arr - background
    mask = rho <= 0.78

    fine = residual - gaussian_blur(residual, 1.25)
    smooth_fine = gaussian_blur(residual, 1.25)
    medium = smooth_fine - gaussian_blur(residual, 6.0)
    fine_values = np.asarray(fine[mask], dtype=np.float64)
    medium_values = np.asarray(medium[mask], dtype=np.float64)
    residual_values = np.asarray(residual[mask], dtype=np.float64)

    fine_std = max(float(np.std(fine_values)), 1e-9)
    fine_centered = fine_values - float(np.mean(fine_values))
    fine_kurtosis = float(np.mean(fine_centered**4) / max(fine_std**4, 1e-12))

    taper = np.clip((0.78 - rho) / 0.10, 0.0, 1.0).astype(np.float32)
    window_y = np.hanning(arr.shape[0]).astype(np.float32)
    window_x = np.hanning(arr.shape[1]).astype(np.float32)
    window = taper * window_y[:, None] * window_x[None, :]
    spectral_input = (residual - float(np.mean(residual_values))) * window
    power = np.abs(np.fft.fft2(spectral_input)) ** 2
    freq_y = np.fft.fftfreq(arr.shape[0])[:, None]
    freq_x = np.fft.fftfreq(arr.shape[1])[None, :]
    frequency = np.sqrt(freq_x**2 + freq_y**2)
    valid_power = (frequency > 1.0 / max(arr.shape)) & (frequency <= 0.5)
    total_power = max(float(np.sum(power[valid_power])), 1e-12)

    def _power_fraction(lo: float, hi: float) -> float:
        band = valid_power & (frequency >= lo) & (frequency < hi)
        return float(np.sum(power[band]) / total_power)

    spectral_centroid = float(np.sum(power[valid_power] * frequency[valid_power]) / total_power)
    return {
        "interior_texture_std_dn": float(np.std(residual_values)),
        "interior_fine_texture_std_dn": fine_std,
        "interior_medium_texture_std_dn": float(np.std(medium_values)),
        "interior_fine_texture_coverage_1dn": float(np.mean(np.abs(fine_values) >= 1.0)),
        "interior_fine_texture_kurtosis": fine_kurtosis,
        "interior_spectral_low_fraction": _power_fraction(0.005, 0.03),
        "interior_spectral_mid_fraction": _power_fraction(0.03, 0.12),
        "interior_spectral_high_fraction": _power_fraction(0.12, 0.50),
        "interior_spectral_centroid_cyc_px": spectral_centroid,
    }


def _per_image_metrics(image: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    dipole_amplitude, dipole_angle = _rim_dipole(image, rho)
    inner_profile = _inner_edge_contrast_profile(image, rho, bins=144)
    inner_angular = _angular_harmonic_metrics(
        inner_profile,
        baseline=0.0,
        max_order=8,
        prefix="inner_edge",
    )
    return {
        **_region_levels(image, rho),
        **_edge_metrics(image, rho),
        **_texture_metrics(image),
        **inner_angular,
        **_interior_texture_metrics(image, rho),
        "rim_angular_cv": _angular_cv(image, rho),
        "rim_dipole_amplitude": dipole_amplitude,
        "rim_dipole_angle_deg": dipole_angle,
        "rim_saturation_fraction": _rim_saturation_fraction(image, rho),
        "interior_p999_dn": _interior_p999(image, rho),
    }


def _median_dict(records: list[dict[str, float]]) -> dict[str, float]:
    keys = records[0].keys()
    return {key: float(np.median([record[key] for record in records])) for key in keys}


def _target_bundle(stack: np.ndarray, rho: np.ndarray) -> dict[str, Any]:
    common = np.median(stack, axis=0).astype(np.float32)
    common_metrics = _per_image_metrics(common, rho)
    distribution_metrics = _median_dict([_per_image_metrics(frame, rho) for frame in stack])
    return {
        "common_image": common,
        "common_metrics": common_metrics,
        "distribution_median": distribution_metrics,
    }


def _camera_is_dn(config_path: Path) -> bool:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    camera = config.get("simulation", {}).get("capture_engine_params", {}).get("camera", {}) or {}
    return str(camera.get("output_mode", "legacy") or "legacy").lower() == "dn"


def _simulate_ensemble(
    config_path: Path,
    *,
    channel: str,
    seed: int,
    ensemble_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    experiment = load_experiment_config(config_path)
    cfg = experiment.simulation
    height = microlens_reference(cfg)
    images: list[np.ndarray] = []
    sampled: list[dict[str, Any]] = []
    for index in range(ensemble_size):
        capture_seed = int(seed + index * 1_000_003)
        print(f"simulate {config_path.name}: {index + 1}/{ensemble_size} seed={capture_seed}", flush=True)
        capture = create_simulation_engine(cfg).simulate_capture(
            height,
            rng=np.random.default_rng(capture_seed),
        )
        if channel not in capture.channels:
            raise ValueError(f"config {config_path} did not emit channel {channel!r}")
        images.append(np.asarray(capture.channels[channel], dtype=np.float32))
        sampled.append(dict(capture.meta.get("sampled_params", {})))
    return np.stack(images, axis=0), _physical_radial_grid(cfg), {
        "sampled_params": sampled,
        "camera_output_mode": "dn" if _camera_is_dn(config_path) else "legacy",
    }


def _profile_comparison(
    real_common: np.ndarray,
    sim_common: np.ndarray,
    *,
    real_rho: np.ndarray,
    sim_rho: np.ndarray,
) -> dict[str, float]:
    bins = np.linspace(0.0, 1.50, 121, dtype=np.float32)
    real_profile = _annulus_profile(real_common, real_rho, bins)
    sim_profile = _annulus_profile(sim_common, sim_rho, bins)
    finite = np.isfinite(real_profile) & np.isfinite(sim_profile)
    rmse = float(np.sqrt(np.mean((real_profile[finite] - sim_profile[finite]) ** 2)))
    corr = float(np.corrcoef(real_profile[finite], sim_profile[finite])[0, 1])
    centers = 0.5 * (bins[:-1] + bins[1:])
    seam = finite & (centers >= 0.82) & (centers <= 1.30)
    seam_rmse = float(np.sqrt(np.mean((real_profile[seam] - sim_profile[seam]) ** 2)))
    seam_corr = float(np.corrcoef(real_profile[seam], sim_profile[seam])[0, 1])
    return {
        "radial_profile_rmse_dn": rmse,
        "radial_profile_corr": corr,
        "seam_profile_rmse_dn": seam_rmse,
        "seam_profile_corr": seam_corr,
    }


def _relative_error(value: float, target: float, floor: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), float(floor))


# Per-term ceiling for the weighted sum.  Without it a single ratio-style term
# can dominate the total: the interior fine-texture kurtosis of a model with a
# near-empty interior reached 234 against a real 3.7, contributing 56% of that
# model's whole score and burying every other term.  Terms are diagnostic of
# "how wrong" only up to a point; beyond a few normalization units the exact
# magnitude carries no additional modelling information.
TERM_CEILING = 4.0


def _periodic_phase_error_deg(value: float, target: float, period: float) -> float:
    half = 0.5 * float(period)
    return abs((float(value) - float(target) + half) % float(period) - half)


def _score(
    real: dict[str, Any],
    simulated: dict[str, Any],
    profile: dict[str, float],
) -> tuple[float, dict[str, float]]:
    rc = real["common_metrics"]
    rd = real["distribution_median"]
    sc = simulated["common_metrics"]
    sd = simulated["distribution_median"]
    harmonic_rmse = float(
        np.sqrt(
            np.mean(
                [
                    (
                        float(sc[f"inner_edge_h{order}_relative_amplitude"])
                        - float(rc[f"inner_edge_h{order}_relative_amplitude"])
                    )
                    ** 2
                    for order in range(1, 9)
                ]
            )
        )
    )
    components = {
        "radial_rmse": profile["radial_profile_rmse_dn"] / 30.0,
        "seam_rmse": profile["seam_profile_rmse_dn"] / 50.0,
        "rim_saturation": abs(sc["rim_saturation_fraction"] - rc["rim_saturation_fraction"]) / 0.15,
        "rim_angular_cv": _relative_error(sc["rim_angular_cv"], rc["rim_angular_cv"], 0.10),
        "rim_dipole": _relative_error(sc["rim_dipole_amplitude"], rc["rim_dipole_amplitude"], 0.10),
        "edge_rise": _relative_error(sd["edge_rise_10_90_um"], rd["edge_rise_10_90_um"], 0.5),
        "rim_fwhm": _relative_error(sd["rim_fwhm_um"], rd["rim_fwhm_um"], 1.0),
        "fixture_corr_length": abs(
            float(np.log(max(sd["fixture_texture_corr_len_px"], 0.5) / max(rd["fixture_texture_corr_len_px"], 0.5)))
        ),
        "fixture_contrast": _relative_error(sd["fixture_contrast"], rd["fixture_contrast"], 0.05),
        "interior_level": abs(sd["interior_median_dn"] - rd["interior_median_dn"]) / 5.0,
        "interior_tail": _relative_error(sd["interior_p999_dn"], rd["interior_p999_dn"], 5.0),
        "fixture_level": _relative_error(sd["fixture_median_dn"], rd["fixture_median_dn"], 10.0),
        # Ordered inner-edge structure: a complete annulus and four localized
        # foci no longer receive the same score merely because their radial
        # averages, CV, or dipole happen to match.
        "inner_edge_harmonic_spectrum": harmonic_rmse / 0.20,
        "inner_edge_h4_phase": _periodic_phase_error_deg(
            sc["inner_edge_h4_peak_phase_deg"],
            rc["inner_edge_h4_peak_phase_deg"],
            90.0,
        )
        / 22.5,
        "inner_edge_focus_coverage": _relative_error(
            sc["inner_edge_focus_coverage_50"],
            rc["inner_edge_focus_coverage_50"],
            0.10,
        ),
        "inner_edge_peak_to_median": _relative_error(
            sc["inner_edge_peak_to_median"],
            rc["inner_edge_peak_to_median"],
            0.50,
        ),
        # Dense interior texture is evaluated both on the common image and on
        # individual captures. Sparse points can no longer satisfy the tail
        # statistic while leaving the log-intensity interior empty.
        "interior_common_fine_texture": _relative_error(
            sc["interior_fine_texture_std_dn"],
            rc["interior_fine_texture_std_dn"],
            0.25,
        ),
        "interior_common_medium_texture": _relative_error(
            sc["interior_medium_texture_std_dn"],
            rc["interior_medium_texture_std_dn"],
            0.20,
        ),
        "interior_common_texture_coverage": abs(
            sc["interior_fine_texture_coverage_1dn"]
            - rc["interior_fine_texture_coverage_1dn"]
        )
        / 0.15,
        "interior_common_texture_kurtosis": _relative_error(
            sc["interior_fine_texture_kurtosis"],
            rc["interior_fine_texture_kurtosis"],
            2.0,
        ),
        "interior_common_spectral_centroid": abs(
            sc["interior_spectral_centroid_cyc_px"]
            - rc["interior_spectral_centroid_cyc_px"]
        )
        / 0.08,
        "interior_distribution_fine_texture": _relative_error(
            sd["interior_fine_texture_std_dn"],
            rd["interior_fine_texture_std_dn"],
            0.50,
        ),
        "interior_distribution_texture_coverage": abs(
            sd["interior_fine_texture_coverage_1dn"]
            - rd["interior_fine_texture_coverage_1dn"]
        )
        / 0.20,
        # Penalize only excess pointiness here. Lower kurtosis is compatible
        # with dense continuous texture; a large positive ratio indicates the
        # isolated-point failure mode called out in the visual audit.
        "interior_distribution_excess_pointiness": max(
            float(
                np.log(
                    max(sd["interior_fine_texture_kurtosis"], 1.0)
                    / max(rd["interior_fine_texture_kurtosis"], 1.0)
                )
            ),
            0.0,
        ),
    }
    # Raw per-term weights.  These express relative importance *within* a
    # family only; the family totals are renormalized below.
    weights = {
        "radial_rmse": 1.0,
        "seam_rmse": 1.0,
        "rim_saturation": 1.5,
        "rim_angular_cv": 1.0,
        "rim_dipole": 0.5,
        "edge_rise": 1.0,
        "rim_fwhm": 0.75,
        "fixture_corr_length": 0.75,
        "fixture_contrast": 0.75,
        "interior_level": 1.0,
        "interior_tail": 0.5,
        "fixture_level": 1.0,
        "inner_edge_harmonic_spectrum": 1.5,
        "inner_edge_h4_phase": 0.5,
        "inner_edge_focus_coverage": 1.5,
        "inner_edge_peak_to_median": 1.0,
        "interior_common_fine_texture": 1.25,
        "interior_common_medium_texture": 0.75,
        "interior_common_texture_coverage": 1.0,
        "interior_common_texture_kurtosis": 0.5,
        "interior_common_spectral_centroid": 0.75,
        "interior_distribution_fine_texture": 1.25,
        "interior_distribution_texture_coverage": 1.0,
        "interior_distribution_excess_pointiness": 1.0,
    }
    capped = {key: min(float(value), TERM_CEILING) for key, value in components.items()}
    total = float(sum(_effective_weights(weights)[key] * value for key, value in capped.items()))
    return total, components


# Terms are grouped by the physical property they measure, and each family is
# renormalized to its own budget below.  Eight of the 24 terms describe interior
# texture; ungrouped they carried 7.5 of 22.75 total weight, so a single
# appearance property outvoted radial and seam profile fidelity combined.
TERM_FAMILIES = {
    "profile": (
        ["radial_rmse", "seam_rmse"],
        2.0,
    ),
    "rim_shape": (
        ["rim_saturation", "rim_angular_cv", "rim_dipole", "edge_rise", "rim_fwhm"],
        2.5,
    ),
    "fixture": (
        ["fixture_corr_length", "fixture_contrast", "fixture_level"],
        1.5,
    ),
    "levels": (
        ["interior_level", "interior_tail"],
        1.0,
    ),
    "inner_edge": (
        [
            "inner_edge_harmonic_spectrum",
            "inner_edge_h4_phase",
            "inner_edge_focus_coverage",
            "inner_edge_peak_to_median",
        ],
        2.5,
    ),
    "interior_texture": (
        [
            "interior_common_fine_texture",
            "interior_common_medium_texture",
            "interior_common_texture_coverage",
            "interior_common_texture_kurtosis",
            "interior_common_spectral_centroid",
            "interior_distribution_fine_texture",
            "interior_distribution_texture_coverage",
            "interior_distribution_excess_pointiness",
        ],
        2.0,
    ),
}


def _effective_weights(raw: Mapping[str, float]) -> dict[str, float]:
    """Renormalize each family so it contributes its family budget, not the
    sum of however many collinear terms happen to describe it."""

    effective: dict[str, float] = {}
    for keys, budget in TERM_FAMILIES.values():
        family_total = sum(float(raw[key]) for key in keys)
        for key in keys:
            effective[key] = float(budget) * float(raw[key]) / family_total
    missing = set(raw) - set(effective)
    if missing:
        raise KeyError(f"score terms not assigned to a family: {sorted(missing)}")
    return effective


def _gate_summary(
    rows: Iterable[Mapping[str, Any]],
    labels: Iterable[str],
) -> dict[str, Any]:
    """Run the veto gate for every non-baseline model against the baseline."""

    rows = list(rows)
    labels = list(labels)
    if "baseline" not in labels:
        return {"verdict": "not_applicable", "note": "no baseline model in this run"}
    baseline = [row for row in rows if row["model"] == "baseline"]
    out: dict[str, Any] = {}
    for label in labels:
        if label == "baseline":
            continue
        candidate = [row for row in rows if row["model"] == label]
        out[label] = _gate_verdict(baseline, candidate)
    return out


def _evaluate_fold(
    sim_stack: np.ndarray,
    sim_rho: np.ndarray,
    *,
    real_fit: np.ndarray,
    real_eval: np.ndarray,
    real_rho: np.ndarray,
    radiometry_mode: str,
) -> tuple[dict[str, Any], np.ndarray]:
    fit_common = np.median(real_fit, axis=0).astype(np.float32)
    sim_common_native = np.median(sim_stack, axis=0).astype(np.float32)
    if radiometry_mode == "dn":
        scale, offset, fit_rmse = 1.0, 0.0, None
    else:
        bins = np.linspace(0.0, 1.50, 121, dtype=np.float32)
        scale, offset, fit_rmse = _fit_scale(
            _annulus_profile(sim_common_native, sim_rho, bins),
            _annulus_profile(fit_common, real_rho, bins),
            max_dn=255.0,
        )
    calibrated = np.clip(scale * sim_stack + offset, 0.0, 255.0).astype(np.float32)
    real_target = _target_bundle(real_eval, real_rho)
    sim_target = _target_bundle(calibrated, sim_rho)
    profile = _profile_comparison(
        real_target["common_image"],
        sim_target["common_image"],
        real_rho=real_rho,
        sim_rho=sim_rho,
    )
    lowpass_real = gaussian_blur(real_target["common_image"], 6.0)
    lowpass_sim = gaussian_blur(sim_target["common_image"], 6.0)
    profile["lowpass_aperture_corr"] = _image_corr(lowpass_real, lowpass_sim, real_rho <= 1.08)
    total_score, score_components = _score(real_target, sim_target, profile)
    result = {
        "radiometry": {
            "mode": radiometry_mode,
            "scale": float(scale),
            "offset": float(offset),
            "fit_profile_rmse_dn": None if fit_rmse is None else float(fit_rmse),
        },
        "profile": profile,
        "real_common_metrics": real_target["common_metrics"],
        "real_distribution_median": real_target["distribution_median"],
        "sim_common_metrics": sim_target["common_metrics"],
        "sim_distribution_median": sim_target["distribution_median"],
        "score": total_score,
        "score_components": score_components,
        "gate_metrics": {key: float(profile[key]) for key in GATE_METRICS},
    }
    return result, np.median(calibrated, axis=0).astype(np.float32)


# Raw profile-fidelity metrics that a composite score must never be allowed to
# trade away silently.  "lower_is_better" fixes the comparison direction.
GATE_METRICS = {
    "radial_profile_rmse_dn": True,
    "radial_profile_corr": False,
    "seam_profile_rmse_dn": True,
    "seam_profile_corr": False,
    "lowpass_aperture_corr": False,
}

# A candidate may lose this much on a gate metric relative to the baseline
# before it counts as a regression.  5% absorbs fold-to-fold noise without
# absorbing the kind of change seen historically (seam corr 0.92 -> 0.46).
GATE_TOLERANCE = 0.05


def _gate_verdict(
    baseline_folds: Iterable[Mapping[str, Any]],
    candidate_folds: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-metric veto on raw profile fidelity.

    The composite score is a weighted mean, so a candidate can improve it while
    degrading every raw profile metric -- which is exactly what happened when a
    candidate improved the total by 71% while seam profile correlation fell from
    0.92 to 0.46 across all 12 folds. Averaging cannot express "this must not
    get worse", so the gate is evaluated outside the score.
    """

    base = list(baseline_folds)
    cand = list(candidate_folds)
    checks: list[dict[str, Any]] = []
    for metric, lower_is_better in GATE_METRICS.items():
        b = float(np.mean([float(f["gate_metrics"][metric]) for f in base]))
        c = float(np.mean([float(f["gate_metrics"][metric]) for f in cand]))
        if lower_is_better:
            allowed = b * (1.0 + GATE_TOLERANCE)
            regressed = c > allowed
            relative = (c - b) / b if b else 0.0
        else:
            allowed = b * (1.0 - GATE_TOLERANCE)
            regressed = c < allowed
            relative = (b - c) / b if b else 0.0
        checks.append(
            {
                "metric": metric,
                "lower_is_better": bool(lower_is_better),
                "baseline_mean": b,
                "candidate_mean": c,
                "allowed": float(allowed),
                "relative_degradation": float(relative),
                "regressed": bool(regressed),
            }
        )
    failed = [c["metric"] for c in checks if c["regressed"]]
    return {
        "tolerance": GATE_TOLERANCE,
        "checks": checks,
        "regressed_metrics": failed,
        "verdict": "gate_fail" if failed else "gate_pass",
        "note": (
            "A gate_fail means the candidate traded raw profile fidelity for "
            "composite score. The composite score alone must not be used to "
            "select a working point when this fails."
        ),
    }


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {}
        _flatten("", row, output)
        flattened.append(output)
    columns = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flattened)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Config-as-is dark-port sim-to-real blocked validation.")
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--channel", choices=("I_x", "I_y"), default="I_x")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.ensemble_size < 1:
        raise ValueError("ensemble-size must be >= 1")

    baseline_experiment = load_experiment_config(args.baseline_config)
    candidate_experiment = load_experiment_config(args.candidate_config)
    if baseline_experiment.simulation.grid_size != candidate_experiment.simulation.grid_size:
        raise ValueError("baseline and candidate grid_size must match")
    crop_radius_scale = 1.0 / max(float(candidate_experiment.simulation.lens_radius_fraction), 1e-12)
    real_stack, real_rho, names = _load_real_crops_dn(
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        grid_size=candidate_experiment.simulation.grid_size,
        crop_radius_scale=crop_radius_scale,
    )
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    _validate_split_manifest(manifest, names)
    calibration_blocks = [[str(name) for name in block] for block in manifest["calibration_blocks"]]
    temporal_test = [str(name) for name in manifest["temporal_test"]]
    calibration_frames = [name for block in calibration_blocks for name in block]

    configs = {
        "baseline": args.baseline_config.resolve(),
        "candidate": args.candidate_config.resolve(),
    }
    implementation_paths = _implementation_paths(configs)
    git_status = _git_status()
    git_diff = _git_diff()
    simulated: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for label, config_path in configs.items():
        print(f"[{label}] {config_path}", flush=True)
        simulated[label] = _simulate_ensemble(
            config_path,
            channel=args.channel,
            seed=args.seed,
            ensemble_size=args.ensemble_size,
        )

    rows: list[dict[str, Any]] = []
    images: dict[tuple[str, str], np.ndarray] = {}
    folds: list[tuple[str, list[str], list[str]]] = []
    for index, eval_frames in enumerate(calibration_blocks):
        fit_frames = [name for name in calibration_frames if name not in set(eval_frames)]
        folds.append((f"calibration_block_{index + 1}", fit_frames, eval_frames))
    folds.append(("temporal_test", calibration_frames, temporal_test))

    for fold_name, fit_frames, eval_frames in folds:
        real_fit = _select(real_stack, names, fit_frames)
        real_eval = _select(real_stack, names, eval_frames)
        for label, (sim_stack, sim_rho, sim_meta) in simulated.items():
            result, sim_common = _evaluate_fold(
                sim_stack,
                sim_rho,
                real_fit=real_fit,
                real_eval=real_eval,
                real_rho=real_rho,
                radiometry_mode=str(sim_meta["camera_output_mode"]),
            )
            row = {
                "model": label,
                "fold": fold_name,
                "fit_frames": fit_frames,
                "eval_frames": eval_frames,
                **result,
            }
            rows.append(row)
            images[(label, fold_name)] = sim_common
            print(f"{label} {fold_name}: score={result['score']:.4f}", flush=True)

    def _aggregate(label: str, prefix: str) -> dict[str, float]:
        selected = [row for row in rows if row["model"] == label and str(row["fold"]).startswith(prefix)]
        return {
            "score_mean": float(np.mean([row["score"] for row in selected])),
            "score_std": float(np.std([row["score"] for row in selected])),
            "radial_rmse_mean": float(np.mean([row["profile"]["radial_profile_rmse_dn"] for row in selected])),
            "seam_rmse_mean": float(np.mean([row["profile"]["seam_profile_rmse_dn"] for row in selected])),
            "lowpass_corr_mean": float(np.mean([row["profile"]["lowpass_aperture_corr"] for row in selected])),
        }

    summary = {
        "scope": "Retrospective config-as-is dark-port blocked validation; historical tuning used all 24 frames.",
        "provenance": {
            "git_commit": _git_commit(),
            "git_dirty": bool(git_status),
            "git_status_porcelain": git_status,
            "working_tree_patch_sha256": hashlib.sha256(git_diff).hexdigest(),
            "implementation_files": {
                key: {
                    "sha256": _sha256(path),
                    "snapshot": f"provenance_sources/{key}",
                }
                for key, path in implementation_paths.items()
            },
            "seed": int(args.seed),
            "ensemble_size": int(args.ensemble_size),
            "channel": args.channel,
            "raw_dir": str(args.raw_dir.resolve()),
            "raw_frames": [
                {
                    "name": name,
                    "size_bytes": int((args.raw_dir / name).stat().st_size),
                    "sha256": _sha256(args.raw_dir / name),
                }
                for name in names
            ],
            "detections": str(args.detections.resolve()),
            "detections_sha256": _sha256(args.detections),
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "configs": {
                label: {"path": str(path), "sha256": _sha256(path)} for label, path in configs.items()
            },
        },
        "split_manifest": manifest,
        "aggregates": {
            label: {
                "blocked_cv": _aggregate(label, "calibration_block_"),
                "temporal_test": next(row for row in rows if row["model"] == label and row["fold"] == "temporal_test"),
            }
            for label in configs
        },
        "gate": _gate_summary(rows, list(configs)),
        "folds": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_implementation(args.output_dir, implementation_paths, git_diff)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "per_fold_metrics.csv", rows)

    real_calibration = np.median(_select(real_stack, names, calibration_frames), axis=0)
    real_temporal = np.median(_select(real_stack, names, temporal_test), axis=0)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    panels = [
        ("real calibration median", real_calibration),
        ("real temporal-test median", real_temporal),
        ("baseline simulation", images[("baseline", "temporal_test")]),
        ("candidate simulation", images[("candidate", "temporal_test")]),
    ]
    for ax, (title, image) in zip(axes[0], panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=255.0)
        ax.set_title(title)
        ax.axis("off")
    for ax, (title, image) in zip(axes[1], panels):
        ax.imshow(np.log1p(image), cmap="magma", vmin=0.0, vmax=np.log(256.0))
        ax.set_title(f"log1p: {title}")
        ax.axis("off")
    fig.savefig(args.output_dir / "fit_vs_holdout.png", dpi=170)
    plt.close(fig)

    print(json.dumps(summary["aggregates"], indent=2), flush=True)
    gate = summary.get("gate") or {}
    for label, verdict in gate.items():
        if not isinstance(verdict, dict) or "verdict" not in verdict:
            continue
        print(f"gate[{label}]: {verdict['verdict']}", flush=True)
        for check in verdict.get("checks", []):
            if check["regressed"]:
                print(
                    f"  REGRESSED {check['metric']}: "
                    f"baseline {check['baseline_mean']:.4f} -> candidate {check['candidate_mean']:.4f} "
                    f"({100.0 * check['relative_degradation']:.1f}% worse)",
                    flush=True,
                )
    print(args.output_dir / "summary.json", flush=True)
    print(args.output_dir / "fit_vs_holdout.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
