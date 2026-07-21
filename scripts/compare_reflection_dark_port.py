from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_capture import _angular_cv, _physical_radial_grid, _radial_grid  # noqa: E402
from mini_grin_rebuild.core.configs import SimulationConfig, load_experiment_config  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_reference  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402

APERTURE_RADIUS_UM = 109.71


def _load_real_crops_dn(
    *,
    raw_dir: Path,
    detections_path: Path,
    grid_size: int,
    crop_radius_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Registered central crops in native DN (no robust normalisation)."""

    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    crops: list[np.ndarray] = []
    names: list[str] = []
    radius_resized = grid_size / (2.0 * crop_radius_scale)
    rho = _radial_grid(grid_size, radius_resized)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    for record in detections:
        center_x = float(record["center_x"])
        center_y = float(record["center_y"])
        radius = float(record["radius"])
        half = crop_radius_scale * radius
        box = (
            int(round(center_x - half)),
            int(round(center_y - half)),
            int(round(center_x + half)),
            int(round(center_y + half)),
        )
        with Image.open(raw_dir / str(record["file"])) as source:
            crop = source.convert("L").crop(box).resize((grid_size, grid_size), resampling)
        crops.append(np.asarray(crop, dtype=np.float32))
        names.append(str(record["file"]))
    return np.stack(crops, axis=0), rho, names


def _annulus_profile(image: np.ndarray, rho: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rho >= lo) & (rho < hi)
        values.append(float(np.median(image[mask])) if np.any(mask) else float("nan"))
    return np.asarray(values, dtype=np.float32)


def _edge_metrics(image: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    """Sharpness of the cap-edge rim: 10-90 rise width and rim FWHM in micrometres."""

    bins = np.arange(0.70, 1.14, 0.004, dtype=np.float32)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = _annulus_profile(image, rho, bins)
    finite = np.isfinite(profile)
    profile = np.where(finite, profile, np.nanmin(profile))
    baseline = float(np.median(profile[centers < 0.85]))
    peak_index = int(np.argmax(profile))
    peak = float(profile[peak_index])
    height = max(peak - baseline, 1e-6)

    def _crossing(level: float) -> float:
        target = baseline + level * height
        for index in range(peak_index, 0, -1):
            if profile[index - 1] <= target <= profile[index]:
                span = profile[index] - profile[index - 1]
                fraction = 0.0 if span <= 1e-9 else (target - profile[index - 1]) / span
                return float(centers[index - 1] + fraction * (centers[index] - centers[index - 1]))
        return float(centers[0])

    rise_10 = _crossing(0.10)
    rise_90 = _crossing(0.90)
    half = baseline + 0.5 * height
    above = np.flatnonzero(profile >= half)
    if above.size:
        fwhm = float(centers[above[-1]] - centers[above[0]])
    else:
        fwhm = float("nan")
    return {
        "rim_baseline_dn": baseline,
        "rim_peak_dn": peak,
        "rim_peak_radius": float(centers[peak_index]),
        "edge_rise_10_90_um": float((rise_90 - rise_10) * APERTURE_RADIUS_UM),
        "rim_fwhm_um": float(fwhm * APERTURE_RADIUS_UM),
    }


def _corner_patches(image: np.ndarray, size: int = 70) -> list[np.ndarray]:
    h, w = image.shape
    patches = [
        image[:size, :size],
        image[:size, w - size :],
        image[h - size :, :size],
        image[h - size :, w - size :],
    ]
    patches = sorted(patches, key=lambda patch: -float(np.median(patch)))
    return [np.asarray(patch, dtype=np.float32) for patch in patches[:3]]


def _texture_metrics(image: np.ndarray) -> dict[str, float]:
    """Fixture texture statistics from the three brightest corner patches."""

    lengths: list[float] = []
    contrasts: list[float] = []
    for patch in _corner_patches(image):
        median = max(float(np.median(patch)), 1e-6)
        contrasts.append(float(np.std(patch)) / median)
        centred = patch - float(np.mean(patch))
        window = np.hanning(patch.shape[0])[:, None] * np.hanning(patch.shape[1])[None, :]
        spectrum = np.abs(np.fft.fft2(centred * window)) ** 2
        autocorr = np.fft.fftshift(np.fft.ifft2(spectrum).real)
        autocorr = autocorr / max(float(autocorr.max()), 1e-12)
        cy, cx = patch.shape[0] // 2, patch.shape[1] // 2
        yy = np.arange(patch.shape[0]) - cy
        xx = np.arange(patch.shape[1]) - cx
        radius = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        length = float(patch.shape[0])
        for lag in range(1, patch.shape[0] // 2):
            ring = (radius >= lag - 0.5) & (radius < lag + 0.5)
            if np.any(ring) and float(np.mean(autocorr[ring])) < 0.5:
                length = float(lag)
                break
        lengths.append(length)
    return {
        "fixture_texture_corr_len_px": float(np.median(lengths)),
        "fixture_contrast": float(np.median(contrasts)),
    }


def _region_levels(image: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    return {
        "interior_median_dn": float(np.median(image[rho <= 0.65])),
        "mid_median_dn": float(np.median(image[(rho >= 0.70) & (rho <= 0.90)])),
        "rim_p95_dn": float(np.quantile(image[(rho >= 0.94) & (rho <= 1.06)], 0.95)),
        "fixture_median_dn": float(np.median(image[(rho >= 1.15) & (rho <= 1.45)])),
    }


def load_exposure_calibration(root: Path) -> tuple[float, float]:
    """Exposure scale (DN per intensity unit) and dark offset from the model fit.

    Downstream seam/defect scripts share this calibration; fail with a clear
    instruction instead of a bare traceback when the fit has not been run yet.
    """

    summary_path = (
        root
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "reflection_darkport_model_fit"
        / "summary.json"
    )
    if not summary_path.is_file():
        raise SystemExit(
            f"Missing exposure calibration: {summary_path}\n"
            "Run scripts/compare_reflection_dark_port.py first to produce the dark-port model fit."
        )
    best = json.loads(summary_path.read_text(encoding="utf-8"))["best_by_total_score"]
    return float(best["exposure_scale_dn_per_unit"]), float(best["dark_offset_dn"])


def _image_corr(real: np.ndarray, sim: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(real[mask], dtype=np.float64)
    b = np.asarray(sim[mask], dtype=np.float64)
    if a.size < 3 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _fit_scale(
    sim_profile: np.ndarray,
    real_profile: np.ndarray,
    *,
    max_dn: float = 255.0,
) -> tuple[float, float, float]:
    finite = np.isfinite(sim_profile) & np.isfinite(real_profile)
    best_scale, best_offset, best_rmse = 1.0, 0.0, float("inf")
    for offset in (0.0, 1.5, 3.0, 4.5):
        for scale in np.geomspace(0.5, 20000.0, 120):
            clipped = np.clip(scale * sim_profile[finite] + offset, 0.0, max_dn)
            rmse = float(np.sqrt(np.mean((clipped - real_profile[finite]) ** 2)))
            if rmse < best_rmse:
                best_scale, best_offset, best_rmse = float(scale), float(offset), rmse
    return best_scale, best_offset, best_rmse


def _simulate_candidate(
    cfg: SimulationConfig,
    *,
    overrides: Mapping[str, Any],
    channel: str,
) -> np.ndarray:
    params = dict(cfg.capture_engine_params)
    params.update({"field_texture": {}, "camera": {}, "coherent_ghost": {}})
    for key, value in overrides.items():
        params[key] = value
    sim_cfg = replace(cfg, capture_engine="optical_leakage_lite", capture_engine_params=params)
    height = microlens_reference(sim_cfg)
    capture = create_simulation_engine(sim_cfg).simulate_capture(height, rng=np.random.default_rng(0))
    return np.asarray(capture.channels[channel], dtype=np.float32)


def _rim_dipole(image: np.ndarray, rho: np.ndarray, *, bins: int = 72) -> tuple[float, float]:
    """First angular harmonic of the rim annulus: (relative amplitude, direction deg)."""

    h, w = image.shape
    yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
    xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
    annulus = (rho >= 0.92) & (rho <= 1.06)
    edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
    values = np.full(bins, np.nan, dtype=np.float64)
    for index in range(bins):
        mask = annulus & (theta >= edges[index]) & (theta < edges[index + 1])
        if np.any(mask):
            values[index] = float(np.median(image[mask]))
    values = np.where(np.isfinite(values), values, np.nanmean(values))
    spectrum = np.fft.rfft(values - np.mean(values))
    scale = max(float(np.mean(values)) * bins / 2.0, 1e-9)
    return float(np.abs(spectrum[1])) / scale, float(np.degrees(np.angle(spectrum[1])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Absolute-DN dark-port model fit with structural metrics (edge sharpness, rim strength, "
            "fixture texture) evaluated against a single-frame reference to avoid median softening."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "reflection_microlens520_actual.json")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "external_data" / "raw" / "wechat_2026-07_15-34" / "extracted" / "15.34",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "valid_sample_detections.json",
    )
    parser.add_argument("--reference-frame", type=str, default="7.bmp")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "reflection_darkport_model_fit",
    )
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    cfg = experiment.simulation
    crop_radius_scale = 1.0 / max(float(cfg.lens_radius_fraction), 1e-12)
    real_stack, rho, names = _load_real_crops_dn(
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        grid_size=cfg.grid_size,
        crop_radius_scale=crop_radius_scale,
    )
    real_median = np.median(real_stack, axis=0).astype(np.float32)
    reference_index = names.index(args.reference_frame)
    real_single = real_stack[reference_index]

    bins = np.linspace(0.0, 1.50, 121, dtype=np.float32)
    centers = 0.5 * (bins[:-1] + bins[1:])
    real_profile = _annulus_profile(real_median, rho, bins)
    real_levels = _region_levels(real_median, rho)
    real_dipole_amp, real_dipole_angle = _rim_dipole(real_single, rho)
    real_structure = {
        **_edge_metrics(real_single, rho),
        **_texture_metrics(real_single),
        "rim_dipole_amplitude": real_dipole_amp,
        "rim_dipole_angle_deg": real_dipole_angle,
        "angular_cv_rim": _angular_cv(real_single, rho),
    }
    median_dipole_amp, median_dipole_angle = _rim_dipole(real_median, rho)
    real_structure_median = {
        **_edge_metrics(real_median, rho),
        **_texture_metrics(real_median),
        "rim_dipole_amplitude": median_dipole_amp,
        "rim_dipole_angle_deg": median_dipole_angle,
        "angular_cv_rim": _angular_cv(real_median, rho),
    }
    sim_rho = _physical_radial_grid(cfg)

    reflectance_base = {
        "enabled": True,
        "lens_amplitude": 0.02,
        "lens_scatter_amplitude": 0.05,
        "lens_scatter_phase_rad": 3.0,
        "lens_texture_sigma_px": 4.0,
        "lens_point_scatter_count": 240,
        "lens_point_scatter_amplitude": 1.0,
        "background_amplitude": 4.0,
        "background_phase_rough_rad": 1.4,
        "background_texture_sigma_px": 13.0,
        "edge_softness_px": 1.5,
        "speckle_realizations": 8,
    }

    candidates: list[dict[str, Any]] = []
    for inner_width in (2.0, 4.0):
        for outer_width in (10.0, 16.0, 24.0):
            for rim_amp in (3.0, 4.5, 6.0):
                for tilt in (0.25,):
                    candidates.append(
                        {
                            "variant": "image_shear",
                            "overrides": {
                                "defocus_strength": 0.0,
                                "raw_blur_sigma_px": 1.0,
                                "dic_blur_sigma_px": 1.0,
                                "reflectance": {
                                    **reflectance_base,
                                    "rim_amplitude": rim_amp,
                                    "rim_inner_width_px": inner_width,
                                    "rim_outer_width_px": outer_width,
                                    "illumination_tilt_strength": tilt,
                                    "illumination_tilt_angle_deg": 180.0,
                                },
                                "dark_port": {
                                    "enabled": True,
                                    "mode": "image_shear",
                                    "leak_amplitude": 0.0,
                                    "leak_phase_rad": 0.0,
                                },
                            },
                            "label": f"imgshear_ri{inner_width:g}_ro{outer_width:g}_ra{rim_amp:g}_tilt{tilt:g}",
                        }
                    )

    records: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {}
    for candidate in candidates:
        simulated = _simulate_candidate(
            cfg,
            overrides=candidate["overrides"],
            channel="I_x",
        )
        sim_profile = _annulus_profile(simulated, sim_rho, bins)
        scale, offset, profile_rmse = _fit_scale(sim_profile, real_profile)
        sim_dn = np.clip(scale * simulated + offset, 0.0, 255.0).astype(np.float32)
        dipole_amp, dipole_angle = _rim_dipole(sim_dn, sim_rho)
        structure = {
            **_edge_metrics(sim_dn, sim_rho),
            **_texture_metrics(sim_dn),
            "rim_dipole_amplitude": dipole_amp,
            "rim_dipole_angle_deg": dipole_angle,
        }
        levels = _region_levels(sim_dn, sim_rho)
        edge_term = abs(structure["edge_rise_10_90_um"] - real_structure["edge_rise_10_90_um"]) / max(
            real_structure["edge_rise_10_90_um"], 0.5
        )
        rim_width_term = abs(structure["rim_fwhm_um"] - real_structure["rim_fwhm_um"]) / max(
            real_structure["rim_fwhm_um"], 1.0
        )
        rim_term = abs(levels["rim_p95_dn"] - real_levels["rim_p95_dn"]) / 255.0
        texture_term = abs(
            float(
                np.log(
                    max(structure["fixture_texture_corr_len_px"], 0.5)
                    / max(real_structure["fixture_texture_corr_len_px"], 0.5)
                )
            )
        )
        contrast_term = abs(structure["fixture_contrast"] - real_structure["fixture_contrast"]) / max(
            real_structure["fixture_contrast"], 0.05
        )
        dipole_term = abs(dipole_amp - real_structure["rim_dipole_amplitude"]) / max(
            real_structure["rim_dipole_amplitude"], 0.05
        )
        total_score = (
            profile_rmse / 30.0
            + edge_term
            + rim_width_term
            + rim_term
            + texture_term
            + contrast_term
            + 0.5 * dipole_term
        )
        record = {
            "label": candidate["label"],
            "variant": candidate["variant"],
            "overrides": candidate["overrides"],
            "exposure_scale_dn_per_unit": scale,
            "dark_offset_dn": offset,
            "profile_rmse_dn": profile_rmse,
            "image_corr_aperture": _image_corr(real_median, sim_dn, rho <= 1.08),
            "angular_cv_rim": _angular_cv(sim_dn, sim_rho),
            "total_score": float(total_score),
            **{f"sim_{key}": value for key, value in levels.items()},
            **{f"sim_{key}": value for key, value in structure.items()},
        }
        records.append(record)
        images[candidate["label"]] = sim_dn

    best = min(records, key=lambda record: float(record["total_score"]))
    best_profile_only = min(records, key=lambda record: float(record["profile_rmse_dn"]))
    best_image = images[best["label"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    display = [
        (f"real single frame ({args.reference_frame})", real_single),
        ("real 24-frame median", real_median),
        (f"best by total score\n{best['label']}", best_image),
    ]
    for ax, (title, image) in zip(axes[0, :3], display):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=255.0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax, (title, image) in zip(axes[1, :3], display):
        ax.imshow(np.log1p(image), cmap="magma", vmin=0.0, vmax=np.log(256.0))
        ax.set_title(f"log1p: {title.splitlines()[0]}", fontsize=9)
        ax.axis("off")

    axes[0, 3].plot(centers, real_profile, color="black", linewidth=2, label="real median")
    axes[0, 3].plot(
        centers,
        _annulus_profile(real_single, rho, bins),
        color="#666666",
        linewidth=1.2,
        label=f"real {args.reference_frame}",
    )
    axes[0, 3].plot(centers, _annulus_profile(best_image, sim_rho, bins), color="#b06060", label="sim best")
    axes[0, 3].set_yscale("symlog", linthresh=1.0)
    axes[0, 3].set_xlabel("normalised aperture radius")
    axes[0, 3].set_ylabel("median DN (symlog)")
    axes[0, 3].legend(fontsize=8)
    axes[0, 3].grid(alpha=0.25)

    table_rows = [
        ("", f"real {args.reference_frame}", "real median", "sim best"),
        (
            "edge rise 10-90 (um)",
            f"{real_structure['edge_rise_10_90_um']:.2f}",
            f"{real_structure_median['edge_rise_10_90_um']:.2f}",
            f"{best['sim_edge_rise_10_90_um']:.2f}",
        ),
        (
            "rim FWHM (um)",
            f"{real_structure['rim_fwhm_um']:.2f}",
            f"{real_structure_median['rim_fwhm_um']:.2f}",
            f"{best['sim_rim_fwhm_um']:.2f}",
        ),
        (
            "rim p95 (DN)",
            f"{_region_levels(real_single, rho)['rim_p95_dn']:.0f}",
            f"{real_levels['rim_p95_dn']:.0f}",
            f"{best['sim_rim_p95_dn']:.0f}",
        ),
        (
            "fixture corr len (px)",
            f"{real_structure['fixture_texture_corr_len_px']:.1f}",
            f"{real_structure_median['fixture_texture_corr_len_px']:.1f}",
            f"{best['sim_fixture_texture_corr_len_px']:.1f}",
        ),
        (
            "fixture contrast",
            f"{real_structure['fixture_contrast']:.2f}",
            f"{real_structure_median['fixture_contrast']:.2f}",
            f"{best['sim_fixture_contrast']:.2f}",
        ),
        (
            "rim dipole amp",
            f"{real_structure['rim_dipole_amplitude']:.2f}",
            f"{real_structure_median['rim_dipole_amplitude']:.2f}",
            f"{best['sim_rim_dipole_amplitude']:.2f}",
        ),
        (
            "angular CV",
            f"{real_structure['angular_cv_rim']:.2f}",
            f"{real_structure_median['angular_cv_rim']:.2f}",
            f"{best['angular_cv_rim']:.2f}",
        ),
        (
            "interior (DN)",
            f"{_region_levels(real_single, rho)['interior_median_dn']:.1f}",
            f"{real_levels['interior_median_dn']:.1f}",
            f"{best['sim_interior_median_dn']:.1f}",
        ),
        ("profile RMSE (DN)", "-", "-", f"{best['profile_rmse_dn']:.2f}"),
        ("aperture corr", "-", "-", f"{best['image_corr_aperture']:.3f}"),
    ]
    axes[1, 3].axis("off")
    rendered = axes[1, 3].table(cellText=[list(row) for row in table_rows], loc="center", cellLoc="center")
    rendered.scale(1.0, 1.4)
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8)
    fig.savefig(args.output_dir / "darkport_model_fit.png", dpi=170)
    plt.close(fig)

    # Zoomed structural comparison: rim segment and fixture corner.
    rim_rows = slice(196, 316)
    rim_cols = slice(430, 512)
    corner = slice(0, 70)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    zoom_panels = [
        (f"real {args.reference_frame}", real_single),
        ("real median", real_median),
        ("sim best", best_image),
    ]
    for ax, (title, image) in zip(axes[0], zoom_panels):
        ax.imshow(image[rim_rows, rim_cols], cmap="gray", vmin=0.0, vmax=255.0)
        ax.set_title(f"rim zoom: {title}", fontsize=9)
        ax.axis("off")
    for ax, (title, image) in zip(axes[1], zoom_panels):
        ax.imshow(image[corner, corner], cmap="gray", vmin=0.0, vmax=255.0)
        ax.set_title(f"fixture corner: {title}", fontsize=9)
        ax.axis("off")
    fig.savefig(args.output_dir / "darkport_structure_zoom.png", dpi=170)
    plt.close(fig)

    summary = {
        "scope": (
            "absolute-DN dark-port fit with structural metrics; sharpness and texture are referenced "
            "to a single real frame because the 24-frame median softens edges (registration jitter) "
            "and averages away per-frame fixture texture"
        ),
        "reference_frame": args.reference_frame,
        "real_levels_dn_median": real_levels,
        "real_structure_single_frame": real_structure,
        "real_structure_median": real_structure_median,
        "best_by_total_score": best,
        "best_by_profile_rmse": best_profile_only,
        "all_candidates_top40": sorted(records, key=lambda record: float(record["total_score"]))[:40],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    condensed = {
        "real_structure_single_frame": real_structure,
        "real_structure_median": real_structure_median,
        "best_by_total_score": {
            key: best[key]
            for key in (
                "label",
                "total_score",
                "profile_rmse_dn",
                "image_corr_aperture",
                "sim_edge_rise_10_90_um",
                "sim_rim_fwhm_um",
                "sim_rim_p95_dn",
                "sim_fixture_texture_corr_len_px",
                "sim_fixture_contrast",
                "sim_rim_dipole_amplitude",
                "sim_rim_dipole_angle_deg",
                "angular_cv_rim",
                "sim_interior_median_dn",
            )
        },
        "best_by_profile_rmse_label": best_profile_only["label"],
    }
    print(json.dumps(condensed, indent=2))
    print(args.output_dir / "darkport_model_fit.png")
    print(args.output_dir / "darkport_structure_zoom.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
