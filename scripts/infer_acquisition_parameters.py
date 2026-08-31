from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]

APERTURE_RADIUS_UM = float(np.sqrt(2.0 * 400.0 * 15.34 - 15.34**2))  # spherical cap, R=400, s=15.34
WAVELENGTH_UM = 0.52


def _load_frame(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("L"), dtype=np.uint8)


def _angular_profile(
    crop: np.ndarray,
    radius_px: float,
    *,
    bins: int = 72,
    rho_lo: float = 0.92,
    rho_hi: float = 1.06,
) -> np.ndarray:
    h, w = crop.shape
    yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
    xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    rho = np.sqrt(x_grid**2 + y_grid**2) / max(radius_px, 1e-6)
    theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
    annulus = (rho >= rho_lo) & (rho <= rho_hi)
    values = np.full(bins, np.nan, dtype=np.float64)
    edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
    arr = crop.astype(np.float32)
    for index in range(bins):
        mask = annulus & (theta >= edges[index]) & (theta < edges[index + 1])
        if np.any(mask):
            values[index] = float(np.median(arr[mask]))
    filled = np.where(np.isfinite(values), values, np.nanmean(values))
    return filled / max(float(np.mean(filled)), 1e-9)


def _harmonics(profile: np.ndarray) -> dict[str, float]:
    spectrum = np.fft.rfft(profile - np.mean(profile))
    scale = max(float(np.mean(profile)) * len(profile) / 2.0, 1e-12)
    out: dict[str, float] = {}
    for order in (1, 2):
        coeff = spectrum[order]
        out[f"h{order}_amplitude"] = float(np.abs(coeff)) / scale
        out[f"h{order}_phase_rad"] = float(np.angle(coeff))
    return out


def _circular_concentration(phases: list[float]) -> tuple[float, float]:
    vec = np.exp(1j * np.asarray(phases, dtype=np.float64))
    mean_vec = complex(np.mean(vec))
    return float(np.abs(mean_vec)), float(np.angle(mean_vec))


def _autocorrelation_profile(
    crop: np.ndarray,
    *,
    highpass_sigma: float = 30.0,
    max_lag: int = 450,
) -> tuple[np.ndarray, np.ndarray]:
    arr = crop.astype(np.float32)
    arr = arr - gaussian_filter(arr, sigma=highpass_sigma, mode="reflect")
    window_y = np.hanning(arr.shape[0]).astype(np.float32)
    window_x = np.hanning(arr.shape[1]).astype(np.float32)
    arr = arr * window_y[:, None] * window_x[None, :]
    spectrum = np.abs(np.fft.fft2(arr)) ** 2
    autocorr = np.fft.fftshift(np.fft.ifft2(spectrum).real)
    autocorr = autocorr / max(float(autocorr.max()), 1e-12)
    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    yy = np.arange(h) - cy
    xx = np.arange(w) - cx
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    radius = np.sqrt(x_grid**2 + y_grid**2)
    lags = np.arange(8, int(max_lag))
    profile = np.zeros(lags.shape, dtype=np.float64)
    for index, lag in enumerate(lags):
        ring = (radius >= lag - 0.5) & (radius < lag + 0.5)
        profile[index] = float(np.max(autocorr[ring])) if np.any(ring) else 0.0
    return lags.astype(np.float64), profile


def _photon_transfer_samples(
    crop: np.ndarray,
    radius_px: float,
    *,
    patch: int = 24,
) -> list[tuple[float, float]]:
    arr = crop.astype(np.float32)
    h, w = arr.shape
    yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
    xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    rho = np.sqrt(x_grid**2 + y_grid**2) / max(radius_px, 1e-6)
    # First horizontal neighbour difference removes smooth structure; /sqrt(2)
    # makes its variance equal to the per-pixel noise variance.
    diff = (arr[:, 1:] - arr[:, :-1]) / np.sqrt(2.0)
    samples: list[tuple[float, float]] = []
    for y0 in range(0, h - patch, patch):
        for x0 in range(0, w - patch, patch):
            if float(np.max(rho[y0 : y0 + patch, x0 : x0 + patch])) > 0.85:
                continue
            block = arr[y0 : y0 + patch, x0 : x0 + patch]
            mean = float(np.mean(block))
            if mean < 8.0 or mean > 235.0:
                continue
            noise = diff[y0 : y0 + patch, x0 : x0 + patch - 1]
            samples.append((mean, float(np.var(noise))))
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Infer acquisition parameters (exposure protocol, illumination direction, sensor response, shear doubling, coherence bound) from the 24 raw BMP frames."
    )
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
    parser.add_argument(
        "--all-circles",
        type=Path,
        default=ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "circle_detections.json",
    )
    parser.add_argument(
        "--ghost-summary",
        type=Path,
        default=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "reflection_localized_ghost_simulation"
        / "summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "acquisition_inference",
    )
    args = parser.parse_args(argv)

    detections = json.loads(args.detections.read_text(encoding="utf-8"))
    all_circles = {
        record["file"]: record["circles"]
        for record in json.loads(args.all_circles.read_text(encoding="utf-8"))
    }

    frame_records: list[dict[str, Any]] = []
    angular_profiles: list[np.ndarray] = []
    autocorr_profiles: list[np.ndarray] = []
    autocorr_lags: np.ndarray | None = None
    transfer_samples: list[tuple[float, float]] = []

    for record in detections:
        name = str(record["file"])
        path = args.raw_dir / name
        frame = _load_frame(path)
        center_x = float(record["center_x"])
        center_y = float(record["center_y"])
        radius = float(record["radius"])

        half = 1.15 * radius
        y0, y1 = int(round(center_y - half)), int(round(center_y + half))
        x0, x1 = int(round(center_x - half)), int(round(center_x + half))
        y0, x0 = max(0, y0), max(0, x0)
        crop = frame[y0:y1, x0:x1]

        # Interior / rim levels in native DN.
        h, w = crop.shape
        yy = np.arange(h, dtype=np.float32) - (center_y - y0)
        xx = np.arange(w, dtype=np.float32) - (center_x - x0)
        y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
        rho = np.sqrt(x_grid**2 + y_grid**2) / radius
        interior = crop[rho <= 0.65].astype(np.float32)
        rim = crop[(rho >= 0.92) & (rho <= 1.06)].astype(np.float32)

        # Fixture/background: bright pixels outside every detected circle.
        background_mask = (frame >= 40) & (frame <= 250)
        for circle in all_circles.get(name, []):
            cx, cy, cr = float(circle["x"]), float(circle["y"]), 1.25 * float(circle["radius"])
            by0, by1 = max(0, int(cy - cr)), min(frame.shape[0], int(cy + cr) + 1)
            bx0, bx1 = max(0, int(cx - cr)), min(frame.shape[1], int(cx + cr) + 1)
            local_y = np.arange(by0, by1, dtype=np.float32) - cy
            local_x = np.arange(bx0, bx1, dtype=np.float32) - cx
            ly, lx = np.meshgrid(local_y, local_x, indexing="ij")
            background_mask[by0:by1, bx0:bx1] &= (ly**2 + lx**2) > cr**2
        background = frame[background_mask].astype(np.float32)

        profile = _angular_profile(crop, radius)
        angular_profiles.append(profile)
        harmonics = _harmonics(profile)

        lags, autocorr = _autocorrelation_profile(crop)
        autocorr_lags = lags
        autocorr_profiles.append(autocorr)

        transfer_samples.extend(_photon_transfer_samples(crop, radius))

        frame_records.append(
            {
                "file": name,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "radius_px": radius,
                "interior_median_dn": float(np.median(interior)),
                "rim_p95_dn": float(np.quantile(rim, 0.95)),
                "background_median_dn": float(np.median(background)) if background.size else float("nan"),
                "background_contrast": (
                    float(np.std(background) / max(np.mean(background), 1e-9)) if background.size else float("nan")
                ),
                "zero_fraction": float(np.mean(frame == 0)),
                "saturated_fraction": float(np.mean(frame >= 254)),
                **harmonics,
            }
        )

    order = np.argsort([record["mtime"] for record in frame_records])
    capture_order = [frame_records[index]["file"] for index in order]
    for rank, index in enumerate(order):
        frame_records[index]["capture_rank"] = int(rank)

    # Exposure-protocol consistency.
    backgrounds = np.asarray([record["background_median_dn"] for record in frame_records], dtype=np.float64)
    interiors = np.asarray([record["interior_median_dn"] for record in frame_records], dtype=np.float64)
    rims = np.asarray([record["rim_p95_dn"] for record in frame_records], dtype=np.float64)
    exposure = {
        "background_median_dn": {
            "mean": float(np.nanmean(backgrounds)),
            "cv": float(np.nanstd(backgrounds) / max(np.nanmean(backgrounds), 1e-9)),
        },
        "interior_median_dn": {
            "mean": float(np.mean(interiors)),
            "cv": float(np.std(interiors) / max(np.mean(interiors), 1e-9)),
        },
        "rim_p95_dn": {
            "mean": float(np.mean(rims)),
            "cv": float(np.std(rims) / max(np.mean(rims), 1e-9)),
        },
    }

    # Illumination direction consistency across frames (camera frame).
    h1_conc, h1_mean = _circular_concentration([record["h1_phase_rad"] for record in frame_records])
    h2_conc, h2_mean = _circular_concentration([record["h2_phase_rad"] for record in frame_records])
    illumination = {
        "harmonic1": {
            "phase_concentration": h1_conc,
            "mean_direction_deg": float(np.degrees(h1_mean)),
            "mean_amplitude": float(np.mean([record["h1_amplitude"] for record in frame_records])),
        },
        "harmonic2": {
            "phase_concentration": h2_conc,
            "mean_axis_deg": float(np.degrees(0.5 * h2_mean)),
            "mean_amplitude": float(np.mean([record["h2_amplitude"] for record in frame_records])),
        },
        "note": (
            "phase_concentration near 1 means the anisotropy pattern is fixed in camera coordinates "
            "(instrument illumination/fixture); near 0 means it varies per frame (sample-specific)"
        ),
    }

    # Photon-transfer estimate from smooth interior patches.
    samples = np.asarray(transfer_samples, dtype=np.float64)
    transfer: dict[str, Any] = {"num_patches": int(samples.shape[0])}
    if samples.shape[0] >= 200:
        bin_edges = np.quantile(samples[:, 0], np.linspace(0.0, 1.0, 17))
        bin_means: list[float] = []
        bin_noise: list[float] = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (samples[:, 0] >= lo) & (samples[:, 0] < hi)
            if int(np.sum(mask)) < 12:
                continue
            bin_means.append(float(np.mean(samples[mask, 0])))
            # 20th percentile per bin: texture only inflates variance upward.
            bin_noise.append(float(np.quantile(samples[mask, 1], 0.20)))
        if len(bin_means) >= 5:
            coeffs = np.polyfit(bin_means, bin_noise, 1)
            predicted = np.polyval(coeffs, bin_means)
            residual = np.asarray(bin_noise) - predicted
            total = np.asarray(bin_noise) - float(np.mean(bin_noise))
            r2 = 1.0 - float(np.sum(residual**2)) / max(float(np.sum(total**2)), 1e-12)
            transfer.update(
                {
                    "bin_means_dn": bin_means,
                    "bin_noise_var_dn2": bin_noise,
                    "slope_var_per_dn": float(coeffs[0]),
                    "intercept_var_dn2": float(coeffs[1]),
                    "fit_r2": r2,
                    "implied_gain_dn_per_electron": float(coeffs[0]),
                    "implied_full_well_at_255_dn_electrons": (
                        float(255.0 / coeffs[0]) if coeffs[0] > 1e-6 else None
                    ),
                    "note": (
                        "variance rising linearly with mean indicates linear (non-gamma) encoding with "
                        "shot noise; slope = DN per photoelectron at the sensor output"
                    ),
                }
            )

    # Shear-doubling search on the mean autocorrelation profile.
    autocorr_stack = np.stack(autocorr_profiles, axis=0)
    autocorr_mean = np.mean(autocorr_stack, axis=0)
    assert autocorr_lags is not None
    peak_indices, properties = find_peaks(autocorr_mean, prominence=0.02)
    peaks = sorted(
        (
            {
                "lag_px": float(autocorr_lags[index]),
                "lag_um_at_0p177": float(autocorr_lags[index] * 0.177),
                "value": float(autocorr_mean[index]),
                "prominence": float(properties["prominences"][list(peak_indices).index(index)]),
            }
            for index in peak_indices
        ),
        key=lambda item: -item["prominence"],
    )[:8]
    shear = {
        "search_range_px": [float(autocorr_lags[0]), float(autocorr_lags[-1])],
        "prominent_peaks": peaks,
        "note": (
            "a Wollaston shear delta would appear as a symmetric autocorrelation peak at the shear "
            "distance in every frame; absence suggests the frames were captured without the shear "
            "element in the path or at a port where doubling is too weak to detect"
        ),
    }

    # Coherence lower bound from the localized stripe fits.
    coherence: dict[str, Any] = {}
    if args.ghost_summary.is_file():
        ghost = json.loads(args.ghost_summary.read_text(encoding="utf-8"))
        stripe_periods_um: list[float] = []
        support_extent_um: list[float] = []
        for fit in ghost.get("fits", {}).values():
            for peak in fit.get("target_spectral_peaks", [])[:2]:
                cycles = float(peak["cycles_per_frame"])
                if cycles > 0:
                    stripe_periods_um.append(240.0 / cycles)
            geometry = fit.get("target_geometry", {})
            support_extent_um.append(4.0 * float(geometry.get("major_sigma_norm", 0.0)) * 120.0)
        if stripe_periods_um and support_extent_um:
            period = float(np.median(stripe_periods_um))
            extent = float(np.median(support_extent_um))
            tilt_rad = WAVELENGTH_UM / period
            opd_um = tilt_rad * extent
            coherence = {
                "stripe_period_um_median": period,
                "stripe_support_extent_um_median": extent,
                "implied_beam_tilt_rad": tilt_rad,
                "implied_opd_span_um": opd_um,
                "coherence_length_lower_bound_um": opd_um,
                "bandwidth_upper_bound_nm": float(1e3 * WAVELENGTH_UM**2 / max(opd_um, 1e-9)),
                "note": (
                    "fringes persisting across the fitted support require a coherence length of at "
                    "least the OPD span; the bandwidth bound assumes the stripes are equal-inclination "
                    "interference of two tilted beams"
                ),
            }

    # Batch tolerance bound from detected aperture radii.
    radii = np.asarray([record["radius_px"] for record in frame_records], dtype=np.float64)
    scale_um_per_px = APERTURE_RADIUS_UM / float(np.median(radii))
    geometry_spread = {
        "aperture_radius_um_nominal": APERTURE_RADIUS_UM,
        "scale_um_per_px": scale_um_per_px,
        "radius_px": {"min": float(radii.min()), "median": float(np.median(radii)), "max": float(radii.max())},
        "aperture_radius_um_range": [float(radii.min() * scale_um_per_px), float(radii.max() * scale_um_per_px)],
        "relative_spread": float((radii.max() - radii.min()) / np.median(radii)),
        "note": "spread mixes true batch tolerance with circle-detection error; treat as an upper bound",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    theta_axis = np.linspace(0.0, 360.0, angular_profiles[0].shape[0], endpoint=False)
    for profile in angular_profiles:
        axes[0, 0].plot(theta_axis, profile, color="#4878a8", alpha=0.25, linewidth=0.8)
    axes[0, 0].plot(theta_axis, np.mean(np.stack(angular_profiles), axis=0), color="black", linewidth=2.0)
    axes[0, 0].set_xlabel("rim angle (deg, camera frame)")
    axes[0, 0].set_ylabel("normalised rim intensity")
    axes[0, 0].set_title(
        f"Rim angular profiles, 24 frames (h1 conc={h1_conc:.2f}, h2 conc={h2_conc:.2f})"
    )
    axes[0, 0].grid(alpha=0.25)

    if "bin_means_dn" in transfer:
        axes[0, 1].scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.15, color="#888888", label="patches")
        axes[0, 1].plot(transfer["bin_means_dn"], transfer["bin_noise_var_dn2"], "o-", color="#b06060", label="binned 20th pct")
        fit_x = np.linspace(min(transfer["bin_means_dn"]), max(transfer["bin_means_dn"]), 50)
        axes[0, 1].plot(
            fit_x,
            np.polyval([transfer["slope_var_per_dn"], transfer["intercept_var_dn2"]], fit_x),
            "--",
            color="black",
            label=f"fit slope={transfer['slope_var_per_dn']:.4f}",
        )
        axes[0, 1].set_ylim(0.0, float(np.quantile(samples[:, 1], 0.98)))
        axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_xlabel("patch mean (DN)")
    axes[0, 1].set_ylabel("noise variance (DN^2)")
    axes[0, 1].set_title("Photon-transfer check (smooth interior patches)")
    axes[0, 1].grid(alpha=0.25)

    for profile in autocorr_profiles:
        axes[1, 0].plot(autocorr_lags, profile, color="#6aa870", alpha=0.2, linewidth=0.8)
    axes[1, 0].plot(autocorr_lags, autocorr_mean, color="black", linewidth=1.8)
    for peak in peaks[:4]:
        axes[1, 0].axvline(peak["lag_px"], color="#b06060", linestyle=":", linewidth=1.0)
    axes[1, 0].set_xlabel("lag (native px)")
    axes[1, 0].set_ylabel("normalised autocorrelation (ring max)")
    axes[1, 0].set_title("Shear-doubling search (high-passed central crops)")
    axes[1, 0].grid(alpha=0.25)

    ranks = [record["capture_rank"] for record in frame_records]
    axes[1, 1].plot(ranks, interiors, "o", label="lens interior median")
    axes[1, 1].plot(ranks, backgrounds, "s", label="fixture background median")
    axes[1, 1].plot(ranks, rims, "^", label="rim p95")
    axes[1, 1].set_xlabel("capture order (from file mtime)")
    axes[1, 1].set_ylabel("native DN")
    axes[1, 1].set_title("Levels vs capture order (exposure-protocol check)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.25)

    fig.savefig(args.output_dir / "acquisition_inference.png", dpi=170)
    plt.close(fig)

    summary = {
        "scope": (
            "data-driven estimates of acquisition parameters from the 24 raw BMP frames; these reduce, "
            "but do not replace, the lab calibration checklist"
        ),
        "capture_order_by_mtime": capture_order,
        "exposure_consistency": exposure,
        "illumination_anisotropy": illumination,
        "photon_transfer": transfer,
        "shear_doubling": shear,
        "coherence_bound": coherence,
        "geometry_spread": geometry_spread,
        "frames": frame_records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    condensed = {
        "exposure_cv": {key: value["cv"] for key, value in exposure.items()},
        "illumination_h1_concentration": h1_conc,
        "illumination_h1_direction_deg": illumination["harmonic1"]["mean_direction_deg"],
        "illumination_h2_concentration": h2_conc,
        "photon_transfer_slope": transfer.get("slope_var_per_dn"),
        "photon_transfer_r2": transfer.get("fit_r2"),
        "top_autocorr_peaks_px": [peak["lag_px"] for peak in peaks[:4]],
        "coherence_length_lower_bound_um": coherence.get("coherence_length_lower_bound_um"),
        "bandwidth_upper_bound_nm": coherence.get("bandwidth_upper_bound_nm"),
        "aperture_radius_relative_spread": geometry_spread["relative_spread"],
    }
    print(json.dumps(condensed, indent=2))
    print(args.output_dir / "acquisition_inference.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
