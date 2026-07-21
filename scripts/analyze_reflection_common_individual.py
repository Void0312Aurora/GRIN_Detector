from __future__ import annotations

"""Decompose the 24 registered dark-port frames into a common component and
per-frame individual components, and characterize both, zone by zone.

The common component is what the deterministic simulation (actual config) must
reproduce; the statistics of the individual components are what the nuisance
distributions (noisy config) must cover.
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_dark_port import _load_real_crops_dn  # noqa: E402

RAW_DIR = ROOT / "external_data" / "raw" / "wechat_2026-07_15-34" / "extracted" / "15.34"
DETECTIONS = ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "valid_sample_detections.json"
OUTPUT_DIR = ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "reflection_common_individual"
UM_PER_PX_NATIVE = 109.71 / 620.0

ZONES = {
    "interior": (0.0, 0.70),
    "arc_zone": (0.70, 0.94),
    "seam": (0.94, 1.06),
    "fixture": (1.10, 1.45),
}


def _autocorr_half_width(patch: np.ndarray) -> float:
    centred = patch - float(np.mean(patch))
    window = np.hanning(patch.shape[0])[:, None] * np.hanning(patch.shape[1])[None, :]
    spectrum = np.abs(np.fft.fft2(centred * window)) ** 2
    autocorr = np.fft.fftshift(np.fft.ifft2(spectrum).real)
    autocorr = autocorr / max(float(autocorr.max()), 1e-12)
    cy, cx = patch.shape[0] // 2, patch.shape[1] // 2
    yy = np.arange(patch.shape[0]) - cy
    xx = np.arange(patch.shape[1]) - cx
    radius = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    for lag in range(1, patch.shape[0] // 2):
        ring = (radius >= lag - 0.5) & (radius < lag + 0.5)
        if np.any(ring) and float(np.mean(autocorr[ring])) < 0.5:
            return float(lag)
    return float(patch.shape[0] // 2)


def _sector_coherence(image: np.ndarray, rho: np.ndarray, lo: float, hi: float, *, sectors: int = 8) -> float:
    h, w = image.shape
    yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
    xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
    bins = np.arange(lo, hi, 0.006, dtype=np.float32)
    profiles = []
    for sector in range(sectors):
        mask = (theta >= 2.0 * np.pi * sector / sectors) & (theta < 2.0 * np.pi * (sector + 1) / sectors)
        profile = np.array(
            [np.mean(image[mask & (rho >= a) & (rho < b)]) for a, b in zip(bins[:-1], bins[1:])]
        )
        profiles.append(profile - gaussian_filter1d(profile, 10))
    stack = np.stack(profiles)
    cors = [
        float(np.corrcoef(stack[i], stack[j])[0, 1])
        for i in range(sectors)
        for j in range(i + 1, sectors)
    ]
    return float(np.mean(cors))


def _rim_dipole_phase(image: np.ndarray, rho: np.ndarray) -> tuple[float, float]:
    h, w = image.shape
    yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
    xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
    annulus = (rho >= 0.92) & (rho <= 1.06)
    bins_count = 72
    edges = np.linspace(0.0, 2.0 * np.pi, bins_count + 1)
    values = np.full(bins_count, np.nan)
    for index in range(bins_count):
        mask = annulus & (theta >= edges[index]) & (theta < edges[index + 1])
        if np.any(mask):
            values[index] = float(np.median(image[mask]))
    values = np.where(np.isfinite(values), values, np.nanmean(values))
    spectrum = np.fft.rfft(values - np.mean(values))
    scale = max(float(np.mean(values)) * bins_count / 2.0, 1e-9)
    return float(np.abs(spectrum[1])) / scale, float(np.degrees(np.angle(spectrum[1])))


def _native_arc_peaks() -> dict[str, object]:
    """Refit the arc spacing law on native-resolution median profiles."""

    detections = json.loads(DETECTIONS.read_text(encoding="utf-8"))
    radii_px = np.arange(0.70 * 620.0, 0.995 * 620.0, 0.5, dtype=np.float64)
    profiles = []
    for record in detections:
        with Image.open(RAW_DIR / str(record["file"])) as source:
            frame = np.asarray(source.convert("L"), dtype=np.float32)
        cy, cx = float(record["center_y"]), float(record["center_x"])
        scale = float(record["radius"]) / 620.0
        size = int(radii_px[-1] * scale + 8)
        y0, y1 = max(0, int(cy) - size), min(frame.shape[0], int(cy) + size + 1)
        x0, x1 = max(0, int(cx) - size), min(frame.shape[1], int(cx) + size + 1)
        sub = frame[y0:y1, x0:x1]
        yy = np.arange(y0, y1, dtype=np.float32) - cy
        xx = np.arange(x0, x1, dtype=np.float32) - cx
        rr = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        profile = np.empty(radii_px.shape)
        for index, radius in enumerate(radii_px * scale):
            ring = (rr >= radius - 0.75) & (rr < radius + 0.75)
            profile[index] = float(np.mean(sub[ring])) if np.any(ring) else np.nan
        filled = np.where(np.isfinite(profile), profile, np.nanmean(profile))
        profiles.append(filled - gaussian_filter1d(filled, 14.0))
    median_profile = np.median(np.stack(profiles), axis=0)
    r_um = radii_px * UM_PER_PX_NATIVE
    peaks, _ = find_peaks(median_profile, prominence=0.35)
    x_from_seam = np.sort(109.71 - r_um[peaks])
    x_valid = x_from_seam[x_from_seam > 1.0]
    result: dict[str, object] = {
        "peak_distances_from_seam_um": [round(float(v), 2) for v in x_valid],
        "num_peaks": int(x_valid.size),
    }
    if x_valid.size >= 5:
        m_index = np.arange(1, x_valid.size + 1, dtype=np.float64)
        coeffs = np.polyfit(m_index, x_valid**2, 1)
        pred = np.polyval(coeffs, m_index)
        ss_res = float(np.sum((x_valid**2 - pred) ** 2))
        ss_tot = max(float(np.sum((x_valid**2 - np.mean(x_valid**2)) ** 2)), 1e-9)
        result["fresnel_fit"] = {
            "dz_um": float(coeffs[0] / 0.52),
            "r_squared": 1.0 - ss_res / ss_tot,
        }
        coeffs_lin = np.polyfit(m_index, x_valid, 1)
        pred_lin = np.polyval(coeffs_lin, m_index)
        ss_res_lin = float(np.sum((x_valid - pred_lin) ** 2))
        ss_tot_lin = max(float(np.sum((x_valid - np.mean(x_valid)) ** 2)), 1e-9)
        result["equal_spacing_fit"] = {
            "spacing_um": float(coeffs_lin[0]),
            "r_squared": 1.0 - ss_res_lin / ss_tot_lin,
        }
    return result


def main() -> int:
    stack, rho, names = _load_real_crops_dn(
        raw_dir=RAW_DIR,
        detections_path=DETECTIONS,
        grid_size=512,
        crop_radius_scale=1.0 / 0.9142652028,
    )
    common = np.median(stack, axis=0).astype(np.float32)
    residuals = stack - common[None, :, :]

    # Zone-wise energy split.
    zone_stats: dict[str, dict[str, float]] = {}
    for zone, (lo, hi) in ZONES.items():
        mask = (rho >= lo) & (rho < hi)
        common_zone = common[mask]
        smooth = gaussian_filter(common, 25.0)[mask]
        structured_common = float(np.std(common_zone - smooth))
        individual_rms = float(np.sqrt(np.mean(residuals[:, mask] ** 2)))
        per_frame_rms = [float(np.sqrt(np.mean(residuals[i][mask] ** 2))) for i in range(len(names))]
        zone_stats[zone] = {
            "common_level_dn": float(np.median(common_zone)),
            "common_structured_std_dn": structured_common,
            "individual_rms_dn": individual_rms,
            "individual_rms_min_dn": float(np.min(per_frame_rms)),
            "individual_rms_max_dn": float(np.max(per_frame_rms)),
            "common_sector_coherence": _sector_coherence(common, rho, max(lo, 0.12), hi),
        }

    # Individual-component character per zone (on a mid-ranked frame residual).
    order_by_total = np.argsort([float(np.sqrt(np.mean(residuals[i] ** 2))) for i in range(len(names))])
    typical_index = int(order_by_total[len(order_by_total) // 2])
    typical = residuals[typical_index]
    corner = typical[:70, :70]
    interior_patch = typical[226:296, 226:296]
    individual_character = {
        "typical_frame": names[typical_index],
        "interior_residual_corr_len_px": _autocorr_half_width(interior_patch),
        "fixture_residual_corr_len_px": _autocorr_half_width(corner),
        "seam_residual_sector_coherence": _sector_coherence(typical, rho, 0.94, 1.06),
        "arc_zone_residual_sector_coherence": _sector_coherence(typical, rho, 0.70, 0.94),
    }

    # Per-frame ranking and dipole drift.
    per_frame = []
    for index, name in enumerate(names):
        amplitude, angle = _rim_dipole_phase(stack[index], rho)
        per_frame.append(
            {
                "file": name,
                "residual_rms_dn": float(np.sqrt(np.mean(residuals[index] ** 2))),
                "interior_residual_rms_dn": float(
                    np.sqrt(np.mean(residuals[index][(rho < 0.85)] ** 2))
                ),
                "dipole_amplitude": amplitude,
                "dipole_angle_deg": angle,
            }
        )
    ranking = sorted(per_frame, key=lambda item: -item["residual_rms_dn"])
    angles = np.deg2rad([item["dipole_angle_deg"] for item in per_frame])
    concentration = float(np.abs(np.mean(np.exp(1j * angles))))

    arcs = _native_arc_peaks()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(np.log1p(common), cmap="magma", vmin=0.0, vmax=np.log(256.0))
    ax.set_title("common component (median, log1p)", fontsize=9)
    ax.axis("off")

    show_frames = ["7.bmp", "5.bmp", "13.bmp"]
    for column, name in enumerate(show_frames, start=1):
        ax = fig.add_subplot(grid[0, column])
        residual = residuals[names.index(name)]
        ax.imshow(residual, cmap="coolwarm", vmin=-30, vmax=30)
        ax.set_title(f"individual: {name} (±30 DN)", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, 0])
    zone_names = list(ZONES.keys())
    x = np.arange(len(zone_names))
    ax.bar(x - 0.2, [zone_stats[z]["common_structured_std_dn"] for z in zone_names], 0.4, label="common structured std")
    ax.bar(x + 0.2, [zone_stats[z]["individual_rms_dn"] for z in zone_names], 0.4, label="individual RMS")
    ax.set_xticks(x)
    ax.set_xticklabels(zone_names, fontsize=8)
    ax.set_ylabel("DN")
    ax.set_title("common vs individual energy by zone", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(grid[1, 1])
    values = [item["residual_rms_dn"] for item in per_frame]
    labels = [item["file"].replace(".bmp", "") for item in per_frame]
    ax.bar(range(len(values)), values, color="#4878a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("individual RMS (DN)")
    ax.set_title("per-frame individual energy", fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(grid[1, 2], projection="polar")
    for item in per_frame:
        ax.plot(
            np.deg2rad(item["dipole_amplitude"] * 0.0 + item["dipole_angle_deg"]),
            item["dipole_amplitude"],
            "o",
            color="#b06060",
            alpha=0.7,
        )
    ax.set_title(f"per-frame rim dipole (concentration {concentration:.2f})", fontsize=9)

    ax = fig.add_subplot(grid[1, 3])
    ax.axis("off")
    lines = [f"{'zone':<9} {'common':>7} {'struct':>7} {'indiv':>7}"]
    for zone in zone_names:
        stats = zone_stats[zone]
        lines.append(
            f"{zone:<9} {stats['common_level_dn']:>7.1f} {stats['common_structured_std_dn']:>7.2f} {stats['individual_rms_dn']:>7.2f}"
        )
    lines.append("")
    lines.append(f"arc peaks from seam: {arcs['num_peaks']}")
    if "fresnel_fit" in arcs:
        lines.append(f"fresnel R2 {arcs['fresnel_fit']['r_squared']:.3f}, dz {arcs['fresnel_fit']['dz_um']:.0f} um")
        lines.append(f"equal-d R2 {arcs['equal_spacing_fit']['r_squared']:.3f}, d {arcs['equal_spacing_fit']['spacing_um']:.2f} um")
    ax.text(0.0, 0.9, "\n".join(lines), fontsize=9, family="monospace", va="top")

    fig.savefig(OUTPUT_DIR / "common_individual_analysis.png", dpi=170)
    plt.close(fig)

    summary = {
        "scope": (
            "common/individual decomposition of the 24 registered dark-port frames; the common part is "
            "the deterministic simulation target, the individual statistics are the nuisance-distribution target"
        ),
        "zone_stats": zone_stats,
        "individual_character": individual_character,
        "per_frame": per_frame,
        "ranking_top6": [item["file"] for item in ranking[:6]],
        "dipole_concentration": concentration,
        "arc_spacing_analysis": arcs,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "zone_stats": zone_stats,
        "individual_character": individual_character,
        "ranking_top6": summary["ranking_top6"],
        "dipole_concentration": concentration,
        "arc_spacing_analysis": arcs,
    }, indent=2))
    print(OUTPUT_DIR / "common_individual_analysis.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
