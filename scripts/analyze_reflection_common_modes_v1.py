from __future__ import annotations

"""Extract shared (common) modes from the 24 registered dark-port frames.

Deepens the earlier median common/individual split with:
  - robust mean (10% trimmed) as a second common estimator
  - PCA/SVD rank-1 common mode
  - zone energy tables (interior / arc / seam / fixture)
  - radial + angular (Fourier) profiles of the common field
  - linear and log1p galleries for common vs frames vs residuals

Outputs land under analysis_outputs/reflection_common_modes_v1/.
"""

import json
from pathlib import Path
import sys

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

from compare_reflection_dark_port import _load_real_crops_dn  # noqa: E402

RAW_DIR = ROOT / "external_data" / "raw" / "wechat_2026-07_15-34" / "extracted" / "15.34"
DETECTIONS = ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "valid_sample_detections.json"
OLD_SUMMARY = (
    ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "reflection_common_individual" / "summary.json"
)
OUTPUT_DIR = ROOT / "analysis_outputs" / "reflection_common_modes_v1"
CROP_RADIUS_SCALE = 1.0 / 0.9142652028
GRID_SIZE = 512
APERTURE_RADIUS_UM = 109.71

ZONES = {
    "interior": (0.0, 0.70),
    "arc_zone": (0.70, 0.94),
    "seam": (0.94, 1.06),
    "fixture": (1.10, 1.45),
}

SHOW_FRAMES = ("7.bmp", "5.bmp", "13.bmp", "21.bmp")


def _trimmed_mean(stack: np.ndarray, *, trim_frac: float = 0.10) -> np.ndarray:
    n = stack.shape[0]
    k = max(1, int(round(trim_frac * n)))
    sorted_stack = np.sort(stack, axis=0)
    return np.mean(sorted_stack[k : n - k], axis=0).astype(np.float32)


def _pca_rank1(stack: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return non-negative PC1 image, explained variance fraction, and loadings."""

    n, h, w = stack.shape
    mean = np.mean(stack, axis=0)
    centered = (stack - mean[None, :, :]).reshape(n, -1)
    # Economy SVD on (n x p) with p huge: use thin SVD via covariance in frame space.
    gram = centered @ centered.T / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = float(np.sum(np.maximum(eigvals, 0.0)))
    frac = float(eigvals[0] / total) if total > 0 else 0.0
    loadings = eigvecs[:, 0]
    # Orient so mean loading is positive (bright structure matches intensity).
    if float(np.mean(loadings)) < 0:
        loadings = -loadings
    scores = centered.T @ loadings
    pc1 = scores.reshape(h, w).astype(np.float32)
    scale = float(np.median(np.abs(loadings))) or 1.0
    pc1_scaled = (mean + pc1 * scale).astype(np.float32)
    return pc1_scaled, frac, loadings.astype(np.float64)


def _zone_stats(
    common: np.ndarray,
    residuals: np.ndarray,
    rho: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for zone, (lo, hi) in ZONES.items():
        mask = (rho >= lo) & (rho < hi)
        common_zone = common[mask]
        smooth = gaussian_filter(common, 25.0)[mask]
        structured = float(np.std(common_zone - smooth))
        indiv_rms = float(np.sqrt(np.mean(residuals[:, mask] ** 2)))
        common_energy = float(np.mean(common_zone**2))
        indiv_energy = float(np.mean(residuals[:, mask] ** 2))
        ratio = float(common_energy / max(indiv_energy, 1e-12))
        out[zone] = {
            "common_level_dn": float(np.median(common_zone)),
            "common_structured_std_dn": structured,
            "individual_rms_dn": indiv_rms,
            "common_energy_dn2": common_energy,
            "individual_energy_dn2": indiv_energy,
            "common_over_individual_energy": ratio,
            "pixel_count": int(np.count_nonzero(mask)),
        }
    return out


def _radial_profile(image: np.ndarray, rho: np.ndarray, *, dr: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0.0, float(np.max(rho)) + dr, dr, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.empty(centers.shape, dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (rho >= lo) & (rho < hi)
        values[i] = float(np.median(image[mask])) if np.any(mask) else np.nan
    return centers.astype(np.float32), values.astype(np.float32)


def _angular_profile(
    image: np.ndarray,
    rho: np.ndarray,
    *,
    rho_lo: float,
    rho_hi: float,
    n_bins: int = 72,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    h, w = image.shape
    yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
    xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
    annulus = (rho >= rho_lo) & (rho < rho_hi)
    edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        mask = annulus & (theta >= edges[i]) & (theta < edges[i + 1])
        if np.any(mask):
            values[i] = float(np.median(image[mask]))
    filled = np.where(np.isfinite(values), values, np.nanmean(values))
    spectrum = np.fft.rfft(filled - np.mean(filled))
    power = np.abs(spectrum) ** 2
    total = float(np.sum(power[1:]))  # exclude DC
    harm = {
        "n_bins": float(n_bins),
        "mean_dn": float(np.mean(filled)),
        "std_dn": float(np.std(filled)),
        "dipole_frac": float(power[1] / max(total, 1e-12)) if total > 0 else 0.0,
        "quadrupole_frac": float(power[2] / max(total, 1e-12)) if total > 0 and power.size > 2 else 0.0,
        "dipole_amp_rel": float(np.abs(spectrum[1]) * 2.0 / max(np.mean(filled) * n_bins, 1e-9)),
        "dipole_angle_deg": float(np.degrees(np.angle(spectrum[1]))),
    }
    for k in range(1, min(7, power.size)):
        harm[f"harmonic_{k}_frac"] = float(power[k] / max(total, 1e-12)) if total > 0 else 0.0
    return centers.astype(np.float32), filled.astype(np.float32), harm


def _save_imshow(path: Path, image: np.ndarray, *, title: str, cmap: str, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    stack, rho, names = _load_real_crops_dn(
        raw_dir=RAW_DIR,
        detections_path=DETECTIONS,
        grid_size=GRID_SIZE,
        crop_radius_scale=CROP_RADIUS_SCALE,
    )
    common_median = np.median(stack, axis=0).astype(np.float32)
    common_robust = _trimmed_mean(stack, trim_frac=0.10)
    pc1_image, pc1_frac, loadings = _pca_rank1(stack)
    residuals = stack - common_median[None, :, :]

    zone_stats = _zone_stats(common_median, residuals, rho)

    # Per-frame residual ranking.
    per_frame = []
    for i, name in enumerate(names):
        entry = {
            "file": name,
            "residual_rms_dn": float(np.sqrt(np.mean(residuals[i] ** 2))),
            "pca_loading": float(loadings[i]),
        }
        for zone, (lo, hi) in ZONES.items():
            mask = (rho >= lo) & (rho < hi)
            entry[f"{zone}_residual_rms_dn"] = float(np.sqrt(np.mean(residuals[i][mask] ** 2)))
            entry[f"{zone}_level_dn"] = float(np.median(stack[i][mask]))
        per_frame.append(entry)
    ranking = sorted(per_frame, key=lambda item: -item["residual_rms_dn"])

    # Profiles of the median common field.
    r_centers, r_profile = _radial_profile(common_median, rho, dr=0.008)
    angular_zones = {
        "interior": (0.20, 0.65),
        "arc_zone": (0.70, 0.94),
        "seam": (0.94, 1.06),
        "fixture": (1.10, 1.40),
    }
    angular_stats: dict[str, dict[str, float]] = {}
    angular_profiles: dict[str, dict[str, list[float]]] = {}
    for zone, (lo, hi) in angular_zones.items():
        theta, vals, harm = _angular_profile(common_median, rho, rho_lo=lo, rho_hi=hi)
        angular_stats[zone] = harm
        angular_profiles[zone] = {
            "theta_rad": [float(v) for v in theta],
            "median_dn": [float(v) for v in vals],
        }

    # Agreement between common estimators.
    def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        aa = a[mask].astype(np.float64)
        bb = b[mask].astype(np.float64)
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        return float(np.dot(aa, bb) / denom) if denom > 0 else 0.0

    estimator_agreement = {
        "median_vs_robust_corr_full": _corr(common_median, common_robust, np.ones_like(rho, dtype=bool)),
        "median_vs_pc1_corr_full": _corr(common_median, pc1_image, np.ones_like(rho, dtype=bool)),
        "median_vs_robust_corr_by_zone": {
            z: _corr(common_median, common_robust, (rho >= lo) & (rho < hi)) for z, (lo, hi) in ZONES.items()
        },
        "median_vs_pc1_corr_by_zone": {
            z: _corr(common_median, pc1_image, (rho >= lo) & (rho < hi)) for z, (lo, hi) in ZONES.items()
        },
        "pca_explained_variance_frac": pc1_frac,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Individual panels ---
    log_vmax = float(np.log1p(255.0))
    _save_imshow(
        OUTPUT_DIR / "common_median_linear.png",
        common_median,
        title="common = median (linear DN)",
        cmap="magma",
        vmin=0,
        vmax=255,
    )
    _save_imshow(
        OUTPUT_DIR / "common_median_log1p.png",
        np.log1p(common_median),
        title="common = median (log1p)",
        cmap="magma",
        vmin=0,
        vmax=log_vmax,
    )
    _save_imshow(
        OUTPUT_DIR / "common_robust_mean_log1p.png",
        np.log1p(np.clip(common_robust, 0, None)),
        title="common = 10% trimmed mean (log1p)",
        cmap="magma",
        vmin=0,
        vmax=log_vmax,
    )
    _save_imshow(
        OUTPUT_DIR / "common_pca_pc1_log1p.png",
        np.log1p(np.clip(pc1_image, 0, None)),
        title=f"PCA rank-1 common (log1p, var={pc1_frac:.2%})",
        cmap="magma",
        vmin=0,
        vmax=log_vmax,
    )

    # Common vs frames vs residuals gallery.
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 5)
    panels = [
        (common_median, "median common\n(linear)", "magma", 0, 255, False),
        (np.log1p(common_median), "median common\n(log1p)", "magma", 0, log_vmax, False),
        (np.log1p(np.clip(common_robust, 0, None)), "robust mean\n(log1p)", "magma", 0, log_vmax, False),
        (np.log1p(np.clip(pc1_image, 0, None)), f"PCA PC1\n(log1p, {pc1_frac:.1%})", "magma", 0, log_vmax, False),
    ]
    for col, (img, title, cmap, vmin, vmax, _) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(grid[0, 4])
    ax.axis("off")
    ax.text(
        0.0,
        0.95,
        "v1 vs old common_individual\n"
        "- same 24 registered crops\n"
        "- same median common + zones\n"
        "- + robust mean, PCA PC1\n"
        "- + log1p galleries\n"
        "- + radial/angular profiles\n"
        "- outputs under analysis_outputs/",
        fontsize=9,
        family="monospace",
        va="top",
    )

    for col, name in enumerate(SHOW_FRAMES):
        idx = names.index(name)
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(np.log1p(stack[idx]), cmap="magma", vmin=0, vmax=log_vmax)
        ax.set_title(f"frame {name}\n(log1p)", fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(grid[2, col])
        ax.imshow(residuals[idx], cmap="coolwarm", vmin=-40, vmax=40)
        ax.set_title(f"individual {name}\n(±40 DN)", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, 4])
    zone_names = list(ZONES.keys())
    x = np.arange(len(zone_names))
    ax.bar(x - 0.2, [zone_stats[z]["common_structured_std_dn"] for z in zone_names], 0.4, label="common struct σ")
    ax.bar(x + 0.2, [zone_stats[z]["individual_rms_dn"] for z in zone_names], 0.4, label="indiv RMS")
    ax.set_xticks(x)
    ax.set_xticklabels(zone_names, fontsize=8)
    ax.set_ylabel("DN")
    ax.set_title("zone energy: common vs individual", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(grid[2, 4])
    vals = [item["residual_rms_dn"] for item in per_frame]
    labels = [item["file"].replace(".bmp", "") for item in per_frame]
    ax.bar(range(len(vals)), vals, color="#4878a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("indiv RMS (DN)")
    ax.set_title("per-frame individual energy", fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.savefig(OUTPUT_DIR / "common_vs_frames_residuals.png", dpi=170)
    plt.close(fig)

    # Radial + angular profile figure.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(r_centers, r_profile, color="#2a4d6e", lw=1.6, label="median common")
    ax.plot(r_centers, np.log1p(np.clip(r_profile, 0, None)), color="#c45c26", lw=1.2, label="log1p(common)")
    for z, (lo, hi) in ZONES.items():
        ax.axvspan(lo, min(hi, float(r_centers[-1])), alpha=0.08, label=z)
    ax.set_xlabel("ρ (aperture radii)")
    ax.set_ylabel("DN / log1p(DN)")
    ax.set_title("radial profile of median common", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1.45)

    ax = axes[0, 1]
    for zone, color in zip(("arc_zone", "seam", "fixture"), ("#2a9d8f", "#e76f51", "#264653")):
        th = np.asarray(angular_profiles[zone]["theta_rad"])
        vv = np.asarray(angular_profiles[zone]["median_dn"])
        ax.plot(np.degrees(th), vv, color=color, lw=1.3, label=zone)
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("median DN")
    ax.set_title("angular profiles (common)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    # Angular power spectrum bars for seam + fixture.
    width = 0.35
    harms = list(range(1, 7))
    seam_p = [angular_stats["seam"].get(f"harmonic_{k}_frac", 0.0) for k in harms]
    fix_p = [angular_stats["fixture"].get(f"harmonic_{k}_frac", 0.0) for k in harms]
    x = np.arange(len(harms))
    ax.bar(x - width / 2, seam_p, width, label="seam", color="#e76f51")
    ax.bar(x + width / 2, fix_p, width, label="fixture", color="#264653")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in harms])
    ax.set_xlabel("angular harmonic k")
    ax.set_ylabel("power fraction (excl. DC)")
    ax.set_title("common-field angular harmonics", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 1]
    ax.axis("off")
    lines = ["zone energy summary (median common)", ""]
    lines.append(f"{'zone':<10} {'lvl':>6} {'struct':>7} {'indiv':>7} {'C/I E':>7}")
    for z in zone_names:
        s = zone_stats[z]
        lines.append(
            f"{z:<10} {s['common_level_dn']:>6.1f} {s['common_structured_std_dn']:>7.2f} "
            f"{s['individual_rms_dn']:>7.2f} {s['common_over_individual_energy']:>7.2f}"
        )
    lines.append("")
    lines.append(f"PCA PC1 variance: {pc1_frac:.3f}")
    lines.append(f"median↔robust corr: {estimator_agreement['median_vs_robust_corr_full']:.4f}")
    lines.append(f"median↔PC1 corr:    {estimator_agreement['median_vs_pc1_corr_full']:.4f}")
    lines.append("")
    lines.append("seam dipole frac:     "
                 f"{angular_stats['seam']['dipole_frac']:.3f}")
    lines.append("fixture dipole frac:  "
                 f"{angular_stats['fixture']['dipole_frac']:.3f}")
    lines.append(f"top indiv frames: {[item['file'] for item in ranking[:4]]}")
    ax.text(0.02, 0.98, "\n".join(lines), fontsize=9, family="monospace", va="top")

    fig.savefig(OUTPUT_DIR / "radial_angular_profiles.png", dpi=170)
    plt.close(fig)

    # Interior detail (log) to highlight sparse shared structure vs dust.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    interior_mask = rho < 0.70
    # High-pass common interior for structure visibility.
    common_hp = common_median - gaussian_filter(common_median, 12.0)
    axes[0].imshow(np.log1p(common_median), cmap="magma", vmin=0, vmax=log_vmax)
    axes[0].set_title("full common (log1p)", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(np.where(interior_mask, common_hp, np.nan), cmap="coolwarm", vmin=-3, vmax=3)
    axes[1].set_title("interior high-pass common (±3 DN)", fontsize=9)
    axes[1].axis("off")
    # Mean |residual| map — idiosyncratic hotspots.
    mean_abs_res = np.mean(np.abs(residuals), axis=0)
    axes[2].imshow(mean_abs_res, cmap="inferno", vmin=0, vmax=np.percentile(mean_abs_res, 98))
    axes[2].set_title("mean |individual| map (idiosyncratic)", fontsize=9)
    axes[2].axis("off")
    fig.savefig(OUTPUT_DIR / "interior_structure_and_idiosyncrasy.png", dpi=170)
    plt.close(fig)

    old_zone = None
    if OLD_SUMMARY.exists():
        old = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
        old_zone = old.get("zone_stats")

    summary = {
        "scope": (
            "common-mode extraction from 24 registered dark-port microlens frames "
            "(wechat_2026-07_15-34); estimators: median, 10% trimmed mean, PCA rank-1"
        ),
        "data": {
            "raw_dir": str(RAW_DIR.relative_to(ROOT)),
            "detections": str(DETECTIONS.relative_to(ROOT)),
            "n_frames": len(names),
            "grid_size": GRID_SIZE,
            "crop_radius_scale": CROP_RADIUS_SCALE,
            "files": names,
        },
        "zone_stats_median_common": zone_stats,
        "angular_harmonics_common": angular_stats,
        "radial_profile_common": {
            "rho": [float(v) for v in r_centers.tolist()],
            "median_dn": [float(v) if np.isfinite(v) else None for v in r_profile.tolist()],
        },
        "estimator_agreement": estimator_agreement,
        "per_frame": per_frame,
        "ranking_top6": [item["file"] for item in ranking[:6]],
        "vs_previous_common_individual": {
            "previous_path": str(OLD_SUMMARY.relative_to(ROOT)) if OLD_SUMMARY.exists() else None,
            "previous_zone_stats": old_zone,
            "deltas": (
                {
                    z: {
                        "common_structured_std_dn": zone_stats[z]["common_structured_std_dn"]
                        - float(old_zone[z]["common_structured_std_dn"]),
                        "individual_rms_dn": zone_stats[z]["individual_rms_dn"]
                        - float(old_zone[z]["individual_rms_dn"]),
                    }
                    for z in ZONES
                }
                if old_zone
                else None
            ),
            "new_in_v1": [
                "robust trimmed-mean common",
                "PCA/SVD rank-1 common + explained variance",
                "log1p galleries and residual panels",
                "radial + angular Fourier profiles",
                "common/individual energy ratios per zone",
                "output under analysis_outputs/reflection_common_modes_v1",
            ],
        },
        "artifacts": [
            "common_median_linear.png",
            "common_median_log1p.png",
            "common_robust_mean_log1p.png",
            "common_pca_pc1_log1p.png",
            "common_vs_frames_residuals.png",
            "radial_angular_profiles.png",
            "interior_structure_and_idiosyncrasy.png",
            "summary.json",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "zone_stats": zone_stats,
        "estimator_agreement": estimator_agreement,
        "ranking_top6": summary["ranking_top6"],
        "angular_harmonics_common": {
            z: {k: angular_stats[z][k] for k in ("mean_dn", "std_dn", "dipole_frac", "quadrupole_frac", "dipole_angle_deg")}
            for z in angular_stats
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
