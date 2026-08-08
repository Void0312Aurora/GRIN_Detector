from __future__ import annotations

"""Extract additional common / base modes beyond v1 (median / robust mean / PC1).

Implements and visualises:
  1. Robust PCA (GoDec-style: low-rank truncated SVD + soft-threshold sparse)
     plus a simple median + soft-threshold residual baseline
  2. Polar / circular-harmonic decomposition of the median common field
     (energy spectrum + reconstructions for k = 0, 1, 4)
  3. Multi-rank PCA (PC1–PC3) and NMF (rank-2 / rank-3) spatial bases

Reuses the same registered-crop loader as v1.
Outputs: analysis_outputs/reflection_common_modes_v2_bases/
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_dark_port import _load_real_crops_dn  # noqa: E402

RAW_DIR = ROOT / "external_data" / "raw" / "wechat_2026-07_15-34" / "extracted" / "15.34"
DETECTIONS = ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "valid_sample_detections.json"
OUTPUT_DIR = ROOT / "analysis_outputs" / "reflection_common_modes_v2_bases"
CROP_RADIUS_SCALE = 1.0 / 0.9142652028
GRID_SIZE = 512

ZONES = {
    "interior": (0.0, 0.70),
    "arc_zone": (0.70, 0.94),
    "seam": (0.94, 1.06),
    "fixture": (1.10, 1.45),
}


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _truncated_svd_lowrank(mat: np.ndarray, rank: int) -> np.ndarray:
    """Economy truncated SVD on (n_frames x pixels) matrix → rank-k reconstruction."""

    n, p = mat.shape
    rank = max(1, min(rank, n, p))
    # Gram in frame space when n << p.
    mean = np.mean(mat, axis=0, keepdims=True)
    centered = mat - mean
    gram = centered @ centered.T
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order[:rank]], 0.0)
    eigvecs = eigvecs[:, order[:rank]]
    # scores: pixels x rank
    denom = np.sqrt(eigvals + 1e-12)
    scores = (centered.T @ eigvecs) / denom[None, :]
    recon = mean + (eigvecs * denom[None, :]) @ scores.T
    return recon.astype(np.float64)


def _robust_pca_godec(
    stack: np.ndarray,
    *,
    rank: int = 2,
    lam: float | None = None,
    n_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """GoDec-style RPCA: alternate low-rank SVD and soft-threshold sparse."""

    n, h, w = stack.shape
    x = stack.reshape(n, -1).astype(np.float64)
    if lam is None:
        # Soft threshold scaled to residual MAD (robust to dust spikes).
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med[None, :])) + 1e-6
        lam = float(2.5 * mad)
    s = np.zeros_like(x)
    l = np.zeros_like(x)
    for _ in range(n_iter):
        l = _truncated_svd_lowrank(x - s, rank)
        s = _soft_threshold(x - l, lam)
    l_stack = l.reshape(n, h, w).astype(np.float32)
    s_stack = s.reshape(n, h, w).astype(np.float32)
    common = np.mean(l_stack, axis=0).astype(np.float32)
    sparse_mean_abs = np.mean(np.abs(s_stack), axis=0).astype(np.float32)
    stats = {
        "rank": float(rank),
        "lambda_dn": float(lam),
        "n_iter": float(n_iter),
        "lowrank_energy_frac": float(np.mean(l**2) / max(np.mean(x**2), 1e-12)),
        "sparse_energy_frac": float(np.mean(s**2) / max(np.mean(x**2), 1e-12)),
        "sparse_nonzero_frac": float(np.mean(np.abs(s) > 1e-6)),
        "common_median_dn": float(np.median(common)),
        "sparse_mean_abs_p98_dn": float(np.percentile(sparse_mean_abs, 98)),
    }
    return common, sparse_mean_abs, stats


def _median_soft_sparse(
    stack: np.ndarray,
    *,
    lam_scale: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Simple robust split: L = pixelwise median, S = soft-threshold residual."""

    common = np.median(stack, axis=0).astype(np.float32)
    resid = stack - common[None, :, :]
    mad = float(np.median(np.abs(resid))) + 1e-6
    lam = lam_scale * mad
    sparse = _soft_threshold(resid, lam).astype(np.float32)
    sparse_mean_abs = np.mean(np.abs(sparse), axis=0).astype(np.float32)
    stats = {
        "lambda_dn": float(lam),
        "mad_dn": float(mad),
        "sparse_energy_frac": float(np.mean(sparse**2) / max(np.mean(stack.astype(np.float64) ** 2), 1e-12)),
        "sparse_nonzero_frac": float(np.mean(np.abs(sparse) > 1e-6)),
        "sparse_mean_abs_p98_dn": float(np.percentile(sparse_mean_abs, 98)),
    }
    return common, sparse_mean_abs, stats


def _pca_multirank(stack: np.ndarray, n_comp: int = 3) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return PC images (mean + scaled mode), explained-variance fractions, loadings (n x k)."""

    n, h, w = stack.shape
    n_comp = min(n_comp, n)
    mean = np.mean(stack, axis=0)
    centered = (stack - mean[None, :, :]).reshape(n, -1)
    gram = centered @ centered.T / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order[:n_comp]], 0.0)
    eigvecs = eigvecs[:, order[:n_comp]]
    total = float(np.sum(np.maximum(np.linalg.eigvalsh(gram), 0.0)))
    fracs = eigvals / max(total, 1e-12)
    modes: list[np.ndarray] = []
    for k in range(n_comp):
        loadings = eigvecs[:, k].copy()
        if float(np.mean(loadings)) < 0:
            loadings = -loadings
            eigvecs[:, k] = loadings
        scores = centered.T @ loadings
        mode = scores.reshape(h, w).astype(np.float32)
        scale = float(np.median(np.abs(loadings))) or 1.0
        # Signed spatial mode for gallery (not forced non-neg).
        modes.append(mode * scale)
        eigvecs[:, k] = loadings
    return modes, fracs.astype(np.float64), eigvecs.astype(np.float64)


def _nmf_bases(stack: np.ndarray, n_comp: int = 3, *, max_iter: int = 1200) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """NMF on frames×pixels → spatial bases (non-negative parts)."""

    n, h, w = stack.shape
    x = np.clip(stack.reshape(n, -1), 0, None).astype(np.float64)
    # Scale to keep NMF numerically stable.
    scale = float(np.median(x[x > 0])) if np.any(x > 0) else 1.0
    scale = max(scale, 1e-3)
    model = NMF(
        n_components=n_comp,
        init="nndsvda",
        max_iter=max_iter,
        random_state=0,
        l1_ratio=0.0,
        alpha_W=0.0,
        alpha_H=0.0,
    )
    w_coeff = model.fit_transform(x / scale)  # n_frames x k
    h_bases = model.components_  # k x pixels
    bases = [(h_bases[k].reshape(h, w) * scale).astype(np.float32) for k in range(n_comp)]
    # Order by total spatial energy descending.
    energies = np.asarray([float(np.mean(b**2)) for b in bases])
    order = np.argsort(energies)[::-1]
    bases = [bases[i] for i in order]
    w_coeff = w_coeff[:, order]
    info = {
        "n_comp": n_comp,
        "reconstruction_err": float(model.reconstruction_err_),
        "n_iter": int(model.n_iter_),
        "scale_dn": scale,
        "component_energy_dn2": [float(e) for e in energies[order]],
        "frame_loadings_mean": [float(np.mean(w_coeff[:, k])) for k in range(n_comp)],
    }
    return bases, w_coeff.astype(np.float64), info


def _polar_coords(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy = np.arange(h, dtype=np.float64) - 0.5 * (h - 1)
    xx = np.arange(w, dtype=np.float64) - 0.5 * (w - 1)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    rho_px = np.sqrt(x_grid**2 + y_grid**2)
    theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
    return rho_px, theta


def _angular_harmonic_decompose(
    image: np.ndarray,
    rho: np.ndarray,
    *,
    n_theta: int = 180,
    n_rho: int = 128,
    rho_max: float = 1.45,
    keep_ks: tuple[int, ...] = (0, 1, 2, 4),
) -> dict:
    """Sample image on polar grid; Fourier in θ; reconstruct selected circular harmonics."""

    h, w = image.shape
    rho_px, theta_xy = _polar_coords(h, w)
    valid = rho_px > 1.0
    scale = float(np.median(rho[valid] / rho_px[valid])) if np.any(valid) else 1.0
    # Vectorised polar binning: image(ρ_ap, θ).
    rho_edges = np.linspace(0.0, rho_max, n_rho + 1)
    rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    theta_centers = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rho_bin = np.clip(np.digitize(rho.ravel(), rho_edges) - 1, 0, n_rho - 1)
    th_bin = np.clip((theta_xy.ravel() / (2.0 * np.pi) * n_theta).astype(np.int64), 0, n_theta - 1)
    flat_idx = rho_bin * n_theta + th_bin
    vals = image.ravel().astype(np.float64)
    # Mean per bin (fast); median of common field is smooth enough for harmonics.
    counts = np.bincount(flat_idx, minlength=n_rho * n_theta).astype(np.float64)
    sums = np.bincount(flat_idx, weights=vals, minlength=n_rho * n_theta)
    polar = np.divide(sums, counts, out=np.zeros(n_rho * n_theta, dtype=np.float64), where=counts > 0)
    polar = polar.reshape(n_rho, n_theta)
    for i in range(n_rho):
        row = polar[i]
        if counts.reshape(n_rho, n_theta)[i].sum() > 0:
            filled = row.copy()
            empty = counts.reshape(n_rho, n_theta)[i] == 0
            if np.any(~empty):
                med = float(np.median(row[~empty]))
                filled[empty] = med
            polar[i] = filled
        else:
            polar[i] = 0.0

    # FFT along θ for each ρ ring.
    centered = polar - np.mean(polar, axis=1, keepdims=True)
    spec = np.fft.rfft(centered, axis=1)
    power = np.abs(spec) ** 2
    # Energy per harmonic (exclude DC for relative, but keep DC separately).
    power_vs_k = np.mean(power, axis=0)
    dc_power = float(np.mean(np.mean(polar, axis=1) ** 2))
    total_ac = float(np.sum(power_vs_k[1:]))
    harm_frac = {
        int(k): float(power_vs_k[k] / max(total_ac, 1e-12)) for k in range(1, min(13, power_vs_k.size))
    }

    # Reconstruct Cartesian fields for selected k.
    reconstructions: dict[str, np.ndarray] = {}
    # Precompute cartesian rho/theta already have.
    for k in keep_ks:
        if k == 0:
            # Pure radial (θ-mean).
            radial = np.mean(polar, axis=1)
            # Map each pixel by rho bin.
            idx = np.clip(np.searchsorted(rho_edges, rho, side="right") - 1, 0, n_rho - 1)
            field = radial[idx].astype(np.float32)
            reconstructions["k0_radial"] = field
            continue
        if k >= spec.shape[1]:
            continue
        # Keep only harmonic k (and conjugate handled by irfft).
        filt = np.zeros_like(spec)
        filt[:, k] = spec[:, k]
        # Also restore ring means for display of absolute intensity for k0 only;
        # for k>0 show signed oscillatory part.
        recon_polar = np.fft.irfft(filt, n=n_theta, axis=1)
        # Interpolate polar → cartesian via nearest bin.
        rho_idx = np.clip(np.searchsorted(rho_edges, rho, side="right") - 1, 0, n_rho - 1)
        th_idx = np.clip(np.round(theta_xy / (2.0 * np.pi) * n_theta).astype(int) % n_theta, 0, n_theta - 1)
        field = recon_polar[rho_idx, th_idx].astype(np.float32)
        reconstructions[f"k{k}"] = field

    # Low-order common: k0 + k1 + k4 (absolute = radial mean + oscillations).
    combo = reconstructions["k0_radial"].astype(np.float64).copy()
    for key in ("k1", "k4"):
        if key in reconstructions:
            combo += reconstructions[key]
    reconstructions["k0_plus_k1_plus_k4"] = combo.astype(np.float32)

    # Energy image: local angular AC power in annuli (for visualisation).
    # Use std across θ as a simple AC energy proxy mapped to cartesian.
    ring_ac = np.std(polar, axis=1).astype(np.float64)
    idx = np.clip(np.searchsorted(rho_edges, rho, side="right") - 1, 0, n_rho - 1)
    ac_map = ring_ac[idx].astype(np.float32)

    return {
        "rho_centers": rho_centers.astype(np.float32),
        "theta_centers": theta_centers.astype(np.float32),
        "polar": polar.astype(np.float32),
        "power_vs_k": power_vs_k.astype(np.float64),
        "harm_frac": harm_frac,
        "dc_power": dc_power,
        "reconstructions": reconstructions,
        "angular_ac_map": ac_map,
        "scale_rho_per_px": scale,
    }


def _save_imshow(path: Path, image: np.ndarray, *, title: str, cmap: str, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    aa = a[mask].astype(np.float64).ravel()
    bb = b[mask].astype(np.float64).ravel()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 0 else 0.0


def main() -> int:
    stack, rho, names = _load_real_crops_dn(
        raw_dir=RAW_DIR,
        detections_path=DETECTIONS,
        grid_size=GRID_SIZE,
        crop_radius_scale=CROP_RADIUS_SCALE,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_vmax = float(np.log1p(255.0))
    common_median = np.median(stack, axis=0).astype(np.float32)

    # --- 1) Robust PCA / sparse split ---
    rpca_common, rpca_sparse_map, rpca_stats = _robust_pca_godec(stack, rank=2, n_iter=25)
    med_common, med_sparse_map, med_sparse_stats = _median_soft_sparse(stack)

    # --- 2) Angular harmonics on median common ---
    harm = _angular_harmonic_decompose(common_median, rho, keep_ks=(0, 1, 2, 4))

    # --- 3) Multi-rank PCA + NMF ---
    pca_modes, pca_fracs, pca_loadings = _pca_multirank(stack, n_comp=3)
    nmf_bases, nmf_loadings, nmf_info = _nmf_bases(stack, n_comp=3)

    # Truncated SVD rank-2 common (mean of low-rank frames) for comparison.
    l2 = _truncated_svd_lowrank(stack.reshape(stack.shape[0], -1), rank=2)
    svd2_common = np.mean(l2, axis=0).reshape(stack.shape[1], stack.shape[2]).astype(np.float32)

    # ========== Figures ==========

    # 1a. RPCA common log1p
    _save_imshow(
        OUTPUT_DIR / "rpca_lowrank_common_log1p.png",
        np.log1p(np.clip(rpca_common, 0, None)),
        title=f"RPCA low-rank common (rank={int(rpca_stats['rank'])}, log1p)",
        cmap="magma",
        vmin=0,
        vmax=log_vmax,
    )
    # 1b. sparse mean-|S|
    _save_imshow(
        OUTPUT_DIR / "rpca_sparse_mean_abs.png",
        rpca_sparse_map,
        title="RPCA mean |sparse| (dust / frame-local)",
        cmap="inferno",
        vmin=0,
        vmax=float(np.percentile(rpca_sparse_map, 99)),
    )
    # 1c. median + soft-threshold sparse
    _save_imshow(
        OUTPUT_DIR / "median_soft_sparse_mean_abs.png",
        med_sparse_map,
        title="median+soft-threshold mean |S|",
        cmap="inferno",
        vmin=0,
        vmax=float(np.percentile(med_sparse_map, 99)),
    )

    # RPCA gallery
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    panels = [
        (np.log1p(common_median), "v1 median common\n(log1p)", "magma", 0, log_vmax),
        (np.log1p(np.clip(rpca_common, 0, None)), "RPCA low-rank common\n(log1p)", "magma", 0, log_vmax),
        (np.log1p(np.clip(svd2_common, 0, None)), "trunc-SVD rank-2 mean\n(log1p)", "magma", 0, log_vmax),
        (rpca_sparse_map, "RPCA mean |S|", "inferno", 0, float(np.percentile(rpca_sparse_map, 99))),
        (med_sparse_map, "median+soft |S|", "inferno", 0, float(np.percentile(med_sparse_map, 99))),
        (np.abs(rpca_common - common_median), "|RPCA−median|", "viridis", 0, float(np.percentile(np.abs(rpca_common - common_median), 99))),
    ]
    for ax, (img, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(OUTPUT_DIR / "rpca_gallery_log1p.png", dpi=170)
    plt.close(fig)

    # 2. Angular harmonics
    recon = harm["reconstructions"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    # Energy bar
    ax = axes[0, 0]
    ks = list(range(1, min(13, harm["power_vs_k"].size)))
    fracs = [harm["harm_frac"].get(k, 0.0) for k in ks]
    colors = ["#e76f51" if k in (1, 4) else "#4878a8" for k in ks]
    ax.bar(ks, fracs, color=colors)
    ax.set_xlabel("angular harmonic k")
    ax.set_ylabel("AC power fraction")
    ax.set_title("circular-harmonic energy (median common)", fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0, 1]
    im = ax.imshow(harm["angular_ac_map"], cmap="magma")
    ax.set_title("angular AC energy map\n(ring std → cartesian)", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[0, 2]
    im = ax.imshow(np.log1p(np.clip(recon["k0_radial"], 0, None)), cmap="magma", vmin=0, vmax=log_vmax)
    ax.set_title("k=0 radial common (log1p)", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # Signed oscillatory harmonics
    for ax, key, title in (
        (axes[1, 0], "k1", "k=1 dipole field"),
        (axes[1, 1], "k4", "k=4 quadrupole×2 field"),
        (axes[1, 2], "k0_plus_k1_plus_k4", "k0+k1+k4 recon (log1p)"),
    ):
        img = recon[key]
        if key.startswith("k0_plus"):
            im = ax.imshow(np.log1p(np.clip(img, 0, None)), cmap="magma", vmin=0, vmax=log_vmax)
        else:
            lim = float(np.percentile(np.abs(img), 99)) or 1.0
            im = ax.imshow(img, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(OUTPUT_DIR / "angular_harmonics_gallery.png", dpi=170)
    plt.close(fig)

    _save_imshow(
        OUTPUT_DIR / "angular_k0_radial_log1p.png",
        np.log1p(np.clip(recon["k0_radial"], 0, None)),
        title="angular k=0 radial common (log1p)",
        cmap="magma",
        vmin=0,
        vmax=log_vmax,
    )
    _save_imshow(
        OUTPUT_DIR / "angular_k0k1k4_recon_log1p.png",
        np.log1p(np.clip(recon["k0_plus_k1_plus_k4"], 0, None)),
        title="k0+k1+k4 angular recon (log1p)",
        cmap="magma",
        vmin=0,
        vmax=log_vmax,
    )

    # 3. PCA PC1–PC3 + NMF
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    for k in range(3):
        mode = pca_modes[k]
        lim = float(np.percentile(np.abs(mode), 99)) or 1.0
        im = axes[0, k].imshow(mode, cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[0, k].set_title(f"PCA PC{k+1} (signed)\nvar={pca_fracs[k]:.1%}", fontsize=9)
        axes[0, k].axis("off")
        fig.colorbar(im, ax=axes[0, k], fraction=0.046, pad=0.03)

        base = nmf_bases[k]
        im = axes[1, k].imshow(np.log1p(np.clip(base, 0, None)), cmap="magma", vmin=0, vmax=log_vmax)
        axes[1, k].set_title(f"NMF W{k+1} (log1p)\nE={nmf_info['component_energy_dn2'][k]:.1f}", fontsize=9)
        axes[1, k].axis("off")
        fig.colorbar(im, ax=axes[1, k], fraction=0.046, pad=0.03)
    fig.savefig(OUTPUT_DIR / "pca_nmf_spatial_modes.png", dpi=170)
    plt.close(fig)

    # Individual log1p NMF / PCA panels
    for k in range(3):
        _save_imshow(
            OUTPUT_DIR / f"pca_pc{k+1}_signed.png",
            pca_modes[k],
            title=f"PCA PC{k+1} signed (var={pca_fracs[k]:.1%})",
            cmap="coolwarm",
            vmin=-float(np.percentile(np.abs(pca_modes[k]), 99)),
            vmax=float(np.percentile(np.abs(pca_modes[k]), 99)),
        )
        _save_imshow(
            OUTPUT_DIR / f"nmf_w{k+1}_log1p.png",
            np.log1p(np.clip(nmf_bases[k], 0, None)),
            title=f"NMF component W{k+1} (log1p)",
            cmap="magma",
            vmin=0,
            vmax=log_vmax,
        )

    # Overview comparison strip
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    overview = [
        (np.log1p(common_median), "v1 median"),
        (np.log1p(np.clip(rpca_common, 0, None)), "RPCA low-rank"),
        (np.log1p(np.clip(recon["k0_plus_k1_plus_k4"], 0, None)), "k0+k1+k4"),
        (np.log1p(np.clip(nmf_bases[0], 0, None)), "NMF W1"),
    ]
    for ax, (img, title) in zip(axes, overview):
        ax.imshow(img, cmap="magma", vmin=0, vmax=log_vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(OUTPUT_DIR / "overview_bases_log1p.png", dpi=170)
    plt.close(fig)

    # Agreement metrics
    agreement = {
        "median_vs_rpca_corr": _corr(common_median, rpca_common),
        "median_vs_svd2_corr": _corr(common_median, svd2_common),
        "median_vs_k0_corr": _corr(common_median, recon["k0_radial"]),
        "median_vs_k0k1k4_corr": _corr(common_median, recon["k0_plus_k1_plus_k4"]),
        "median_vs_nmf_w1_corr": _corr(common_median, nmf_bases[0]),
        "pca_pc1_vs_nmf_w1_corr": _corr(pca_modes[0], nmf_bases[0]),
        "by_zone_median_vs_rpca": {
            z: _corr(common_median, rpca_common, (rho >= lo) & (rho < hi)) for z, (lo, hi) in ZONES.items()
        },
    }

    method_menu = {
        "implemented_in_v2": [
            {
                "name": "Robust PCA (GoDec-style)",
                "why": "low-rank shared field vs sparse dust/scratches across 24 frames",
            },
            {
                "name": "median + soft-threshold sparse residual",
                "why": "lightweight robust split without joint SVD",
            },
            {
                "name": "circular harmonics (k=0,1,4) on median common",
                "why": "microlens/fixture geometry is approximately polar-separable",
            },
            {
                "name": "PCA PC1–PC3 spatial modes",
                "why": "v1 only showed PC1; PC2/PC3 capture secondary shared asymmetries",
            },
            {
                "name": "NMF rank-3 non-negative parts",
                "why": "intensity is non-negative; parts often split seam vs fixture vs glow",
            },
        ],
        "listed_not_run_now": [
            {
                "name": "ICA / sparse coding",
                "reason": "24 frames is small for stable ICA; modes tend to overfit dust; defer until more frames or patch-based ICA",
            },
            {
                "name": "polar-separable SVD (full radial×angular matrix SVD)",
                "reason": "partially covered by circular harmonics; full separable SVD needs denser polar resampling QA first",
            },
            {
                "name": "multi-scale Gaussian/wavelet common",
                "reason": "v1 already high-passed interior; pyramid common adds little until RPCA/NMF interpreted",
            },
            {
                "name": "temporal clustering then templates",
                "reason": "frame order here is sample index not true time; cluster templates useful after labeling capture batches",
            },
            {
                "name": "DMD / dynamic modes",
                "reason": "no meaningful temporal dynamics in this 24-frame static stack",
            },
            {
                "name": "full PCP (nuclear-norm RPCA)",
                "reason": "GoDec rank-k + soft-threshold is the practical equivalent at this size; exact PCP is heavier with similar maps",
            },
        ],
    }

    summary = {
        "scope": (
            "v2 base-mode extraction on the same 24 registered dark-port microlens frames; "
            "methods beyond v1 median / trimmed-mean / PC1"
        ),
        "data": {
            "raw_dir": str(RAW_DIR.relative_to(ROOT)),
            "detections": str(DETECTIONS.relative_to(ROOT)),
            "n_frames": len(names),
            "grid_size": GRID_SIZE,
            "crop_radius_scale": CROP_RADIUS_SCALE,
            "files": names,
        },
        "rpca": rpca_stats,
        "median_soft_sparse": med_sparse_stats,
        "angular_harmonics": {
            "harm_frac_k1_to_k12": harm["harm_frac"],
            "dc_power": harm["dc_power"],
            "highlighted": {
                "k1": harm["harm_frac"].get(1, 0.0),
                "k2": harm["harm_frac"].get(2, 0.0),
                "k4": harm["harm_frac"].get(4, 0.0),
            },
        },
        "pca": {
            "explained_variance_frac": [float(v) for v in pca_fracs],
            "loadings_mean": [float(np.mean(pca_loadings[:, k])) for k in range(pca_loadings.shape[1])],
        },
        "nmf": nmf_info,
        "estimator_agreement": agreement,
        "method_menu": method_menu,
        "artifacts": sorted(p.name for p in OUTPUT_DIR.glob("*.png")) + ["summary.json"],
        "vs_v1": {
            "v1_had": ["median", "10% trimmed mean", "PCA PC1 only", "radial/angular profiles of median"],
            "v2_adds": [
                "RPCA low-rank common + sparse dust map",
                "median+soft-threshold sparse map",
                "explicit k=0/1/4 circular-harmonic fields + energy bars",
                "PCA PC2/PC3 signed spatial modes",
                "NMF W1–W3 non-negative parts",
            ],
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "rpca": rpca_stats,
                "angular_highlighted": summary["angular_harmonics"]["highlighted"],
                "pca_fracs": summary["pca"]["explained_variance_frac"],
                "nmf_energies": nmf_info["component_energy_dn2"],
                "agreement": agreement,
                "n_png": len(list(OUTPUT_DIR.glob("*.png"))),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
