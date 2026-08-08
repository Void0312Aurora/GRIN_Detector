"""Random real frames minus common bases → residual matrix figure.

Rows: randomly sampled registered dark-port crops.
Columns: frame (log1p) | residual vs each basis (log1p domain).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

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

from analyze_reflection_common_modes_v2_bases import (  # noqa: E402
    CROP_RADIUS_SCALE,
    DETECTIONS,
    GRID_SIZE,
    RAW_DIR,
    _angular_harmonic_decompose,
    _nmf_bases,
    _robust_pca_godec,
)
from compare_reflection_dark_port import _load_real_crops_dn  # noqa: E402

OUTPUT_DIR = ROOT / "analysis_outputs" / "reflection_common_modes_v2_bases"
N_SAMPLE = 4
SEED = 20260724


def _log1p_clip(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x.astype(np.float64), 0.0, None))


def main() -> int:
    stack, rho, names = _load_real_crops_dn(
        raw_dir=RAW_DIR,
        detections_path=DETECTIONS,
        grid_size=GRID_SIZE,
        crop_radius_scale=CROP_RADIUS_SCALE,
    )
    names = list(names)
    n = stack.shape[0]
    rng = np.random.default_rng(SEED)
    idxs = np.sort(rng.choice(n, size=min(N_SAMPLE, n), replace=False))

    median = np.median(stack, axis=0).astype(np.float32)
    rpca_common, _, rpca_stats = _robust_pca_godec(stack, rank=2, n_iter=25)
    harm = _angular_harmonic_decompose(median, rho, keep_ks=(0, 1, 2, 4))
    k014 = harm["reconstructions"]["k0_plus_k1_plus_k4"]
    nmf_bases, _, nmf_info = _nmf_bases(stack, n_comp=3)
    nmf_w1 = nmf_bases[0]

    bases = [
        ("median", median),
        ("RPCA-L", rpca_common),
        ("k0+k1+k4", k014),
        ("NMF-W1", nmf_w1),
    ]

    # Residuals in log1p domain: log1p(frame) - log1p(basis)
    log_vmax = float(np.log1p(255.0))
    residuals = []
    for i in idxs:
        frame = stack[i]
        row_res = []
        for _, basis in bases:
            diff = _log1p_clip(frame) - _log1p_clip(basis)
            row_res.append(diff.astype(np.float32))
        residuals.append(row_res)

    # Shared clim from aperture interior+rim of all residuals
    aperture = rho <= 1.08
    all_abs = np.concatenate([np.abs(d[aperture]).ravel() for row in residuals for d in row])
    clim = float(np.percentile(all_abs, 99.5)) if all_abs.size else 1.0
    clim = max(clim, 0.2)

    n_rows = len(idxs)
    n_cols = 1 + len(bases)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.asarray([axes])

    for r, i in enumerate(idxs):
        frame = stack[i]
        ax0 = axes[r, 0]
        im0 = ax0.imshow(_log1p_clip(frame), cmap="magma", vmin=0.0, vmax=log_vmax)
        ax0.set_ylabel(names[i], fontsize=9)
        if r == 0:
            ax0.set_title("real (log1p)", fontsize=9)
        ax0.set_xticks([])
        ax0.set_yticks([])
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

        for c, ((bname, _), diff) in enumerate(zip(bases, residuals[r])):
            ax = axes[r, c + 1]
            im = ax.imshow(diff, cmap="coolwarm", vmin=-clim, vmax=clim)
            if r == 0:
                ax.set_title(f"− {bname}\n(log1p)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"Random frames − bases (seed={SEED}, log1p residual, clim=±{clim:.2f})",
        fontsize=11,
    )
    out_png = OUTPUT_DIR / "random_frame_minus_bases_matrix.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    # Also DN residual matrix for reference (aperture clim)
    residuals_dn = []
    for i in idxs:
        frame = stack[i]
        residuals_dn.append([(frame.astype(np.float64) - b.astype(np.float64)).astype(np.float32) for _, b in bases])
    all_abs_dn = np.concatenate([np.abs(d[aperture]).ravel() for row in residuals_dn for d in row])
    clim_dn = float(np.percentile(all_abs_dn, 99.0)) if all_abs_dn.size else 50.0

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.asarray([axes])
    for r, i in enumerate(idxs):
        ax0 = axes[r, 0]
        im0 = ax0.imshow(stack[i], cmap="gray", vmin=0, vmax=255)
        ax0.set_ylabel(names[i], fontsize=9)
        if r == 0:
            ax0.set_title("real (DN)", fontsize=9)
        ax0.set_xticks([])
        ax0.set_yticks([])
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)
        for c, ((bname, _), diff) in enumerate(zip(bases, residuals_dn[r])):
            ax = axes[r, c + 1]
            im = ax.imshow(diff, cmap="coolwarm", vmin=-clim_dn, vmax=clim_dn)
            if r == 0:
                ax.set_title(f"− {bname}\n(DN)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"Random frames − bases DN (seed={SEED}, clim=±{clim_dn:.1f})",
        fontsize=11,
    )
    out_dn = OUTPUT_DIR / "random_frame_minus_bases_matrix_dn.png"
    fig.savefig(out_dn, dpi=170)
    plt.close(fig)

    # Per-cell RMSE summary (log1p, aperture)
    rows_meta = []
    for r, i in enumerate(idxs):
        entry = {"file": names[i], "index": int(i), "rmse_log1p_aperture": {}}
        for (bname, _), diff in zip(bases, residuals[r]):
            entry["rmse_log1p_aperture"][bname] = float(np.sqrt(np.mean(diff[aperture] ** 2)))
        rows_meta.append(entry)

    summary = {
        "seed": SEED,
        "n_sample": int(len(idxs)),
        "sampled_files": [names[i] for i in idxs],
        "bases": [b[0] for b in bases],
        "residual_domain_primary": "log1p(frame)-log1p(basis)",
        "clim_log1p": clim,
        "clim_dn": clim_dn,
        "rpca_rank": rpca_stats.get("rank"),
        "nmf_w1_energy": nmf_info["component_energy_dn2"][0],
        "per_frame": rows_meta,
        "artifacts": [
            str(out_png.relative_to(ROOT)).replace("\\", "/"),
            str(out_dn.relative_to(ROOT)).replace("\\", "/"),
        ],
    }
    out_json = OUTPUT_DIR / "random_frame_minus_bases_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_png}")
    print(f"wrote {out_dn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
