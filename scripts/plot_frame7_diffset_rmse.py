"""Use frame 7 as reference base; build difference set and RMSE."""

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
)
from compare_reflection_dark_port import _load_real_crops_dn  # noqa: E402

OUTPUT_DIR = ROOT / "analysis_outputs" / "reflection_common_modes_v2_bases"
REF_NAME = "7.bmp"
# Show a compact matrix: ref + a few frames; RMSE covers all frames.
SHOW_NAMES = ["7.bmp", "5.bmp", "6.bmp", "14.bmp", "21.bmp", "1.bmp"]


def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x.astype(np.float64), 0.0, None))


def _rmse(a: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a[mask] ** 2)))


def main() -> int:
    stack, rho, names = _load_real_crops_dn(
        raw_dir=RAW_DIR,
        detections_path=DETECTIONS,
        grid_size=GRID_SIZE,
        crop_radius_scale=CROP_RADIUS_SCALE,
    )
    names = list(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    if REF_NAME not in name_to_idx:
        raise SystemExit(f"reference {REF_NAME} not found in {names}")
    ref_i = name_to_idx[REF_NAME]
    ref = stack[ref_i].astype(np.float64)
    ref_log = _log1p(ref)

    aperture = rho <= 1.08
    interior = rho < 0.70
    seam = (rho >= 0.94) & (rho < 1.06)

    per_frame = []
    diffs_log = np.zeros_like(stack, dtype=np.float32)
    diffs_dn = np.zeros_like(stack, dtype=np.float32)
    for i, name in enumerate(names):
        frame = stack[i].astype(np.float64)
        d_dn = (frame - ref).astype(np.float32)
        d_log = (_log1p(frame) - ref_log).astype(np.float32)
        diffs_dn[i] = d_dn
        diffs_log[i] = d_log
        per_frame.append(
            {
                "file": name,
                "index": i,
                "is_reference": name == REF_NAME,
                "rmse_dn_aperture": _rmse(d_dn, aperture),
                "rmse_dn_interior": _rmse(d_dn, interior),
                "rmse_dn_seam": _rmse(d_dn, seam),
                "rmse_log1p_aperture": _rmse(d_log, aperture),
                "rmse_log1p_interior": _rmse(d_log, interior),
                "rmse_log1p_seam": _rmse(d_log, seam),
            }
        )

    # Exclude self-ref (zero) from aggregate stats over others
    others = [p for p in per_frame if not p["is_reference"]]
    agg = {
        "rmse_log1p_aperture_mean": float(np.mean([p["rmse_log1p_aperture"] for p in others])),
        "rmse_log1p_aperture_median": float(np.median([p["rmse_log1p_aperture"] for p in others])),
        "rmse_dn_aperture_mean": float(np.mean([p["rmse_dn_aperture"] for p in others])),
        "rmse_dn_aperture_median": float(np.median([p["rmse_dn_aperture"] for p in others])),
        "rmse_log1p_interior_mean": float(np.mean([p["rmse_log1p_interior"] for p in others])),
        "rmse_dn_interior_mean": float(np.mean([p["rmse_dn_interior"] for p in others])),
    }
    ranking = sorted(others, key=lambda p: p["rmse_log1p_aperture"], reverse=True)

    # Matrix figure: rows = selected frames; cols = real log1p | diff log1p | |diff| log
    show_idxs = []
    for n in SHOW_NAMES:
        if n in name_to_idx:
            show_idxs.append(name_to_idx[n])
    # ensure ref first
    show_idxs = [ref_i] + [i for i in show_idxs if i != ref_i]

    clim = float(
        np.percentile(np.abs(diffs_log[:, aperture]), 99.5)
    )
    clim = max(clim, 0.2)
    log_vmax = float(np.log1p(255.0))

    n_rows = len(show_idxs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9.6, 2.9 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])
    for r, i in enumerate(show_idxs):
        ax0, ax1, ax2 = axes[r]
        im0 = ax0.imshow(_log1p(stack[i]), cmap="magma", vmin=0, vmax=log_vmax)
        label = names[i] + (" [REF]" if i == ref_i else "")
        ax0.set_ylabel(label, fontsize=9)
        if r == 0:
            ax0.set_title("frame (log1p)", fontsize=9)
        ax0.set_xticks([])
        ax0.set_yticks([])
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

        d = diffs_log[i]
        im1 = ax1.imshow(d, cmap="coolwarm", vmin=-clim, vmax=clim)
        if r == 0:
            ax1.set_title(f"frame − {REF_NAME}\n(log1p)", fontsize=9)
        ax1.set_xticks([])
        ax1.set_yticks([])
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

        im2 = ax2.imshow(np.abs(d), cmap="inferno", vmin=0, vmax=clim)
        rmse = per_frame[i]["rmse_log1p_aperture"]
        if r == 0:
            ax2.set_title("|diff| log1p", fontsize=9)
        ax2.set_xlabel(f"RMSE_ap={rmse:.3f}", fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"Difference set vs {REF_NAME} (log1p), clim=±{clim:.2f}",
        fontsize=11,
    )
    out_png = OUTPUT_DIR / "frame7_diffset_matrix_log1p.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    # Bar chart of all-frame RMSE
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    xs = np.arange(len(names))
    vals = [p["rmse_log1p_aperture"] for p in per_frame]
    colors = ["#e76f51" if p["is_reference"] else "#4878a8" for p in per_frame]
    ax.bar(xs, vals, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace(".bmp", "") for n in names], rotation=90, fontsize=8)
    ax.set_ylabel("RMSE log1p (aperture ρ≤1.08)")
    ax.set_title(f"Per-frame RMSE vs reference {REF_NAME}")
    ax.axhline(agg["rmse_log1p_aperture_mean"], color="gray", ls="--", lw=1, label="mean (excl ref)")
    ax.legend(fontsize=8)
    out_bar = OUTPUT_DIR / "frame7_diffset_rmse_bars.png"
    fig.savefig(out_bar, dpi=160)
    plt.close(fig)

    summary = {
        "reference": REF_NAME,
        "n_frames": len(names),
        "residual": "frame - ref (DN) and log1p(frame)-log1p(ref)",
        "aggregate_excl_ref": agg,
        "ranking_highest_rmse_log1p_aperture": [
            {"file": p["file"], "rmse_log1p_aperture": p["rmse_log1p_aperture"]} for p in ranking[:6]
        ],
        "ranking_lowest_rmse_log1p_aperture": [
            {"file": p["file"], "rmse_log1p_aperture": p["rmse_log1p_aperture"]}
            for p in sorted(others, key=lambda p: p["rmse_log1p_aperture"])[:6]
        ],
        "per_frame": per_frame,
        "artifacts": [
            str(out_png.relative_to(ROOT)).replace("\\", "/"),
            str(out_bar.relative_to(ROOT)).replace("\\", "/"),
        ],
    }
    out_json = OUTPUT_DIR / "frame7_diffset_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("reference", "aggregate_excl_ref", "ranking_highest_rmse_log1p_aperture", "ranking_lowest_rmse_log1p_aperture", "artifacts")}, indent=2))
    print(f"wrote {out_png}")
    print(f"wrote {out_bar}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
