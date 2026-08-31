from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from mini_grin_rebuild.core.configs import load_experiment_config
from mini_grin_rebuild.data.datasets import DefectDataset
from mini_grin_rebuild.evaluation.metrics import defect_mask


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p75": None, "p90": None, "p95": None, "p99": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "mean": float(np.mean(arr)),
    }


def _hotspot_mask(defect: np.ndarray, phase_scale: float, dx: float, aperture: np.ndarray) -> np.ndarray:
    peak_h = float(np.max(np.abs(defect)))
    height_hot = np.abs(defect) > max(1e-4, 0.2 * peak_h)

    defect_phase = defect * phase_scale
    gy, gx = np.gradient(defect_phase, dx)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_peak = float(np.max(grad_mag))
    grad_hot = grad_mag >= 0.2 * grad_peak if grad_peak > 0 else np.zeros_like(grad_mag, dtype=bool)
    return aperture & (height_hot | grad_hot)


def _accumulate_axis(
    stats: dict[str, list[float] | int],
    *,
    g_std: np.ndarray,
    d_def: np.ndarray,
    g_test: np.ndarray,
    mask: np.ndarray,
) -> None:
    if not np.any(mask):
        return

    g_abs = np.abs(g_std[mask])
    d_abs = np.abs(d_def[mask])
    t_abs = np.abs(g_test[mask])
    ratio = d_abs / np.maximum(g_abs, 1e-12)
    residual = np.abs(g_abs - d_abs) / np.maximum(g_abs, 1e-12)
    same_sign = np.sign(g_std[mask]) == np.sign(g_test[mask])
    flip = np.sign(g_std[mask]) != np.sign(g_test[mask])
    dominance = d_abs < g_abs
    dominance_half = d_abs < 0.5 * g_abs
    near_equal = d_abs >= 0.9 * g_abs
    defect_dominates = d_abs > g_abs

    cast = lambda key: stats[key]  # noqa: E731
    cast("g_std_abs").extend(g_abs.tolist())
    cast("d_def_abs").extend(d_abs.tolist())
    cast("g_test_abs").extend(t_abs.tolist())
    cast("ratio_abs_d_over_g").extend(ratio.tolist())
    cast("residual_abs_g_minus_d_over_g").extend(residual.tolist())
    stats["count"] += int(mask.sum())
    stats["same_sign_count"] += int(np.sum(same_sign))
    stats["flip_count"] += int(np.sum(flip))
    stats["dominance_count"] += int(np.sum(dominance))
    stats["dominance_half_count"] += int(np.sum(dominance_half))
    stats["near_equal_count"] += int(np.sum(near_equal))
    stats["defect_dominates_count"] += int(np.sum(defect_dominates))


def _new_axis_bucket() -> dict[str, list[float] | int]:
    return {
        "g_std_abs": [],
        "d_def_abs": [],
        "g_test_abs": [],
        "ratio_abs_d_over_g": [],
        "residual_abs_g_minus_d_over_g": [],
        "count": 0,
        "same_sign_count": 0,
        "flip_count": 0,
        "dominance_count": 0,
        "dominance_half_count": 0,
        "near_equal_count": 0,
        "defect_dominates_count": 0,
    }


def _summarize_axis(stats: dict[str, list[float] | int]) -> dict[str, object]:
    count = int(stats["count"])
    if count == 0:
        return {
            "count": 0,
            "flip_frac": None,
            "same_sign_frac": None,
            "abs_d_lt_abs_g_frac": None,
            "abs_d_lt_half_abs_g_frac": None,
            "abs_d_ge_0p9_abs_g_frac": None,
            "abs_d_gt_abs_g_frac": None,
            "|g_std|": _quantiles([]),
            "|d_defect|": _quantiles([]),
            "|g_test|": _quantiles([]),
            "|d|/|g_std|": _quantiles([]),
            "||g_std|-|d||/|g_std|": _quantiles([]),
        }

    return {
        "count": count,
        "flip_frac": float(stats["flip_count"]) / count,
        "same_sign_frac": float(stats["same_sign_count"]) / count,
        "abs_d_lt_abs_g_frac": float(stats["dominance_count"]) / count,
        "abs_d_lt_half_abs_g_frac": float(stats["dominance_half_count"]) / count,
        "abs_d_ge_0p9_abs_g_frac": float(stats["near_equal_count"]) / count,
        "abs_d_gt_abs_g_frac": float(stats["defect_dominates_count"]) / count,
        "|g_std|": _quantiles(stats["g_std_abs"]),
        "|d_defect|": _quantiles(stats["d_def_abs"]),
        "|g_test|": _quantiles(stats["g_test_abs"]),
        "|d|/|g_std|": _quantiles(stats["ratio_abs_d_over_g"]),
        "||g_std|-|d||/|g_std|": _quantiles(stats["residual_abs_g_minus_d_over_g"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit in-wrap phase-gradient relationships for sign feasibility.")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON.")
    parser.add_argument("--dataset-root", required=True, help="Root containing wrap-mixed datasets.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory.")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    dataset_root = Path(args.dataset_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (Path(cfg.paths.runs_dir) / f"{timestamp}_in_wrap_phase_gradient_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    from mini_grin_rebuild.physics.phase import phase_scale as _shared_phase_scale

    phase_scale = _shared_phase_scale(cfg.simulation)
    dx = float(cfg.simulation.dx)
    h = int(cfg.simulation.grid_size)
    w = int(cfg.simulation.grid_size)
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h), np.linspace(-1.0, 1.0, w), indexing="ij")
    aperture = (xx**2 + yy**2) <= (float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0) ** 2 + 1e-12)

    region_names = ("defect_mask", "hotspot")
    subset_names = ("all", "std_active")
    axis_names = ("x", "y")

    overall = {
        region: {subset: {axis: _new_axis_bucket() for axis in axis_names} for subset in subset_names}
        for region in region_names
    }

    payload: dict[str, object] = {
        "created_local": datetime.now().isoformat(),
        "config": str(Path(args.config).resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "selection": "in_wrap test samples only",
        "definitions": {
            "g_std": "phase gradient of standard surface",
            "d_defect": "phase gradient of defect phase = gradient(test_phase - std_phase)",
            "g_test": "phase gradient of test surface = g_std + d_defect",
            "sign_feasibility_focus": "whether sign(g_test) can stay aligned with sign(g_std) inside defect-related regions",
            "std_active_mask": "|g_std| >= max(0.05 * peak(|g_std|), 1e-6) on each axis of each sample",
        },
        "simulation": {
            "dx_um": dx,
            "phase_scale_rad_per_um": phase_scale,
            "grid_size": h,
        },
        "datasets": {},
    }

    for dataset_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        ds = DefectDataset(dataset_dir, "test")
        buckets = {
            region: {subset: {axis: _new_axis_bucket() for axis in axis_names} for subset in subset_names}
            for region in region_names
        }
        sample_count = 0

        for i in range(len(ds)):
            sample = ds[i]
            if int(sample["wrap_class_id"].item()) != 0:
                continue
            sample_count += 1

            standard = sample["standard"].squeeze(0).cpu().numpy().astype(np.float64)
            test = sample["test"].squeeze(0).cpu().numpy().astype(np.float64)
            defect = sample["defect"].squeeze(0).cpu().numpy().astype(np.float64)

            std_phase = standard * phase_scale
            test_phase = test * phase_scale
            g_y, g_x = np.gradient(std_phase, dx)
            t_y, t_x = np.gradient(test_phase, dx)
            d_y = t_y - g_y
            d_x = t_x - g_x

            gx_active_thr = max(0.05 * float(np.max(np.abs(g_x))), 1e-6)
            gy_active_thr = max(0.05 * float(np.max(np.abs(g_y))), 1e-6)
            gx_active = np.abs(g_x) >= gx_active_thr
            gy_active = np.abs(g_y) >= gy_active_thr

            defect_true_ts = sample["defect"].unsqueeze(0)
            defect_region = defect_mask(
                defect_true_ts,
                abs_threshold=float(cfg.training.eval_defect_abs_threshold),
                rel_threshold=float(cfg.training.eval_defect_rel_threshold),
                dilate_px=int(cfg.training.eval_defect_dilate_px),
            ).cpu().numpy().astype(bool)
            hot_region = _hotspot_mask(defect, phase_scale, dx, aperture)

            for region_name, region_mask in (("defect_mask", defect_region), ("hotspot", hot_region)):
                for subset_name, mask_x, mask_y in (
                    ("all", region_mask, region_mask),
                    ("std_active", region_mask & gx_active, region_mask & gy_active),
                ):
                    _accumulate_axis(
                        buckets[region_name][subset_name]["x"],
                        g_std=g_x,
                        d_def=d_x,
                        g_test=t_x,
                        mask=mask_x,
                    )
                    _accumulate_axis(
                        buckets[region_name][subset_name]["y"],
                        g_std=g_y,
                        d_def=d_y,
                        g_test=t_y,
                        mask=mask_y,
                    )
                    _accumulate_axis(
                        overall[region_name][subset_name]["x"],
                        g_std=g_x,
                        d_def=d_x,
                        g_test=t_x,
                        mask=mask_x,
                    )
                    _accumulate_axis(
                        overall[region_name][subset_name]["y"],
                        g_std=g_y,
                        d_def=d_y,
                        g_test=t_y,
                        mask=mask_y,
                    )

        payload["datasets"][dataset_dir.name] = {
            "sample_count": sample_count,
            "regions": {
                region: {
                    subset: {axis: _summarize_axis(buckets[region][subset][axis]) for axis in axis_names}
                    for subset in subset_names
                }
                for region in region_names
            },
        }

    payload["overall"] = {
        "regions": {
            region: {
                subset: {axis: _summarize_axis(overall[region][subset][axis]) for axis in axis_names}
                for subset in subset_names
            }
            for region in region_names
        }
    }

    out_path = out_dir / "in_wrap_phase_gradient_audit.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
