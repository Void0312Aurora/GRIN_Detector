from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from mini_grin_rebuild.core.configs import load_experiment_config
from mini_grin_rebuild.data.datasets import DefectDataset
from mini_grin_rebuild.evaluation.metrics import defect_mask
from mini_grin_rebuild.physics.factory import create_forward_model
from mini_grin_rebuild.reconstruction import (
    reconstruct_defect_first_order_poisson,
    reconstruct_defect_first_order_sign_quadratic_poisson,
    reconstruct_defect_pseudo_poisson,
)
from mini_grin_rebuild.reconstruction.pseudo_poisson import reconstruct_defect_unwrap_poisson


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: Callable


METHODS = (
    MethodSpec("pseudo_poisson", reconstruct_defect_pseudo_poisson),
    MethodSpec("first_order_poisson", reconstruct_defect_first_order_poisson),
    MethodSpec("first_order_sign_quadratic_poisson", reconstruct_defect_first_order_sign_quadratic_poisson),
    MethodSpec("unwrap_poisson", reconstruct_defect_unwrap_poisson),
)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr))


def _rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    diff = pred[mask] - target[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def _corr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if int(np.sum(mask)) < 2:
        return None
    p = pred[mask].reshape(-1)
    t = target[mask].reshape(-1)
    p = p - np.mean(p)
    t = t - np.mean(t)
    den = float(np.sqrt(np.sum(p * p)) * np.sqrt(np.sum(t * t)))
    if den < 1e-12:
        return 0.0
    return float(np.sum(p * t) / den)


def _slope(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    yt = target[mask]
    yp = pred[mask]
    den = float(np.sum(yt * yt)) + 1e-8
    return float(np.sum(yt * yp) / den)


def _peak_ratio(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    t_peak = float(np.max(np.abs(target[mask])))
    if t_peak < 1e-12:
        return None
    p_peak = float(np.max(np.abs(pred[mask])))
    return p_peak / t_peak


def _hotspot_mask(defect: np.ndarray, phase_scale: float, dx: float, aperture: np.ndarray) -> np.ndarray:
    peak_h = float(np.max(np.abs(defect)))
    height_hot = np.abs(defect) > max(1e-4, 0.2 * peak_h)

    defect_phase = defect * phase_scale
    gy, gx = np.gradient(defect_phase, dx)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_peak = float(np.max(grad_mag))
    grad_hot = grad_mag >= 0.2 * grad_peak if grad_peak > 0 else np.zeros_like(grad_mag, dtype=bool)
    return aperture & (height_hot | grad_hot)


def _branch_and_defect_sign_stats(
    sample: dict[str, torch.Tensor],
    *,
    phase_scale: float,
    dx: float,
    aperture: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    standard = sample["standard"].squeeze(0).cpu().numpy().astype(np.float64)
    test = sample["test"].squeeze(0).cpu().numpy().astype(np.float64)
    defect = sample["defect"].squeeze(0).cpu().numpy().astype(np.float64)
    diff_ix = sample["inputs"][0].cpu().numpy().astype(np.float64)
    diff_iy = sample["inputs"][1].cpu().numpy().astype(np.float64)

    std_phase = standard * phase_scale
    test_phase = test * phase_scale
    g_y, g_x = np.gradient(std_phase, dx)
    t_y, t_x = np.gradient(test_phase, dx)
    d_true_y = t_y - g_y
    d_true_x = t_x - g_x

    gx_abs = np.abs(g_x)
    gy_abs = np.abs(g_y)
    gx_floor = max(float(gx_abs.max()) * 0.05, 1e-6)
    gy_floor = max(float(gy_abs.max()) * 0.05, 1e-6)
    denom_x = 2.0 * np.where(gx_abs >= gx_floor, g_x, np.sign(g_x) * gx_floor)
    denom_y = 2.0 * np.where(gy_abs >= gy_floor, g_y, np.sign(g_y) * gy_floor)
    denom_x = np.where(denom_x == 0.0, 2.0 * gx_floor, denom_x)
    denom_y = np.where(denom_y == 0.0, 2.0 * gy_floor, denom_y)
    d_first_x = diff_ix / denom_x
    d_first_y = diff_iy / denom_y

    std_sign_x = np.where(np.sign(g_x) == 0.0, 1.0, np.sign(g_x))
    std_sign_y = np.where(np.sign(g_y) == 0.0, 1.0, np.sign(g_y))
    branch_first_x = np.where(np.sign(g_x + d_first_x) == 0.0, std_sign_x, np.sign(g_x + d_first_x))
    branch_first_y = np.where(np.sign(g_y + d_first_y) == 0.0, std_sign_y, np.sign(g_y + d_first_y))
    d_quad_x = std_sign_x * np.sqrt(np.clip(g_x**2 + diff_ix, 0.0, None)) - g_x
    d_quad_y = std_sign_y * np.sqrt(np.clip(g_y**2 + diff_iy, 0.0, None)) - g_y

    hot = _hotspot_mask(defect, phase_scale, dx, aperture)
    stats: dict[str, dict[str, float | int | None]] = {}
    for axis, g_test, d_true, branch_quad, branch_first, defect_quad, defect_first in (
        ("x", t_x, d_true_x, std_sign_x, branch_first_x, np.sign(d_quad_x), np.sign(d_first_x)),
        ("y", t_y, d_true_y, std_sign_y, branch_first_y, np.sign(d_quad_y), np.sign(d_first_y)),
    ):
        branch_truth = np.sign(g_test)
        defect_truth = np.sign(d_true)
        branch_valid = hot & (np.abs(g_test) > max(1e-8, 1e-4 * float(np.max(np.abs(g_test)))))
        defect_valid = hot & (np.abs(d_true) > max(1e-9, 1e-4 * float(np.max(np.abs(d_true)))))
        stats[axis] = {
            "branch_count": int(np.sum(branch_valid)),
            "branch_quadratic_ok": int(np.sum(branch_quad[branch_valid] == branch_truth[branch_valid])),
            "branch_first_order_ok": int(np.sum(branch_first[branch_valid] == branch_truth[branch_valid])),
            "defect_count": int(np.sum(defect_valid)),
            "defect_quadratic_ok": int(np.sum(defect_quad[defect_valid] == defect_truth[defect_valid])),
            "defect_first_order_ok": int(np.sum(defect_first[defect_valid] == defect_truth[defect_valid])),
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit in-wrap hotspot sign and reconstruction metrics.")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing wrap-mixed datasets.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory.")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    dataset_root = Path(args.dataset_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (Path(cfg.paths.runs_dir) / f"{timestamp}_in_wrap_hotspot_audit_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    physics = create_forward_model(cfg.simulation, cfg.training, device=device, freeze=True)
    from mini_grin_rebuild.physics.phase import phase_scale as _shared_phase_scale

    phase_scale = _shared_phase_scale(cfg.simulation)

    results: dict[str, object] = {
        "config": str(Path(args.config).resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "created_local": datetime.now().isoformat(),
        "datasets": {},
    }
    rows: list[dict[str, object]] = []

    for dataset_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        ds = DefectDataset(dataset_dir, "test")
        h = int(cfg.simulation.grid_size)
        w = int(cfg.simulation.grid_size)
        yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h), np.linspace(-1.0, 1.0, w), indexing="ij")
        aperture = (xx**2 + yy**2) <= (float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0) ** 2 + 1e-12)

        sign_acc = {
            axis: {
                "branch_count": 0,
                "branch_quadratic_ok": 0,
                "branch_first_order_ok": 0,
                "defect_count": 0,
                "defect_quadratic_ok": 0,
                "defect_first_order_ok": 0,
            }
            for axis in ("x", "y")
        }
        method_metrics = {
            spec.name: {
                "default_rmse": [],
                "default_slope": [],
                "default_corr": [],
                "default_peak_ratio": [],
                "hot_rmse": [],
                "hot_slope": [],
                "hot_corr": [],
                "hot_peak_ratio": [],
            }
            for spec in METHODS
        }

        in_wrap_count = 0
        for i in range(len(ds)):
            sample = ds[i]
            if int(sample["wrap_class_id"].item()) != 0:
                continue
            in_wrap_count += 1

            defect_true_ts = sample["defect"].unsqueeze(0)
            defect_true = defect_true_ts.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
            hot = _hotspot_mask(defect_true, phase_scale, float(cfg.simulation.dx), aperture)
            default_mask = defect_mask(
                defect_true_ts,
                abs_threshold=float(cfg.training.eval_defect_abs_threshold),
                rel_threshold=float(cfg.training.eval_defect_rel_threshold),
                dilate_px=int(cfg.training.eval_defect_dilate_px),
            ).cpu().numpy().astype(bool)

            sign_stats = _branch_and_defect_sign_stats(
                sample,
                phase_scale=phase_scale,
                dx=float(cfg.simulation.dx),
                aperture=aperture,
            )
            for axis in ("x", "y"):
                for key, value in sign_stats[axis].items():
                    sign_acc[axis][key] += int(value or 0)

            batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
            diff_ts = {"I_x": batch["inputs"][:, 0], "I_y": batch["inputs"][:, 1]}
            standard = batch["standard"]

            for spec in METHODS:
                defect_pred_ts = spec.fn(
                    physics=physics,
                    standard_height=standard,
                    diff_ts=diff_ts,
                    defect_roi_radius=float(cfg.training.defect_roi_radius),
                    apply_edge_offset=True,
                    poisson_pad=int(getattr(cfg.training, "pseudo_poisson_poisson_pad", 0) or 0),
                    pad_mode=str(getattr(cfg.training, "pseudo_poisson_pad_mode", "reflect")),
                    apply_edge_taper=bool(getattr(cfg.training, "pseudo_poisson_apply_edge_taper", False)),
                    taper_margin=float(getattr(cfg.training, "pseudo_poisson_taper_margin", 0.25)),
                )
                defect_pred = defect_pred_ts.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
                mm = method_metrics[spec.name]
                for prefix, mask in (("default", default_mask), ("hot", hot)):
                    rmse_v = _rmse(defect_pred, defect_true, mask)
                    slope_v = _slope(defect_pred, defect_true, mask)
                    corr_v = _corr(defect_pred, defect_true, mask)
                    peak_v = _peak_ratio(defect_pred, defect_true, mask)
                    if rmse_v is not None:
                        mm[f"{prefix}_rmse"].append(rmse_v)
                    if slope_v is not None:
                        mm[f"{prefix}_slope"].append(slope_v)
                    if corr_v is not None:
                        mm[f"{prefix}_corr"].append(corr_v)
                    if peak_v is not None:
                        mm[f"{prefix}_peak_ratio"].append(peak_v)

        dataset_result = {
            "in_wrap_count": in_wrap_count,
            "hotspot_sign": {
                axis: {
                    "branch_quadratic_acc": (100.0 * vals["branch_quadratic_ok"] / vals["branch_count"]) if vals["branch_count"] else None,
                    "branch_first_order_acc": (100.0 * vals["branch_first_order_ok"] / vals["branch_count"]) if vals["branch_count"] else None,
                    "defect_quadratic_acc": (100.0 * vals["defect_quadratic_ok"] / vals["defect_count"]) if vals["defect_count"] else None,
                    "defect_first_order_acc": (100.0 * vals["defect_first_order_ok"] / vals["defect_count"]) if vals["defect_count"] else None,
                    "counts": vals,
                }
                for axis, vals in sign_acc.items()
            },
            "methods": {
                name: {metric: _mean(values) for metric, values in metrics.items()}
                for name, metrics in method_metrics.items()
            },
        }
        results["datasets"][dataset_dir.name] = dataset_result

        for axis in ("x", "y"):
            hs = dataset_result["hotspot_sign"][axis]
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "kind": "hotspot_sign",
                    "axis": axis,
                    "method": "quadratic_vs_first_order",
                    "branch_quadratic_acc": hs["branch_quadratic_acc"],
                    "branch_first_order_acc": hs["branch_first_order_acc"],
                    "defect_quadratic_acc": hs["defect_quadratic_acc"],
                    "defect_first_order_acc": hs["defect_first_order_acc"],
                    "default_rmse": None,
                    "default_slope": None,
                    "hot_rmse": None,
                    "hot_slope": None,
                    "hot_corr": None,
                    "hot_peak_ratio": None,
                }
            )
        for method_name, metrics in dataset_result["methods"].items():
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "kind": "reconstruction",
                    "axis": "",
                    "method": method_name,
                    "branch_quadratic_acc": None,
                    "branch_first_order_acc": None,
                    "defect_quadratic_acc": None,
                    "defect_first_order_acc": None,
                    "default_rmse": metrics["default_rmse"],
                    "default_slope": metrics["default_slope"],
                    "hot_rmse": metrics["hot_rmse"],
                    "hot_slope": metrics["hot_slope"],
                    "hot_corr": metrics["hot_corr"],
                    "hot_peak_ratio": metrics["hot_peak_ratio"],
                }
            )

    json_path = out_dir / "in_wrap_hotspot_audit_full.json"
    csv_path = out_dir / "in_wrap_hotspot_audit_full.csv"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(str(out_dir))


if __name__ == "__main__":
    main()
