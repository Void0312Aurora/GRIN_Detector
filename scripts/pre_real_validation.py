from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

import argparse
import math
from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.core.configs import ExperimentConfig, load_experiment_config
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.core.runs import create_run
from mini_grin_rebuild.data.generate_dataset import generate_dataset
from mini_grin_rebuild.evaluation.evaluator import evaluate_checkpoint


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_update(dict(out[k]), v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _safe_get(d: Mapping[str, Any], path: str) -> float:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return float("nan")
        cur = cur[part]
    try:
        return float(cur)
    except Exception:
        return float("nan")


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], *, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows:
        vals: list[str] = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("")
                else:
                    vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fov_um(cfg: ExperimentConfig) -> float:
    return float(cfg.simulation.dx) * float(cfg.simulation.grid_size)


def main() -> int:
    p = argparse.ArgumentParser(description="Minimal pre-real-data validation sweeps (resolution/noise/calibration).")
    p.add_argument("--config", required=True, help="Base experiment config (should match checkpoint training flags).")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path to evaluate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-count", type=int, default=40, help="Number of synthetic test samples per scenario.")
    p.add_argument("--device", default=None, help="Override device (e.g. cuda/cpu).")
    p.add_argument("--name", default="pre_real_validation", help="Run name suffix.")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser()
    base_cfg = load_experiment_config(cfg_path)
    project_root = cfg_path.resolve().parents[1] if cfg_path.parent.name == "configs" else cfg_path.resolve().parent
    runs_root = project_root / base_cfg.paths.runs_dir
    run = create_run(runs_root, name=str(args.name), argv=None, config_snapshot=base_cfg.to_dict())

    ckpt = str(Path(args.checkpoint).expanduser())

    datasets_root = run.root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)

    # Common helpers for scenario creation.
    base_fov = _fov_um(base_cfg)
    base_dict = base_cfg.to_dict()
    if args.device is not None:
        base_dict.setdefault("training", {})["device"] = str(args.device)

    scenarios: list[dict[str, Any]] = []

    # A) Resolution transfer (same FOV, different grid_size).
    for grid, count, bs in [(64, int(args.test_count), 8), (256, int(args.test_count), 2), (512, min(20, int(args.test_count)), 1)]:
        dx = float(base_fov) / float(grid)
        name = f"res_grid{grid}_dx{dx:.6g}"
        cfg_d = _deep_update(
            base_dict,
            {"simulation": {"grid_size": int(grid), "dx": float(dx)}, "training": {"batch_size": int(bs)}},
        )
        scenarios.append({"group": "resolution", "name": name, "cfg": ExperimentConfig.from_dict(cfg_d), "count": int(count)})

    # B) Noise sweep (keep grid=64, vary noise_level > 0 to keep RNG stream comparable).
    for nl in [0.002, 0.005, 0.01, 0.02]:
        name = f"noise_sigma{nl:g}"
        cfg_d = _deep_update(base_dict, {"simulation": {"noise_level": float(nl)}, "training": {"batch_size": 8}})
        scenarios.append({"group": "noise", "name": name, "cfg": ExperimentConfig.from_dict(cfg_d), "count": int(args.test_count)})

    # C) Aperture radius sensitivity (QC only; reuse the base dataset).
    for ap in [0.98, 1.0, 1.02, 0.95, 1.05]:
        name = f"aperture_r{ap:g}"
        cfg_d = _deep_update(base_dict, {"simulation": {"lens_radius_fraction": float(ap)}, "training": {"batch_size": 8}})
        scenarios.append({"group": "aperture", "name": name, "cfg": ExperimentConfig.from_dict(cfg_d), "count": int(args.test_count), "reuse_dataset": True})

    # D) dx calibration sensitivity (affects phase inputs + pseudo-poisson prior scaling).
    for s in [0.98, 1.0, 1.02, 0.95, 1.05]:
        dx = float(base_cfg.simulation.dx) * float(s)
        name = f"dx_scale{s:g}"
        cfg_d = _deep_update(base_dict, {"simulation": {"dx": float(dx)}, "training": {"batch_size": 8}})
        scenarios.append({"group": "dx", "name": name, "cfg": ExperimentConfig.from_dict(cfg_d), "count": int(args.test_count), "reuse_dataset": True})

    # Generate datasets + run eval.
    rows: list[dict[str, Any]] = []
    dataset_cache: dict[str, Path] = {}

    for s in scenarios:
        cfg = s["cfg"]
        name = str(s["name"])
        group = str(s["group"])
        count = int(s["count"])
        reuse = bool(s.get("reuse_dataset", False))
        ds_sim_cfg = base_cfg.simulation if reuse else cfg.simulation

        # Dataset key: scenarios that only change QC masks should reuse the same synthetic data.
        ds_key = f"grid{ds_sim_cfg.grid_size}_dx{ds_sim_cfg.dx}_noise{ds_sim_cfg.noise_level}"

        if ds_key not in dataset_cache:
            ds_root = datasets_root / ds_key
            generate_dataset(
                ds_sim_cfg,
                output_root=ds_root,
                train=0,
                val=0,
                test=count,
                seed=int(args.seed),
                config_snapshot=asdict(ds_sim_cfg),
            )
            dataset_cache[ds_key] = ds_root

        ds_root = dataset_cache[ds_key]
        out_dir = run.root / "eval" / group / name
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_result = evaluate_checkpoint(
            cfg,
            data_root=ds_root,
            split="test",
            checkpoint_path=ckpt,
            out_dir=out_dir,
            num_plots=0,
        )

        row = {
            "group": group,
            "name": name,
            "grid": int(ds_sim_cfg.grid_size),
            "dx": float(ds_sim_cfg.dx),
            "noise": float(ds_sim_cfg.noise_level),
            "lens_r": float(cfg.simulation.lens_radius_fraction),
            "batch": int(cfg.training.batch_size),
            "qc_pass_rate": _safe_get(eval_result, "qc.pass_rate"),
            "qc_edge_p95_abs_p95": _safe_get(eval_result, "qc.summary.edge_p95_abs_p95"),
            "qc_outside_p95_abs_p95": _safe_get(eval_result, "qc.summary.outside_p95_abs_p95"),
            "qc_physics_p95_abs_p95": _safe_get(eval_result, "qc.summary.physics_p95_abs_p95"),
            "defect_rmse": _safe_get(eval_result, "summary_defect.rmse"),
            "defect_f1": _safe_get(eval_result, "summary_defect.f1"),
            "defect_auprc": _safe_get(eval_result, "summary_defect.auprc"),
            "global_rmse": _safe_get(eval_result, "summary.rmse"),
            "eval_dir": str(out_dir),
            "dataset_root": str(ds_root),
        }
        rows.append(row)

    columns = [
        "group",
        "name",
        "grid",
        "dx",
        "noise",
        "lens_r",
        "batch",
        "qc_pass_rate",
        "qc_physics_p95_abs_p95",
        "qc_edge_p95_abs_p95",
        "qc_outside_p95_abs_p95",
        "defect_f1",
        "defect_auprc",
        "defect_rmse",
        "global_rmse",
        "eval_dir",
        "dataset_root",
    ]
    _write_markdown_table(run.root / "validation_table.md", rows, columns=columns)
    write_json(run.root / "validation_report.json", {"config": str(cfg_path), "checkpoint": ckpt, "rows": rows})
    print(str(run.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
