from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

import argparse
import math
from typing import Any, Mapping

from mini_grin_rebuild.core.configs import ExperimentConfig, load_experiment_config
from mini_grin_rebuild.core.json_io import read_json, write_json
from mini_grin_rebuild.core.runs import create_run
from mini_grin_rebuild.data.generate_dataset import generate_dataset
from mini_grin_rebuild.evaluation.evaluator import evaluate_checkpoint, evaluate_pseudo_poisson
from mini_grin_rebuild.training.trainer import train_dataset


def _infer_project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


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


def _fmt(v: float) -> str:
    if not math.isfinite(v):
        return ""
    return f"{v:.6g}"


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
                vals.append(_fmt(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_dataset(
    cfg: ExperimentConfig,
    *,
    output_root: Path,
    train: int,
    val: int,
    test: int,
    seed: int,
    overwrite: bool,
) -> None:
    output_root = Path(output_root)
    if overwrite and output_root.exists():
        import shutil

        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.rglob("*.npz")) and not overwrite:
        return
    generate_dataset(
        cfg.simulation,
        output_root=output_root,
        train=train,
        val=val,
        test=test,
        seed=seed,
        config_snapshot=cfg.to_dict(),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Train a 20x (FOV~200um) model and evaluate defect-distribution OOD sets.")
    p.add_argument("--config", default="mini_grin_rebuild/configs/benchmark_microlens200_srt_20x.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-count", type=int, default=240)
    p.add_argument("--val-count", type=int, default=60)
    p.add_argument("--test-count", type=int, default=60)
    p.add_argument("--ood-test-count", type=int, default=60)
    p.add_argument("--num-plots", type=int, default=8)
    p.add_argument("--device", default=None, help="Override device (cuda/cpu).")
    p.add_argument("--name", default="20x_ood_suite")
    p.add_argument("--overwrite-datasets", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser()
    base_cfg = load_experiment_config(cfg_path)
    if args.device is not None:
        cfg_dict = base_cfg.to_dict()
        cfg_dict.setdefault("training", {})["device"] = str(args.device)
        base_cfg = ExperimentConfig.from_dict(cfg_dict)

    project_root = _infer_project_root(cfg_path)
    runs_root = project_root / base_cfg.paths.runs_dir
    run = create_run(runs_root, name=str(args.name), argv=None, config_snapshot=base_cfg.to_dict())

    datasets_root = run.root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)

    # 20x convention here: keep FOV ~200um but use smaller dx (higher sampling).
    fov_um = float(base_cfg.simulation.dx) * float(base_cfg.simulation.grid_size)
    write_json(
        run.root / "suite_meta.json",
        {
            "base_config_path": str(cfg_path),
            "fov_um": fov_um,
            "seed": int(args.seed),
            "counts": {
                "train": int(args.train_count),
                "val": int(args.val_count),
                "test": int(args.test_count),
                "ood_test": int(args.ood_test_count),
            },
        },
    )

    # 1) ID dataset + training.
    ds_id = datasets_root / "id"
    _ensure_dataset(
        base_cfg,
        output_root=ds_id,
        train=int(args.train_count),
        val=int(args.val_count),
        test=int(args.test_count),
        seed=int(args.seed),
        overwrite=bool(args.overwrite_datasets),
    )

    outs = train_dataset(base_cfg, data_root=ds_id, run=run, resume=None)
    write_json(
        run.root / "artifacts.json",
        {
            "dataset_root": str(ds_id),
            "best_checkpoint": str(outs.best_checkpoint),
            "last_checkpoint": str(outs.last_checkpoint),
        },
    )

    ckpt = outs.best_checkpoint

    # 2) Define defect-distribution OOD scenarios (physics unchanged; only sampling ranges differ).
    scenarios: list[dict[str, Any]] = [
        {
            "name": "ood_scratch_only",
            "overrides": {"simulation": {"defect_scratch_prob": 1.0}},
        },
        {
            "name": "ood_dot_only",
            "overrides": {"simulation": {"defect_scratch_prob": 0.0}},
        },
        {
            "name": "ood_small",
            "overrides": {
                "simulation": {
                    "defect_sigma_min_um": 0.2,
                    "defect_sigma_max_um": 0.8,
                    "scratch_width_min_um": 0.2,
                    "scratch_width_max_um": 0.8,
                    "scratch_length_min_um": 5.0,
                    "scratch_length_max_um": 20.0,
                    "defect_amplitude_wrap_min": 0.05,
                    "defect_amplitude_wrap_max": 0.3
                }
            },
        },
        {
            "name": "ood_large",
            "overrides": {
                "simulation": {
                    "defect_sigma_min_um": 3.0,
                    "defect_sigma_max_um": 7.0,
                    "scratch_width_min_um": 3.0,
                    "scratch_width_max_um": 7.0,
                    "scratch_length_min_um": 80.0,
                    "scratch_length_max_um": 150.0,
                    "defect_amplitude_wrap_min": 0.3,
                    "defect_amplitude_wrap_max": 1.0
                }
            },
        },
        {
            "name": "ood_pos_uniform",
            "overrides": {
                "simulation": {
                    "defect_center_sigma_norm": 0.0,
                    "defect_center_max_radius_norm": 0.5
                }
            },
        },
    ]

    # 3) Evaluate ID + OOD datasets with pseudo-poisson baseline and the trained NN.
    rows: list[dict[str, Any]] = []

    def _eval_one(tag: str, *, cfg: ExperimentConfig, dataset_root: Path, split: str, num_plots: int) -> None:
        eval_root = run.root / "eval" / tag
        eval_root.mkdir(parents=True, exist_ok=True)

        base = evaluate_pseudo_poisson(cfg, data_root=dataset_root, split=split, out_dir=eval_root / "pseudo_poisson", num_plots=num_plots)
        nn = evaluate_checkpoint(
            cfg,
            data_root=dataset_root,
            split=split,
            checkpoint_path=ckpt,
            out_dir=eval_root / "nn",
            num_plots=num_plots,
        )

        for method, res, subdir in [
            ("pseudo_poisson", base, "pseudo_poisson"),
            ("nn", nn, "nn"),
        ]:
            defect_rmse_um = _safe_get(res, "summary_defect.rmse")
            global_rmse_um = _safe_get(res, "summary.rmse")
            edge_p95_abs = _safe_get(res, "artifacts.edge_p95_abs")
            rows.append(
                {
                    "scenario": tag,
                    "method": method,
                    "count": int(res.get("count", 0)),
                    "defect_rmse_um": defect_rmse_um,
                    "defect_rmse_nm": defect_rmse_um * 1e3 if math.isfinite(defect_rmse_um) else float("nan"),
                    "defect_auprc": _safe_get(res, "summary_defect.auprc"),
                    "defect_f1": _safe_get(res, "summary_defect.f1"),
                    "global_rmse_um": global_rmse_um,
                    "global_rmse_nm": global_rmse_um * 1e3 if math.isfinite(global_rmse_um) else float("nan"),
                    "edge_art_p95_abs": edge_p95_abs,
                    "eval_dir": str(eval_root / subdir),
                    "plots_dir": str(eval_root / subdir / "plots"),
                }
            )

    # ID
    _eval_one("id", cfg=base_cfg, dataset_root=ds_id, split="test", num_plots=int(args.num_plots))

    # OOD
    for sc in scenarios:
        name = str(sc["name"])
        cfg_dict = _deep_update(base_cfg.to_dict(), sc.get("overrides") or {})
        sc_cfg = ExperimentConfig.from_dict(cfg_dict)

        ds_root = datasets_root / name
        _ensure_dataset(
            sc_cfg,
            output_root=ds_root,
            train=0,
            val=0,
            test=int(args.ood_test_count),
            seed=int(args.seed),
            overwrite=bool(args.overwrite_datasets),
        )

        _eval_one(name, cfg=base_cfg, dataset_root=ds_root, split="test", num_plots=max(2, int(args.num_plots // 2)))

    columns = [
        "scenario",
        "method",
        "count",
        "defect_rmse_um",
        "defect_rmse_nm",
        "defect_auprc",
        "defect_f1",
        "global_rmse_um",
        "global_rmse_nm",
        "edge_art_p95_abs",
        "eval_dir",
        "plots_dir",
    ]
    _write_markdown_table(run.root / "ood_report.md", rows, columns=columns)

    # Attach dataset meta for auditability.
    datasets: dict[str, Any] = {}
    for tag in ["id", *[str(s["name"]) for s in scenarios]]:
        meta_path = datasets_root / tag / "dataset_meta.json"
        datasets[tag] = {"root": str(datasets_root / tag), "meta": read_json(meta_path) if meta_path.exists() else None}

    write_json(
        run.root / "ood_report.json",
        {
            "run_root": str(run.root),
            "checkpoint": str(ckpt),
            "base_config_path": str(cfg_path),
            "base_config": base_cfg.to_dict(),
            "rows": rows,
            "datasets": datasets,
        },
    )
    print(str(run.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

