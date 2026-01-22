from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from mini_grin_rebuild.core.configs import ExperimentConfig, load_experiment_config
from mini_grin_rebuild.core.json_io import read_json, write_json
from mini_grin_rebuild.core.runs import RunPaths, collect_run_meta, create_run
from mini_grin_rebuild.evaluation.evaluator import evaluate_checkpoint, evaluate_oracle_poisson, evaluate_pseudo_poisson
from mini_grin_rebuild.training.trainer import train_dataset


def _infer_project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            base[k] = _deep_update(dict(base[k]), v)
        else:
            base[k] = v
    return base


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    return [int(values)]


def _metric_get(d: Mapping[str, Any], path: str) -> float:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return float("nan")
        cur = cur[part]
    try:
        return float(cur)
    except Exception:
        return float("nan")


def _safe_std(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) <= 1:
        return float("nan")
    return float(np.std(vals, ddof=1))


def _safe_nanmean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _write_csv(path: Path, rows: list[dict[str, Any]], *, columns: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], *, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows:
        vals = []
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


@dataclass(frozen=True)
class SuiteResult:
    suite_run: RunPaths
    summary_path: Path


def run_ablation_suite(
    *,
    base_config_path: str | Path,
    suite_path: str | Path,
    data_root: str | Path,
    eval_split: str = "test",
    num_plots: int = 0,
    name: str | None = None,
) -> SuiteResult:
    base_config_path = Path(base_config_path).expanduser()
    suite_path = Path(suite_path).expanduser()
    data_root = Path(data_root).expanduser()

    base_cfg = load_experiment_config(base_config_path)
    suite_spec = read_json(suite_path)

    suite_name = str(suite_spec.get("name") or suite_path.stem)
    seeds = _as_int_list(suite_spec.get("seeds"))
    if not seeds:
        seed0 = base_cfg.training.seed if base_cfg.training.seed is not None else 42
        seeds = [int(seed0)]

    project_root = _infer_project_root(base_config_path)
    runs_root = project_root / base_cfg.paths.runs_dir

    suite_run = create_run(
        runs_root,
        name=f"suite_{suite_name}" + (f"_{name}" if name else ""),
        argv=None,
        config_snapshot={"base_config": base_cfg.to_dict(), "suite": suite_spec, "suite_path": str(suite_path)},
    )

    # Record suite meta in one place for reproducibility.
    write_json(
        suite_run.root / "suite_meta.json",
        {
            "suite_name": suite_name,
            "suite_path": str(suite_path),
            "base_config_path": str(base_config_path),
            "data_root": str(data_root),
            "eval_split": eval_split,
            "seeds": seeds,
            "run_meta": collect_run_meta(argv=None),
        },
    )

    results: list[dict[str, Any]] = []

    baselines = suite_spec.get("baselines", [])
    for item in baselines:
        if not isinstance(item, Mapping):
            raise ValueError("Each baselines[] entry must be an object")
        if item.get("enabled", True) is False:
            continue
        method = str(item.get("method") or item.get("name") or "")
        label = str(item.get("name") or method)
        if method not in {"pseudo_poisson", "oracle_poisson"}:
            raise ValueError(f"Unknown baseline method {method!r} in suite baselines")

        run = create_run(
            runs_root,
            name=f"{suite_run.root.name}_{label}_{eval_split}",
            argv=None,
            config_snapshot={"base_config": base_cfg.to_dict(), "suite": suite_spec, "entry": dict(item)},
        )
        if method == "pseudo_poisson":
            eval_result = evaluate_pseudo_poisson(
                base_cfg,
                data_root=data_root,
                split=eval_split,
                out_dir=run.root,
                num_plots=int(num_plots),
            )
        else:
            eval_result = evaluate_oracle_poisson(
                base_cfg,
                data_root=data_root,
                split=eval_split,
                out_dir=run.root,
                num_plots=int(num_plots),
            )

        write_json(run.root / "artifacts.json", {"dataset_root": str(data_root), "method": method, "split": eval_split})
        results.append(
            {
                "kind": "baseline",
                "name": label,
                "seed": None,
                "method": method,
                "run_dir": str(run.root),
                "eval_dir": str(run.root),
                "eval_metrics_path": str(run.root / "eval_metrics.json"),
                "eval": eval_result,
            }
        )

    experiments = suite_spec.get("experiments", [])
    for exp in experiments:
        if not isinstance(exp, Mapping):
            raise ValueError("Each experiments[] entry must be an object")
        if exp.get("enabled", True) is False:
            continue
        exp_name = str(exp.get("name") or "")
        if not exp_name:
            raise ValueError("Each experiments[] entry must have a non-empty name")
        overrides = exp.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, Mapping):
            raise ValueError(f"experiments[{exp_name}].overrides must be an object")

        for seed in seeds:
            cfg_dict = base_cfg.to_dict()
            cfg_dict = _deep_update(cfg_dict, overrides)
            cfg_dict.setdefault("training", {})["seed"] = int(seed)
            cfg = ExperimentConfig.from_dict(cfg_dict)

            run = create_run(
                runs_root,
                name=f"{suite_run.root.name}_{exp_name}_s{seed}",
                argv=None,
                config_snapshot=cfg.to_dict(),
            )
            outs = train_dataset(cfg, data_root=data_root, run=run, resume=None)
            write_json(
                run.root / "artifacts.json",
                {"dataset_root": str(data_root), "best_checkpoint": str(outs.best_checkpoint), "last_checkpoint": str(outs.last_checkpoint)},
            )
            eval_result = evaluate_checkpoint(
                cfg,
                data_root=data_root,
                split=eval_split,
                checkpoint_path=outs.best_checkpoint,
                out_dir=run.root / "eval",
                num_plots=int(num_plots),
            )

            results.append(
                {
                    "kind": "train",
                    "name": exp_name,
                    "seed": int(seed),
                    "method": "nn",
                    "run_dir": str(run.root),
                    "checkpoint": str(outs.best_checkpoint),
                    "eval_dir": str(run.root / "eval"),
                    "eval_metrics_path": str(run.root / "eval" / "eval_metrics.json"),
                    "eval": eval_result,
                }
            )

    summary = {
        "suite_name": suite_name,
        "suite_path": str(suite_path),
        "base_config_path": str(base_config_path),
        "data_root": str(data_root),
        "eval_split": eval_split,
        "seeds": seeds,
        "results": results,
    }
    summary_path = suite_run.root / "suite_summary.json"
    write_json(summary_path, summary)

    # Tabular export.
    columns = [
        "kind",
        "name",
        "seed",
        "method",
        "qc_pass_rate",
        "qc_physics_p95_abs_p95",
        "qc_edge_p95_abs_p95",
        "qc_outside_p95_abs_p95",
        "defect_f1",
        "defect_iou",
        "defect_auprc",
        "defect_auroc",
        "defect_psnr",
        "defect_rmse",
        "defect_corr",
        "aperture_rmse",
        "edge_rmse",
        "center_rmse",
        "edge_artifact_mean_abs",
        "edge_artifact_p95_abs",
        "global_psnr",
        "global_rmse",
        "run_dir",
    ]
    table_rows: list[dict[str, Any]] = []
    for r in results:
        e = r.get("eval") or {}
        row = {
            "kind": r.get("kind"),
            "name": r.get("name"),
            "seed": r.get("seed"),
            "method": r.get("method"),
            "qc_pass_rate": _metric_get(e, "qc.pass_rate"),
            "qc_physics_p95_abs_p95": _metric_get(e, "qc.summary.physics_p95_abs_p95"),
            "qc_edge_p95_abs_p95": _metric_get(e, "qc.summary.edge_p95_abs_p95"),
            "qc_outside_p95_abs_p95": _metric_get(e, "qc.summary.outside_p95_abs_p95"),
            "defect_f1": _metric_get(e, "summary_defect.f1"),
            "defect_iou": _metric_get(e, "summary_defect.iou"),
            "defect_auprc": _metric_get(e, "summary_defect.auprc"),
            "defect_auroc": _metric_get(e, "summary_defect.auroc"),
            "defect_psnr": _metric_get(e, "summary_defect.psnr"),
            "defect_rmse": _metric_get(e, "summary_defect.rmse"),
            "defect_corr": _metric_get(e, "summary_defect.corr"),
            "aperture_rmse": _metric_get(e, "summary_regions.aperture.rmse"),
            "edge_rmse": _metric_get(e, "summary_regions.edge.rmse"),
            "center_rmse": _metric_get(e, "summary_regions.center.rmse"),
            "edge_artifact_mean_abs": _metric_get(e, "artifacts.edge_mean_abs"),
            "edge_artifact_p95_abs": _metric_get(e, "artifacts.edge_p95_abs"),
            "global_psnr": _metric_get(e, "summary.psnr"),
            "global_rmse": _metric_get(e, "summary.rmse"),
            "run_dir": r.get("run_dir"),
        }
        table_rows.append(row)

    _write_csv(suite_run.root / "suite_table.csv", table_rows, columns=columns)
    _write_markdown_table(suite_run.root / "suite_table.md", table_rows, columns=columns)

    # Aggregate across seeds if needed.
    if len(seeds) > 1:
        aggregates: list[dict[str, Any]] = []
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in table_rows:
            if row["kind"] != "train":
                continue
            key = (str(row["name"]), str(row["method"]))
            groups.setdefault(key, []).append(row)
        for (exp_name, method), rows in sorted(groups.items()):
            agg = {"name": exp_name, "method": method, "n": len(rows)}
            for col in (
                "defect_f1",
                "defect_iou",
                "defect_auprc",
                "defect_auroc",
                "defect_psnr",
                "defect_rmse",
                "defect_corr",
                "aperture_rmse",
                "edge_rmse",
                "center_rmse",
                "edge_artifact_mean_abs",
                "edge_artifact_p95_abs",
            ):
                vals = [float(r[col]) for r in rows if isinstance(r.get(col), float)]
                agg[col] = _safe_nanmean(vals)
                agg[col + "_std"] = _safe_std(vals)
            aggregates.append(agg)
        write_json(suite_run.root / "suite_aggregate.json", aggregates)

    return SuiteResult(suite_run=suite_run, summary_path=summary_path)


__all__ = ["SuiteResult", "run_ablation_suite"]
