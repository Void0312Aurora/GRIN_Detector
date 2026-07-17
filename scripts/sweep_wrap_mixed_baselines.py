from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

import argparse
from typing import Any, Mapping

from mini_grin_rebuild.core.configs import ExperimentConfig, load_experiment_config
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.core.runs import create_run
from mini_grin_rebuild.data.generate_dataset import generate_dataset
from mini_grin_rebuild.evaluation.evaluator import (
    evaluate_pseudo_poisson,
    evaluate_residue_cut_poisson,
)


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


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep mixed-wrap coarse baselines on microlens_srt data.")
    p.add_argument("--config", default="configs/wrap_stress_mixed_v1.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-count", type=int, default=240)
    p.add_argument("--val-count", type=int, default=60)
    p.add_argument("--test-count", type=int, default=60)
    p.add_argument("--num-plots", type=int, default=0)
    p.add_argument("--name", default="wrap_mixed_baseline_sweep")
    p.add_argument("--overwrite-datasets", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser()
    base_cfg = load_experiment_config(cfg_path)
    base_cfg = ExperimentConfig.from_dict(
        _deep_update(base_cfg.to_dict(), {"training": {"device": str(args.device)}})
    )

    project_root = _infer_project_root(cfg_path)
    runs_root = project_root / base_cfg.paths.runs_dir
    sweep_run = create_run(runs_root, name=str(args.name), argv=None, config_snapshot=base_cfg.to_dict())

    datasets_root = sweep_run.root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)

    large_ranges = [
        (1.2, 1.6),
        (1.6, 2.0),
        (2.0, 2.4),
        (2.4, 2.8),
    ]
    rows: list[dict[str, Any]] = []

    for idx, (lo, hi) in enumerate(large_ranges):
        cfg = ExperimentConfig.from_dict(
            _deep_update(
                base_cfg.to_dict(),
                {
                    "simulation": {
                        "large_defect_amplitude_wrap_min": float(lo),
                        "large_defect_amplitude_wrap_max": float(hi),
                    }
                },
            )
        )
        ds_root = datasets_root / f"large_{lo:.1f}_{hi:.1f}"
        _ensure_dataset(
            cfg,
            output_root=ds_root,
            train=int(args.train_count),
            val=int(args.val_count),
            test=int(args.test_count),
            seed=int(args.seed) + 1000 + idx,
            overwrite=bool(args.overwrite_datasets),
        )

        eval_root = sweep_run.root / "eval" / f"large_{lo:.1f}_{hi:.1f}"
        pseudo = evaluate_pseudo_poisson(cfg, data_root=ds_root, split="test", out_dir=eval_root / "pseudo_poisson", num_plots=int(args.num_plots))
        residue_cut = evaluate_residue_cut_poisson(
            cfg,
            data_root=ds_root,
            split="test",
            out_dir=eval_root / "residue_cut_poisson",
            num_plots=int(args.num_plots),
        )
        rows.append(
            {
                "large_wrap_min": float(lo),
                "large_wrap_max": float(hi),
                "pseudo_cross_rmse": _safe_get(pseudo, "summary_wrap_groups.cross_wrap.summary_defect.rmse"),
                "pseudo_cross_slope": _safe_get(pseudo, "summary_wrap_groups.cross_wrap.summary_defect.slope"),
                "pseudo_in_rmse": _safe_get(pseudo, "summary_wrap_groups.in_wrap.summary_defect.rmse"),
                "residue_cut_cross_rmse": _safe_get(residue_cut, "summary_wrap_groups.cross_wrap.summary_defect.rmse"),
                "residue_cut_cross_slope": _safe_get(residue_cut, "summary_wrap_groups.cross_wrap.summary_defect.slope"),
                "residue_cut_in_rmse": _safe_get(residue_cut, "summary_wrap_groups.in_wrap.summary_defect.rmse"),
                "pseudo_cross_count": _safe_get(pseudo, "summary_wrap_groups.cross_wrap.count"),
                "pseudo_in_count": _safe_get(pseudo, "summary_wrap_groups.in_wrap.count"),
            }
        )

    write_json(
        sweep_run.root / "wrap_mixed_baseline_sweep_summary.json",
        {
            "large_ranges": large_ranges,
            "rows": rows,
        },
    )
    print(str(sweep_run.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
