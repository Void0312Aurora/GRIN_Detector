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
from mini_grin_rebuild.core.json_io import write_json
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
    p = argparse.ArgumentParser(description="Train/eval current model under large-defect phase-jump stress.")
    p.add_argument("--config", default="configs/benchmark_microlens200_srt_gated_q99_best.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-count", type=int, default=240)
    p.add_argument("--val-count", type=int, default=60)
    p.add_argument("--test-count", type=int, default=60)
    p.add_argument("--num-plots", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default="phase_jump_suite")
    p.add_argument("--overwrite-datasets", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser()
    base_cfg = load_experiment_config(cfg_path)
    base_dict = base_cfg.to_dict()
    if args.device is not None:
        base_dict.setdefault("training", {})["device"] = str(args.device)
    base_cfg = ExperimentConfig.from_dict(base_dict)

    project_root = _infer_project_root(cfg_path)
    runs_root = project_root / base_cfg.paths.runs_dir
    run = create_run(runs_root, name=str(args.name), argv=None, config_snapshot=base_cfg.to_dict())

    datasets_root = run.root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)

    id_cfg = base_cfg
    jump_cfg = ExperimentConfig.from_dict(
        _deep_update(
            base_cfg.to_dict(),
            {
                "simulation": {
                    "large_defect_prob": 1.0,
                    "large_defect_sigma_min_um": 2.5,
                    "large_defect_sigma_max_um": 5.0,
                    "large_scratch_width_min_um": 1.5,
                    "large_scratch_width_max_um": 4.0,
                    "large_scratch_length_min_um": 15.0,
                    "large_scratch_length_max_um": 28.0,
                    "large_defect_amplitude_wrap_min": 1.1,
                    "large_defect_amplitude_wrap_max": 2.2,
                    "defect_center_sigma_norm": 0.12,
                    "defect_center_max_radius_norm": 0.35,
                }
            },
        )
    )

    ds_id = datasets_root / "id"
    ds_jump = datasets_root / "phase_jump_large"
    _ensure_dataset(
        id_cfg,
        output_root=ds_id,
        train=int(args.train_count),
        val=int(args.val_count),
        test=int(args.test_count),
        seed=int(args.seed),
        overwrite=bool(args.overwrite_datasets),
    )
    _ensure_dataset(
        jump_cfg,
        output_root=ds_jump,
        train=int(args.train_count),
        val=int(args.val_count),
        test=int(args.test_count),
        seed=int(args.seed) + 1000,
        overwrite=bool(args.overwrite_datasets),
    )

    outs = train_dataset(id_cfg, data_root=ds_id, run=run, resume=None)
    ckpt = outs.best_checkpoint

    eval_id_pseudo = evaluate_pseudo_poisson(id_cfg, data_root=ds_id, split="test", out_dir=run.root / "eval" / "id" / "pseudo_poisson", num_plots=int(args.num_plots))
    eval_id_nn = evaluate_checkpoint(id_cfg, data_root=ds_id, split="test", checkpoint_path=ckpt, out_dir=run.root / "eval" / "id" / "nn", num_plots=int(args.num_plots))
    eval_jump_pseudo = evaluate_pseudo_poisson(jump_cfg, data_root=ds_jump, split="test", out_dir=run.root / "eval" / "phase_jump_large" / "pseudo_poisson", num_plots=int(args.num_plots))
    eval_jump_nn = evaluate_checkpoint(jump_cfg, data_root=ds_jump, split="test", checkpoint_path=ckpt, out_dir=run.root / "eval" / "phase_jump_large" / "nn", num_plots=int(args.num_plots))

    summary = {
        "checkpoint": str(ckpt),
        "datasets": {
            "id": str(ds_id),
            "phase_jump_large": str(ds_jump),
        },
        "results": {
            "id": {
                "pseudo_poisson": eval_id_pseudo,
                "nn": eval_id_nn,
            },
            "phase_jump_large": {
                "pseudo_poisson": eval_jump_pseudo,
                "nn": eval_jump_nn,
            },
        },
        "key_metrics": {
            "id": {
                "pseudo_defect_f1": _safe_get(eval_id_pseudo, "summary_defect.f1"),
                "nn_defect_f1": _safe_get(eval_id_nn, "summary_defect.f1"),
                "pseudo_defect_rmse_um": _safe_get(eval_id_pseudo, "summary_defect.rmse"),
                "nn_defect_rmse_um": _safe_get(eval_id_nn, "summary_defect.rmse"),
            },
            "phase_jump_large": {
                "pseudo_defect_f1": _safe_get(eval_jump_pseudo, "summary_defect.f1"),
                "nn_defect_f1": _safe_get(eval_jump_nn, "summary_defect.f1"),
                "pseudo_defect_rmse_um": _safe_get(eval_jump_pseudo, "summary_defect.rmse"),
                "nn_defect_rmse_um": _safe_get(eval_jump_nn, "summary_defect.rmse"),
                "pseudo_qc_pass_rate": _safe_get(eval_jump_pseudo, "gate.pass_rate"),
                "nn_qc_pass_rate": _safe_get(eval_jump_nn, "gate.pass_rate"),
            },
        },
    }
    write_json(run.root / "phase_jump_summary.json", summary)

    print(str(run.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
