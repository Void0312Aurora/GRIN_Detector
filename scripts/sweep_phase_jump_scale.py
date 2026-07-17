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
from mini_grin_rebuild.evaluation.evaluator import evaluate_checkpoint, evaluate_first_order_poisson, evaluate_pseudo_poisson
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
    p = argparse.ArgumentParser(description="Sweep phase-jump scale and observe degradation.")
    p.add_argument("--config", default="configs/benchmark_microlens200_srt_gated_q99_best.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-count", type=int, default=120)
    p.add_argument("--val-count", type=int, default=30)
    p.add_argument("--test-count", type=int, default=30)
    p.add_argument("--num-plots", type=int, default=1)
    p.add_argument("--name", default="phase_jump_scale_sweep")
    p.add_argument("--overwrite-datasets", action="store_true")
    p.add_argument(
        "--train-per-scale",
        action="store_true",
        help="Train one model per phase-jump scale instead of reusing a single ID-trained checkpoint.",
    )
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

    id_cfg = base_cfg
    ds_id = datasets_root / "id"
    shared_checkpoint = None
    shared_train_run = None
    if not bool(args.train_per_scale):
        _ensure_dataset(
            id_cfg,
            output_root=ds_id,
            train=int(args.train_count),
            val=int(args.val_count),
            test=int(args.test_count),
            seed=int(args.seed),
            overwrite=bool(args.overwrite_datasets),
        )

        shared_train_run = create_run(
            runs_root,
            name=f"{sweep_run.root.name}_train",
            argv=None,
            config_snapshot=id_cfg.to_dict(),
        )
        outs = train_dataset(id_cfg, data_root=ds_id, run=shared_train_run, resume=None)
        shared_checkpoint = outs.best_checkpoint

    rows: list[dict[str, Any]] = []
    scales = list(range(2, 16, 2))
    checkpoints: dict[str, str] = {}
    for scale in scales:
        jump_cfg = ExperimentConfig.from_dict(
            _deep_update(
                base_cfg.to_dict(),
                {
                    "simulation": {
                        "large_defect_prob": 1.0,
                        "allow_defect_wrap_exceed": True,
                        "large_defect_sigma_min_um": 2.5,
                        "large_defect_sigma_max_um": 5.0,
                        "large_scratch_width_min_um": 1.5,
                        "large_scratch_width_max_um": 4.0,
                        "large_scratch_length_min_um": 15.0,
                        "large_scratch_length_max_um": 28.0,
                        "large_defect_amplitude_wrap_min": float(scale),
                        "large_defect_amplitude_wrap_max": float(scale),
                        "defect_center_sigma_norm": 0.12,
                        "defect_center_max_radius_norm": 0.35,
                    }
                },
            )
        )
        ds_jump = datasets_root / f"scale_{scale:02d}"
        _ensure_dataset(
            jump_cfg,
            output_root=ds_jump,
            train=int(args.train_count),
            val=int(args.val_count),
            test=int(args.test_count),
            seed=int(args.seed) + 1000 + int(scale),
            overwrite=bool(args.overwrite_datasets),
        )

        if bool(args.train_per_scale):
            train_run = create_run(
                runs_root,
                name=f"{sweep_run.root.name}_scale_{scale:02d}_train",
                argv=None,
                config_snapshot=jump_cfg.to_dict(),
            )
            outs = train_dataset(jump_cfg, data_root=ds_jump, run=train_run, resume=None)
            ckpt = outs.best_checkpoint
        else:
            if shared_checkpoint is None:
                raise RuntimeError("shared_checkpoint missing while train_per_scale=False")
            ckpt = shared_checkpoint
        checkpoints[f"scale_{scale:02d}"] = str(ckpt)

        eval_root = sweep_run.root / "eval" / f"scale_{scale:02d}"
        nn = evaluate_checkpoint(
            jump_cfg,
            data_root=ds_jump,
            split="test",
            checkpoint_path=ckpt,
            out_dir=eval_root / "nn",
            num_plots=int(args.num_plots),
        )
        pseudo = evaluate_pseudo_poisson(
            jump_cfg,
            data_root=ds_jump,
            split="test",
            out_dir=eval_root / "pseudo_poisson",
            num_plots=int(args.num_plots),
        )
        first_order = evaluate_first_order_poisson(
            jump_cfg,
            data_root=ds_jump,
            split="test",
            out_dir=eval_root / "first_order_poisson",
            num_plots=int(args.num_plots),
        )
        rows.append(
            {
                "scale_wrap": int(scale),
                "nn_defect_f1": _safe_get(nn, "summary_defect.f1"),
                "nn_defect_iou": _safe_get(nn, "summary_defect.iou"),
                "nn_defect_rmse_um": _safe_get(nn, "summary_defect.rmse"),
                "nn_global_rmse_um": _safe_get(nn, "summary.rmse"),
                "nn_qc_pass_rate": _safe_get(nn, "qc.pass_rate"),
                "pseudo_defect_f1": _safe_get(pseudo, "summary_defect.f1"),
                "pseudo_defect_iou": _safe_get(pseudo, "summary_defect.iou"),
                "pseudo_defect_rmse_um": _safe_get(pseudo, "summary_defect.rmse"),
                "pseudo_global_rmse_um": _safe_get(pseudo, "summary.rmse"),
                "pseudo_qc_pass_rate": _safe_get(pseudo, "qc.pass_rate"),
                "first_order_defect_f1": _safe_get(first_order, "summary_defect.f1"),
                "first_order_defect_iou": _safe_get(first_order, "summary_defect.iou"),
                "first_order_defect_rmse_um": _safe_get(first_order, "summary_defect.rmse"),
                "first_order_global_rmse_um": _safe_get(first_order, "summary.rmse"),
                "first_order_qc_pass_rate": _safe_get(first_order, "qc.pass_rate"),
            }
        )

    write_json(
        sweep_run.root / "phase_jump_scale_sweep_summary.json",
        {
            "checkpoint": str(shared_checkpoint) if shared_checkpoint is not None else None,
            "train_run": str(shared_train_run.root) if shared_train_run is not None else None,
            "train_per_scale": bool(args.train_per_scale),
            "checkpoints": checkpoints,
            "scales": scales,
            "rows": rows,
        },
    )
    print(str(sweep_run.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
