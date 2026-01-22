from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

from mini_grin_rebuild.core.configs import ExperimentConfig, load_experiment_config
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.core.runs import create_run


def _default_config_path() -> Path | None:
    candidates = []
    here = Path(__file__).resolve()
    # Source layout: mini_grin_rebuild/src/mini_grin_rebuild/cli/main.py -> mini_grin_rebuild/
    candidates.append(here.parents[3] / "configs" / "default.json")
    # Repo root layout: ./mini_grin_rebuild/configs/default.json
    candidates.append(Path.cwd() / "mini_grin_rebuild" / "configs" / "default.json")
    # Local layout: ./configs/default.json
    candidates.append(Path.cwd() / "configs" / "default.json")
    for path in candidates:
        if path.exists():
            return path
    return None


def _infer_project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def cmd_init_run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config).expanduser()
    cfg = load_experiment_config(cfg_path)
    project_root = _infer_project_root(cfg_path)
    runs_root = project_root / cfg.paths.runs_dir
    run = create_run(
        runs_root,
        name=args.name,
        argv=sys.argv,
        config_snapshot=cfg.to_dict(),
    )
    print(str(run.root))
    return 0


def cmd_print_config(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config).expanduser()
    cfg = load_experiment_config(cfg_path)
    import json

    json.dump(cfg.to_dict(), sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def cmd_generate_dataset(args: argparse.Namespace) -> int:
    from mini_grin_rebuild.data.generate_dataset import generate_dataset

    cfg_path = Path(args.config).expanduser()
    cfg = load_experiment_config(cfg_path)
    project_root = _infer_project_root(cfg_path)

    runs_root = project_root / cfg.paths.runs_dir
    run = create_run(runs_root, name=args.name or "generate", argv=sys.argv, config_snapshot=cfg.to_dict())

    data_root = project_root / cfg.paths.data_dir
    output = Path(args.output) if args.output is not None else (data_root / "mini_grin_dataset")
    output = output.expanduser()

    if output.exists() and any(output.rglob("*.npz")) and not args.overwrite:
        raise SystemExit(f"Dataset already exists at {output}; pass --overwrite to replace.")
    if args.overwrite and output.exists():
        import shutil

        shutil.rmtree(output)

    meta = generate_dataset(
        cfg.simulation,
        output_root=output,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
        config_snapshot=cfg.to_dict(),
    )
    write_json(run.root / "dataset.json", {"dataset_root": str(output), "meta": meta})
    print(str(run.root))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from mini_grin_rebuild.training.trainer import train_dataset

    cfg_path = Path(args.config).expanduser()
    cfg = load_experiment_config(cfg_path)
    project_root = _infer_project_root(cfg_path)

    runs_root = project_root / cfg.paths.runs_dir
    run = create_run(runs_root, name=args.name or "train", argv=sys.argv, config_snapshot=cfg.to_dict())

    data_root = project_root / cfg.paths.data_dir
    dataset_root = Path(args.data_root) if args.data_root is not None else (data_root / "mini_grin_dataset")
    dataset_root = dataset_root.expanduser()

    outs = train_dataset(cfg, data_root=dataset_root, run=run, resume=args.resume)
    write_json(
        run.root / "artifacts.json",
        {"dataset_root": str(dataset_root), "best_checkpoint": str(outs.best_checkpoint), "last_checkpoint": str(outs.last_checkpoint)},
    )
    print(str(run.root))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from mini_grin_rebuild.evaluation.evaluator import evaluate_checkpoint

    cfg_path = Path(args.config).expanduser()
    cfg = load_experiment_config(cfg_path)
    project_root = _infer_project_root(cfg_path)

    runs_root = project_root / cfg.paths.runs_dir
    run = create_run(runs_root, name=args.name or f"eval_{args.split}", argv=sys.argv, config_snapshot=cfg.to_dict())

    data_root = project_root / cfg.paths.data_dir
    dataset_root = Path(args.data_root) if args.data_root is not None else (data_root / "mini_grin_dataset")
    dataset_root = dataset_root.expanduser()

    evaluate_checkpoint(
        cfg,
        data_root=dataset_root,
        split=args.split,
        checkpoint_path=args.checkpoint,
        out_dir=run.root,
        num_plots=args.num_plots,
    )
    write_json(run.root / "artifacts.json", {"dataset_root": str(dataset_root), "checkpoint": str(args.checkpoint), "split": args.split})
    print(str(run.root))
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from mini_grin_rebuild.evaluation.evaluator import evaluate_oracle_poisson, evaluate_pseudo_poisson

    cfg_path = Path(args.config).expanduser()
    cfg = load_experiment_config(cfg_path)
    project_root = _infer_project_root(cfg_path)

    runs_root = project_root / cfg.paths.runs_dir
    run = create_run(runs_root, name=args.name or f"baseline_{args.split}", argv=sys.argv, config_snapshot=cfg.to_dict())

    data_root = project_root / cfg.paths.data_dir
    dataset_root = Path(args.data_root) if args.data_root is not None else (data_root / "mini_grin_dataset")
    dataset_root = dataset_root.expanduser()

    method = str(getattr(args, "method", "pseudo_poisson"))
    if method == "pseudo_poisson":
        evaluate_pseudo_poisson(
            cfg,
            data_root=dataset_root,
            split=args.split,
            out_dir=run.root,
            num_plots=args.num_plots,
        )
    elif method == "oracle_poisson":
        evaluate_oracle_poisson(
            cfg,
            data_root=dataset_root,
            split=args.split,
            out_dir=run.root,
            num_plots=args.num_plots,
        )
    else:
        raise SystemExit(f"Unknown baseline method: {method!r}")

    write_json(run.root / "artifacts.json", {"dataset_root": str(dataset_root), "method": method, "split": args.split})
    print(str(run.root))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from mini_grin_rebuild.evaluation.evaluator import evaluate_checkpoint
    from mini_grin_rebuild.training.trainer import train_dataset

    cfg_path = Path(args.config).expanduser()
    base_cfg = load_experiment_config(cfg_path)
    project_root = _infer_project_root(cfg_path)

    data_root = project_root / base_cfg.paths.data_dir
    dataset_root = Path(args.data_root) if args.data_root is not None else (data_root / "mini_grin_dataset")
    dataset_root = dataset_root.expanduser()

    field = args.field
    values = [float(v) for v in args.values.split(",") if v]

    summary: Dict[str, Any] = {"field": field, "values": values, "runs": []}
    for v in values:
        cfg_dict = base_cfg.to_dict()
        cfg_dict.setdefault("training", {})[field] = v
        cfg = ExperimentConfig.from_dict(cfg_dict)

        runs_root = project_root / cfg.paths.runs_dir
        run = create_run(
            runs_root,
            name=f"scan_{field}={v}",
            argv=sys.argv,
            config_snapshot=cfg.to_dict(),
        )
        outs = train_dataset(cfg, data_root=dataset_root, run=run, resume=None)
        eval_result = evaluate_checkpoint(
            cfg,
            data_root=dataset_root,
            split=args.eval_split,
            checkpoint_path=outs.best_checkpoint,
            out_dir=run.root / "eval",
            num_plots=0,
        )
        summary["runs"].append({"value": v, "run": str(run.root), "best_checkpoint": str(outs.best_checkpoint), "eval": eval_result})

    out_path = Path(args.summary_json).expanduser() if args.summary_json else (project_root / base_cfg.paths.runs_dir / "scan_summary.json")
    write_json(out_path, summary)
    print(str(out_path))
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    from mini_grin_rebuild.experiments.suite import run_ablation_suite

    cfg_path = Path(args.config).expanduser()
    base_cfg = load_experiment_config(cfg_path)
    project_root = _infer_project_root(cfg_path)

    data_root = project_root / base_cfg.paths.data_dir
    dataset_root = Path(args.data_root) if args.data_root is not None else (data_root / "mini_grin_dataset")
    dataset_root = dataset_root.expanduser()

    res = run_ablation_suite(
        base_config_path=cfg_path,
        suite_path=args.suite,
        data_root=dataset_root,
        eval_split=args.eval_split,
        num_plots=args.num_plots,
        name=args.name,
    )
    print(str(res.suite_run.root))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-grin", description="mini_grin_rebuild CLI (rewrite-first)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    default_cfg = _default_config_path()

    p_init = sub.add_parser("init-run", help="Create a new audited run directory")
    p_init.add_argument("--config", type=str, default=str(default_cfg) if default_cfg else None, required=default_cfg is None)
    p_init.add_argument("--name", type=str, default=None, help="Optional run name suffix")
    p_init.set_defaults(func=cmd_init_run)

    p_cfg = sub.add_parser("print-config", help="Print resolved config JSON")
    p_cfg.add_argument("--config", type=str, default=str(default_cfg) if default_cfg else None, required=default_cfg is None)
    p_cfg.set_defaults(func=cmd_print_config)

    p_gen = sub.add_parser("generate-dataset", help="Generate legacy-compatible `.npz` dataset")
    p_gen.add_argument("--config", type=str, default=str(default_cfg) if default_cfg else None, required=default_cfg is None)
    p_gen.add_argument("--output", type=str, default=None, help="Dataset output root (defaults to <project>/data/mini_grin_dataset)")
    p_gen.add_argument("--train", type=int, default=200)
    p_gen.add_argument("--val", type=int, default=40)
    p_gen.add_argument("--test", type=int, default=40)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset directory")
    p_gen.add_argument("--name", type=str, default=None)
    p_gen.set_defaults(func=cmd_generate_dataset)

    p_train = sub.add_parser("train", help="Train on dataset and write audited run artifacts")
    p_train.add_argument("--config", type=str, default=str(default_cfg) if default_cfg else None, required=default_cfg is None)
    p_train.add_argument("--data-root", type=str, default=None, help="Dataset root (defaults to <project>/data/mini_grin_dataset)")
    p_train.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint path")
    p_train.add_argument("--name", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint on a dataset split")
    p_eval.add_argument("--config", type=str, default=str(default_cfg) if default_cfg else None, required=default_cfg is None)
    p_eval.add_argument("--data-root", type=str, default=None, help="Dataset root (defaults to <project>/data/mini_grin_dataset)")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--num-plots", type=int, default=3)
    p_eval.add_argument("--name", type=str, default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_base = sub.add_parser("baseline", help="Evaluate pseudo-Poisson baseline reconstruction")
    p_base.add_argument("--config", type=str, default=str(default_cfg) if default_cfg else None, required=default_cfg is None)
    p_base.add_argument("--data-root", type=str, default=None, help="Dataset root (defaults to <project>/data/mini_grin_dataset)")
    p_base.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_base.add_argument("--method", type=str, default="pseudo_poisson", choices=["pseudo_poisson", "oracle_poisson"])
    p_base.add_argument("--num-plots", type=int, default=3)
    p_base.add_argument("--name", type=str, default=None)
    p_base.set_defaults(func=cmd_baseline)

    p_scan = sub.add_parser("scan", help="Sweep a single training.* field and record eval summaries")
    p_scan.add_argument("--config", type=str, default=str(default_cfg) if default_cfg else None, required=default_cfg is None)
    p_scan.add_argument("--data-root", type=str, default=None, help="Dataset root (defaults to <project>/data/mini_grin_dataset)")
    p_scan.add_argument("--field", type=str, required=True, help="Training field name (e.g., curl_weight)")
    p_scan.add_argument("--values", type=str, required=True, help="Comma-separated values (e.g., 0,1e-4,5e-4)")
    p_scan.add_argument("--eval-split", type=str, default="val", choices=["train", "val", "test"])
    p_scan.add_argument("--summary-json", type=str, default=None, help="Where to write scan summary JSON")
    p_scan.set_defaults(func=cmd_scan)

    p_suite = sub.add_parser("suite", help="Run an ablation/benchmark suite (train + baselines + summary table)")
    p_suite.add_argument("--config", type=str, default=str(default_cfg) if default_cfg else None, required=default_cfg is None)
    p_suite.add_argument("--suite", type=str, required=True, help="Path to suite JSON (experiments + baselines)")
    p_suite.add_argument("--data-root", type=str, default=None, help="Dataset root (defaults to <project>/data/mini_grin_dataset)")
    p_suite.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    p_suite.add_argument("--num-plots", type=int, default=0)
    p_suite.add_argument("--name", type=str, default=None, help="Optional suffix for suite run name")
    p_suite.set_defaults(func=cmd_suite)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
