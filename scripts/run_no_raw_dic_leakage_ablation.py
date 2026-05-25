from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402
from mini_grin_rebuild.core.json_io import read_json, write_json  # noqa: E402
from mini_grin_rebuild.data.generate_dataset import generate_dataset  # noqa: E402
from mini_grin_rebuild.experiments.suite import run_ablation_suite  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _metric_get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _summarize_suite(summary_path: Path) -> list[dict[str, Any]]:
    summary = read_json(summary_path)
    rows: list[dict[str, Any]] = []
    for item in summary.get("results", []):
        eval_payload = item.get("eval", {}) or {}
        row = {
            "name": item.get("name"),
            "kind": item.get("kind"),
            "method": item.get("method"),
            "defect_rmse": _metric_get(eval_payload, "summary_defect.rmse"),
            "defect_corr": _metric_get(eval_payload, "summary_defect.corr"),
            "sign_x_global": _metric_get(eval_payload, "summary_sign_gradient.x.global.accuracy"),
            "sign_y_global": _metric_get(eval_payload, "summary_sign_gradient.y.global.accuracy"),
            "sign_x_defect_local": _metric_get(eval_payload, "summary_sign_gradient.x.defect_local.accuracy"),
            "sign_y_defect_local": _metric_get(eval_payload, "summary_sign_gradient.y.defect_local.accuracy"),
            "sign_x_branch_flip": _metric_get(eval_payload, "summary_sign_gradient.x.branch_flip.accuracy"),
            "sign_y_branch_flip": _metric_get(eval_payload, "summary_sign_gradient.y.branch_flip.accuracy"),
            "sign_x_branch_flip_count": _metric_get(eval_payload, "summary_sign_gradient.x.branch_flip.count"),
            "sign_y_branch_flip_count": _metric_get(eval_payload, "summary_sign_gradient.y.branch_flip.count"),
            "run_dir": item.get("run_dir"),
        }
        rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Run no-raw DIC leakage smoke ablations.")
    p.add_argument("--train", type=int, default=24)
    p.add_argument("--val", type=int, default=8)
    p.add_argument("--test", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-plots", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_root = Path(args.out).expanduser() if args.out else ROOT / "runs" / "no_raw_dic_leakage_ablation"
    out_root.mkdir(parents=True, exist_ok=True)

    configs = {
        "ideal_dic": ROOT / "configs" / "ideal_dic_smoke64.json",
        "nonideal_dic_no_raw": ROOT / "configs" / "nonideal_dic_no_raw_smoke64.json",
    }
    suite_path = ROOT / "configs" / "suites" / "no_raw_dic_leakage_smoke.json"

    results: dict[str, Any] = {
        "datasets": {},
        "suites": {},
        "rows": {},
        "settings": {
            "train": int(args.train),
            "val": int(args.val),
            "test": int(args.test),
            "seed": int(args.seed),
            "num_plots": int(args.num_plots),
        },
    }
    for label, cfg_path in configs.items():
        cfg = load_experiment_config(cfg_path)
        data_root = out_root / "data" / label
        meta = generate_dataset(
            cfg.simulation,
            output_root=data_root,
            train=int(args.train),
            val=int(args.val),
            test=int(args.test),
            seed=int(args.seed),
            config_snapshot=cfg.to_dict(),
        )
        suite = run_ablation_suite(
            base_config_path=cfg_path,
            suite_path=suite_path,
            data_root=data_root,
            eval_split="test",
            num_plots=int(args.num_plots),
            name=label,
        )
        rows = _summarize_suite(suite.summary_path)
        results["datasets"][label] = {"root": str(data_root), "meta": meta}
        results["suites"][label] = {"root": str(suite.suite_run.root), "summary_path": str(suite.summary_path)}
        results["rows"][label] = rows

    out_path = out_root / "no_raw_dic_leakage_ablation_summary.json"
    write_json(out_path, results)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
