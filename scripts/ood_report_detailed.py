from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


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


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.6g}"
    return str(v)


def _write_md_table(path: Path, rows: list[dict[str, Any]], *, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a detailed OOD report from an existing suite run directory.")
    p.add_argument("--run", required=True, help="Run directory containing ood_report.json (suite root).")
    p.add_argument("--out", default=None, help="Output markdown path (default: <run>/ood_report_detailed.md).")
    args = p.parse_args()

    run_root = Path(args.run).expanduser().resolve()
    report_path = run_root / "ood_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing {report_path}")
    report = _load_json(report_path)
    base_rows = report.get("rows", [])
    if not isinstance(base_rows, list) or not base_rows:
        raise ValueError(f"No rows found in {report_path}")

    rows: list[dict[str, Any]] = []
    for r in base_rows:
        if not isinstance(r, Mapping):
            continue
        eval_dir = Path(str(r.get("eval_dir", "")))
        metrics_path = eval_dir / "eval_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_json(metrics_path)

        defect_rmse_um = _safe_get(metrics, "summary_defect.rmse")
        edge_p95_um = _safe_get(metrics, "artifacts.edge_p95_abs")
        row: dict[str, Any] = {
            "scenario": str(r.get("scenario", "")),
            "method": str(r.get("method", "")),
            "count": int(metrics.get("count", r.get("count", 0) or 0)),
            "defect_rmse_nm": defect_rmse_um * 1e3 if math.isfinite(defect_rmse_um) else float("nan"),
            "defect_psnr": _safe_get(metrics, "summary_defect.psnr"),
            "defect_corr": _safe_get(metrics, "summary_defect.corr"),
            "defect_auprc": _safe_get(metrics, "summary_defect.auprc"),
            "defect_auroc": _safe_get(metrics, "summary_defect.auroc"),
            "defect_iou": _safe_get(metrics, "summary_defect.iou"),
            "defect_f1": _safe_get(metrics, "summary_defect.f1"),
            "defect_precision": _safe_get(metrics, "summary_defect.precision"),
            "defect_recall": _safe_get(metrics, "summary_defect.recall"),
            "defect_support_frac": _safe_get(metrics, "summary_defect.support_frac"),
            "defect_volume_rel_error": _safe_get(metrics, "summary_defect.volume_rel_error"),
            "edge_p95_nm": edge_p95_um * 1e3 if math.isfinite(edge_p95_um) else float("nan"),
            "edge_mean_nm": _safe_get(metrics, "artifacts.edge_mean_abs") * 1e3,
            "center_f1": _safe_get(metrics, "summary_regions.center.f1"),
            "center_auprc": _safe_get(metrics, "summary_regions.center.auprc"),
            "eval_dir": str(eval_dir),
            "plots_dir": str(eval_dir / "plots"),
        }

        if "qc" in metrics and isinstance(metrics["qc"], Mapping):
            row["qc_pass_rate"] = _safe_get(metrics, "qc.pass_rate")
            row["qc_physics_rmse_mean"] = _safe_get(metrics, "qc.summary.physics_rmse_mean")
            row["qc_physics_p95_mean"] = _safe_get(metrics, "qc.summary.physics_p95_abs_mean")
            row["qc_edge_mean_nm"] = _safe_get(metrics, "qc.summary.edge_mean_abs_mean") * 1e3
            row["qc_outside_mean_nm"] = _safe_get(metrics, "qc.summary.outside_mean_abs_mean") * 1e3
            logvar_mean = _safe_get(metrics, "qc.summary.logvar_mean_mean")
            if math.isfinite(logvar_mean):
                row["qc_logvar_mean"] = logvar_mean
        rows.append(row)

    out_path = Path(args.out).expanduser().resolve() if args.out else (run_root / "ood_report_detailed.md")
    columns = [
        "scenario",
        "method",
        "count",
        "defect_rmse_nm",
        "defect_psnr",
        "defect_corr",
        "defect_auprc",
        "defect_auroc",
        "defect_iou",
        "defect_f1",
        "defect_precision",
        "defect_recall",
        "defect_support_frac",
        "defect_volume_rel_error",
        "edge_mean_nm",
        "edge_p95_nm",
        "center_f1",
        "center_auprc",
        "eval_dir",
        "plots_dir",
    ]
    _write_md_table(out_path, rows, columns=columns)
    (out_path.parent / "ood_report_detailed.json").write_text(
        json.dumps({"run_root": str(run_root), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
