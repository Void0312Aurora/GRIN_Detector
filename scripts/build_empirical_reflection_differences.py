from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_reflection_capture import _load_real_crops  # noqa: E402
from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402


def _standard_scores(stack: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Rank frames by robust interior distance to the across-sample median."""

    template = np.median(stack, axis=0).astype(np.float32)
    mask = rho <= 0.85
    records: list[dict[str, float]] = []
    for image in stack:
        residual = np.abs(np.asarray(image - template, dtype=np.float32)[mask])
        p95 = float(np.quantile(residual, 0.95))
        p99 = float(np.quantile(residual, 0.99))
        p999 = float(np.quantile(residual, 0.999))
        records.append(
            {
                "mean_abs": float(np.mean(residual)),
                "p95_abs": p95,
                "p99_abs": p99,
                "p999_abs": p999,
                "standard_score": p95 + 0.35 * p99 + 0.10 * p999,
            }
        )
    return template, records


def _difference_metrics(diff: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    mask = rho <= 0.88
    values = np.asarray(diff[mask], dtype=np.float32)
    absolute = np.abs(values)
    return {
        "mean_abs": float(np.mean(absolute)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "p999_abs": float(np.quantile(absolute, 0.999)),
        "positive_mean": float(np.mean(np.clip(values, 0.0, None))),
        "negative_mean_abs": float(np.mean(np.clip(-values, 0.0, None))),
    }


def _save_gray(path: Path, image: np.ndarray) -> None:
    shown = np.clip(np.asarray(image, dtype=np.float32) / 1.5, 0.0, 1.0)
    Image.fromarray(np.round(255.0 * shown).astype(np.uint8), mode="L").save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build cross-sample empirical differences from registered raw reflection frames."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "reflection_microlens520_actual.json")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "external_data" / "raw" / "wechat_2026-07_15-34" / "extracted" / "15.34",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "valid_sample_detections.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "empirical_reflection_differences",
    )
    parser.add_argument(
        "--standard",
        type=str,
        default="auto",
        help="BMP filename to use as the empirical standard, or 'auto' for the robust interior medoid.",
    )
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.config).simulation
    crop_radius_scale = 1.0 / max(float(cfg.lens_radius_fraction), 1e-12)
    crops, rho, names = _load_real_crops(
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        grid_size=cfg.grid_size,
        crop_radius_scale=crop_radius_scale,
    )
    stack = np.stack(crops, axis=0).astype(np.float32)
    template, score_records = _standard_scores(stack, rho)
    for name, record in zip(names, score_records):
        record["file"] = name  # type: ignore[assignment]

    if args.standard.lower() == "auto":
        standard_index = int(np.argmin([float(record["standard_score"]) for record in score_records]))
    else:
        if args.standard not in names:
            raise ValueError(f"Unknown standard frame {args.standard!r}; available: {', '.join(names)}")
        standard_index = names.index(args.standard)

    standard_name = names[standard_index]
    standard = stack[standard_index]
    differences = stack - standard[None, :, :]
    valid_mask = rho <= 0.88
    masked_differences = differences.copy()
    masked_differences[:, ~valid_mask] = 0.0

    metrics: list[dict[str, float | str | bool]] = []
    for name, diff in zip(names, masked_differences):
        record: dict[str, float | str | bool] = {
            "file": name,
            "is_standard": name == standard_name,
            **_difference_metrics(diff, rho),
        }
        metrics.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_gray(args.output_dir / "empirical_standard.png", standard)
    _save_gray(args.output_dir / "robust_median_template.png", template)
    np.savez_compressed(
        args.output_dir / "empirical_differences.npz",
        names=np.asarray(names),
        registered=stack,
        standard=standard,
        robust_median_template=template,
        differences=differences,
        masked_differences=masked_differences,
        rho=rho.astype(np.float32),
        valid_mask=valid_mask,
        standard_index=np.asarray(standard_index, dtype=np.int32),
    )

    fieldnames = ["file", "is_standard", "mean_abs", "p95_abs", "p99_abs", "p999_abs", "positive_mean", "negative_mean_abs"]
    with (args.output_dir / "difference_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    abs_values = np.abs(masked_differences[:, valid_mask])
    display_limit = max(float(np.quantile(abs_values, 0.995)), 1e-3)
    fig, axes = plt.subplots(6, 4, figsize=(12, 17), constrained_layout=True)
    for ax, name, diff, record in zip(axes.flat, names, masked_differences, metrics):
        ax.imshow(diff, cmap="coolwarm", vmin=-display_limit, vmax=display_limit)
        suffix = " (standard)" if bool(record["is_standard"]) else ""
        ax.set_title(f"{name}{suffix}\np99={float(record['p99_abs']):.4f}", fontsize=9)
        ax.axis("off")
    fig.savefig(args.output_dir / "difference_contact_sheet.png", dpi=170)
    plt.close(fig)

    ranking = sorted(
        (
            {
                "file": names[index],
                **score_records[index],
            }
            for index in range(len(names))
        ),
        key=lambda item: float(item["standard_score"]),
    )
    anomaly_ranking = sorted(
        (record for record in metrics if not bool(record["is_standard"])),
        key=lambda item: float(item["p99_abs"]),
        reverse=True,
    )
    summary = {
        "scope": (
            "cross-sample empirical subtraction after circle registration and robust intensity normalisation; "
            "not a same-sample before/after measurement and not quantitative height ground truth"
        ),
        "standard_file": standard_name,
        "standard_selection": "robust interior medoid" if args.standard.lower() == "auto" else "user-selected",
        "valid_radius_fraction": 0.88,
        "standard_ranking": ranking,
        "anomaly_ranking": anomaly_ranking,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"standard={standard_name}")
    print("top_anomalies=" + ", ".join(str(item["file"]) for item in anomaly_ranking[:8]))
    print(args.output_dir / "difference_contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
