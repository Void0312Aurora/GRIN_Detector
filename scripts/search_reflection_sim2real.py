from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_reflection_sim2real import (  # noqa: E402
    _evaluate_fold,
    _load_real_crops_dn,
    _select,
    _validate_split_manifest,
)
from compare_reflection_capture import _physical_radial_grid  # noqa: E402
from mini_grin_rebuild.core.configs import ExperimentConfig  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_reference  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402


def _deep_set(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid override key {dotted_key!r}")
    cursor: dict[str, Any] = mapping
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise KeyError(f"override path does not resolve to a mapping: {dotted_key!r}")
        cursor = child
    cursor[parts[-1]] = value


def _config_digest(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _camera_mode(data: dict[str, Any]) -> str:
    camera = data.get("simulation", {}).get("capture_engine_params", {}).get("camera", {}) or {}
    return "dn" if str(camera.get("output_mode", "legacy")).lower() == "dn" else "legacy"


def _simulate(
    data: dict[str, Any],
    *,
    channel: str,
    seed: int,
    ensemble_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    experiment = ExperimentConfig.from_dict(data)
    cfg = experiment.simulation
    height = microlens_reference(cfg)
    images: list[np.ndarray] = []
    for index in range(ensemble_size):
        capture_seed = int(seed + index * 1_000_003)
        capture = create_simulation_engine(cfg).simulate_capture(
            height,
            rng=np.random.default_rng(capture_seed),
        )
        if channel not in capture.channels:
            raise ValueError(f"candidate did not emit channel {channel!r}")
        images.append(np.asarray(capture.channels[channel], dtype=np.float32))
    return np.stack(images, axis=0), _physical_radial_grid(cfg)


def _mean_nested(rows: list[dict[str, Any]], group: str) -> dict[str, float]:
    keys = rows[0][group].keys()
    return {key: float(np.mean([row[group][key] for row in rows])) for key in keys}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-aware calibration-block search for the dark-port sim-to-real model."
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--search-manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--channel", choices=("I_x", "I_y"), default="I_x")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.ensemble_size < 1:
        raise ValueError("ensemble-size must be >= 1")

    base_data = json.loads(args.base_config.read_text(encoding="utf-8"))
    search = json.loads(args.search_manifest.read_text(encoding="utf-8"))
    candidates = search.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("search manifest requires a non-empty candidates list")
    names = [str(candidate.get("name", "")) for candidate in candidates]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("candidate names must be non-empty and unique")

    base_experiment = ExperimentConfig.from_dict(base_data)
    base_cfg = base_experiment.simulation
    crop_radius_scale = 1.0 / max(float(base_cfg.lens_radius_fraction), 1e-12)
    real_stack, real_rho, real_names = _load_real_crops_dn(
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        grid_size=base_cfg.grid_size,
        crop_radius_scale=crop_radius_scale,
    )
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    _validate_split_manifest(split, real_names)
    calibration_blocks = [[str(name) for name in block] for block in split["calibration_blocks"]]
    calibration_frames = [name for block in calibration_blocks for name in block]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "candidate_results.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    compact_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        name = str(candidate["name"])
        overrides = candidate.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"candidate {name!r} overrides must be a mapping")
        data = copy.deepcopy(base_data)
        for key, value in overrides.items():
            _deep_set(data, str(key), value)
        print(f"[{index + 1}/{len(candidates)}] {name}: {json.dumps(overrides, sort_keys=True)}", flush=True)
        sim_stack, sim_rho = _simulate(
            data,
            channel=args.channel,
            seed=args.seed,
            ensemble_size=args.ensemble_size,
        )

        fold_rows: list[dict[str, Any]] = []
        for block_index, eval_frames in enumerate(calibration_blocks):
            eval_set = set(eval_frames)
            fit_frames = [frame for frame in calibration_frames if frame not in eval_set]
            result, _ = _evaluate_fold(
                sim_stack,
                sim_rho,
                real_fit=_select(real_stack, real_names, fit_frames),
                real_eval=_select(real_stack, real_names, eval_frames),
                real_rho=real_rho,
                radiometry_mode=_camera_mode(data),
            )
            fold_rows.append({"fold": block_index + 1, **result})

        summary = {
            "name": name,
            "overrides": overrides,
            "config_sha256": _config_digest(data),
            "score_mean": float(np.mean([row["score"] for row in fold_rows])),
            "score_std": float(np.std([row["score"] for row in fold_rows])),
            "profile_mean": _mean_nested(fold_rows, "profile"),
            "score_components_mean": _mean_nested(fold_rows, "score_components"),
            "sim_common_metrics_mean": _mean_nested(fold_rows, "sim_common_metrics"),
            "sim_distribution_median_mean": _mean_nested(fold_rows, "sim_distribution_median"),
            "folds": fold_rows,
        }
        full_rows.append(summary)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
        compact = {
            "name": name,
            "score_mean": summary["score_mean"],
            "score_std": summary["score_std"],
            "radial_rmse_mean": summary["profile_mean"]["radial_profile_rmse_dn"],
            "seam_rmse_mean": summary["profile_mean"]["seam_profile_rmse_dn"],
            "lowpass_corr_mean": summary["profile_mean"]["lowpass_aperture_corr"],
            **{
                f"component_{key}": value
                for key, value in summary["score_components_mean"].items()
            },
        }
        compact_rows.append(compact)
        _write_csv(args.output_dir / "candidate_summary.csv", compact_rows)
        print(f"{name}: blocked score={summary['score_mean']:.4f} +/- {summary['score_std']:.4f}", flush=True)

    ranked = sorted(full_rows, key=lambda row: float(row["score_mean"]))
    output = {
        "scope": "Calibration blocks only; temporal_test is intentionally excluded from selection.",
        "seed": int(args.seed),
        "ensemble_size": int(args.ensemble_size),
        "base_config": str(args.base_config.resolve()),
        "search_manifest": str(args.search_manifest.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "ranking": [
            {"name": row["name"], "score_mean": row["score_mean"], "score_std": row["score_std"]}
            for row in ranked
        ],
        "best": ranked[0],
    }
    (args.output_dir / "search_summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output["ranking"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
