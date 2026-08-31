from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_dark_port import _load_real_crops_dn  # noqa: E402
from evaluate_reflection_sim2real import (  # noqa: E402
    _inner_edge_contrast_profile,
    _interior_texture_metrics,
    _per_image_metrics,
    _select,
    _simulate_ensemble,
    _target_bundle,
    _validate_split_manifest,
)
from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402
from mini_grin_rebuild.simulation.transforms.utils import gaussian_blur  # noqa: E402


def _common_profile(image: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = _inner_edge_contrast_profile(image, rho, bins=144)
    angles = (np.arange(signal.size, dtype=np.float32) + 0.5) * (360.0 / signal.size)
    return angles, signal


def _radial_residual(image: np.ndarray, rho: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    bins = np.linspace(0.0, 0.82, 83, dtype=np.float32)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.asarray(
        [
            float(np.median(arr[(rho >= lo) & (rho < hi)]))
            if np.any((rho >= lo) & (rho < hi))
            else float("nan")
            for lo, hi in zip(bins[:-1], bins[1:])
        ],
        dtype=np.float32,
    )
    finite = np.flatnonzero(np.isfinite(profile))
    profile = np.interp(np.arange(profile.size), finite, profile[finite])
    background = np.interp(np.clip(rho, centers[0], centers[-1]), centers, profile)
    residual = arr - background
    residual[rho > 0.78] = np.nan
    return residual


def _ordered_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "inner_edge_focus_coverage_50",
        "inner_edge_peak_to_median",
        *[f"inner_edge_h{order}_relative_amplitude" for order in range(1, 9)],
        *[f"inner_edge_h{order}_peak_phase_deg" for order in range(1, 9)],
        "interior_texture_std_dn",
        "interior_fine_texture_std_dn",
        "interior_medium_texture_std_dn",
        "interior_fine_texture_coverage_1dn",
        "interior_fine_texture_kurtosis",
        "interior_spectral_low_fraction",
        "interior_spectral_mid_fraction",
        "interior_spectral_high_fraction",
        "interior_spectral_centroid_cyc_px",
    ]
    return {key: float(metrics[key]) for key in keys}


def _frame_distribution(stack: np.ndarray, rho: np.ndarray) -> dict[str, Any]:
    records = [_per_image_metrics(frame, rho) for frame in stack]
    output: dict[str, Any] = {
        "count": len(records),
        "median": {
            key: float(np.median([record[key] for record in records]))
            for key in _ordered_metrics(records[0])
        },
    }
    for order in range(1, 9):
        phases = np.deg2rad([record[f"inner_edge_h{order}_peak_phase_deg"] for record in records])
        concentration = abs(np.mean(np.exp(1j * order * phases)))
        output[f"h{order}_phase_concentration"] = float(concentration)
    return output


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure inner-edge angular structure and dense lens texture")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", default="I_x")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--ensemble-size", type=int, default=4)
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    crop_radius_scale = 1.0 / max(
        float(experiment.simulation.lens_radius_fraction),
        1e-12,
    )
    real_stack, real_rho, names = _load_real_crops_dn(
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        grid_size=experiment.simulation.grid_size,
        crop_radius_scale=crop_radius_scale,
    )
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    _validate_split_manifest(split, names)
    calibration_names = [str(name) for block in split["calibration_blocks"] for name in block]
    temporal_names = [str(name) for name in split["temporal_test"]]
    calibration_stack = _select(real_stack, names, calibration_names)
    temporal_stack = _select(real_stack, names, temporal_names)
    sim_stack, sim_rho, sim_meta = _simulate_ensemble(
        args.config,
        channel=str(args.channel),
        seed=int(args.seed),
        ensemble_size=int(args.ensemble_size),
    )

    real_calibration = _target_bundle(calibration_stack, real_rho)
    real_temporal = _target_bundle(temporal_stack, real_rho)
    simulated = _target_bundle(sim_stack, sim_rho)
    summary = {
        "scope": "Visual-structure audit: inner-edge ordered harmonics and dense interior texture",
        "config": str(args.config.resolve()),
        "seed": int(args.seed),
        "ensemble_size": int(args.ensemble_size),
        "channel": str(args.channel),
        "real_calibration_common": _ordered_metrics(real_calibration["common_metrics"]),
        "real_temporal_common": _ordered_metrics(real_temporal["common_metrics"]),
        "simulation_common": _ordered_metrics(simulated["common_metrics"]),
        "real_calibration_distribution": _frame_distribution(calibration_stack, real_rho),
        "real_temporal_distribution": _frame_distribution(temporal_stack, real_rho),
        "simulation_distribution": _frame_distribution(sim_stack, sim_rho),
        "simulation_meta": sim_meta,
    }

    common_images = [
        ("real calibration", real_calibration["common_image"], real_rho),
        ("real temporal", real_temporal["common_image"], real_rho),
        ("simulation", simulated["common_image"], sim_rho),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    for column, (title, image, rho) in enumerate(common_images):
        axes[0, column].imshow(np.log1p(image), cmap="magma", vmin=0.0, vmax=np.log(256.0))
        axes[0, column].set_title(f"log1p: {title}")
        axes[0, column].axis("off")
        residual = _radial_residual(image, rho)
        axes[1, column].imshow(residual, cmap="coolwarm", vmin=-3.0, vmax=3.0)
        axes[1, column].set_title(f"radial-detrended interior: {title}")
        axes[1, column].axis("off")

    for title, image, rho in common_images:
        angles, signal = _common_profile(image, rho)
        axes[2, 0].plot(angles, signal, label=title)
    axes[2, 0].set_title("inner-edge angular signal")
    axes[2, 0].set_xlabel("camera angle (deg)")
    axes[2, 0].set_ylabel("p90 minus interior baseline (DN)")
    axes[2, 0].grid(alpha=0.25)
    axes[2, 0].legend(fontsize=8)

    orders = np.arange(1, 9)
    width = 0.25
    for index, (title, key) in enumerate(
        [
            ("real calibration", "real_calibration_common"),
            ("real temporal", "real_temporal_common"),
            ("simulation", "simulation_common"),
        ]
    ):
        values = [summary[key][f"inner_edge_h{order}_relative_amplitude"] for order in orders]
        axes[2, 1].bar(orders + (index - 1) * width, values, width=width, label=title)
    axes[2, 1].set_title("ordered inner-edge harmonics")
    axes[2, 1].set_xlabel("harmonic order")
    axes[2, 1].set_ylabel("relative amplitude")
    axes[2, 1].set_xticks(orders)
    axes[2, 1].legend(fontsize=8)

    texture_keys = [
        "interior_fine_texture_std_dn",
        "interior_medium_texture_std_dn",
        "interior_fine_texture_coverage_1dn",
        "interior_fine_texture_kurtosis",
    ]
    x = np.arange(len(texture_keys))
    for index, (title, key) in enumerate(
        [
            ("real calibration", "real_calibration_common"),
            ("real temporal", "real_temporal_common"),
            ("simulation", "simulation_common"),
        ]
    ):
        axes[2, 2].bar(
            x + (index - 1) * width,
            [summary[key][metric] for metric in texture_keys],
            width=width,
            label=title,
        )
    axes[2, 2].set_title("interior texture morphology")
    axes[2, 2].set_xticks(x, ["fine std", "medium std", "coverage", "kurtosis"], rotation=20)
    axes[2, 2].legend(fontsize=8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "visual_structure_audit.png"
    summary_path = args.output_dir / "summary.json"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    summary_path.write_text(
        json.dumps(_json_sanitize(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(summary_path, flush=True)
    print(figure_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
