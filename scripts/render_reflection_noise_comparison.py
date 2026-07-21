from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_capture import _normalise_observation, _physical_radial_grid  # noqa: E402
from mini_grin_rebuild.core.configs import SimulationConfig, load_experiment_config  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_reference  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402
from mini_grin_rebuild.simulation.transforms.utils import gaussian_blur  # noqa: E402


def _profile_cfg(base: SimulationConfig, *, textured: bool) -> SimulationConfig:
    params = dict(base.capture_engine_params)
    if textured:
        params["field_texture"] = {
            "amplitude_strength": 0.005,
            "phase_strength_rad": 0.010,
            "correlation_sigma_px": 1.5,
            "correlation_sigma_x_px": 0.7,
            "correlation_sigma_y_px": 2.2,
            "correlation_angle_deg": [-25.0, 25.0],
            "illumination_strength": 0.010,
            "illumination_sigma_px": 40.0,
            "shared_fraction": 0.0,
        }
        params["camera"] = {
            "shot_noise": True,
            "photon_gain": 1000.0,
            "read_noise_std": 0.015,
            "saturation_level": 2.0,
            "bit_depth": 8,
            "bad_pixel_fraction": 0.0,
            "post_blur_sigma_px": 0.7,
        }
    else:
        params["field_texture"] = {
            "amplitude_strength": 0.0,
            "phase_strength_rad": 0.0,
            "illumination_strength": 0.0,
            "shared_fraction": 0.0,
        }
        params["camera"] = {
            "shot_noise": True,
            "photon_gain": 4000.0,
            "read_noise_std": 0.006,
            "saturation_level": 2.0,
            "bit_depth": 8,
            "bad_pixel_fraction": 0.0,
            "post_blur_sigma_px": 0.8,
        }
    return replace(base, capture_engine_params=params)


def _simulate_raw(cfg: SimulationConfig, *, seed: int, rho: np.ndarray) -> np.ndarray:
    capture = create_simulation_engine(cfg).simulate_capture(
        microlens_reference(cfg),
        rng=np.random.default_rng(seed),
    )
    return _normalise_observation(np.asarray(capture.channels["I_raw"], dtype=np.float32), rho)


def _high_pass(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) - gaussian_blur(np.asarray(image, dtype=np.float32), 5.0)


def _lag_correlation(values: np.ndarray, mask: np.ndarray, *, dy: int, dx: int) -> float:
    a = values[max(0, dy) : values.shape[0] + min(0, dy), max(0, dx) : values.shape[1] + min(0, dx)]
    b = values[max(0, -dy) : values.shape[0] + min(0, -dy), max(0, -dx) : values.shape[1] + min(0, -dx)]
    ma = mask[max(0, dy) : mask.shape[0] + min(0, dy), max(0, dx) : mask.shape[1] + min(0, dx)]
    mb = mask[max(0, -dy) : mask.shape[0] + min(0, -dy), max(0, -dx) : mask.shape[1] + min(0, -dx)]
    active = ma & mb
    x = np.asarray(a[active], dtype=np.float64)
    y = np.asarray(b[active], dtype=np.float64)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def _texture_metrics(image: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    high = _high_pass(image)
    mask = rho <= 0.85
    return {
        "highpass_std": float(np.std(high[mask])),
        "lag1_x": _lag_correlation(high, mask, dy=0, dx=1),
        "lag1_y": _lag_correlation(high, mask, dy=1, dx=0),
    }


def _save_gray(path: Path, image: np.ndarray) -> None:
    shown = np.clip(np.asarray(image, dtype=np.float32) / 1.5, 0.0, 1.0)
    Image.fromarray(np.round(255.0 * shown).astype(np.uint8), mode="L").save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render reflection noise profiles against real frames 7, 13 and 14.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "reflection_microlens520_actual.json")
    parser.add_argument("--noisy-config", type=Path, default=ROOT / "configs" / "reflection_microlens520_noisy.json")
    parser.add_argument("--ensemble-count", type=int, default=24)
    parser.add_argument(
        "--empirical-npz",
        type=Path,
        default=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "empirical_reflection_differences"
        / "empirical_differences.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "reflection_noise_simulation",
    )
    args = parser.parse_args(argv)

    base_cfg = load_experiment_config(args.config).simulation
    domain_cfg = load_experiment_config(args.noisy_config).simulation
    sim_rho = _physical_radial_grid(base_cfg)
    clean_cfg = _profile_cfg(base_cfg, textured=False)
    textured_cfg = domain_cfg
    sim_clean = _simulate_raw(clean_cfg, seed=7, rho=sim_rho)
    sim_textured_a = _simulate_raw(textured_cfg, seed=13, rho=sim_rho)
    sim_textured_b = _simulate_raw(textured_cfg, seed=14, rho=sim_rho)

    with np.load(args.empirical_npz) as data:
        names = [str(item) for item in data["names"].tolist()]
        registered = np.asarray(data["registered"], dtype=np.float32)
        real_rho = np.asarray(data["rho"], dtype=np.float32)
    lookup = {name: registered[index] for index, name in enumerate(names)}
    real_7 = lookup["7.bmp"]
    real_13 = lookup["13.bmp"]
    real_14 = lookup["14.bmp"]

    images = {
        "real_7": real_7,
        "real_13": real_13,
        "real_14": real_14,
        "sim_clean": sim_clean,
        "sim_textured_a": sim_textured_a,
        "sim_textured_b": sim_textured_b,
    }
    metrics = {
        "real_7": _texture_metrics(real_7, real_rho),
        "real_13": _texture_metrics(real_13, real_rho),
        "real_14": _texture_metrics(real_14, real_rho),
        "sim_clean": _texture_metrics(sim_clean, sim_rho),
        "sim_textured_a": _texture_metrics(sim_textured_a, sim_rho),
        "sim_textured_b": _texture_metrics(sim_textured_b, sim_rho),
    }

    domain_rho = _physical_radial_grid(domain_cfg)
    domain_height = microlens_reference(domain_cfg)
    ensemble_images: list[np.ndarray] = []
    ensemble_metrics: list[dict[str, float]] = []
    for index in range(max(int(args.ensemble_count), 1)):
        image = _simulate_raw(domain_cfg, seed=1000 + index, rho=domain_rho)
        ensemble_images.append(image)
        ensemble_metrics.append(_texture_metrics(image, domain_rho))

    real_metrics_all = [_texture_metrics(image, real_rho) for image in registered]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        _save_gray(args.output_dir / f"{name}.png", image)

    high_images = {name: _high_pass(image) for name, image in images.items()}
    real_hp_limit = max(
        float(np.quantile(np.abs(high_images["real_13"][real_rho <= 0.85]), 0.995)),
        float(np.quantile(np.abs(high_images["real_14"][real_rho <= 0.85]), 0.995)),
    )
    sim_hp_limit = max(
        float(np.quantile(np.abs(high_images["sim_textured_a"][sim_rho <= 0.85]), 0.995)),
        float(np.quantile(np.abs(high_images["sim_textured_b"][sim_rho <= 0.85]), 0.995)),
    )

    fig, axes = plt.subplots(4, 4, figsize=(16, 15), constrained_layout=True)
    row1 = [("real 7.bmp", real_7), ("real 13.bmp", real_13), ("real 14.bmp", real_14), ("simulated clean", sim_clean)]
    for ax, (title, image) in zip(axes[0], row1):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.5)
        ax.set_title(title)
        ax.axis("off")

    for ax, key in zip(axes[1, :3], ("real_7", "real_13", "real_14")):
        ax.imshow(high_images[key], cmap="coolwarm", vmin=-real_hp_limit, vmax=real_hp_limit)
        ax.set_title(f"high-pass {key}\nstd={metrics[key]['highpass_std']:.4f}")
        ax.axis("off")
    axes[1, 3].text(0.0, 1.0, json.dumps({key: metrics[key] for key in ("real_7", "real_13", "real_14")}, indent=2), va="top", family="monospace", fontsize=10)
    axes[1, 3].axis("off")

    sim_keys = ("sim_clean", "sim_textured_a", "sim_textured_b")
    for ax, key in zip(axes[2, :3], sim_keys):
        ax.imshow(high_images[key], cmap="coolwarm", vmin=-sim_hp_limit, vmax=sim_hp_limit)
        ax.set_title(f"high-pass {key}\nstd={metrics[key]['highpass_std']:.4f}")
        ax.axis("off")
    axes[2, 3].text(0.0, 1.0, json.dumps({key: metrics[key] for key in sim_keys}, indent=2), va="top", family="monospace", fontsize=10)
    axes[2, 3].axis("off")

    differences = [
        ("real 13 - 7", real_13 - real_7, real_rho),
        ("real 14 - 7", real_14 - real_7, real_rho),
        ("sim textured A - clean", sim_textured_a - sim_clean, sim_rho),
        ("sim textured B - clean", sim_textured_b - sim_clean, sim_rho),
    ]
    diff_limit = max(
        float(np.quantile(np.abs(diff[rho <= 0.88]), 0.995))
        for _, diff, rho in differences
    )
    for ax, (title, diff, rho) in zip(axes[3], differences):
        shown = diff.copy()
        shown[rho > 0.88] = 0.0
        ax.imshow(shown, cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit)
        ax.set_title(title)
        ax.axis("off")

    fig.savefig(args.output_dir / "noise_comparison.png", dpi=180)
    plt.close(fig)

    ensemble_high = [_high_pass(image) for image in ensemble_images]
    ensemble_limit = max(
        float(np.quantile(np.abs(image[domain_rho <= 0.85]), 0.995))
        for image in ensemble_high
    )
    rows = int(np.ceil(len(ensemble_high) / 4.0))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3.0 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(rows, 4)
    for index, ax in enumerate(axes_array.flat):
        if index >= len(ensemble_high):
            ax.axis("off")
            continue
        ax.imshow(ensemble_high[index], cmap="coolwarm", vmin=-ensemble_limit, vmax=ensemble_limit)
        ax.set_title(f"sim {index + 1}: std={ensemble_metrics[index]['highpass_std']:.4f}", fontsize=9)
        ax.axis("off")
    fig.savefig(args.output_dir / "simulated_noise_contact_sheet.png", dpi=170)
    plt.close(fig)

    metric_keys = ("highpass_std", "lag1_x", "lag1_y")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    rng = np.random.default_rng(20260721)
    for ax, key in zip(axes, metric_keys):
        real_values = np.asarray([item[key] for item in real_metrics_all], dtype=np.float64)
        sim_values = np.asarray([item[key] for item in ensemble_metrics], dtype=np.float64)
        ax.boxplot([real_values, sim_values], tick_labels=["real 24", "simulated 24"], showfliers=False)
        ax.scatter(1.0 + rng.normal(0.0, 0.035, real_values.size), real_values, s=18, alpha=0.65)
        ax.scatter(2.0 + rng.normal(0.0, 0.035, sim_values.size), sim_values, s=18, alpha=0.65)
        ax.set_title(key)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(args.output_dir / "metric_distributions.png", dpi=180)
    plt.close(fig)

    def _distribution_summary(items: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for key in metric_keys:
            values = np.asarray([item[key] for item in items], dtype=np.float64)
            result[key] = {
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
            }
        return result

    summary = {
        "scope": "noise-texture shape comparison after the same robust aperture normalisation",
        "clean_profile": clean_cfg.capture_engine_params,
        "textured_profile": textured_cfg.capture_engine_params,
        "metrics": metrics,
        "ensemble": {
            "config": str(args.noisy_config.resolve()),
            "count": len(ensemble_metrics),
            "real_distribution": _distribution_summary(real_metrics_all),
            "simulated_distribution": _distribution_summary(ensemble_metrics),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(args.output_dir / "noise_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
