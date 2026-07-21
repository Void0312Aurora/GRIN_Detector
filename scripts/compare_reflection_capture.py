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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mini_grin_rebuild.core.configs import SimulationConfig, load_experiment_config  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_reference  # noqa: E402
from mini_grin_rebuild.physics.phase import phase_scale  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _radial_grid(size: int, radius_px: float) -> np.ndarray:
    coord = np.arange(size, dtype=np.float32) - 0.5 * (size - 1)
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    return np.sqrt(xx**2 + yy**2) / max(float(radius_px), 1e-12)


def _physical_radial_grid(cfg: SimulationConfig) -> np.ndarray:
    radius = float(cfg.lens_curvature_radius_um or 0.0)
    sag = float(cfg.lens_sag_um or 0.0)
    aperture_radius = np.sqrt(max(2.0 * radius * sag - sag * sag, 0.0))
    coord = (np.arange(cfg.grid_size, dtype=np.float32) - 0.5 * (cfg.grid_size - 1)) * float(cfg.dx)
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    return np.sqrt(xx**2 + yy**2) / max(float(aperture_radius), 1e-12)


def _normalise_observation(image: np.ndarray, rho: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    interior = arr[rho <= 0.65]
    edge = arr[(rho >= 0.82) & (rho <= 1.03)]
    baseline = float(np.median(interior)) if interior.size else float(np.median(arr))
    edge_level = float(np.quantile(edge, 0.95)) if edge.size else float(np.quantile(arr, 0.99))
    scale = max(edge_level - baseline, 1e-8)
    return np.clip((arr - baseline) / scale, 0.0, 3.0).astype(np.float32)


def _radial_profile(image: np.ndarray, rho: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rho >= lo) & (rho < hi)
        values.append(float(np.median(image[mask])) if np.any(mask) else float("nan"))
    return np.asarray(values, dtype=np.float32)


def _profile_metrics(real: np.ndarray, sim: np.ndarray, centers: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(real) & np.isfinite(sim) & (centers <= 1.08)
    real_v = real[mask].astype(np.float64)
    sim_v = sim[mask].astype(np.float64)
    if real_v.size < 3:
        return {"profile_rmse": float("nan"), "profile_corr": float("nan"), "edge_peak_radius": float("nan")}
    rmse = float(np.sqrt(np.mean((np.clip(real_v, 0.0, 2.0) - np.clip(sim_v, 0.0, 2.0)) ** 2)))
    if float(np.std(real_v)) > 1e-12 and float(np.std(sim_v)) > 1e-12:
        corr = float(np.corrcoef(real_v, sim_v)[0, 1])
    else:
        corr = 0.0
    edge_mask = mask & (centers >= 0.65)
    edge_indices = np.flatnonzero(edge_mask)
    peak_radius = float(centers[edge_indices[np.nanargmax(sim[edge_indices])]]) if edge_indices.size else float("nan")
    return {"profile_rmse": rmse, "profile_corr": corr, "edge_peak_radius": peak_radius}


def _image_metrics(real: np.ndarray, sim: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    def _corr(mask: np.ndarray) -> float:
        a = np.asarray(real[mask], dtype=np.float64)
        b = np.asarray(sim[mask], dtype=np.float64)
        if a.size < 3 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    aperture = rho <= 1.08
    edge = (rho >= 0.88) & (rho <= 1.08)
    diff = np.clip(real[aperture], 0.0, 2.0) - np.clip(sim[aperture], 0.0, 2.0)
    return {
        "image_rmse_aperture": float(np.sqrt(np.mean(np.asarray(diff, dtype=np.float64) ** 2))),
        "image_corr_aperture": _corr(aperture),
        "image_corr_edge": _corr(edge),
    }


def _angular_cv(image: np.ndarray, rho: np.ndarray, *, bins: int = 72) -> float:
    coord = np.arange(image.shape[0], dtype=np.float32) - 0.5 * (image.shape[0] - 1)
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    theta = np.mod(np.arctan2(yy, xx), 2.0 * np.pi)
    annulus = (rho >= 0.96) & (rho <= 1.04)
    values: list[float] = []
    for index in range(bins):
        lo = 2.0 * np.pi * index / bins
        hi = 2.0 * np.pi * (index + 1) / bins
        mask = annulus & (theta >= lo) & (theta < hi)
        if np.any(mask):
            values.append(float(np.median(image[mask])))
    arr = np.asarray(values, dtype=np.float64)
    return float(np.std(arr) / max(float(np.mean(arr)), 1e-12))


def _load_real_crops(
    *,
    raw_dir: Path,
    detections_path: Path,
    grid_size: int,
    crop_radius_scale: float,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    crops: list[np.ndarray] = []
    names: list[str] = []
    radius_resized = grid_size / (2.0 * crop_radius_scale)
    rho = _radial_grid(grid_size, radius_resized)
    resampling = getattr(Image, "Resampling", Image).BILINEAR

    for record in detections:
        name = str(record["file"])
        center_x = float(record["center_x"])
        center_y = float(record["center_y"])
        radius = float(record["radius"])
        half = crop_radius_scale * radius
        box = (
            int(round(center_x - half)),
            int(round(center_y - half)),
            int(round(center_x + half)),
            int(round(center_y + half)),
        )
        with Image.open(raw_dir / name) as source:
            crop = source.convert("L").crop(box).resize((grid_size, grid_size), resampling)
            arr = np.asarray(crop, dtype=np.float32)
        crops.append(_normalise_observation(arr, rho))
        names.append(name)
    return crops, rho, names


def _simulate(
    cfg: SimulationConfig,
    *,
    defocus: float,
    raw_blur_sigma_px: float | None = None,
    channel: str = "I_raw",
    shear_px: float | None = None,
) -> np.ndarray:
    params = dict(cfg.capture_engine_params)
    params["defocus_strength"] = float(defocus)
    if raw_blur_sigma_px is not None:
        params["raw_blur_sigma_px"] = float(raw_blur_sigma_px)
        if channel != "I_raw":
            # The blur sweep must reach the sheared-difference (dark-port) channels.
            params["dic_blur_sigma_px"] = float(raw_blur_sigma_px)
    if shear_px is not None:
        params["shear_px"] = float(shear_px)
    sim_cfg = replace(cfg, capture_engine="optical_leakage_lite", capture_engine_params=params)
    height = microlens_reference(sim_cfg)
    capture = create_simulation_engine(sim_cfg).simulate_capture(height, rng=np.random.default_rng(0))
    return np.asarray(capture.channels[channel], dtype=np.float32)


def _save_gray(path: Path, image: np.ndarray) -> None:
    shown = np.clip(np.asarray(image, dtype=np.float32) / 1.5, 0.0, 1.0)
    Image.fromarray(np.round(255.0 * shown).astype(np.uint8), mode="L").save(path)


def _best(records: list[dict[str, object]]) -> dict[str, object]:
    return min(records, key=lambda item: float(item["profile_rmse"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare reflection simulations with the 24 central BMP samples.")
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
        / "reflection_simulation_comparison",
    )
    parser.add_argument("--defocus-values", type=_csv_floats, default=[-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0])
    parser.add_argument("--raw-blur-values", type=_csv_floats, default=[0.8, 2.0, 4.0, 8.0])
    parser.add_argument(
        "--channel",
        type=str,
        default="I_raw",
        choices=("I_raw", "I_x", "I_y"),
        help="Engine output channel to compare: I_raw for the bright port, I_x/I_y for the sheared dark port.",
    )
    parser.add_argument(
        "--shear-px",
        type=float,
        default=None,
        help="Override the sheared-difference split in simulation pixels (10 um at dx=0.46875 is 21.33 px).",
    )
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    cfg = experiment.simulation
    if cfg.phase_mode != "reflection":
        raise ValueError("comparison config must use simulation.phase_mode='reflection'")
    if cfg.scene != "microlens_spherical_cap":
        raise ValueError("comparison config must use scene='microlens_spherical_cap'")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_radius_scale = 1.0 / max(float(cfg.lens_radius_fraction), 1e-12)
    real_crops, real_rho, names = _load_real_crops(
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        grid_size=cfg.grid_size,
        crop_radius_scale=crop_radius_scale,
    )
    real_stack = np.stack(real_crops, axis=0)
    real_median = np.median(real_stack, axis=0).astype(np.float32)

    bins = np.linspace(0.0, 1.15, 151, dtype=np.float32)
    centers = 0.5 * (bins[:-1] + bins[1:])
    real_profiles = np.stack([_radial_profile(crop, real_rho, bins) for crop in real_crops], axis=0)
    real_profile = np.nanmedian(real_profiles, axis=0).astype(np.float32)
    sim_rho = _physical_radial_grid(cfg)

    images: dict[str, np.ndarray] = {}
    profiles: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    for defocus in args.defocus_values:
        for raw_blur in args.raw_blur_values:
            simulated = _simulate(
                cfg,
                defocus=float(defocus),
                raw_blur_sigma_px=float(raw_blur),
                channel=str(args.channel),
                shear_px=args.shear_px,
            )
            raw_name = f"{args.channel}_defocus_{defocus:g}_blur_{raw_blur:g}"
            raw = _normalise_observation(simulated, sim_rho)
            images[raw_name] = raw
            profiles[raw_name] = _radial_profile(raw, sim_rho, bins)
            records.append(
                {
                    "name": raw_name,
                    "defocus_strength": float(defocus),
                    "raw_blur_sigma_px": float(raw_blur),
                    **_profile_metrics(real_profile, profiles[raw_name], centers),
                    **_image_metrics(real_median, raw, real_rho),
                }
            )

    best_raw = _best(records)
    best_raw_image = images[str(best_raw["name"])]

    _save_gray(args.output_dir / "real_median.png", real_median)
    _save_gray(args.output_dir / "best_raw.png", best_raw_image)
    for stale_name in ("best_dic_sum.png", "best_ideal_gradient_sum.png"):
        (args.output_dir / stale_name).unlink(missing_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = [
        ("24 raw crops: pixel median", real_median),
        (f"Simulated {args.channel} reflection", best_raw_image),
        ("Absolute shape residual", np.abs(real_median - best_raw_image)),
    ]
    for ax, (title, image) in zip(axes[0, :3], panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.5)
        ax.set_title(title)
        ax.axis("off")

    axes[0, 3].plot(centers, real_profile, color="black", linewidth=2, label="real raw median")
    axes[0, 3].plot(centers, profiles[str(best_raw["name"])], label="simulated raw")
    axes[0, 3].set_xlim(0.0, 1.15)
    axes[0, 3].set_ylim(0.0, 2.2)
    axes[0, 3].set_xlabel("normalised aperture radius")
    axes[0, 3].set_ylabel("robust normalised intensity")
    axes[0, 3].grid(alpha=0.25)
    axes[0, 3].legend(fontsize=8)

    sample_indices = [0, 4, 6, 18]
    for ax, index in zip(axes[1], sample_indices):
        ax.imshow(real_crops[index], cmap="gray", vmin=0.0, vmax=1.5)
        ax.set_title(f"real {names[index]}")
        ax.axis("off")
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    plt.close(fig)

    summary = {
        "config": str(args.config.resolve()),
        "channel": str(args.channel),
        "shear_px_override": args.shear_px,
        "real_sample_count": len(real_crops),
        "comparison_scope": (
            "single-frame raw reflection shape only; test-standard differences are constructed separately "
            "after empirical standard selection"
        ),
        "phase_scale_rad_per_um": phase_scale(cfg),
        "geometry": {
            "wavelength_um": cfg.wavelength,
            "numerical_aperture": cfg.numerical_aperture,
            "curvature_radius_um": cfg.lens_curvature_radius_um,
            "sag_um": cfg.lens_sag_um,
            "fov_um": cfg.grid_size * cfg.dx,
            "dx_um": cfg.dx,
            "lens_radius_fraction": cfg.lens_radius_fraction,
        },
        "best_raw": best_raw,
        "angular_variation_cv": {
            "real_median": _angular_cv(real_median, real_rho),
            "simulated_raw": _angular_cv(best_raw_image, real_rho),
        },
        "all_candidates": sorted(records, key=lambda item: float(item["profile_rmse"])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(best_raw, indent=2))
    print(args.output_dir / "comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
