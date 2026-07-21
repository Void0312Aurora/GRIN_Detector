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

from compare_reflection_capture import _physical_radial_grid  # noqa: E402
from compare_reflection_dark_port import (  # noqa: E402
    _annulus_profile,
    _fit_scale,
    _load_real_crops_dn,
)
from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_reference  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402

# One 2*pi phase wrap corresponds to lambda/2 of height in front-surface reflection.
WRAP_HEIGHT_UM = 0.52 / 2.0


def _gaussian_pit(
    shape: tuple[int, int],
    *,
    center_offset_px: tuple[float, float],
    sigma_px: float,
    depth_um: float,
) -> np.ndarray:
    h, w = shape
    cy = 0.5 * (h - 1) + float(center_offset_px[0])
    cx = 0.5 * (w - 1) + float(center_offset_px[1])
    yy, xx = np.indices(shape, dtype=np.float32)
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (-float(depth_um) * np.exp(-0.5 * rr2 / float(sigma_px) ** 2)).astype(np.float32)


def _real_frame_anomaly_stats(image: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    interior = (rho <= 0.85)
    values = np.asarray(image[interior], dtype=np.float64)
    baseline = float(np.median(values))
    return {
        "interior_median_dn": baseline,
        "interior_p99_dn": float(np.quantile(values, 0.99)),
        "interior_p999_dn": float(np.quantile(values, 0.999)),
        "anomaly_area_px": int(np.sum(values > baseline + 4.0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Defect visibility under the calibrated dark-port radiometry: inject Gaussian pits of "
            "known depth into the clean lens and measure the DN response against the camera floor."
        )
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
        default=ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "reflection_defect_visibility",
    )
    parser.add_argument("--defect-sigma-um", type=float, default=2.0)
    parser.add_argument("--defect-offset-radius-frac", type=float, default=0.15)
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    cfg = experiment.simulation
    sim_rho = _physical_radial_grid(cfg)

    # Exposure calibration against the 24-frame median, same protocol as the model fit.
    crop_radius_scale = 1.0 / max(float(cfg.lens_radius_fraction), 1e-12)
    real_stack, rho, names = _load_real_crops_dn(
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        grid_size=cfg.grid_size,
        crop_radius_scale=crop_radius_scale,
    )
    real_median = np.median(real_stack, axis=0).astype(np.float32)
    bins = np.linspace(0.0, 1.50, 121, dtype=np.float32)
    real_profile = _annulus_profile(real_median, rho, bins)

    engine = create_simulation_engine(cfg)
    standard_height = microlens_reference(cfg)
    clean_bundle = engine.simulate_bundle(
        {"standard": standard_height, "test": standard_height},
        rng=np.random.default_rng(0),
    )
    clean_ix = np.asarray(clean_bundle.captures["standard"].channels["I_x"], dtype=np.float32)
    scale, offset, _ = _fit_scale(_annulus_profile(clean_ix, sim_rho, bins), real_profile)

    def _to_dn(intensity: np.ndarray) -> np.ndarray:
        return np.clip(scale * np.asarray(intensity, dtype=np.float32) + offset, 0.0, 255.0)

    # Photon floor: shot + read noise at the interior level, from the measured transfer.
    interior_dn = float(np.median(_to_dn(clean_ix)[sim_rho <= 0.65]))
    noise_sigma_dn = float(np.sqrt(max(0.46 * interior_dn, 0.0) + 1.0 / 12.0))
    visibility_threshold_dn = 3.0 * noise_sigma_dn

    radius_px = float(cfg.lens_radius_fraction) * 0.5 * float(cfg.grid_size)
    offset_px = (0.0, float(args.defect_offset_radius_frac) * radius_px)
    sigma_px = float(args.defect_sigma_um) / float(cfg.dx)

    wrap_fractions = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    records: list[dict[str, Any]] = []
    differential_maps: list[np.ndarray] = []
    for wrap_fraction in wrap_fractions:
        depth_um = wrap_fraction * WRAP_HEIGHT_UM
        defect = _gaussian_pit(
            standard_height.shape,
            center_offset_px=offset_px,
            sigma_px=sigma_px,
            depth_um=depth_um,
        )
        bundle = engine.simulate_bundle(
            {"standard": standard_height, "test": standard_height + defect},
            rng=np.random.default_rng(0),
        )
        std_dn = _to_dn(bundle.captures["standard"].channels["I_x"])
        test_dn = _to_dn(bundle.captures["test"].channels["I_x"])
        diff = test_dn - std_dn
        cy = int(round(0.5 * (cfg.grid_size - 1) + offset_px[0]))
        cx = int(round(0.5 * (cfg.grid_size - 1) + offset_px[1]))
        half = int(round(4.0 * sigma_px))
        roi = diff[cy - half : cy + half + 1, cx - half : cx + half + 1]
        records.append(
            {
                "wrap_fraction": wrap_fraction,
                "depth_um": depth_um,
                "depth_nm": 1e3 * depth_um,
                "defect_peak_diff_dn": float(np.max(np.abs(roi))),
                "visible_area_px": int(np.sum(np.abs(roi) >= visibility_threshold_dn)),
                "test_peak_dn_in_roi": float(np.max(test_dn[cy - half : cy + half + 1, cx - half : cx + half + 1])),
                "visible": bool(float(np.max(np.abs(roi))) >= visibility_threshold_dn),
            }
        )
        differential_maps.append(diff)

    # Real defect-bearing frames for context.
    real_stats = {}
    for name in ("7.bmp", "5.bmp", "13.bmp", "19.bmp", "20.bmp"):
        index = names.index(name)
        real_stats[name] = _real_frame_anomaly_stats(real_stack[index], rho)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(17, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, max(len(wrap_fractions), 4))
    show_limit = max(visibility_threshold_dn * 4.0, 12.0)
    for column, (record, diff) in enumerate(zip(records, differential_maps)):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(diff, cmap="coolwarm", vmin=-show_limit, vmax=show_limit)
        ax.set_title(
            f"depth {record['depth_nm']:.0f} nm\npeak {record['defect_peak_diff_dn']:.1f} DN",
            fontsize=8,
        )
        ax.axis("off")

    ax_curve = fig.add_subplot(grid[1, : max(len(wrap_fractions) // 2, 2)])
    depths = [record["depth_nm"] for record in records]
    peaks = [record["defect_peak_diff_dn"] for record in records]
    ax_curve.plot(depths, peaks, "o-", color="#b06060", label="sim defect peak |ΔDN|")
    ax_curve.axhline(
        visibility_threshold_dn,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"3σ floor ≈ {visibility_threshold_dn:.1f} DN",
    )
    for name, stats in real_stats.items():
        if name == "7.bmp":
            continue
        ax_curve.axhline(
            stats["interior_p999_dn"] - stats["interior_median_dn"],
            color="#4878a8",
            alpha=0.45,
            linewidth=1.0,
        )
    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("defect depth (nm)")
    ax_curve.set_ylabel("peak |ΔDN| in defect ROI")
    ax_curve.set_title("Dark-port defect response vs camera floor (blue: real frames' interior p99.9 - median)")
    ax_curve.grid(alpha=0.3, which="both")
    ax_curve.legend(fontsize=8)

    ax_real = fig.add_subplot(grid[1, max(len(wrap_fractions) // 2, 2) :])
    index_5 = names.index("5.bmp")
    ax_real.imshow(np.clip(real_stack[index_5], 0, 60), cmap="gray", vmin=0, vmax=60)
    ax_real.set_title("real 5.bmp (0-60 DN stretch): central pit anomaly", fontsize=9)
    ax_real.axis("off")

    fig.savefig(args.output_dir / "defect_visibility.png", dpi=170)
    plt.close(fig)

    visible_records = [record for record in records if record["visible"]]
    minimum_visible_depth_nm = visible_records[0]["depth_nm"] if visible_records else None
    summary = {
        "scope": (
            "synthetic Gaussian pits injected into the clean dark-port model with calibrated exposure; "
            "visibility judged against the shot+quantization floor from the measured photon transfer"
        ),
        "exposure_scale_dn_per_unit": scale,
        "dark_offset_dn": offset,
        "interior_floor_dn": interior_dn,
        "noise_sigma_dn": noise_sigma_dn,
        "visibility_threshold_dn": visibility_threshold_dn,
        "defect_sigma_um": float(args.defect_sigma_um),
        "records": records,
        "minimum_visible_depth_nm": minimum_visible_depth_nm,
        "real_frame_interior_stats": real_stats,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "visibility_threshold_dn": visibility_threshold_dn,
                "minimum_visible_depth_nm": minimum_visible_depth_nm,
                "response": {f"{r['depth_nm']:.0f}nm": round(r["defect_peak_diff_dn"], 2) for r in records},
                "real_frame_interior_p999_minus_median": {
                    name: round(stats["interior_p999_dn"] - stats["interior_median_dn"], 1)
                    for name, stats in real_stats.items()
                },
            },
            indent=2,
        )
    )
    print(args.output_dir / "defect_visibility.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
