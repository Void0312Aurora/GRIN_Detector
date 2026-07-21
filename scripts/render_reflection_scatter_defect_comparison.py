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
from mini_grin_rebuild.simulation.engines.optical_leakage_lite import OpticalLeakageLiteEngine  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402


def _scatter_defect_modifier(
    shape: tuple[int, int],
    *,
    center_offset_px: tuple[float, float],
    sigma_px: float,
    rough_rad: float,
    amplitude_factor: float,
    texture_sigma_px: float,
    seed: int,
) -> np.ndarray:
    """Complex field modifier of a localized scattering defect.

    Inside a Gaussian support: correlated rough phase (surface damage /
    contamination texture) plus an optional local amplitude change.
    """

    h, w = shape
    cy = 0.5 * (h - 1) + float(center_offset_px[0])
    cx = 0.5 * (w - 1) + float(center_offset_px[1])
    yy, xx = np.indices(shape, dtype=np.float32)
    support = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / float(sigma_px) ** 2).astype(np.float32)
    rough_basis = OpticalLeakageLiteEngine._normalized_random_field(
        np.random.default_rng(int(seed)),
        shape,
        sigma_x_px=float(texture_sigma_px),
        sigma_y_px=float(texture_sigma_px),
    )
    phase = float(rough_rad) * support * rough_basis
    amplitude = 1.0 + (float(amplitude_factor) - 1.0) * support
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate scattering-defect strength against the real anomalous frames: sweep local rough-phase "
            "and amplitude perturbations, render dark-port differentials in DN, and find the parameter range "
            "that reproduces the observed 120-140 DN anomaly contrast."
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
        default=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "reflection_scatter_defect_comparison",
    )
    parser.add_argument("--defect-sigma-um", type=float, default=3.0)
    parser.add_argument("--defect-offset-radius-frac", type=float, default=0.15)
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    cfg = experiment.simulation
    sim_rho = _physical_radial_grid(cfg)

    real_stack, rho, names = _load_real_crops_dn(
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        grid_size=cfg.grid_size,
        crop_radius_scale=1.0 / max(float(cfg.lens_radius_fraction), 1e-12),
    )
    real_median = np.median(real_stack, axis=0).astype(np.float32)
    bins = np.linspace(0.0, 1.50, 121, dtype=np.float32)
    real_profile = _annulus_profile(real_median, rho, bins)

    engine = create_simulation_engine(cfg)
    standard_height = microlens_reference(cfg)
    clean = engine.simulate_capture(standard_height, rng=np.random.default_rng(0))
    clean_ix = np.asarray(clean.channels["I_x"], dtype=np.float32)
    scale, offset, _ = _fit_scale(_annulus_profile(clean_ix, sim_rho, bins), real_profile)

    def _to_dn(intensity: np.ndarray) -> np.ndarray:
        return np.clip(scale * np.asarray(intensity, dtype=np.float32) + offset, 0.0, 255.0)

    radius_px = float(cfg.lens_radius_fraction) * 0.5 * float(cfg.grid_size)
    offset_px = (0.0, float(args.defect_offset_radius_frac) * radius_px)
    sigma_px = float(args.defect_sigma_um) / float(cfg.dx)
    cy = int(round(0.5 * (cfg.grid_size - 1) + offset_px[0]))
    cx = int(round(0.5 * (cfg.grid_size - 1) + offset_px[1]))
    half = int(round(4.0 * sigma_px))

    # Real anomaly contrast references (interior p99.9 above interior median).
    real_refs: dict[str, float] = {}
    for name in ("5.bmp", "19.bmp", "20.bmp"):
        image = real_stack[names.index(name)]
        interior = image[rho <= 0.85].astype(np.float64)
        real_refs[name] = float(np.quantile(interior, 0.999) - np.median(interior))

    # Sweep in absolute scattering amplitude so the calibration is invariant to
    # the configured smooth-lens amplitude (the multiplier depends on it).
    lens_amplitude = float(cfg.capture_engine_params["reflectance"]["lens_amplitude"])
    absolute_amplitudes = (0.2, 0.25, 0.3, 0.4)
    amplitude_factors = tuple(round(a / lens_amplitude, 2) for a in absolute_amplitudes)
    rough_values = (0.3, 0.6, 1.2, 2.4)
    records: list[dict[str, Any]] = []
    example_maps: dict[str, np.ndarray] = {}
    for amplitude_factor in amplitude_factors:
        for rough in rough_values:
            scatter = _scatter_defect_modifier(
                tuple(standard_height.shape),
                center_offset_px=offset_px,
                sigma_px=sigma_px,
                rough_rad=rough,
                amplitude_factor=amplitude_factor,
                texture_sigma_px=1.5,
                seed=97,
            )
            bundle = engine.simulate_bundle(
                {"standard": standard_height, "test": standard_height},
                rng=np.random.default_rng(0),
                extra_field_modifiers={"test": scatter},
            )
            std_dn = _to_dn(bundle.captures["standard"].channels["I_x"])
            test_dn = _to_dn(bundle.captures["test"].channels["I_x"])
            diff = test_dn - std_dn
            roi = diff[cy - half : cy + half + 1, cx - half : cx + half + 1]
            label = f"amp{amplitude_factor:g}_rough{rough:g}"
            records.append(
                {
                    "label": label,
                    "amplitude_factor": float(amplitude_factor),
                    "rough_rad": float(rough),
                    "defect_peak_diff_dn": float(np.max(np.abs(roi))),
                    "defect_p99_diff_dn": float(np.quantile(np.abs(roi), 0.99)),
                    "test_peak_dn_in_roi": float(
                        np.max(test_dn[cy - half : cy + half + 1, cx - half : cx + half + 1])
                    ),
                }
            )
            example_maps[label] = diff

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(17, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 5)
    show = [
        (f"amp{amplitude_factors[1]:g}_rough0.6", f"|A|={absolute_amplitudes[1]:g}, 0.6 rad"),
        (f"amp{amplitude_factors[2]:g}_rough0.6", f"|A|={absolute_amplitudes[2]:g}, 0.6 rad"),
        (f"amp{amplitude_factors[2]:g}_rough1.2", f"|A|={absolute_amplitudes[2]:g}, 1.2 rad"),
        (f"amp{amplitude_factors[3]:g}_rough1.2", f"|A|={absolute_amplitudes[3]:g}, 1.2 rad"),
    ]
    for column, (label, title) in enumerate(show):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(example_maps[label], cmap="coolwarm", vmin=-140, vmax=140)
        ax.set_title(f"{title}\npeak {next(r['defect_peak_diff_dn'] for r in records if r['label'] == label):.0f} DN", fontsize=8)
        ax.axis("off")

    ax_real = fig.add_subplot(grid[0, 4])
    image_5 = real_stack[names.index("5.bmp")]
    ax_real.imshow(np.clip(image_5, 0, 140), cmap="gray", vmin=0, vmax=140)
    ax_real.set_title("real 5.bmp (0-140 DN)", fontsize=9)
    ax_real.axis("off")

    ax_curve = fig.add_subplot(grid[1, :3])
    for amplitude_factor, absolute, color in zip(
        amplitude_factors, absolute_amplitudes, ("#4878a8", "#6aa870", "#b06060", "#8060a8")
    ):
        subset = [r for r in records if r["amplitude_factor"] == amplitude_factor]
        ax_curve.plot(
            [r["rough_rad"] for r in subset],
            [r["defect_peak_diff_dn"] for r in subset],
            "o-",
            color=color,
            label=f"|A|={absolute:g} (x{amplitude_factor:g} lens)",
        )
    band = [min(real_refs.values()), max(real_refs.values())]
    ax_curve.axhspan(band[0], band[1], color="#c9a04e", alpha=0.25, label="real anomaly contrast 5/19/20.bmp")
    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("local rough phase strength (rad)")
    ax_curve.set_ylabel("defect peak |ΔDN|")
    ax_curve.set_title("Scattering-defect response vs real anomaly contrast")
    ax_curve.grid(alpha=0.3, which="both")
    ax_curve.legend(fontsize=8)

    ax_note = fig.add_subplot(grid[1, 3:])
    ax_note.axis("off")
    background_amplitude = float(cfg.capture_engine_params["reflectance"]["background_amplitude"])
    matching = [
        r for r in records if band[0] * 0.8 <= r["defect_peak_diff_dn"] <= band[1] * 1.25
    ]
    matching_text = "\n".join(
        f"  {r['label']}: peak {r['defect_peak_diff_dn']:.0f} DN"
        for r in sorted(matching, key=lambda r: r["defect_peak_diff_dn"])
    ) or "  (none in sweep)"
    height_equivalent = (
        "height-only pit (260 nm, full wrap): peak 4 DN\n"
        f"real anomaly contrast: {band[0]:.0f}-{band[1]:.0f} DN\n"
        f"lens specular amplitude: {lens_amplitude:g} (fixture: {background_amplitude:g})\n"
        "candidates inside the real contrast band:\n"
        f"{matching_text}"
    )
    ax_note.text(0.02, 0.6, height_equivalent, fontsize=10, family="monospace", va="center")

    fig.savefig(args.output_dir / "scatter_defect_comparison.png", dpi=170)
    plt.close(fig)

    summary = {
        "scope": (
            "scattering-defect calibration: local correlated rough phase and amplitude perturbations "
            "rendered through the calibrated dark-port model and compared with the anomaly contrast of "
            "the real defective frames"
        ),
        "exposure_scale_dn_per_unit": scale,
        "dark_offset_dn": offset,
        "defect_sigma_um": float(args.defect_sigma_um),
        "lens_amplitude": lens_amplitude,
        "absolute_scatter_amplitudes": [float(a) for a in absolute_amplitudes],
        "real_anomaly_contrast_dn": real_refs,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "real_anomaly_contrast_dn": {k: round(v, 1) for k, v in real_refs.items()},
                "response_peak_dn": {r["label"]: round(r["defect_peak_diff_dn"], 1) for r in records},
            },
            indent=2,
        )
    )
    print(args.output_dir / "scatter_defect_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
