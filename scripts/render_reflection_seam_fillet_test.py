from __future__ import annotations

"""Render the dark-port model with a seam fillet/shoulder relief in the height
map and compare the seam-neighbourhood bands with the real median frame."""

import json
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_capture import _physical_radial_grid  # noqa: E402
from compare_reflection_dark_port import _load_real_crops_dn  # noqa: E402
from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_reference  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402

OUTPUT_DIR = ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "reflection_seam_fillet_test"


def _seam_profile(image: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.82, 1.18, 0.004, dtype=np.float32)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.array(
        [np.median(image[(rho >= lo) & (rho < hi)]) for lo, hi in zip(bins[:-1], bins[1:])]
    )
    return (centers - 1.0) * 109.71, profile


def main() -> int:
    stack, rho, names = _load_real_crops_dn(
        raw_dir=ROOT / "external_data" / "raw" / "wechat_2026-07_15-34" / "extracted" / "15.34",
        detections_path=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "valid_sample_detections.json",
        grid_size=512,
        crop_radius_scale=1.0 / 0.9142652028,
    )
    real_median = np.median(stack, axis=0).astype(np.float32)

    cfg0 = load_experiment_config(ROOT / "configs" / "reflection_microlens520_actual.json").simulation
    sim_rho = _physical_radial_grid(cfg0)
    fit = json.loads(
        (
            ROOT
            / "external_data"
            / "processed"
            / "wechat_2026-07_15-34"
            / "reflection_darkport_model_fit"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )["best_by_total_score"]
    scale = float(fit["exposure_scale_dn_per_unit"])
    offset = float(fit["dark_offset_dn"])

    def render(seam_kwargs: dict, rim_amplitude: float | None = None) -> np.ndarray:
        params = dict(cfg0.capture_engine_params)
        params.update({"field_texture": {}, "camera": {}, "coherent_ghost": {}})
        reflectance = dict(params["reflectance"])
        reflectance["lens_point_scatter_count"] = 0
        if rim_amplitude is not None:
            reflectance["rim_amplitude"] = rim_amplitude
        params["reflectance"] = reflectance
        cfg = replace(cfg0, capture_engine_params=params, **seam_kwargs)
        capture = create_simulation_engine(cfg).simulate_capture(
            microlens_reference(cfg), rng=np.random.default_rng(0)
        )
        return np.clip(scale * np.asarray(capture.channels["I_x"], dtype=np.float32) + offset, 0.0, 255.0)

    variants: dict[str, dict] = {
        "no relief": {"seam": {}, "rim": None},
        "fillet+shoulder": {
            "seam": {
                "seam_fillet_width_um": 2.0,
                "seam_shoulder_height_um": 0.35,
                "seam_shoulder_offset_um": 6.0,
                "seam_shoulder_width_um": 4.0,
            },
            "rim": None,
        },
        "calibrated A": {
            "seam": {
                "seam_fillet_width_um": 2.5,
                "seam_shoulder_height_um": 0.18,
                "seam_shoulder_offset_um": 6.0,
                "seam_shoulder_width_um": 4.5,
                "seam_trench_depth_um": 0.06,
                "seam_trench_offset_um": 5.0,
                "seam_trench_width_um": 3.0,
            },
            "rim": 2.0,
        },
        "calibrated B": {
            "seam": {
                "seam_fillet_width_um": 3.0,
                "seam_shoulder_height_um": 0.12,
                "seam_shoulder_offset_um": 7.0,
                "seam_shoulder_width_um": 5.5,
            },
            "rim": 1.5,
        },
    }
    images = {name: render(spec["seam"], spec["rim"]) for name, spec in variants.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 5)
    rows, cols = slice(60, 220), slice(60, 220)
    panels = [("real median", real_median)] + list(images.items())
    for column, (title, image) in enumerate(panels):
        high = image - gaussian_filter(image, 6.0)
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(high[rows, cols], cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_title(f"{title}\nseam patch high-pass ±3 DN", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :3])
    x_real, p_real = _seam_profile(real_median, rho)
    ax.plot(x_real, p_real, "k-", linewidth=2.2, label="real median")
    colors = {"no relief": "#999999", "fillet+shoulder": "#4878a8", "calibrated A": "#b06060", "calibrated B": "#6aa870"}
    for name, image in images.items():
        x_sim, p_sim = _seam_profile(image, sim_rho)
        ax.plot(x_sim, p_sim, color=colors[name], linewidth=1.4, label=name)
    ax.set_xlabel("distance from nominal seam (um)")
    ax.set_ylabel("median DN")
    ax.set_title("Seam radial profile: real vs seam-relief variants")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, 3:])
    for name, spec in variants.items():
        if not spec["seam"]:
            continue
        cfg = replace(cfg0, **spec["seam"])
        height = microlens_reference(cfg)
        n = height.shape[0]
        row = height[n // 2, :]
        coords = (np.arange(n) - 0.5 * (n - 1)) * float(cfg0.dx)
        mask = (coords > 85) & (coords < 135)
        ax.plot(coords[mask] - 109.71, row[mask], color=colors[name], label=name)
    ax.set_xlabel("distance from nominal seam (um)")
    ax.set_ylabel("height (um)")
    ax.set_title("Injected seam relief profiles")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.savefig(OUTPUT_DIR / "seam_fillet_test.png", dpi=170)
    plt.close(fig)

    # Quantitative: correlation of the seam-zone profile shape (0.9-1.15).
    results = {}
    zone = (x_real > -12) & (x_real < 17)
    for name, image in images.items():
        x_sim, p_sim = _seam_profile(image, sim_rho)
        corr = float(np.corrcoef(p_real[zone], p_sim[zone])[0, 1])
        results[name] = {"seam_profile_corr": round(corr, 4)}
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"variants": {k: v for k, v in variants.items()}, "results": results}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    print(OUTPUT_DIR / "seam_fillet_test.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
