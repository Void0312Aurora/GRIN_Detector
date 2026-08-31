from __future__ import annotations

"""Decisive test for the seam-band mechanism and the shear reference plane.

The real frames show common-mode bands next to the seam with a dominant period
of about 13 um, wavevector perpendicular to the seam. Candidate mechanisms:

- dark-port sheared double image of the coherent seam ring (band spacing tied to
  the object-plane shear; for a pure x-shear the doubling vanishes where the
  seam normal is perpendicular to the shear axis);
- defocused edge-wave interference (isotropic around the ring).

The test renders the calibrated dark-port model with a coherent seam component
under both shear readings and measures (a) the band period next to the seam and
(b) the angular distribution of band strength, for the simulations and for the
real median frame.
"""

import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_capture import _physical_radial_grid  # noqa: E402
from compare_reflection_dark_port import _load_real_crops_dn, load_exposure_calibration  # noqa: E402
from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_reference  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402

OUTPUT_DIR = ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "reflection_seam_band_test"
UM_PER_CROP_PX = 0.46875  # object-plane micrometres per 512-grid pixel


def _sector_band_strength(image: np.ndarray, rho: np.ndarray, *, sectors: int = 12) -> list[float]:
    """Arc/band strength per angular sector: std of the detrended radial profile
    in the zone 0.70-0.94."""

    h, w = image.shape
    yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
    xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    theta = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
    bins = np.arange(0.70, 0.94, 0.006, dtype=np.float32)
    strengths: list[float] = []
    for sector in range(sectors):
        mask = (theta >= 2.0 * np.pi * sector / sectors) & (theta < 2.0 * np.pi * (sector + 1) / sectors)
        profile = np.array(
            [np.mean(image[mask & (rho >= a) & (rho < b)]) for a, b in zip(bins[:-1], bins[1:])]
        )
        strengths.append(float(np.std(profile - gaussian_filter1d(profile, 10))))
    return strengths


def _patch_dominant_period(image: np.ndarray, rows: slice, cols: slice) -> tuple[float, float]:
    """Dominant spatial period (um) and orientation (deg) of a high-passed patch."""

    patch = image[rows, cols].astype(np.float32)
    patch = patch - gaussian_filter(patch, 6.0)
    n = patch.shape[0]
    window = np.hanning(n)[:, None] * np.hanning(patch.shape[1])[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * window)))
    cy, cx = spectrum.shape[0] // 2, spectrum.shape[1] // 2
    spectrum[cy - 3 : cy + 4, cx - 3 : cx + 4] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    fy = (iy - cy) / float(spectrum.shape[0])
    fx = (ix - cx) / float(spectrum.shape[1])
    frequency = float(np.hypot(fy, fx))
    period_um = UM_PER_CROP_PX / frequency if frequency > 0 else float("nan")
    orientation = float(np.degrees(np.arctan2(fy, fx)))
    return period_um, orientation


def _simulate(cfg0: Any, *, shear_px: float, rim_coherent: float, scale: float, offset: float) -> np.ndarray:
    params = dict(cfg0.capture_engine_params)
    params.update({"field_texture": {}, "camera": {}, "coherent_ghost": {}})
    params["shear_px"] = shear_px
    reflectance = dict(params["reflectance"])
    reflectance["rim_coherent_amplitude"] = rim_coherent
    # Keep the frame clean of per-lens dust for this deterministic test.
    reflectance["lens_point_scatter_count"] = 0
    params["reflectance"] = reflectance
    cfg = replace(cfg0, capture_engine_params=params)
    capture = create_simulation_engine(cfg).simulate_capture(
        microlens_reference(cfg), rng=np.random.default_rng(0)
    )
    intensity = np.asarray(capture.channels["I_x"], dtype=np.float32)
    return np.clip(scale * intensity + offset, 0.0, 255.0)


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
    scale, offset = load_exposure_calibration(ROOT)

    variants = {
        "shear1.57_coh0": {"shear_px": 1.5733333333, "rim_coherent": 0.0},
        "shear1.57_coh2": {"shear_px": 1.5733333333, "rim_coherent": 2.0},
        "shear21.3_coh0": {"shear_px": 21.3333333333, "rim_coherent": 0.0},
        "shear21.3_coh2": {"shear_px": 21.3333333333, "rim_coherent": 2.0},
    }
    images = {
        name: _simulate(cfg0, scale=scale, offset=offset, **settings)
        for name, settings in variants.items()
    }

    # Patch next to the seam, upper-left diagonal (same as the real inspection).
    rows, cols = slice(60, 220), slice(60, 220)
    results: dict[str, Any] = {}
    real_period, real_orientation = _patch_dominant_period(real_median, rows, cols)
    real_sectors = _sector_band_strength(real_median, rho)
    results["real_median"] = {
        "band_period_um": real_period,
        "band_orientation_deg": real_orientation,
        "sector_strengths_dn": [round(v, 3) for v in real_sectors],
        "sector_anisotropy": float(np.std(real_sectors) / max(np.mean(real_sectors), 1e-9)),
    }
    for name, image in images.items():
        period, orientation = _patch_dominant_period(image, rows, cols)
        sectors = _sector_band_strength(image, sim_rho)
        results[name] = {
            "band_period_um": period,
            "band_orientation_deg": orientation,
            "sector_strengths_dn": [round(v, 3) for v in sectors],
            "sector_anisotropy": float(np.std(sectors) / max(np.mean(sectors), 1e-9)),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5), constrained_layout=True)
    panels = [("real median", real_median)] + [(name, images[name]) for name in variants]
    for column, (title, image) in enumerate(panels):
        high = image - gaussian_filter(image, 6.0)
        axes[0, column].imshow(high, cmap="coolwarm", vmin=-3, vmax=3)
        axes[0, column].set_title(f"{title}\nhigh-pass ±3 DN", fontsize=9)
        axes[0, column].axis("off")
        axes[1, column].imshow(high[rows, cols], cmap="coolwarm", vmin=-3, vmax=3)
        entry = results["real_median" if column == 0 else panels[column][0]]
        axes[1, column].set_title(
            f"seam patch\nperiod {entry['band_period_um']:.1f} um, aniso {entry['sector_anisotropy']:.2f}",
            fontsize=9,
        )
        axes[1, column].axis("off")
    fig.savefig(OUTPUT_DIR / "seam_band_test.png", dpi=170)
    plt.close(fig)

    # Angular strength comparison plot.
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    angles = np.arange(12) * 30.0 + 15.0
    ax.plot(angles, real_sectors, "ko-", label="real median")
    for name in variants:
        ax.plot(angles, results[name]["sector_strengths_dn"], "o--", alpha=0.75, label=name)
    ax.set_xlabel("sector angle (deg, camera frame)")
    ax.set_ylabel("band strength (DN)")
    ax.set_title("Seam-band strength vs angle: x-shear doubling predicts nulls at 90/270 deg")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(OUTPUT_DIR / "seam_band_angular.png", dpi=170)
    plt.close(fig)

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sector_strengths_dn"} for k, v in results.items()}, indent=2))
    print(OUTPUT_DIR / "seam_band_test.png")
    print(OUTPUT_DIR / "seam_band_angular.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
