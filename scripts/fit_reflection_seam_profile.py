from __future__ import annotations

"""Joint fit of the seam relief and seam scattering parameters against the real
median seam profile: coarse random search followed by Nelder-Mead refinement.
The exposure calibration stays fixed; interior and fixture levels enter the
objective as soft constraints so the optimiser cannot trade them away."""

import json
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

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

OUTPUT_DIR = ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "reflection_seam_profile_fit"

PARAM_NAMES = [
    "seam_fillet_width_um",
    "seam_shoulder_height_um",
    "seam_shoulder_offset_um",
    "seam_shoulder_width_um",
    "seam_trench_depth_um",
    "seam_trench_offset_um",
    "seam_trench_width_um",
    "rim_amplitude",
    "rim_inner_width_px",
    "rim_outer_width_px",
]
BOUNDS = np.array(
    [
        [0.5, 5.0],
        [0.0, 0.6],
        [2.0, 12.0],
        [2.0, 8.0],
        [0.0, 0.3],
        [2.0, 10.0],
        [1.5, 6.0],
        [0.5, 8.0],
        [1.0, 8.0],
        [6.0, 40.0],
    ]
)


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
    fit_meta = json.loads(
        (
            ROOT
            / "external_data"
            / "processed"
            / "wechat_2026-07_15-34"
            / "reflection_darkport_model_fit"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )["best_by_total_score"]
    scale = float(fit_meta["exposure_scale_dn_per_unit"])
    offset = float(fit_meta["dark_offset_dn"])

    bins = np.arange(0.82, 1.30, 0.004, dtype=np.float32)
    centers = (0.5 * (bins[:-1] + bins[1:]) - 1.0) * 109.71

    def profile_of(image: np.ndarray, rr: np.ndarray) -> np.ndarray:
        return np.array(
            [np.median(image[(rr >= lo) & (rr < hi)]) for lo, hi in zip(bins[:-1], bins[1:])]
        )

    real_profile = profile_of(real_median, rho)
    real_interior = float(np.median(real_median[rho <= 0.65]))
    real_fixture = float(np.median(real_median[(rho >= 1.32) & (rho <= 1.45)]))

    def render(vector: np.ndarray) -> np.ndarray:
        values = {name: float(v) for name, v in zip(PARAM_NAMES, vector)}
        params = dict(cfg0.capture_engine_params)
        params.update({"field_texture": {}, "camera": {}, "coherent_ghost": {}})
        reflectance = dict(params["reflectance"])
        reflectance["lens_point_scatter_count"] = 0
        reflectance["rim_amplitude"] = values["rim_amplitude"]
        reflectance["rim_inner_width_px"] = values["rim_inner_width_px"]
        reflectance["rim_outer_width_px"] = values["rim_outer_width_px"]
        params["reflectance"] = reflectance
        cfg = replace(
            cfg0,
            capture_engine_params=params,
            seam_fillet_width_um=values["seam_fillet_width_um"],
            seam_shoulder_height_um=values["seam_shoulder_height_um"],
            seam_shoulder_offset_um=values["seam_shoulder_offset_um"],
            seam_shoulder_width_um=values["seam_shoulder_width_um"],
            seam_trench_depth_um=values["seam_trench_depth_um"],
            seam_trench_offset_um=values["seam_trench_offset_um"],
            seam_trench_width_um=values["seam_trench_width_um"],
        )
        capture = create_simulation_engine(cfg).simulate_capture(
            microlens_reference(cfg), rng=np.random.default_rng(0)
        )
        return np.clip(scale * np.asarray(capture.channels["I_x"], dtype=np.float32) + offset, 0.0, 255.0)

    evaluations: list[tuple[float, np.ndarray]] = []

    def objective(vector: np.ndarray) -> float:
        clipped = np.clip(vector, BOUNDS[:, 0], BOUNDS[:, 1])
        image = render(clipped)
        sim_profile = profile_of(image, sim_rho)
        profile_rmse = float(np.sqrt(np.mean((sim_profile - real_profile) ** 2)))
        interior_penalty = abs(float(np.median(image[sim_rho <= 0.65])) - real_interior)
        fixture_penalty = abs(
            float(np.median(image[(sim_rho >= 1.32) & (sim_rho <= 1.45)])) - real_fixture
        )
        cost = profile_rmse + 2.0 * interior_penalty + 0.5 * fixture_penalty
        evaluations.append((cost, clipped.copy()))
        return cost

    rng = np.random.default_rng(7)
    best_cost, best_vector = np.inf, None
    for index in range(70):
        candidate = BOUNDS[:, 0] + rng.random(len(PARAM_NAMES)) * (BOUNDS[:, 1] - BOUNDS[:, 0])
        cost = objective(candidate)
        if cost < best_cost:
            best_cost, best_vector = cost, candidate
    assert best_vector is not None

    result = minimize(
        objective,
        best_vector,
        method="Nelder-Mead",
        options={"maxfev": 80, "xatol": 0.05, "fatol": 0.2},
    )
    final_vector = np.clip(result.x, BOUNDS[:, 0], BOUNDS[:, 1])
    final_cost = objective(final_vector)
    if final_cost > best_cost:
        final_vector, final_cost = best_vector, best_cost

    fitted = {name: round(float(v), 4) for name, v in zip(PARAM_NAMES, final_vector)}
    final_image = render(final_vector)
    final_profile = profile_of(final_image, sim_rho)
    baseline_image = render(
        np.array([0.5, 0.0, 5.0, 4.0, 0.0, 4.0, 3.0, 6.0, 4.0, 24.0])
    )
    baseline_profile = profile_of(baseline_image, sim_rho)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    rows, cols = slice(60, 220), slice(60, 220)
    panels = [
        ("real median", real_median),
        ("previous model", baseline_image),
        ("fitted seam", final_image),
    ]
    for column, (title, image) in enumerate(panels):
        ax = fig.add_subplot(grid[0, column])
        high = image - gaussian_filter(image, 6.0)
        ax.imshow(high[rows, cols], cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_title(f"{title}\nseam patch high-pass", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(grid[0, 3])
    ax.imshow(np.log1p(final_image), cmap="magma", vmin=0, vmax=np.log(256.0))
    ax.set_title("fitted seam: full view (log1p)", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(grid[1, :])
    ax.plot(centers, real_profile, "k-", linewidth=2.4, label="real median")
    ax.plot(centers, baseline_profile, color="#999999", linewidth=1.3, label="previous model")
    ax.plot(centers, final_profile, color="#b06060", linewidth=1.8, label="fitted seam")
    ax.set_xlabel("distance from nominal seam (um)")
    ax.set_ylabel("median DN")
    profile_rmse = float(np.sqrt(np.mean((final_profile - real_profile) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((baseline_profile - real_profile) ** 2)))
    ax.set_title(
        f"Seam profile after joint fit: RMSE {baseline_rmse:.1f} -> {profile_rmse:.1f} DN "
        f"(corr {np.corrcoef(final_profile, real_profile)[0, 1]:.3f})"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig(OUTPUT_DIR / "seam_profile_fit.png", dpi=170)
    plt.close(fig)

    summary = {
        "scope": "joint Nelder-Mead fit of seam relief + rim scattering against the real median seam profile",
        "fitted_parameters": fitted,
        "profile_rmse_dn": profile_rmse,
        "baseline_profile_rmse_dn": baseline_rmse,
        "profile_corr": float(np.corrcoef(final_profile, real_profile)[0, 1]),
        "num_evaluations": len(evaluations),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(OUTPUT_DIR / "seam_profile_fit.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
