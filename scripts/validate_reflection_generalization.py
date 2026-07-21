from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

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

from compare_reflection_capture import (  # noqa: E402
    _csv_floats,
    _image_metrics,
    _load_real_crops,
    _normalise_observation,
    _physical_radial_grid,
    _profile_metrics,
    _radial_profile,
    _simulate,
)
from render_reflection_localized_ghost_comparison import (  # noqa: E402
    _correlation,
    _high_pass,
    _local_energy,
    _prepare_base,
    _render_component,
    _spectral_descriptor,
    _weighted_geometry,
)
from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402


def _aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _candidate_label(candidate: Mapping[str, Any]) -> str:
    return f"defocus={candidate['defocus']:g}, blur={candidate['blur']:g}"


def _holdout_defocus_blur(
    cfg: Any,
    *,
    raw_dir: Path,
    detections_path: Path,
    defocus_values: list[float],
    raw_blur_values: list[float],
    num_splits: int,
) -> dict[str, Any]:
    """Held-out validation of the global defocus/blur nuisance selection."""

    crop_radius_scale = 1.0 / max(float(cfg.lens_radius_fraction), 1e-12)
    crops, rho, names = _load_real_crops(
        raw_dir=raw_dir,
        detections_path=detections_path,
        grid_size=cfg.grid_size,
        crop_radius_scale=crop_radius_scale,
    )
    stack = np.stack(crops, axis=0).astype(np.float32)
    bins = np.linspace(0.0, 1.15, 151, dtype=np.float32)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profiles = np.stack([_radial_profile(crop, rho, bins) for crop in crops], axis=0)
    sim_rho = _physical_radial_grid(cfg)

    candidates: list[dict[str, Any]] = []
    for defocus in defocus_values:
        for blur in raw_blur_values:
            image = _normalise_observation(
                _simulate(cfg, defocus=float(defocus), raw_blur_sigma_px=float(blur)),
                sim_rho,
            )
            candidates.append(
                {
                    "defocus": float(defocus),
                    "blur": float(blur),
                    "image": image,
                    "profile": _radial_profile(image, sim_rho, bins),
                }
            )

    def _profile_rmse_against(profile: np.ndarray, candidate: Mapping[str, Any]) -> float:
        return float(_profile_metrics(profile, candidate["profile"], centers)["profile_rmse"])

    def _select(profile: np.ndarray) -> int:
        return int(np.argmin([_profile_rmse_against(profile, candidate) for candidate in candidates]))

    def _evaluate(index: int, profile: np.ndarray, median_image: np.ndarray) -> dict[str, float]:
        candidate = candidates[index]
        return {
            **_profile_metrics(profile, candidate["profile"], centers),
            **_image_metrics(median_image, candidate["image"], rho),
        }

    # In-sample reference on all 24 frames (mirrors compare_reflection_capture).
    profile_all = np.nanmedian(profiles, axis=0).astype(np.float32)
    median_all = np.median(stack, axis=0).astype(np.float32)
    index_all = _select(profile_all)
    in_sample = {
        "selected": _candidate_label(candidates[index_all]),
        **_evaluate(index_all, profile_all, median_all),
    }

    # Random half/half splits: select on the fit half, evaluate on the held-out half.
    rng = np.random.default_rng(20260721)
    split_records: list[dict[str, Any]] = []
    selection_counts: dict[str, int] = {}
    for split_index in range(int(num_splits)):
        perm = rng.permutation(len(names))
        half = len(names) // 2
        fit_idx, eval_idx = perm[:half], perm[half:]
        fit_profile = np.nanmedian(profiles[fit_idx], axis=0).astype(np.float32)
        eval_profile = np.nanmedian(profiles[eval_idx], axis=0).astype(np.float32)
        eval_median = np.median(stack[eval_idx], axis=0).astype(np.float32)
        selected = _select(fit_profile)
        oracle = _select(eval_profile)
        heldout = _evaluate(selected, eval_profile, eval_median)
        oracle_rmse = _profile_rmse_against(eval_profile, candidates[oracle])
        label = _candidate_label(candidates[selected])
        selection_counts[label] = selection_counts.get(label, 0) + 1
        split_records.append(
            {
                "split": split_index,
                "selected": label,
                "oracle": _candidate_label(candidates[oracle]),
                "heldout_profile_rmse": float(heldout["profile_rmse"]),
                "heldout_profile_corr": float(heldout["profile_corr"]),
                "heldout_image_corr_aperture": float(heldout["image_corr_aperture"]),
                "heldout_image_corr_edge": float(heldout["image_corr_edge"]),
                "oracle_profile_rmse": float(oracle_rmse),
                "selection_regret_rmse": float(heldout["profile_rmse"] - oracle_rmse),
            }
        )

    # Leave-one-out: select on 23 frames, evaluate on the single held-out frame.
    loo_records: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        fit_idx = np.asarray([i for i in range(len(names)) if i != index], dtype=np.int64)
        fit_profile = np.nanmedian(profiles[fit_idx], axis=0).astype(np.float32)
        selected = _select(fit_profile)
        heldout = _evaluate(selected, profiles[index], stack[index])
        label = _candidate_label(candidates[selected])
        selection_counts[label] = selection_counts.get(label, 0) + 1
        loo_records.append(
            {
                "file": name,
                "selected": label,
                "heldout_profile_rmse": float(heldout["profile_rmse"]),
                "heldout_profile_corr": float(heldout["profile_corr"]),
                "heldout_image_corr_aperture": float(heldout["image_corr_aperture"]),
                "heldout_image_corr_edge": float(heldout["image_corr_edge"]),
            }
        )

    return {
        "candidate_grid": {
            "defocus_values": [float(v) for v in defocus_values],
            "raw_blur_values": [float(v) for v in raw_blur_values],
        },
        "in_sample_all_24": {
            key: value for key, value in in_sample.items() if not isinstance(value, np.ndarray)
        },
        "selection_counts": selection_counts,
        "random_half_splits": {
            "num_splits": int(num_splits),
            "heldout_profile_rmse": _aggregate([r["heldout_profile_rmse"] for r in split_records]),
            "heldout_profile_corr": _aggregate([r["heldout_profile_corr"] for r in split_records]),
            "heldout_image_corr_aperture": _aggregate(
                [r["heldout_image_corr_aperture"] for r in split_records]
            ),
            "heldout_image_corr_edge": _aggregate([r["heldout_image_corr_edge"] for r in split_records]),
            "selection_regret_rmse": _aggregate([r["selection_regret_rmse"] for r in split_records]),
            "records": split_records,
        },
        "leave_one_out": {
            "heldout_profile_rmse": _aggregate([r["heldout_profile_rmse"] for r in loo_records]),
            "heldout_profile_corr": _aggregate([r["heldout_profile_corr"] for r in loo_records]),
            "heldout_image_corr_aperture": _aggregate(
                [r["heldout_image_corr_aperture"] for r in loo_records]
            ),
            "worst_frames_by_profile_rmse": sorted(
                loo_records, key=lambda r: float(r["heldout_profile_rmse"]), reverse=True
            )[:6],
            "records": loo_records,
        },
        "_loo_records": loo_records,
    }


def _ghost_transfer(
    cfg: Any,
    *,
    empirical_npz: Path,
    ghost_summary_path: Path,
) -> dict[str, Any]:
    """Cross-frame transfer check for the localized ghost fits of 13.bmp and 14.bmp."""

    with np.load(empirical_npz) as data:
        names = [str(item) for item in data["names"].tolist()]
        registered = np.asarray(data["registered"], dtype=np.float32)
        rho = np.asarray(data["rho"], dtype=np.float32)
    lookup = {name: registered[index] for index, name in enumerate(names)}
    high_stack = np.stack([_high_pass(image) for image in registered], axis=0)
    robust_high = np.median(high_stack, axis=0).astype(np.float32)
    aperture_mask = rho <= 0.82

    ghost_summary = json.loads(ghost_summary_path.read_text(encoding="utf-8"))
    fits = ghost_summary["fits"]
    frames = tuple(sorted(fits.keys()))

    engine, optics, field, propagated = _prepare_base(cfg)
    sim_clean_raw = gaussian_filter(np.abs(propagated) ** 2, sigma=1.0, mode="reflect")
    sim_clean = _normalise_observation(sim_clean_raw, rho)
    sim_clean_high = _high_pass(sim_clean)

    targets: dict[str, dict[str, Any]] = {}
    for name in frames:
        signal = (_high_pass(lookup[name]) - robust_high).astype(np.float32)
        energy = _local_energy(signal)
        weight = energy / max(float(np.quantile(energy[aperture_mask], 0.99)), 1e-8)
        weight = np.clip(weight, 0.0, 1.0).astype(np.float32)
        descriptor, band, _ = _spectral_descriptor(signal, weight)
        targets[name] = {
            "signal": signal,
            "energy": energy,
            "weight": weight,
            "descriptor": descriptor,
            "band": band,
            "geometry": _weighted_geometry(energy, aperture_mask),
        }

    renders: dict[str, dict[str, Any]] = {}
    for name in frames:
        component = fits[name]["component"]
        texture_seed = int(fits[name]["source_texture_seed"])
        simulated = _render_component(
            engine,
            optics,
            field,
            propagated,
            component,
            rho,
            texture_seed=texture_seed,
        )
        signal = (_high_pass(simulated) - sim_clean_high).astype(np.float32)
        renders[name] = {"signal": signal, "energy": _local_energy(signal)}

    matrix: dict[str, dict[str, float]] = {}
    for fit_name in frames:
        for eval_name in frames:
            target = targets[eval_name]
            render = renders[fit_name]
            descriptor_sim, _, _ = _spectral_descriptor(render["signal"], target["weight"])
            geometry_sim = _weighted_geometry(render["energy"], aperture_mask)
            matrix[f"{fit_name}->{eval_name}"] = {
                "energy_map_correlation": _correlation(target["energy"], render["energy"], aperture_mask),
                "spectrum_correlation": _correlation(target["descriptor"], descriptor_sim, target["band"]),
                "center_distance_norm": float(
                    np.hypot(
                        geometry_sim["center_x_norm"] - target["geometry"]["center_x_norm"],
                        geometry_sim["center_y_norm"] - target["geometry"]["center_y_norm"],
                    )
                ),
                "in_sample": fit_name == eval_name,
            }

    return {
        "frames": list(frames),
        "ghost_summary": str(ghost_summary_path.resolve()),
        "matrix": matrix,
    }


def _plot(
    output_path: Path,
    holdout: Mapping[str, Any],
    ghost: Mapping[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)

    loo_records = holdout["_loo_records"]
    labels = [str(record["file"]).replace(".bmp", "") for record in loo_records]
    values = [float(record["heldout_profile_rmse"]) for record in loo_records]
    in_sample_rmse = float(holdout["in_sample_all_24"]["profile_rmse"])
    axes[0].bar(range(len(labels)), values, color="#4878a8")
    axes[0].axhline(in_sample_rmse, color="black", linestyle="--", linewidth=1.2, label="in-sample (24-frame median)")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].set_xlabel("held-out frame")
    axes[0].set_ylabel("held-out radial profile RMSE")
    axes[0].set_title("Leave-one-out: global defocus/blur")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, axis="y")

    counts = holdout["selection_counts"]
    count_labels = list(counts.keys())
    axes[1].barh(range(len(count_labels)), [counts[k] for k in count_labels], color="#6aa870")
    axes[1].set_yticks(range(len(count_labels)))
    axes[1].set_yticklabels(count_labels, fontsize=8)
    axes[1].set_xlabel("times selected (splits + leave-one-out)")
    axes[1].set_title("Selection stability of (defocus, blur)")
    axes[1].grid(alpha=0.25, axis="x")

    pairs = list(ghost["matrix"].keys())
    energy_values = [float(ghost["matrix"][pair]["energy_map_correlation"]) for pair in pairs]
    spectrum_values = [float(ghost["matrix"][pair]["spectrum_correlation"]) for pair in pairs]
    x = np.arange(len(pairs))
    width = 0.36
    axes[2].bar(x - 0.5 * width, energy_values, width, label="energy-map corr", color="#b06060")
    axes[2].bar(x + 0.5 * width, spectrum_values, width, label="localized spectrum corr", color="#c9a04e")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([pair.replace(".bmp", "") for pair in pairs], fontsize=8)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Localized ghost: in-sample vs cross-frame transfer")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25, axis="y")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Held-out generalization checks for the reflection nuisance fits."
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
        "--ghost-summary",
        type=Path,
        default=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "reflection_localized_ghost_simulation"
        / "summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "external_data"
        / "processed"
        / "wechat_2026-07_15-34"
        / "reflection_generalization_check",
    )
    parser.add_argument(
        "--defocus-values",
        type=_csv_floats,
        default=[-200.0, -150.0, -100.0, -75.0, -50.0, 0.0, 50.0, 100.0, 200.0],
    )
    parser.add_argument("--raw-blur-values", type=_csv_floats, default=[0.8, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    parser.add_argument("--num-splits", type=int, default=12)
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    cfg = experiment.simulation
    if cfg.phase_mode != "reflection":
        raise ValueError("generalization check config must use simulation.phase_mode='reflection'")

    holdout = _holdout_defocus_blur(
        cfg,
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        defocus_values=list(args.defocus_values),
        raw_blur_values=list(args.raw_blur_values),
        num_splits=args.num_splits,
    )
    ghost = _ghost_transfer(
        cfg,
        empirical_npz=args.empirical_npz,
        ghost_summary_path=args.ghost_summary,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot(args.output_dir / "generalization_check.png", holdout, ghost)

    holdout_public = {key: value for key, value in holdout.items() if not key.startswith("_")}
    summary = {
        "scope": (
            "held-out validation of the raw reflection nuisance fits: random half splits and leave-one-out "
            "for the global defocus/blur selection, and cross-frame transfer for the localized ghost fits"
        ),
        "config": str(Path(args.config).resolve()),
        "global_defocus_blur": holdout_public,
        "localized_ghost_transfer": ghost,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    condensed = {
        "in_sample_profile_rmse": holdout_public["in_sample_all_24"]["profile_rmse"],
        "half_split_heldout_profile_rmse_mean": holdout_public["random_half_splits"]["heldout_profile_rmse"]["mean"],
        "half_split_selection_regret_rmse_mean": holdout_public["random_half_splits"]["selection_regret_rmse"]["mean"],
        "loo_heldout_profile_rmse_mean": holdout_public["leave_one_out"]["heldout_profile_rmse"]["mean"],
        "selection_counts": holdout_public["selection_counts"],
        "ghost_transfer": {
            pair: {
                "energy_map_correlation": metrics["energy_map_correlation"],
                "spectrum_correlation": metrics["spectrum_correlation"],
            }
            for pair, metrics in ghost["matrix"].items()
        },
    }
    print(json.dumps(condensed, indent=2))
    print(args.output_dir / "generalization_check.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
