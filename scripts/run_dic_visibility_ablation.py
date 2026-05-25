from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from mini_grin_rebuild.core.configs import SimulationConfig, load_experiment_config  # noqa: E402
from mini_grin_rebuild.core.json_io import write_json  # noqa: E402
from mini_grin_rebuild.data.generate_dataset import generate_dataset  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PROFILES: dict[str, dict[str, float | bool]] = {
    "near_identity": {
        "emit_raw": False,
        "defocus_strength": 0.0,
        "aperture_sigma_freq": 999.0,
        "aberration_strength": 0.0,
        "raw_blur_sigma_px": 0.0,
        "dic_blur_sigma_px": 0.0,
        "raw_gain": 1.0,
        "raw_bias": 0.0,
        "shear_px": 1.5,
    },
    "weak": {
        "emit_raw": False,
        "defocus_strength": 100.0,
        "aperture_sigma_freq": 0.24,
        "aberration_strength": 0.01,
        "raw_blur_sigma_px": 0.0,
        "dic_blur_sigma_px": 0.05,
        "raw_gain": 1.0,
        "raw_bias": 0.0,
        "shear_px": 1.5,
    },
    "mid": {
        "emit_raw": False,
        "defocus_strength": 300.0,
        "aperture_sigma_freq": 0.16,
        "aberration_strength": 0.025,
        "raw_blur_sigma_px": 0.0,
        "dic_blur_sigma_px": 0.1,
        "raw_gain": 1.0,
        "raw_bias": 0.0,
        "shear_px": 1.5,
    },
    "default": {
        "emit_raw": False,
        "defocus_strength": 600.0,
        "aperture_sigma_freq": 0.08,
        "aberration_strength": 0.05,
        "raw_blur_sigma_px": 0.3,
        "dic_blur_sigma_px": 0.2,
        "raw_gain": 1.0,
        "raw_bias": 0.0,
        "shear_px": 1.5,
    },
    "lowpass_only": {
        "emit_raw": False,
        "defocus_strength": 0.0,
        "aperture_sigma_freq": 0.08,
        "aberration_strength": 0.0,
        "raw_blur_sigma_px": 0.0,
        "dic_blur_sigma_px": 0.0,
        "raw_gain": 1.0,
        "raw_bias": 0.0,
        "shear_px": 1.5,
    },
}


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _defect_mask(defect: np.ndarray, *, abs_threshold: float = 1e-4, rel_threshold: float = 0.05) -> np.ndarray:
    peak = float(np.max(np.abs(defect)))
    threshold = max(float(abs_threshold), float(rel_threshold) * peak)
    return np.abs(defect) > threshold


def _di_mag(sample: dict[str, np.ndarray]) -> np.ndarray:
    return np.sqrt(sample["diff_ix_st"] ** 2 + sample["diff_iy_st"] ** 2)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def _sample_visibility_metrics(nonideal_path: Path, ideal_path: Path) -> dict[str, Any]:
    non = _load_npz(nonideal_path)
    ideal = _load_npz(ideal_path)
    defect = non["defect"]
    mask = _defect_mask(defect)
    bg = ~mask
    if int(np.sum(mask)) <= 0 or int(np.sum(bg)) <= 0:
        return {
            "sample": nonideal_path.stem,
            "defect_peak_abs": float(np.max(np.abs(defect))),
            "defect_support_px": int(np.sum(mask)),
            "visibility_z": float("nan"),
        }

    mag_non = _di_mag(non)
    mag_ideal = _di_mag(ideal)
    non_mask_mean = float(np.mean(mag_non[mask]))
    non_bg_mean = float(np.mean(mag_non[bg]))
    non_bg_std = float(np.std(mag_non[bg]))
    ideal_mask_mean = float(np.mean(mag_ideal[mask]))
    ideal_bg_mean = float(np.mean(mag_ideal[bg]))
    ideal_bg_std = float(np.std(mag_ideal[bg]))

    delta_ix = non["diff_ix_st"] - ideal["diff_ix_st"]
    delta_iy = non["diff_iy_st"] - ideal["diff_iy_st"]
    delta_mag = np.sqrt(delta_ix**2 + delta_iy**2)

    return {
        "sample": nonideal_path.stem,
        "defect_peak_abs": float(np.max(np.abs(defect))),
        "defect_support_px": int(np.sum(mask)),
        "visibility_z": float((non_mask_mean - non_bg_mean) / (non_bg_std + 1e-12)),
        "visibility_ratio": float(non_mask_mean / (non_bg_mean + 1e-12)),
        "mask_mean_di": non_mask_mean,
        "bg_mean_di": non_bg_mean,
        "bg_std_di": non_bg_std,
        "ideal_visibility_z": float((ideal_mask_mean - ideal_bg_mean) / (ideal_bg_std + 1e-12)),
        "ideal_visibility_ratio": float(ideal_mask_mean / (ideal_bg_mean + 1e-12)),
        "ideal_mask_mean_di": ideal_mask_mean,
        "ideal_bg_mean_di": ideal_bg_mean,
        "corr_ix_vs_ideal": _safe_corr(non["diff_ix_st"], ideal["diff_ix_st"]),
        "corr_iy_vs_ideal": _safe_corr(non["diff_iy_st"], ideal["diff_iy_st"]),
        "relative_delta_mean": float(np.mean(delta_mag) / max(float(np.mean(mag_ideal)), 1e-12)),
        "relative_delta_mask_mean": float(np.mean(delta_mag[mask]) / max(float(np.mean(mag_ideal[mask])), 1e-12)),
        "signal_retention_mask": float(non_mask_mean / max(ideal_mask_mean, 1e-12)),
    }


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _finite_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _finite_quantile(values: list[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "visibility_z",
        "visibility_ratio",
        "ideal_visibility_z",
        "ideal_visibility_ratio",
        "relative_delta_mean",
        "relative_delta_mask_mean",
        "signal_retention_mask",
        "corr_ix_vs_ideal",
        "corr_iy_vs_ideal",
        "mask_mean_di",
        "bg_mean_di",
    )
    out: dict[str, Any] = {"count": len(samples)}
    for key in keys:
        vals = [float(item.get(key, float("nan"))) for item in samples]
        out[f"{key}_mean"] = _finite_mean(vals)
        out[f"{key}_median"] = _finite_median(vals)
        out[f"{key}_p25"] = _finite_quantile(vals, 0.25)
        out[f"{key}_p75"] = _finite_quantile(vals, 0.75)
    return out


def _fov_um(base_cfg: SimulationConfig) -> float:
    return float(base_cfg.grid_size) * float(base_cfg.dx)


def _resolution_cfg(base_cfg: SimulationConfig, resolution: int) -> SimulationConfig:
    fov = _fov_um(base_cfg)
    return replace(base_cfg, grid_size=int(resolution), dx=float(fov / int(resolution)))


def _ideal_cfg(sim_cfg: SimulationConfig) -> SimulationConfig:
    return replace(sim_cfg, capture_engine="ideal_gradient", capture_engine_params={})


def _profile_cfg(sim_cfg: SimulationConfig, profile: str) -> SimulationConfig:
    params = dict(DEFAULT_PROFILES[profile])
    return replace(sim_cfg, capture_engine="optical_leakage_lite", capture_engine_params=params)


def _dataset_root(out_root: Path, resolution: int, profile: str) -> Path:
    return out_root / "data" / f"res{resolution}_{profile}"


def _generate_if_needed(
    cfg: SimulationConfig,
    *,
    root: Path,
    test_count: int,
    seed: int,
    force: bool,
) -> None:
    sample = root / "test" / "sample_0000.npz"
    if sample.is_file() and not force:
        return
    generate_dataset(
        cfg,
        output_root=root,
        train=0,
        val=0,
        test=int(test_count),
        seed=int(seed),
        config_snapshot={"simulation": cfg.__dict__},
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _plot_visibility_heatmap(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    resolutions = sorted({int(row["resolution"]) for row in summary_rows})
    profiles = [name for name in DEFAULT_PROFILES if any(row["profile"] == name for row in summary_rows)]
    matrix = np.full((len(profiles), len(resolutions)), np.nan, dtype=float)
    for row in summary_rows:
        r = profiles.index(str(row["profile"]))
        c = resolutions.index(int(row["resolution"]))
        matrix[r, c] = float(row["visibility_z_median"])

    fig, ax = plt.subplots(figsize=(1.8 * len(resolutions) + 3, 0.55 * len(profiles) + 2.5), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(resolutions)), labels=[str(r) for r in resolutions])
    ax.set_yticks(np.arange(len(profiles)), labels=profiles)
    ax.set_xlabel("grid resolution, fixed FOV")
    ax.set_title("Median nonideal defect visibility z-score")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < np.nanmax(matrix) * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="(mean defect dI - mean bg dI) / std bg dI")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _crop_from_mask(mask: np.ndarray, *, pad_y: int = 8, pad_x: int = 8) -> tuple[slice, slice]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(mask.shape[0], int(ys.max()) + pad_y + 1)
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(mask.shape[1], int(xs.max()) + pad_x + 1)
    return slice(y0, y1), slice(x0, x1)


def _plot_sample_grid(
    *,
    out_root: Path,
    resolutions: list[int],
    profiles: list[str],
    sample_index: int,
    output_path: Path,
) -> None:
    rows = len(resolutions)
    cols = len(profiles) + 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]

    for r_idx, resolution in enumerate(resolutions):
        ideal_path = _dataset_root(out_root, resolution, "ideal") / "test" / f"sample_{sample_index:04d}.npz"
        ideal = _load_npz(ideal_path)
        defect = ideal["defect"]
        mask = _defect_mask(defect)
        crop = _crop_from_mask(mask)
        defect_crop = defect[crop]
        ideal_mag_crop = _di_mag(ideal)[crop]
        vmax_def = max(float(np.max(np.abs(defect_crop))), 1e-12)
        vmax_mag = max(float(np.quantile(np.abs(ideal_mag_crop), 0.99)), 1e-12)

        ax = axes[r_idx, 0]
        ax.imshow(defect_crop, cmap="magma", vmin=0.0, vmax=vmax_def)
        ax.contour(mask[crop].astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title(f"res {resolution}\ntrue defect")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[r_idx, 1]
        ax.imshow(ideal_mag_crop, cmap="magma", vmin=0.0, vmax=vmax_mag)
        ax.contour(mask[crop].astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title("ideal |dI|")
        ax.set_xticks([])
        ax.set_yticks([])

        for p_idx, profile in enumerate(profiles, start=2):
            sample_path = _dataset_root(out_root, resolution, profile) / "test" / f"sample_{sample_index:04d}.npz"
            sample = _load_npz(sample_path)
            mag = _di_mag(sample)[crop]
            ax = axes[r_idx, p_idx]
            ax.imshow(mag, cmap="magma", vmin=0.0, vmax=vmax_mag)
            ax.contour(mask[crop].astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
            ax.set_title(f"{profile} |dI|")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Matched sample {sample_index:04d}: DIC visibility under resolution/profile ablation", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled visibility ablation for no-raw DIC physical leakage.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "nonideal_dic_no_raw_smoke64.json"))
    parser.add_argument("--out", default="/tmp/dic_visibility_ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--resolutions", default="64,128,256")
    parser.add_argument("--profiles", default="near_identity,weak,mid,default,lowpass_only")
    parser.add_argument("--sample-index", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    out_root = Path(args.out).expanduser().resolve()
    resolutions = _parse_ints(args.resolutions)
    profiles = [part.strip() for part in str(args.profiles).split(",") if part.strip()]
    unknown = [profile for profile in profiles if profile not in DEFAULT_PROFILES]
    if unknown:
        raise ValueError(f"Unknown profile(s): {unknown}. Available: {sorted(DEFAULT_PROFILES)}")

    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "config": str(Path(args.config).expanduser().resolve()),
        "out_root": str(out_root),
        "seed": int(args.seed),
        "test_count": int(args.test),
        "resolutions": resolutions,
        "profiles": profiles,
        "profile_params": {name: DEFAULT_PROFILES[name] for name in profiles},
        "conditions": [],
    }

    for resolution in resolutions:
        sim_cfg = _resolution_cfg(cfg.simulation, resolution)
        ideal_root = _dataset_root(out_root, resolution, "ideal")
        _generate_if_needed(
            _ideal_cfg(sim_cfg),
            root=ideal_root,
            test_count=int(args.test),
            seed=int(args.seed),
            force=bool(args.force),
        )
        for profile in profiles:
            profile_root = _dataset_root(out_root, resolution, profile)
            profile_cfg = _profile_cfg(sim_cfg, profile)
            _generate_if_needed(
                profile_cfg,
                root=profile_root,
                test_count=int(args.test),
                seed=int(args.seed),
                force=bool(args.force),
            )
            sample_metrics: list[dict[str, Any]] = []
            for idx in range(int(args.test)):
                item = _sample_visibility_metrics(
                    profile_root / "test" / f"sample_{idx:04d}.npz",
                    ideal_root / "test" / f"sample_{idx:04d}.npz",
                )
                item.update(
                    {
                        "resolution": int(resolution),
                        "dx": float(sim_cfg.dx),
                        "profile": profile,
                        "index": int(idx),
                    }
                )
                sample_metrics.append(item)
                sample_rows.append(item)
            summary = _summarize_samples(sample_metrics)
            summary.update({"resolution": int(resolution), "dx": float(sim_cfg.dx), "profile": profile})
            summary_rows.append(summary)
            result["conditions"].append(
                {
                    "resolution": int(resolution),
                    "dx": float(sim_cfg.dx),
                    "profile": profile,
                    "data_root": str(profile_root),
                    "ideal_data_root": str(ideal_root),
                    "summary": summary,
                    "sample_metrics": sample_metrics,
                }
            )

    sample_fields = [
        "resolution",
        "dx",
        "profile",
        "index",
        "sample",
        "defect_peak_abs",
        "defect_support_px",
        "visibility_z",
        "visibility_ratio",
        "ideal_visibility_z",
        "ideal_visibility_ratio",
        "signal_retention_mask",
        "relative_delta_mean",
        "relative_delta_mask_mean",
        "corr_ix_vs_ideal",
        "corr_iy_vs_ideal",
        "mask_mean_di",
        "bg_mean_di",
        "bg_std_di",
    ]
    summary_fields = [
        "resolution",
        "dx",
        "profile",
        "count",
        "visibility_z_mean",
        "visibility_z_median",
        "visibility_z_p25",
        "visibility_z_p75",
        "ideal_visibility_z_median",
        "visibility_ratio_median",
        "signal_retention_mask_median",
        "relative_delta_mean_median",
        "relative_delta_mask_mean_median",
        "corr_ix_vs_ideal_median",
        "corr_iy_vs_ideal_median",
    ]
    _write_csv(out_root / "sample_metrics.csv", sample_rows, sample_fields)
    _write_csv(out_root / "summary_metrics.csv", summary_rows, summary_fields)
    _plot_visibility_heatmap(summary_rows, out_root / "visibility_heatmap.png")
    _plot_sample_grid(
        out_root=out_root,
        resolutions=resolutions,
        profiles=profiles,
        sample_index=int(args.sample_index),
        output_path=out_root / f"sample_{int(args.sample_index):04d}_ablation_grid.png",
    )

    result["summary_rows"] = summary_rows
    result["artifacts"] = {
        "summary_csv": str(out_root / "summary_metrics.csv"),
        "sample_csv": str(out_root / "sample_metrics.csv"),
        "heatmap": str(out_root / "visibility_heatmap.png"),
        "sample_grid": str(out_root / f"sample_{int(args.sample_index):04d}_ablation_grid.png"),
    }
    summary_path = out_root / "dic_visibility_ablation_summary.json"
    write_json(summary_path, result)
    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
