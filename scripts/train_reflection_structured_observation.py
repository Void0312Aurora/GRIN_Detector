from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

# Required before the first CUDA context when deterministic linear algebra is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_dark_port import _load_real_crops_dn  # noqa: E402
from evaluate_reflection_sim2real import _simulate_ensemble  # noqa: E402
from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402
from mini_grin_rebuild.models.structured_observation import (  # noqa: E402
    decode_linear_factor,
    deterministic_median,
    extract_polar_harmonics,
    fit_edge_harmonics,
    fit_linear_factor,
    fit_texture_spectrum,
    project_linear_factor,
    reconstruct_edge_harmonics,
    reconstruct_polar_harmonics,
    sample_correlated_texture,
    sample_factor_scores,
)
from train_reflection_observation_pilot import (  # noqa: E402
    _angular_edge_profile,
    _feature_matrix,
    _fit_pca,
    _indices,
    _log_normalize,
    _log_to_dn,
    _mean_pairwise_rmse,
    _nearest_train_rmse,
    _pca_reconstruct,
    _pca_sample,
    _reconstruction_metrics,
    _resize_map,
    _resize_stack,
    _split_names,
    _write_metrics_csv,
    sliced_wasserstein_distance,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "<unavailable>"


def _set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(threads)))
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _parse_harmonics(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not values or any(value < 1 for value in values):
        raise ValueError("harmonics must contain positive comma-separated integers")
    return values


def _safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def _shift_corr(image: np.ndarray, rho: np.ndarray, lag: int, axis: int) -> float:
    mask = rho <= 0.78
    if axis == 0:
        first = image[:-lag, :]
        second = image[lag:, :]
        valid = mask[:-lag, :] & mask[lag:, :]
    else:
        first = image[:, :-lag]
        second = image[:, lag:]
        valid = mask[:, :-lag] & mask[:, lag:]
    return _safe_corr(first[valid], second[valid]) if np.any(valid) else 0.0


def _texture_feature_vector(image: np.ndarray, rho: np.ndarray) -> np.ndarray:
    mask = rho <= 0.78
    values = image[mask]
    gy, gx = np.gradient(image)
    gradient = np.sqrt(gx * gx + gy * gy)[mask]
    laplacian = (
        np.roll(image, 1, axis=0)
        + np.roll(image, -1, axis=0)
        + np.roll(image, 1, axis=1)
        + np.roll(image, -1, axis=1)
        - 4.0 * image
    )[mask]
    features = [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.quantile(values, 0.90) - np.quantile(values, 0.10)),
        float(np.std(gradient)),
        float(np.quantile(gradient, 0.90)),
        float(np.std(laplacian)),
    ]
    for lag in (1, 2, 4, 8, 16):
        features.extend((_shift_corr(image, rho, lag, 0), _shift_corr(image, rho, lag, 1)))
    return np.asarray(features, dtype=np.float32)


def _texture_feature_matrix(images: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return np.stack([_texture_feature_vector(image, rho) for image in images], axis=0)


def _edge_harmonic_summary(images: np.ndarray, rho: np.ndarray) -> np.ndarray:
    profiles = np.stack([_angular_edge_profile(image, rho, bins=48) for image in images], axis=0)
    centered = profiles - np.mean(profiles, axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) / max(centered.shape[1], 1)
    return spectrum[:, (1, 2, 4)].astype(np.float32)


def _specialized_distribution_metrics(
    generated: np.ndarray,
    holdout: np.ndarray,
    rho: np.ndarray,
) -> dict[str, float]:
    generated_texture = _texture_feature_matrix(generated, rho)
    holdout_texture = _texture_feature_matrix(holdout, rho)
    generated_edge = _edge_harmonic_summary(generated, rho)
    holdout_edge = _edge_harmonic_summary(holdout, rho)
    return {
        "texture_feature_rmse_to_holdout_mean": float(
            np.sqrt(np.mean((np.mean(generated_texture, axis=0) - np.mean(holdout_texture, axis=0)) ** 2))
        ),
        "texture_feature_spread_ratio": float(
            np.mean(np.std(generated_texture, axis=0))
            / max(float(np.mean(np.std(holdout_texture, axis=0))), 1e-9)
        ),
        "edge_k124_rmse_to_holdout_mean": float(
            np.sqrt(np.mean((np.mean(generated_edge, axis=0) - np.mean(holdout_edge, axis=0)) ** 2))
        ),
        "edge_k4_amplitude": float(np.mean(generated_edge[:, 2])),
        "holdout_edge_k4_amplitude": float(np.mean(holdout_edge[:, 2])),
    }


def _extended_features(images: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return np.concatenate((_feature_matrix(images, rho), _texture_feature_matrix(images, rho)), axis=1)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def _plot_comparison(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(
        2,
        len(panels),
        figsize=(3.0 * len(panels), 6.1),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (title, image) in enumerate(panels):
        axes[0, column].imshow(_log_to_dn(image), cmap="gray", vmin=0.0, vmax=255.0)
        axes[0, column].set_title(title, fontsize=8)
        axes[0, column].axis("off")
        axes[1, column].imshow(image, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, column].set_title(f"log1p: {title}", fontsize=8)
        axes[1, column].axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_components(path: Path, components: list[tuple[str, np.ndarray]]) -> None:
    columns = len(components)
    fig, axes = plt.subplots(1, columns, figsize=(3.6 * columns, 3.8), squeeze=False, constrained_layout=True)
    for index, (title, image) in enumerate(components):
        axes[0, index].imshow(image, cmap="coolwarm", vmin=-np.percentile(np.abs(image), 99), vmax=np.percentile(np.abs(image), 99))
        axes[0, index].set_title(title, fontsize=9)
        axes[0, index].axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geometry-aware static observation-model feasibility pilot.")
    parser.add_argument("--simulation-config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", choices=("I_x", "I_y"), default="I_x")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--harmonics", default="1,2,4")
    parser.add_argument("--radial-bins", type=int, default=128)
    parser.add_argument("--harmonic-rank", type=int, default=3)
    parser.add_argument("--texture-rank", type=int, default=2)
    parser.add_argument(
        "--explicit-edge-correction",
        action="store_true",
        help="fit a separate fixed-width edge-band harmonic correction before texture factors",
    )
    parser.add_argument("--edge-center", type=float, default=1.0)
    parser.add_argument("--edge-width", type=float, default=0.075)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--generated-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.image_size < 32 or args.image_size % 16 != 0:
        raise ValueError("image-size must be >=32 and divisible by 16")
    if args.generated_samples < 4 or args.ensemble_size < 1:
        raise ValueError("generated-samples must be >=4 and ensemble-size must be positive")
    harmonics = _parse_harmonics(args.harmonics)
    _set_seed(args.seed, args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    if device.type == "cuda":
        print(
            f"device={device} gpu={torch.cuda.get_device_name(device)} "
            f"torch={torch.__version__} cuda_build={torch.version.cuda}",
            flush=True,
        )
    else:
        print(f"device={device} torch={torch.__version__}", flush=True)
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiment = load_experiment_config(args.simulation_config)
    crop_radius_scale = 1.0 / max(float(experiment.simulation.lens_radius_fraction), 1e-12)
    real_stack_dn, real_rho, names = _load_real_crops_dn(
        raw_dir=args.raw_dir,
        detections_path=args.detections,
        grid_size=experiment.simulation.grid_size,
        crop_radius_scale=crop_radius_scale,
    )
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_names, holdout_names = _split_names(manifest, names)
    train_index = _indices(names, train_names)
    holdout_index = _indices(names, holdout_names)

    simulated, _, simulation_meta = _simulate_ensemble(
        args.simulation_config,
        channel=args.channel,
        seed=args.seed,
        ensemble_size=args.ensemble_size,
    )
    candidate_dn = np.median(simulated, axis=0).astype(np.float32)
    real_stack_dn = _resize_stack(real_stack_dn, args.image_size)
    candidate_dn = _resize_stack(candidate_dn[None], args.image_size)[0]
    rho_np = _resize_map(real_rho, args.image_size)
    real_log = _log_normalize(real_stack_dn)
    candidate_log = _log_normalize(candidate_dn)
    residual = real_log - candidate_log[None]
    train_real = real_log[train_index]
    holdout_real = real_log[holdout_index]
    train_residual = residual[train_index]
    holdout_residual = residual[holdout_index]

    rho = torch.from_numpy(rho_np).to(device=device, dtype=torch.float32)
    train_tensor = torch.from_numpy(train_residual).to(device=device, dtype=torch.float32)
    holdout_tensor = torch.from_numpy(holdout_residual).to(device=device, dtype=torch.float32)

    edge_state = None
    edge_state_holdout = None
    if args.explicit_edge_correction:
        print("fit explicit edge harmonics", flush=True)
        edge_state, edge_correction_train = fit_edge_harmonics(
            train_tensor,
            rho,
            harmonics=harmonics,
            center=args.edge_center,
            width=args.edge_width,
        )
        edge_state_holdout, edge_correction_holdout = fit_edge_harmonics(
            holdout_tensor,
            rho,
            harmonics=harmonics,
            center=args.edge_center,
            width=args.edge_width,
        )
        fit_train_tensor = train_tensor - edge_correction_train
        fit_holdout_tensor = holdout_tensor - edge_correction_holdout
        edge_common = reconstruct_edge_harmonics(
            edge_state,
            torch.mean(edge_state.coefficients, dim=0, keepdim=True),
        )
    else:
        edge_correction_train = torch.zeros_like(train_tensor)
        edge_correction_holdout = torch.zeros_like(holdout_tensor)
        fit_train_tensor = train_tensor
        fit_holdout_tensor = holdout_tensor
        edge_common = torch.zeros((1, *rho.shape), device=device, dtype=train_tensor.dtype)

    print("fit polar harmonics", flush=True)
    train_harmonic_coeff, train_harmonic_full = extract_polar_harmonics(
        fit_train_tensor,
        rho,
        harmonics=harmonics,
        radial_bins=args.radial_bins,
        rho_max=float(np.max(rho_np)),
    )
    holdout_harmonic_coeff, _ = extract_polar_harmonics(
        fit_holdout_tensor,
        rho,
        harmonics=harmonics,
        radial_bins=args.radial_bins,
        rho_max=float(np.max(rho_np)),
    )
    harmonic_factor = fit_linear_factor(train_harmonic_coeff, args.harmonic_rank)
    train_harmonic_scores, train_harmonic_projected = project_linear_factor(
        train_harmonic_coeff,
        harmonic_factor,
    )
    _, holdout_harmonic_projected = project_linear_factor(
        holdout_harmonic_coeff,
        harmonic_factor,
    )
    harmonic_common_coeff = harmonic_factor.mean.reshape(1, *train_harmonic_coeff.shape[1:])
    harmonic_common = reconstruct_polar_harmonics(
        harmonic_common_coeff,
        rho,
        harmonics=harmonics,
        rho_max=float(np.max(rho_np)),
    )
    train_harmonic_projected_image = reconstruct_polar_harmonics(
        train_harmonic_projected.reshape(train_harmonic_projected.shape[0], *train_harmonic_coeff.shape[1:]),
        rho,
        harmonics=harmonics,
        rho_max=float(np.max(rho_np)),
    )
    holdout_harmonic_projected_image = reconstruct_polar_harmonics(
        holdout_harmonic_projected.reshape(holdout_harmonic_projected.shape[0], *holdout_harmonic_coeff.shape[1:]),
        rho,
        harmonics=harmonics,
        rho_max=float(np.max(rho_np)),
    )

    # Keep the non-harmonic common texture deterministic; only the remaining
    # lens-to-lens texture is given a stochastic factor/spectrum model.
    common_texture = deterministic_median(fit_train_tensor - train_harmonic_full, dim=0)
    train_texture_residual = fit_train_tensor - train_harmonic_full - common_texture[None]
    holdout_texture_residual = fit_holdout_tensor - holdout_harmonic_projected_image - common_texture[None]
    texture_factor = fit_linear_factor(train_texture_residual, args.texture_rank)
    train_texture_scores, train_texture_projected = project_linear_factor(
        train_texture_residual,
        texture_factor,
    )
    _, holdout_texture_projected = project_linear_factor(
        holdout_texture_residual,
        texture_factor,
    )
    grf_state = fit_texture_spectrum(
        train_texture_residual - train_texture_projected,
        rho,
    )

    structured_train_oracle = (
        edge_correction_train
        + train_harmonic_projected_image
        + common_texture[None]
        + train_texture_projected
    )
    structured_holdout_oracle = (
        edge_correction_holdout
        + holdout_harmonic_projected_image
        + common_texture[None]
        + holdout_texture_projected
    )
    structured_common = edge_common + harmonic_common + common_texture[None]

    pca = _fit_pca(train_residual, max(args.harmonic_rank + args.texture_rank, 1))
    pca_train_residual = _pca_reconstruct(train_residual, pca)
    pca_holdout_residual = _pca_reconstruct(holdout_residual, pca)
    pca_generator = np.random.default_rng(args.seed + 11)
    pca_prior_residual = _pca_sample(
        pca,
        count=args.generated_samples,
        image_shape=(args.image_size, args.image_size),
        rng=pca_generator,
    )

    harmonic_generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    texture_generator = torch.Generator(device=device).manual_seed(args.seed + 19)
    grf_generator = torch.Generator(device=device).manual_seed(args.seed + 23)
    edge_generator = torch.Generator(device=device).manual_seed(args.seed + 13)
    sampled_harmonic_scores = sample_factor_scores(
        train_harmonic_scores,
        args.generated_samples,
        generator=harmonic_generator,
    )
    sampled_texture_scores = sample_factor_scores(
        train_texture_scores,
        args.generated_samples,
        generator=texture_generator,
    )
    sampled_harmonic_coeff = decode_linear_factor(
        sampled_harmonic_scores,
        harmonic_factor,
        tuple(train_harmonic_coeff.shape[1:]),
    )
    sampled_harmonic_image = reconstruct_polar_harmonics(
        sampled_harmonic_coeff,
        rho,
        harmonics=harmonics,
        rho_max=float(np.max(rho_np)),
    )
    sampled_texture = decode_linear_factor(
        sampled_texture_scores,
        texture_factor,
        (args.image_size, args.image_size),
    )
    sampled_grf = sample_correlated_texture(
        grf_state,
        args.generated_samples,
        generator=grf_generator,
    )
    if edge_state is not None:
        sampled_edge_coefficients = sample_factor_scores(
            edge_state.coefficients,
            args.generated_samples,
            generator=edge_generator,
        )
        sampled_edge = reconstruct_edge_harmonics(edge_state, sampled_edge_coefficients)
    else:
        sampled_edge_coefficients = torch.zeros(
            (args.generated_samples, 1 + 2 * len(harmonics)),
            device=device,
            dtype=train_tensor.dtype,
        )
        sampled_edge = torch.zeros_like(sampled_harmonic_image)

    predictions = {
        "physics": {
            "train": np.repeat(candidate_log[None], len(train_index), axis=0),
            "holdout": np.repeat(candidate_log[None], len(holdout_index), axis=0),
        },
        "train_median": {
            "train": np.repeat(np.median(train_real, axis=0)[None], len(train_index), axis=0),
            "holdout": np.repeat(np.median(train_real, axis=0)[None], len(holdout_index), axis=0),
        },
        "pca_oracle": {
            "train": candidate_log[None] + pca_train_residual,
            "holdout": candidate_log[None] + pca_holdout_residual,
        },
        "structured_common": {
            "train": np.repeat(
                (candidate_log + _to_numpy(structured_common[0]))[None],
                len(train_index),
                axis=0,
            ),
            "holdout": np.repeat(
                (candidate_log + _to_numpy(structured_common[0]))[None],
                len(holdout_index),
                axis=0,
            ),
        },
        "structured_oracle": {
            "train": candidate_log[None] + _to_numpy(structured_train_oracle),
            "holdout": candidate_log[None] + _to_numpy(structured_holdout_oracle),
        },
    }
    for model_predictions in predictions.values():
        for split_name in model_predictions:
            model_predictions[split_name] = np.clip(model_predictions[split_name], 0.0, 1.0)

    metric_rows: list[dict[str, Any]] = []
    for model_name, model_predictions in predictions.items():
        for split_name, target in (("train", train_real), ("holdout", holdout_real)):
            metric_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    **_reconstruction_metrics(target, model_predictions[split_name], rho_np),
                }
            )

    generated = {
        "physics": np.repeat(candidate_log[None], args.generated_samples, axis=0),
        "train_median": np.repeat(np.median(train_real, axis=0)[None], args.generated_samples, axis=0),
        "pca_prior": np.clip(candidate_log[None] + pca_prior_residual, 0.0, 1.0),
        "structured_no_grf": np.clip(
            candidate_log[None]
            + _to_numpy(
                sampled_edge + sampled_harmonic_image + common_texture[None] + sampled_texture
            ),
            0.0,
            1.0,
        ),
        "structured_grf": np.clip(
            candidate_log[None]
            + _to_numpy(
                sampled_edge
                + sampled_harmonic_image
                + common_texture[None]
                + sampled_texture
                + sampled_grf
            ),
            0.0,
            1.0,
        ),
    }
    train_extended = _extended_features(train_real, rho_np)
    holdout_extended = _extended_features(holdout_real, rho_np)
    feature_mean = np.mean(train_extended, axis=0, keepdims=True)
    feature_std = np.maximum(np.std(train_extended, axis=0, keepdims=True), 1e-4)
    holdout_standardized = (holdout_extended - feature_mean) / feature_std
    holdout_diversity = _mean_pairwise_rmse(holdout_real)
    holdout_nearest_train = _nearest_train_rmse(holdout_real, train_real)
    generation_rows: list[dict[str, Any]] = []
    for model_name, images in generated.items():
        standardized = (_extended_features(images, rho_np) - feature_mean) / feature_std
        row = {
            "model": model_name,
            "extended_feature_swd_to_holdout": sliced_wasserstein_distance(
                standardized,
                holdout_standardized,
                seed=args.seed + 29,
            ),
            "mean_log_rmse_to_holdout_mean": float(
                np.sqrt(np.mean((np.mean(images, axis=0) - np.mean(holdout_real, axis=0)) ** 2))
            ),
            "diversity_ratio_to_holdout": float(_mean_pairwise_rmse(images) / max(holdout_diversity, 1e-9)),
            "nearest_train_log_rmse": _nearest_train_rmse(images, train_real),
            **_specialized_distribution_metrics(images, holdout_real, rho_np),
        }
        generation_rows.append(row)

    by_key = {(row["model"], row["split"]): row for row in metric_rows}
    pca_holdout_rmse = float(by_key[("pca_oracle", "holdout")]["log_rmse"])
    structured_holdout_rmse = float(by_key[("structured_oracle", "holdout")]["log_rmse"])
    pca_swd = next(row for row in generation_rows if row["model"] == "pca_prior")["extended_feature_swd_to_holdout"]
    structured_swd = next(row for row in generation_rows if row["model"] == "structured_grf")["extended_feature_swd_to_holdout"]
    structured_memory_ratio = next(row for row in generation_rows if row["model"] == "structured_grf")["nearest_train_log_rmse"] / max(
        holdout_nearest_train,
        1e-9,
    )
    criteria = {
        "structured_oracle_within_5pct_of_pca": bool(structured_holdout_rmse <= 1.05 * pca_holdout_rmse),
        "structured_grf_extended_swd_beats_pca": bool(structured_swd <= pca_swd),
        "structured_grf_not_near_copy": bool(structured_memory_ratio >= 0.5),
        "structured_grf_texture_error_beats_physics": bool(
            next(row for row in generation_rows if row["model"] == "structured_grf")["texture_feature_rmse_to_holdout_mean"]
            < next(row for row in generation_rows if row["model"] == "physics")["texture_feature_rmse_to_holdout_mean"]
        ),
    }
    verdict = "structured_pilot_pass" if all(criteria.values()) else "structured_pilot_fail"

    np.savez_compressed(
        args.output_dir / "structured_factors.npz",
        harmonic_mean=_to_numpy(harmonic_factor.mean),
        harmonic_components=_to_numpy(harmonic_factor.components),
        harmonic_scores=_to_numpy(harmonic_factor.scores),
        texture_mean=_to_numpy(texture_factor.mean),
        texture_components=_to_numpy(texture_factor.components),
        texture_scores=_to_numpy(texture_factor.scores),
        common_texture=_to_numpy(common_texture),
        edge_coefficients=(
            _to_numpy(edge_state.coefficients)
            if edge_state is not None
            else np.zeros((0, 1 + 2 * len(harmonics)), dtype=np.float32)
        ),
        edge_sampled_coefficients=_to_numpy(sampled_edge_coefficients),
        rho=rho_np,
        harmonics=np.asarray(harmonics, dtype=np.int64),
    )
    np.savez_compressed(
        args.output_dir / "texture_spectrum.npz",
        spectra=_to_numpy(grf_state.spectra),
        target_std=_to_numpy(grf_state.target_std),
        zone_masks=_to_numpy(grf_state.zone_masks),
    )
    np.savez_compressed(args.output_dir / "pca_baseline.npz", **pca)
    _write_metrics_csv(args.output_dir / "reconstruction_metrics.csv", metric_rows)
    _write_metrics_csv(args.output_dir / "generation_metrics.csv", generation_rows)
    panels = [
        ("real holdout median", np.median(holdout_real, axis=0)),
        ("physics candidate", candidate_log),
        ("train median", np.median(train_real, axis=0)),
        ("structured common", np.clip(candidate_log + _to_numpy(structured_common[0]), 0.0, 1.0)),
        ("PCA holdout oracle", np.median(predictions["pca_oracle"]["holdout"], axis=0)),
        ("structured holdout oracle", np.median(predictions["structured_oracle"]["holdout"], axis=0)),
        ("PCA prior sample", generated["pca_prior"][0]),
        ("structured no-GRF sample", generated["structured_no_grf"][0]),
        ("structured GRF sample", generated["structured_grf"][0]),
    ]
    _plot_comparison(args.output_dir / "comparison.png", panels)
    component_images = [
        ("common texture", _to_numpy(common_texture)),
        ("harmonic common", _to_numpy(harmonic_common[0])),
        ("edge correction", _to_numpy(edge_common[0])),
        ("texture factor 1", _to_numpy(texture_factor.components[0].reshape(args.image_size, args.image_size))),
        ("GRF target std by zone", np.broadcast_to(_to_numpy(grf_state.target_std)[:, None], (3, 3))),
    ]
    _plot_components(args.output_dir / "components.png", component_images)

    duration = time.time() - started
    summary = {
        "scope": (
            "Static geometry-aware observation pilot: robust shared texture, circular harmonics k=1/2/4, "
            "low-dimensional lens factors, and zone-conditioned correlated residual fields."
            + (" An explicit edge-band harmonic correction is enabled." if args.explicit_edge_correction else "")
        ),
        "verdict": verdict,
        "predeclared_criteria": criteria,
        "diagnostics": {
            "pca_holdout_log_rmse": pca_holdout_rmse,
            "structured_holdout_log_rmse": structured_holdout_rmse,
            "structured_over_pca_rmse_ratio": structured_holdout_rmse / max(pca_holdout_rmse, 1e-9),
            "pca_prior_extended_swd": pca_swd,
            "structured_grf_extended_swd": structured_swd,
            "holdout_nearest_train_log_rmse": holdout_nearest_train,
            "structured_grf_nearest_train_ratio": structured_memory_ratio,
            "holdout_diversity": holdout_diversity,
        },
        "reconstruction_metrics": metric_rows,
        "generation_metrics": generation_rows,
        "data": {
            "raw_frame_count": len(names),
            "train_frames": train_names,
            "holdout_frames": holdout_names,
            "raw_dir": str(args.raw_dir.resolve()),
            "detections": str(args.detections.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "simulation_config": str(args.simulation_config.resolve()),
            "channel": args.channel,
        },
        "model": {
            "class": "PolarHarmonicFactorGRF",
            "harmonics": list(harmonics),
            "explicit_edge_correction": bool(args.explicit_edge_correction),
            "edge_center": args.edge_center,
            "edge_width": args.edge_width,
            "radial_bins": args.radial_bins,
            "harmonic_rank": harmonic_factor.rank,
            "texture_rank": texture_factor.rank,
            "learned_parameter_count": int(
                harmonic_factor.mean.numel()
                + harmonic_factor.components.numel()
                + texture_factor.mean.numel()
                + texture_factor.components.numel()
                + grf_state.spectra.numel()
            ),
        },
        "training": {
            "seed": args.seed,
            "threads": args.threads,
            "duration_seconds": duration,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "simulation_meta": simulation_meta,
        "provenance": {
            "git_commit": _git_text("rev-parse", "HEAD"),
            "git_status": _git_text("status", "--short"),
            "files": {
                "script": _sha256(Path(__file__)),
                "model": _sha256(ROOT / "src/mini_grin_rebuild/models/structured_observation.py"),
                "simulation_config": _sha256(args.simulation_config),
                "detections": _sha256(args.detections),
                "split_manifest": _sha256(args.split_manifest),
            },
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "criteria": criteria,
                "diagnostics": summary["diagnostics"],
            },
            indent=2,
        )
    )
    print(args.output_dir / "summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
