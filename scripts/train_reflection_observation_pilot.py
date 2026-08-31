from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable

# Required by CuBLAS before the first CUDA context is created when PyTorch
# deterministic algorithms are enabled.  Keep this in the executable rather
# than relying on an interactive shell setting so recorded runs reproduce.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_reflection_dark_port import _load_real_crops_dn  # noqa: E402
from evaluate_reflection_sim2real import _simulate_ensemble  # noqa: E402
from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402
from mini_grin_rebuild.models.observation_residual import ObservationResidualVAE  # noqa: E402


LOG_MAX = float(np.log1p(255.0))


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


def _resize_stack(stack: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.as_tensor(stack, dtype=torch.float32).unsqueeze(1)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="area")
    return resized.squeeze(1).cpu().numpy().astype(np.float32)


def _resize_map(image: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.as_tensor(image, dtype=torch.float32)[None, None]
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized[0, 0].cpu().numpy().astype(np.float32)


def _log_normalize(images_dn: np.ndarray) -> np.ndarray:
    return (np.log1p(np.clip(images_dn, 0.0, 255.0)) / LOG_MAX).astype(np.float32)


def _log_to_dn(images_log: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.clip(images_log, 0.0, 1.0) * LOG_MAX), 0.0, 255.0).astype(np.float32)


def _indices(names: list[str], selected: Iterable[str]) -> list[int]:
    lookup = {name: index for index, name in enumerate(names)}
    return [lookup[str(name)] for name in selected]


def _split_names(manifest: dict[str, Any], available: list[str]) -> tuple[list[str], list[str]]:
    train = [str(name) for block in manifest["calibration_blocks"] for name in block]
    holdout = [str(name) for name in manifest["temporal_test"]]
    combined = train + holdout
    if len(combined) != len(set(combined)):
        raise ValueError("split manifest contains duplicate frames")
    if set(combined) != set(available):
        raise ValueError(
            f"split mismatch: missing={sorted(set(available) - set(combined))}, "
            f"extra={sorted(set(combined) - set(available))}"
        )
    return train, holdout


def _weight_map(rho: np.ndarray) -> torch.Tensor:
    weights = np.ones_like(rho, dtype=np.float32)
    weights[(rho >= 0.84) & (rho <= 1.12)] = 2.0
    weights[rho > 1.12] = 0.55
    weights /= max(float(np.mean(weights)), 1e-9)
    return torch.from_numpy(weights)[None, None]


def _gradient_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    true_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    true_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    return torch.mean(torch.abs(pred_y - true_y)) + torch.mean(torch.abs(pred_x - true_x))


def _vae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    *,
    weights: torch.Tensor,
    beta_kl: float,
    gradient_weight: float,
    lowpass_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon = torch.mean(torch.abs(prediction - target) * weights)
    gradient = _gradient_l1(prediction, target)
    lowpass_prediction = F.avg_pool2d(prediction, kernel_size=9, stride=1, padding=4)
    lowpass_target = F.avg_pool2d(target, kernel_size=9, stride=1, padding=4)
    lowpass = torch.mean(torch.abs(lowpass_prediction - lowpass_target))
    kl = -0.5 * torch.mean(1.0 + logvar - mean.square() - logvar.exp())
    total = recon + gradient_weight * gradient + lowpass_weight * lowpass + beta_kl * kl
    return total, {
        "total": float(total.detach()),
        "reconstruction": float(recon.detach()),
        "gradient": float(gradient.detach()),
        "lowpass": float(lowpass.detach()),
        "kl": float(kl.detach()),
    }


@torch.no_grad()
def _reconstruct(model: ObservationResidualVAE, residuals: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, residuals.shape[0], batch_size):
        batch = residuals[start : start + batch_size]
        mean, _ = model.encode(batch)
        output.append(model.decode(mean).cpu().numpy()[:, 0])
    return np.concatenate(output, axis=0).astype(np.float32)


def _fit_pca(train_residual: np.ndarray, latent_dim: int) -> dict[str, np.ndarray]:
    flat = train_residual.reshape(train_residual.shape[0], -1).astype(np.float64)
    mean = np.mean(flat, axis=0, keepdims=True)
    centered = flat - mean
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(int(latent_dim), centered.shape[0] - 1, vh.shape[0])
    components = vh[:rank]
    scores = centered @ components.T
    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "scores": scores.astype(np.float32),
        "singular_values": singular[:rank].astype(np.float32),
    }


def _pca_reconstruct(residual: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    shape = residual.shape
    flat = residual.reshape(shape[0], -1)
    centered = flat - pca["mean"]
    scores = centered @ pca["components"].T
    reconstructed = pca["mean"] + scores @ pca["components"]
    return reconstructed.reshape(shape).astype(np.float32)


def _pca_sample(
    pca: dict[str, np.ndarray],
    *,
    count: int,
    image_shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    scores = np.asarray(pca["scores"], dtype=np.float64)
    if scores.shape[1] == 1:
        sampled = rng.normal(float(np.mean(scores)), max(float(np.std(scores)), 1e-6), size=(count, 1))
    else:
        covariance = np.cov(scores, rowvar=False) + 1e-6 * np.eye(scores.shape[1])
        sampled = rng.multivariate_normal(np.mean(scores, axis=0), covariance, size=count)
    flat = pca["mean"] + sampled @ pca["components"]
    return flat.reshape(count, *image_shape).astype(np.float32)


@torch.no_grad()
def _sample_empirical_latent_prior(
    model: ObservationResidualVAE,
    train_residuals: torch.Tensor,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Decode a Gaussian fitted only to training-set posterior means.

    This is an exploratory prior-mismatch diagnostic, not a replacement for
    the predeclared standard-normal VAE prior.  Posterior variances are
    intentionally excluded: including them would confound the distribution
    of observation-specific codes with encoder uncertainty under a weak KL
    penalty.
    """

    model.eval()
    latent_mean, latent_logvar = model.encode(train_residuals)
    codes = latent_mean.detach().cpu().numpy().astype(np.float64)
    logvar = latent_logvar.detach().cpu().numpy().astype(np.float64)
    center = np.mean(codes, axis=0)
    if codes.shape[0] <= 1:
        covariance = np.eye(codes.shape[1], dtype=np.float64) * 1e-6
    else:
        covariance = np.atleast_2d(np.cov(codes, rowvar=False, ddof=1))
        scale = max(float(np.trace(covariance) / codes.shape[1]), 1e-6)
        covariance = covariance + np.eye(codes.shape[1], dtype=np.float64) * (1e-6 * scale)
    sampled_codes = rng.multivariate_normal(center, covariance, size=count)
    sampled_tensor = torch.as_tensor(
        sampled_codes,
        dtype=train_residuals.dtype,
        device=train_residuals.device,
    )
    decoded = model.decode(sampled_tensor).cpu().numpy()[:, 0].astype(np.float32)
    statistics = {
        "posterior_means": codes.astype(np.float32),
        "posterior_logvars": logvar.astype(np.float32),
        "empirical_mean": center.astype(np.float32),
        "empirical_covariance": covariance.astype(np.float32),
        "sampled_codes": sampled_codes.astype(np.float32),
    }
    return decoded, statistics


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if mask is not None:
        aa = aa[mask]
        bb = bb[mask]
    aa = aa.ravel() - float(np.mean(aa))
    bb = bb.ravel() - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0


def _blur_stack(images: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    tensor = torch.as_tensor(images, dtype=torch.float32)[:, None]
    return F.avg_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)[:, 0].numpy()


def _angular_edge_profile(image: np.ndarray, rho: np.ndarray, bins: int = 48) -> np.ndarray:
    height, width = image.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    theta = np.mod(np.arctan2(yy - 0.5 * (height - 1), xx - 0.5 * (width - 1)), 2.0 * np.pi)
    output = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        lo = 2.0 * np.pi * index / bins
        hi = 2.0 * np.pi * (index + 1) / bins
        edge = (rho >= 0.92) & (rho <= 1.06) & (theta >= lo) & (theta < hi)
        inner = (rho >= 0.80) & (rho <= 0.89) & (theta >= lo) & (theta < hi)
        edge_value = float(np.quantile(image[edge], 0.90)) if np.any(edge) else 0.0
        inner_value = float(np.median(image[inner])) if np.any(inner) else 0.0
        output[index] = max(edge_value - inner_value, 0.0)
    return output


def _radial_profile(image: np.ndarray, rho: np.ndarray, bins: int = 28) -> np.ndarray:
    edges = np.linspace(0.0, 1.4, bins + 1)
    values = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (rho >= lo) & (rho < hi)
        values.append(float(np.median(image[mask])) if np.any(mask) else 0.0)
    return np.asarray(values, dtype=np.float32)


def _spectral_features(image: np.ndarray, rho: np.ndarray, bins: int = 8) -> np.ndarray:
    mask = (rho <= 0.78).astype(np.float32)
    values = image * mask
    values = values - float(np.sum(values) / max(float(np.sum(mask)), 1.0)) * mask
    power = np.abs(np.fft.fft2(values)) ** 2
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    frequency = np.sqrt(fx**2 + fy**2)
    edges = np.geomspace(1.0 / max(image.shape), 0.5, bins + 1)
    result = []
    total = max(float(np.sum(power[(frequency > 0) & (frequency <= 0.5)])), 1e-12)
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (frequency >= lo) & (frequency < hi)
        result.append(float(np.sum(power[band]) / total))
    return np.asarray(result, dtype=np.float32)


def _feature_vector(image: np.ndarray, rho: np.ndarray) -> np.ndarray:
    regions = ((rho <= 0.78), ((rho >= 0.84) & (rho <= 1.12)), (rho > 1.12))
    region_stats: list[float] = []
    for mask in regions:
        values = image[mask]
        region_stats.extend([float(np.mean(values)), float(np.std(values))])
    return np.concatenate(
        [
            np.asarray(region_stats, dtype=np.float32),
            _radial_profile(image, rho),
            _angular_edge_profile(image, rho),
            _spectral_features(image, rho),
        ]
    )


def _feature_matrix(images: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return np.stack([_feature_vector(image, rho) for image in images], axis=0)


def sliced_wasserstein_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int,
    projections: int = 256,
) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("feature matrices must be [N,D] with matching D")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, a.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    quantiles = np.linspace(0.0, 1.0, max(a.shape[0], b.shape[0], 32))
    distances = []
    for direction in directions:
        pa = a @ direction
        pb = b @ direction
        qa = np.quantile(pa, quantiles)
        qb = np.quantile(pb, quantiles)
        distances.append(float(np.mean(np.abs(qa - qb))))
    return float(np.mean(distances))


def _mean_pairwise_rmse(first: np.ndarray, second: np.ndarray | None = None) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = a if second is None else np.asarray(second, dtype=np.float32)
    values = []
    for i, image in enumerate(a):
        start = i + 1 if second is None else 0
        for j in range(start, b.shape[0]):
            values.append(float(np.sqrt(np.mean((image - b[j]) ** 2))))
    return float(np.mean(values)) if values else 0.0


def _nearest_train_rmse(generated: np.ndarray, train: np.ndarray) -> float:
    values = []
    for image in generated:
        distances = np.sqrt(np.mean((train - image[None]) ** 2, axis=(1, 2)))
        values.append(float(np.min(distances)))
    return float(np.mean(values))


def _reconstruction_metrics(target_log: np.ndarray, predicted_log: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    target = np.asarray(target_log, dtype=np.float32)
    predicted = np.asarray(predicted_log, dtype=np.float32)
    if target.shape != predicted.shape:
        raise ValueError(f"target/prediction mismatch: {target.shape} != {predicted.shape}")
    low_target = _blur_stack(target)
    low_predicted = _blur_stack(predicted)
    high_target = target - low_target
    high_predicted = predicted - low_predicted
    mask = rho <= 1.20
    edge_errors = []
    for real_image, predicted_image in zip(target, predicted):
        edge_errors.append(
            float(
                np.sqrt(
                    np.mean(
                        (
                            _angular_edge_profile(real_image, rho)
                            - _angular_edge_profile(predicted_image, rho)
                        )
                        ** 2
                    )
                )
            )
        )
    target_dn = _log_to_dn(target)
    predicted_dn = _log_to_dn(predicted)
    return {
        "log_rmse": float(np.sqrt(np.mean((target - predicted) ** 2))),
        "log_mae": float(np.mean(np.abs(target - predicted))),
        "dn_rmse": float(np.sqrt(np.mean((target_dn - predicted_dn) ** 2))),
        "lowpass_corr": float(np.mean([_corr(a, b, mask) for a, b in zip(low_target, low_predicted)])),
        "highpass_corr": float(np.mean([_corr(a, b, mask) for a, b in zip(high_target, high_predicted)])),
        "edge_profile_rmse": float(np.mean(edge_errors)),
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(
    path: Path,
    *,
    holdout_real_log: np.ndarray,
    candidate_log: np.ndarray,
    mean_log: np.ndarray,
    pca_reconstruction_log: np.ndarray,
    vae_reconstruction_log: np.ndarray,
    pca_sample_log: np.ndarray,
    vae_sample_log: np.ndarray,
    vae_empirical_sample_log: np.ndarray,
) -> None:
    panels = [
        ("real holdout median", np.median(holdout_real_log, axis=0)),
        ("physics candidate", candidate_log),
        ("train-mean baseline", mean_log),
        ("PCA holdout recon (oracle)", np.median(pca_reconstruction_log, axis=0)),
        ("VAE holdout recon (oracle)", np.median(vae_reconstruction_log, axis=0)),
        ("PCA prior sample", pca_sample_log),
        ("VAE N(0,I) prior sample", vae_sample_log),
        ("VAE empirical-latent sample", vae_empirical_sample_log),
    ]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 6.4), constrained_layout=True)
    for column, (title, image_log) in enumerate(panels):
        axes[0, column].imshow(_log_to_dn(image_log), cmap="gray", vmin=0.0, vmax=255.0)
        axes[0, column].set_title(title, fontsize=9)
        axes[0, column].axis("off")
        axes[1, column].imshow(image_log, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, column].set_title(f"log1p: {title}", fontsize=9)
        axes[1, column].axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_history(path: Path, history: list[dict[str, float]]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_total"] for row in history], label="train")
    axes[0].plot(epochs, [row["holdout_total"] for row in history], label="holdout")
    axes[0].set_title("VAE objective (holdout monitored, not selected)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_reconstruction"] for row in history], label="train")
    axes[1].plot(epochs, [row["holdout_reconstruction"] for row in history], label="holdout")
    axes[1].set_title("Weighted reconstruction L1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Appearance-only neural observation-model feasibility pilot.")
    parser.add_argument("--simulation-config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", choices=("I_x", "I_y"), default="I_x")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta-kl", type=float, default=2e-4)
    parser.add_argument("--gradient-weight", type=float, default=0.35)
    parser.add_argument("--lowpass-weight", type=float, default=0.25)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--generated-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)

    if args.epochs < 1 or args.batch_size < 1 or args.generated_samples < 4:
        raise ValueError("epochs/batch-size must be positive and generated-samples must be >= 4")
    _set_seed(args.seed, args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this Python environment has no usable CUDA PyTorch build; "
            "the pilot refuses to fall back to CPU silently"
        )
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
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_names, holdout_names = _split_names(split, names)
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
    rho = _resize_map(real_rho, args.image_size)
    real_log = _log_normalize(real_stack_dn)
    candidate_log = _log_normalize(candidate_dn)
    residual = real_log - candidate_log[None]
    train_real = real_log[train_index]
    holdout_real = real_log[holdout_index]
    train_residual = residual[train_index]
    holdout_residual = residual[holdout_index]

    pca = _fit_pca(train_residual, args.latent_dim)
    pca_train_residual = _pca_reconstruct(train_residual, pca)
    pca_holdout_residual = _pca_reconstruct(holdout_residual, pca)
    mean_residual = np.mean(train_residual, axis=0)

    model = ObservationResidualVAE(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_tensor = torch.from_numpy(train_residual)[:, None].to(device)
    holdout_tensor = torch.from_numpy(holdout_residual)[:, None].to(device)
    weights = _weight_map(rho).to(device)
    order_generator = torch.Generator().manual_seed(args.seed + 17)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train_tensor.shape[0], generator=order_generator)
        train_accumulator = {key: 0.0 for key in ("total", "reconstruction", "gradient", "lowpass", "kl")}
        batches = 0
        for start in range(0, order.numel(), args.batch_size):
            batch = train_tensor[order[start : start + args.batch_size]]
            output = model(batch)
            loss, parts = _vae_loss(
                output.residual,
                batch,
                output.mean,
                output.logvar,
                weights=weights,
                beta_kl=args.beta_kl,
                gradient_weight=args.gradient_weight,
                lowpass_weight=args.lowpass_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in parts.items():
                train_accumulator[key] += value
            batches += 1

        model.eval()
        with torch.no_grad():
            holdout_output = model(holdout_tensor)
            _, holdout_parts = _vae_loss(
                holdout_output.residual,
                holdout_tensor,
                holdout_output.mean,
                holdout_output.logvar,
                weights=weights,
                beta_kl=args.beta_kl,
                gradient_weight=args.gradient_weight,
                lowpass_weight=args.lowpass_weight,
            )
        row: dict[str, float] = {"epoch": float(epoch)}
        for key, value in train_accumulator.items():
            row[f"train_{key}"] = value / max(batches, 1)
        for key, value in holdout_parts.items():
            row[f"holdout_{key}"] = value
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train={row['train_total']:.6f} "
                f"holdout={row['holdout_total']:.6f} kl={row['train_kl']:.6f}",
                flush=True,
            )

    vae_train_residual = _reconstruct(model, train_tensor, args.batch_size)
    vae_holdout_residual = _reconstruct(model, holdout_tensor, args.batch_size)
    prior_generator = torch.Generator(device=device).manual_seed(args.seed + 29)
    with torch.no_grad():
        vae_prior_residual = model.sample(
            args.generated_samples,
            generator=prior_generator,
        ).cpu().numpy()[:, 0].astype(np.float32)
    rng = np.random.default_rng(args.seed + 31)
    pca_prior_residual = _pca_sample(
        pca,
        count=args.generated_samples,
        image_shape=(args.image_size, args.image_size),
        rng=rng,
    )
    empirical_prior_residual, empirical_latent = _sample_empirical_latent_prior(
        model,
        train_tensor,
        count=args.generated_samples,
        rng=np.random.default_rng(args.seed + 37),
    )

    predictions = {
        "physics": {
            "train": np.repeat(candidate_log[None], len(train_index), axis=0),
            "holdout": np.repeat(candidate_log[None], len(holdout_index), axis=0),
        },
        "train_mean": {
            "train": np.repeat((candidate_log + mean_residual)[None], len(train_index), axis=0),
            "holdout": np.repeat((candidate_log + mean_residual)[None], len(holdout_index), axis=0),
        },
        "pca_oracle": {
            "train": candidate_log[None] + pca_train_residual,
            "holdout": candidate_log[None] + pca_holdout_residual,
        },
        "vae_oracle": {
            "train": candidate_log[None] + vae_train_residual,
            "holdout": candidate_log[None] + vae_holdout_residual,
        },
    }
    for model_predictions in predictions.values():
        for split_name in model_predictions:
            model_predictions[split_name] = np.clip(model_predictions[split_name], 0.0, 1.0)

    metric_rows: list[dict[str, Any]] = []
    for model_name, model_predictions in predictions.items():
        for split_name, target in (("train", train_real), ("holdout", holdout_real)):
            row = {
                "model": model_name,
                "split": split_name,
                **_reconstruction_metrics(target, model_predictions[split_name], rho),
            }
            metric_rows.append(row)

    train_features = _feature_matrix(train_real, rho)
    feature_mean = np.mean(train_features, axis=0, keepdims=True)
    feature_std = np.maximum(np.std(train_features, axis=0, keepdims=True), 1e-4)
    holdout_features = (_feature_matrix(holdout_real, rho) - feature_mean) / feature_std
    generation_sets = {
        "physics": np.repeat(candidate_log[None], args.generated_samples, axis=0),
        "train_mean": np.repeat((candidate_log + mean_residual)[None], args.generated_samples, axis=0),
        "pca_prior": np.clip(candidate_log[None] + pca_prior_residual, 0.0, 1.0),
        "vae_prior": np.clip(candidate_log[None] + vae_prior_residual, 0.0, 1.0),
        "vae_empirical_prior": np.clip(
            candidate_log[None] + empirical_prior_residual,
            0.0,
            1.0,
        ),
    }
    generation_metrics: dict[str, dict[str, float]] = {}
    holdout_diversity = _mean_pairwise_rmse(holdout_real)
    holdout_nearest_train = _nearest_train_rmse(holdout_real, train_real)
    for model_name, generated in generation_sets.items():
        generated_features = (_feature_matrix(generated, rho) - feature_mean) / feature_std
        generation_metrics[model_name] = {
            "feature_swd_to_holdout": sliced_wasserstein_distance(
                generated_features,
                holdout_features,
                seed=args.seed + 43,
            ),
            "mean_log_rmse_to_holdout_mean": float(
                np.sqrt(np.mean((np.mean(generated, axis=0) - np.mean(holdout_real, axis=0)) ** 2))
            ),
            "diversity_ratio_to_holdout": float(
                _mean_pairwise_rmse(generated) / max(holdout_diversity, 1e-9)
            ),
            "nearest_train_log_rmse": _nearest_train_rmse(generated, train_real),
        }

    reconstruction_by_key = {(row["model"], row["split"]): row for row in metric_rows}
    pca_holdout_rmse = float(reconstruction_by_key[("pca_oracle", "holdout")]["log_rmse"])
    vae_holdout_rmse = float(reconstruction_by_key[("vae_oracle", "holdout")]["log_rmse"])
    pca_swd = generation_metrics["pca_prior"]["feature_swd_to_holdout"]
    vae_swd = generation_metrics["vae_prior"]["feature_swd_to_holdout"]
    empirical_vae_swd = generation_metrics["vae_empirical_prior"]["feature_swd_to_holdout"]
    vae_memory_ratio = generation_metrics["vae_prior"]["nearest_train_log_rmse"] / max(
        holdout_nearest_train,
        1e-9,
    )
    train_gap = float(reconstruction_by_key[("vae_oracle", "holdout")]["log_rmse"]) / max(
        float(reconstruction_by_key[("vae_oracle", "train")]["log_rmse"]),
        1e-9,
    )
    criteria = {
        "vae_oracle_beats_pca_by_5pct": bool(vae_holdout_rmse <= 0.95 * pca_holdout_rmse),
        "vae_prior_swd_beats_pca_by_5pct": bool(vae_swd <= 0.95 * pca_swd),
        "vae_train_holdout_gap_at_most_1p5": bool(train_gap <= 1.5),
        "vae_not_near_copy_of_training_set": bool(vae_memory_ratio >= 0.5),
    }
    verdict = "pilot_pass" if all(criteria.values()) else "pilot_fail"
    exploratory_diagnostics = {
        "empirical_prior_swd_beats_standard_normal": bool(empirical_vae_swd < vae_swd),
        "empirical_prior_swd_beats_pca": bool(empirical_vae_swd < pca_swd),
        "empirical_prior_feature_swd": empirical_vae_swd,
        "standard_normal_prior_feature_swd": vae_swd,
        "pca_prior_feature_swd": pca_swd,
    }

    checkpoint = {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": {
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
        },
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "observation_residual_vae.pt")
    np.savez_compressed(args.output_dir / "pca_baseline.npz", **pca)
    np.savez_compressed(args.output_dir / "vae_empirical_latent.npz", **empirical_latent)
    _write_metrics_csv(args.output_dir / "reconstruction_metrics.csv", metric_rows)
    _write_metrics_csv(args.output_dir / "training_history.csv", history)
    _plot_history(args.output_dir / "training_history.png", history)
    _plot_comparison(
        args.output_dir / "comparison.png",
        holdout_real_log=holdout_real,
        candidate_log=candidate_log,
        mean_log=np.clip(candidate_log + mean_residual, 0.0, 1.0),
        pca_reconstruction_log=predictions["pca_oracle"]["holdout"],
        vae_reconstruction_log=predictions["vae_oracle"]["holdout"],
        pca_sample_log=generation_sets["pca_prior"][0],
        vae_sample_log=generation_sets["vae_prior"][0],
        vae_empirical_sample_log=generation_sets["vae_empirical_prior"][0],
    )

    duration = time.time() - started
    summary = {
        "scope": (
            "Appearance-only feasibility pilot. The 20/4 lens split is retrospective and the VAE oracle "
            "reconstruction encodes the target; only prior-sample metrics assess standalone generation."
        ),
        "verdict": verdict,
        "predeclared_criteria": criteria,
        "exploratory_diagnostics": exploratory_diagnostics,
        "diagnostics": {
            "pca_holdout_log_rmse": pca_holdout_rmse,
            "vae_holdout_log_rmse": vae_holdout_rmse,
            "pca_prior_feature_swd": pca_swd,
            "vae_prior_feature_swd": vae_swd,
            "vae_train_holdout_gap": train_gap,
            "holdout_nearest_train_log_rmse": holdout_nearest_train,
            "vae_nearest_train_ratio": vae_memory_ratio,
            "holdout_diversity": holdout_diversity,
        },
        "reconstruction_metrics": metric_rows,
        "generation_metrics": generation_metrics,
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
            "class": "ObservationResidualVAE",
            "parameters": model.parameter_count(),
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "image_size": args.image_size,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "beta_kl": args.beta_kl,
            "gradient_weight": args.gradient_weight,
            "lowpass_weight": args.lowpass_weight,
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
                "model": _sha256(ROOT / "src/mini_grin_rebuild/models/observation_residual.py"),
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
                "exploratory_diagnostics": exploratory_diagnostics,
            },
            indent=2,
        )
    )
    print(args.output_dir / "summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
