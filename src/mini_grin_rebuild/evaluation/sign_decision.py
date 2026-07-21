from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from mini_grin_rebuild.core.configs import ExperimentConfig
from mini_grin_rebuild.data.generate_dataset import random_triplet
from mini_grin_rebuild.data.virtual_objects import microlens_standard
from mini_grin_rebuild.evaluation.metrics import defect_mask
from mini_grin_rebuild.physics.factory import create_forward_model
from mini_grin_rebuild.physics.phase import phase_scale
from mini_grin_rebuild.simulation.factory import create_simulation_engine
from mini_grin_rebuild.simulation.transforms.utils import gaussian_blur
from mini_grin_rebuild.simulation.types import CaptureBundle


def aperture_mask(grid_size: int, *, radius_fraction: float = 1.0) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float32),
        np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float32),
        indexing="ij",
    )
    return (xx**2 + yy**2) <= (float(radius_fraction) ** 2 + 1e-12)


def defect_local_mask(defect: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    return (
        defect_mask(
            torch.from_numpy(np.asarray(defect, dtype=np.float32)).unsqueeze(0).unsqueeze(0),
            abs_threshold=float(cfg.training.eval_defect_abs_threshold),
            rel_threshold=float(cfg.training.eval_defect_rel_threshold),
            dilate_px=int(cfg.training.eval_defect_dilate_px),
        )
        .cpu()
        .numpy()
        .astype(bool)
    )


def _phase_scale(cfg: ExperimentConfig) -> float:
    return phase_scale(cfg.simulation)


def _spawn_rng(master: np.random.Generator) -> np.random.Generator:
    return np.random.default_rng(int(master.integers(0, 2**32 - 1)))


def _simulate_bundle(
    engine: Any,
    heights: Mapping[str, np.ndarray],
    *,
    rng: np.random.Generator,
) -> CaptureBundle:
    if hasattr(engine, "simulate_bundle"):
        return engine.simulate_bundle(heights, rng=rng)  # type: ignore[no-any-return, attr-defined]

    captures = {
        str(name): engine.simulate_capture(height, rng=_spawn_rng(rng), meta={"frame_name": str(name)})
        for name, height in heights.items()
    }
    return CaptureBundle(captures=captures, meta={"engine_name": getattr(engine, "name", "unknown")})


def _binary_sign(values: np.ndarray, fallback: np.ndarray | float = 1.0) -> np.ndarray:
    arr = np.sign(np.asarray(values, dtype=np.float64))
    fb = np.asarray(fallback, dtype=np.float64)
    return np.where(arr == 0.0, np.sign(np.where(fb == 0.0, 1.0, fb)), arr).astype(np.int8)


def _valid_gradient_mask(values: np.ndarray, *, rel_floor: float = 1e-4, abs_floor: float = 1e-8) -> np.ndarray:
    arr = np.abs(np.asarray(values, dtype=np.float64))
    peak = float(np.max(arr)) if arr.size else 0.0
    thr = max(float(abs_floor), float(rel_floor) * peak)
    return arr > thr


def _sampled_grid_mask(
    *,
    aperture: np.ndarray,
    stride: int,
    border: int,
) -> np.ndarray:
    mask = np.zeros_like(aperture, dtype=bool)
    h, w = aperture.shape
    y0 = max(int(border), 0)
    x0 = max(int(border), 0)
    y1 = max(y0, h - int(border))
    x1 = max(x0, w - int(border))
    if stride <= 0:
        stride = 1
    mask[y0:y1:stride, x0:x1:stride] = True
    return mask & aperture


def _gaussian_patch_basis(
    shape: tuple[int, int],
    *,
    center_y: int,
    center_x: int,
    sigma_px: float,
    axis: str,
    dx_phys: float,
    phase_scale: float,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    dy = yy - float(center_y)
    dx = xx - float(center_x)
    sigma2 = max(float(sigma_px) ** 2, 1e-6)
    gauss = np.exp(-0.5 * (dx * dx + dy * dy) / sigma2).astype(np.float32)
    if axis == "x":
        basis = -(dx / sigma2) * gauss
    elif axis == "y":
        basis = -(dy / sigma2) * gauss
    else:
        raise ValueError(f"Unknown axis={axis!r}")
    if axis == "x":
        left = max(center_x - 1, 0)
        right = min(center_x + 1, shape[1] - 1)
        phase_grad = phase_scale * float(basis[center_y, right] - basis[center_y, left]) / max(2.0 * float(dx_phys), 1e-12)
    else:
        up = max(center_y - 1, 0)
        down = min(center_y + 1, shape[0] - 1)
        phase_grad = phase_scale * float(basis[down, center_x] - basis[up, center_x]) / max(2.0 * float(dx_phys), 1e-12)
    scale = 1.0 / max(abs(phase_grad), 1e-6)
    return (basis * scale).astype(np.float32)


def extract_test_gradient_sign_map_first_order(
    *,
    physics: Any,
    standard_height: np.ndarray,
    diff_ix: np.ndarray,
    diff_iy: np.ndarray,
    grad_floor_ratio: float = 0.05,
    grad_floor_abs: float = 1e-6,
) -> dict[str, np.ndarray]:
    std = torch.from_numpy(np.asarray(standard_height, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    std_phase = physics._phase(std).squeeze(1)
    g_y_t, g_x_t = physics._gradient(std_phase)
    g_x = g_x_t.squeeze(0).cpu().numpy().astype(np.float64)
    g_y = g_y_t.squeeze(0).cpu().numpy().astype(np.float64)

    diff_x = np.asarray(diff_ix, dtype=np.float64)
    diff_y = np.asarray(diff_iy, dtype=np.float64)

    gx_abs = np.abs(g_x)
    gy_abs = np.abs(g_y)
    gx_floor = max(float(np.max(gx_abs)) * float(grad_floor_ratio), float(grad_floor_abs))
    gy_floor = max(float(np.max(gy_abs)) * float(grad_floor_ratio), float(grad_floor_abs))

    denom_x = 2.0 * np.where(gx_abs >= gx_floor, g_x, np.sign(g_x) * gx_floor)
    denom_y = 2.0 * np.where(gy_abs >= gy_floor, g_y, np.sign(g_y) * gy_floor)
    denom_x = np.where(denom_x == 0.0, 2.0 * gx_floor, denom_x)
    denom_y = np.where(denom_y == 0.0, 2.0 * gy_floor, denom_y)

    d_first_x = diff_x / denom_x
    d_first_y = diff_y / denom_y

    sign_x = _binary_sign(g_x + d_first_x, fallback=g_x)
    sign_y = _binary_sign(g_y + d_first_y, fallback=g_y)
    return {
        "x": sign_x,
        "y": sign_y,
        "standard_grad_x": g_x.astype(np.float32),
        "standard_grad_y": g_y.astype(np.float32),
    }


def extract_test_gradient_sign_map_raw_branch(
    *,
    cfg: ExperimentConfig,
    engine: Any,
    standard_height: np.ndarray,
    observed_raw_diff: np.ndarray,
    diff_ix: np.ndarray,
    diff_iy: np.ndarray,
    g_sx: np.ndarray,
    g_sy: np.ndarray,
    rng: np.random.Generator,
    sample_stride: int = 4,
    patch_radius: int = 3,
    basis_sigma_px: float = 1.5,
    candidate_blur_sigma_px: float = 0.6,
) -> dict[str, np.ndarray]:
    if sample_stride <= 0:
        raise ValueError("sample_stride must be >= 1")
    if patch_radius < 0:
        raise ValueError("patch_radius must be >= 0")

    standard = np.asarray(standard_height, dtype=np.float32)
    raw_obs = np.asarray(observed_raw_diff, dtype=np.float64)
    diff_x = np.asarray(diff_ix, dtype=np.float64)
    diff_y = np.asarray(diff_iy, dtype=np.float64)
    g_x = np.asarray(g_sx, dtype=np.float64)
    g_y = np.asarray(g_sy, dtype=np.float64)
    phase_scale = _phase_scale(cfg)
    dx_phys = float(cfg.simulation.dx)

    mag_x = np.sqrt(np.clip(g_x**2 + diff_x, a_min=0.0, a_max=None))
    mag_y = np.sqrt(np.clip(g_y**2 + diff_y, a_min=0.0, a_max=None))

    aperture = aperture_mask(
        int(cfg.simulation.grid_size),
        radius_fraction=float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0),
    )
    sample_mask = _sampled_grid_mask(aperture=aperture, stride=int(sample_stride), border=int(patch_radius) + 1)
    valid_x = sample_mask & _valid_gradient_mask(mag_x)
    valid_y = sample_mask & _valid_gradient_mask(mag_y)

    sign_x = np.zeros_like(g_x, dtype=np.int8)
    sign_y = np.zeros_like(g_y, dtype=np.int8)

    h, w = standard.shape

    def choose_axis(axis: str, valid_mask: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        out = np.zeros((h, w), dtype=np.int8)
        ys, xs = np.nonzero(valid_mask)
        for cy, cx in zip(ys.tolist(), xs.tolist()):
            amp = float(magnitudes[cy, cx])
            if amp <= 0.0:
                out[cy, cx] = 1
                continue

            basis = _gaussian_patch_basis(
                (h, w),
                center_y=cy,
                center_x=cx,
                sigma_px=float(basis_sigma_px),
                axis=axis,
                dx_phys=dx_phys,
                phase_scale=phase_scale,
            )
            std_grad = float(g_x[cy, cx] if axis == "x" else g_y[cy, cx])
            plus = standard + float(amp - std_grad) * basis
            minus = standard + float(-amp - std_grad) * basis
            bundle = _simulate_bundle(
                engine,
                {"standard": standard, "plus": plus, "minus": minus},
                rng=_spawn_rng(rng),
            )
            std_raw = np.asarray(bundle.captures["standard"].channels["I_raw"], dtype=np.float64)
            plus_raw = np.asarray(bundle.captures["plus"].channels["I_raw"], dtype=np.float64) - std_raw
            minus_raw = np.asarray(bundle.captures["minus"].channels["I_raw"], dtype=np.float64) - std_raw

            if candidate_blur_sigma_px > 0.0:
                plus_raw = gaussian_blur(plus_raw.astype(np.float32), float(candidate_blur_sigma_px)).astype(np.float64)
                minus_raw = gaussian_blur(minus_raw.astype(np.float32), float(candidate_blur_sigma_px)).astype(np.float64)

            y0 = cy - int(patch_radius)
            y1 = cy + int(patch_radius) + 1
            x0 = cx - int(patch_radius)
            x1 = cx + int(patch_radius) + 1
            obs_patch = raw_obs[y0:y1, x0:x1]
            plus_patch = plus_raw[y0:y1, x0:x1]
            minus_patch = minus_raw[y0:y1, x0:x1]

            err_plus = float(np.sum((obs_patch - plus_patch) ** 2))
            err_minus = float(np.sum((obs_patch - minus_patch) ** 2))
            out[cy, cx] = 1 if err_plus <= err_minus else -1
        return out

    sign_x = choose_axis("x", valid_x, mag_x)
    sign_y = choose_axis("y", valid_y, mag_y)
    return {
        "x": sign_x,
        "y": sign_y,
        "sample_mask_x": valid_x,
        "sample_mask_y": valid_y,
        "mag_x": mag_x.astype(np.float32),
        "mag_y": mag_y.astype(np.float32),
    }


@dataclass(frozen=True)
class SignAccuracy:
    count: int
    correct: int
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "correct": int(self.correct),
            "accuracy": float(self.accuracy),
        }


@dataclass(frozen=True)
class AxisMethodMetrics:
    global_metrics: SignAccuracy
    defect_local_metrics: SignAccuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "global": self.global_metrics.to_dict(),
            "defect_local": self.defect_local_metrics.to_dict(),
        }


@dataclass(frozen=True)
class SampleSignComparison:
    index: int
    defect_peak_abs: float
    open_method: dict[str, AxisMethodMetrics]
    physical_leakage: dict[str, AxisMethodMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "defect_peak_abs": float(self.defect_peak_abs),
            "open_method": {axis: metrics.to_dict() for axis, metrics in self.open_method.items()},
            "physical_leakage": {axis: metrics.to_dict() for axis, metrics in self.physical_leakage.items()},
        }


def _evaluate_sign_accuracy(
    *,
    pred_sign: np.ndarray,
    true_sign: np.ndarray,
    eval_mask: np.ndarray,
) -> SignAccuracy:
    mask = np.asarray(eval_mask, dtype=bool)
    count = int(np.sum(mask))
    if count <= 0:
        return SignAccuracy(count=0, correct=0, accuracy=float("nan"))
    pred = np.asarray(pred_sign, dtype=np.int8)[mask]
    truth = np.asarray(true_sign, dtype=np.int8)[mask]
    correct = int(np.sum(pred == truth))
    return SignAccuracy(count=count, correct=correct, accuracy=float(correct / count))


def compare_sign_methods_on_sample(
    *,
    cfg: ExperimentConfig,
    physics: Any,
    engine: Any,
    standard: np.ndarray,
    defect: np.ndarray,
    rng: np.random.Generator,
    index: int = 0,
    sample_stride: int = 4,
    patch_radius: int = 3,
    basis_sigma_px: float = 1.5,
    candidate_blur_sigma_px: float = 0.6,
) -> SampleSignComparison:
    standard = np.asarray(standard, dtype=np.float32)
    defect = np.asarray(defect, dtype=np.float32)
    test = standard + defect

    bundle = _simulate_bundle(engine, {"standard": standard, "test": test}, rng=rng)
    std = bundle.captures["standard"].channels
    tst = bundle.captures["test"].channels

    diff_ix = np.asarray(tst["I_x"] - std["I_x"], dtype=np.float32)
    diff_iy = np.asarray(tst["I_y"] - std["I_y"], dtype=np.float32)
    raw_diff = np.asarray(tst["I_raw"] - std["I_raw"], dtype=np.float32)

    open_maps = extract_test_gradient_sign_map_first_order(
        physics=physics,
        standard_height=standard,
        diff_ix=diff_ix,
        diff_iy=diff_iy,
    )
    raw_maps = extract_test_gradient_sign_map_raw_branch(
        cfg=cfg,
        engine=engine,
        standard_height=standard,
        observed_raw_diff=raw_diff,
        diff_ix=diff_ix,
        diff_iy=diff_iy,
        g_sx=open_maps["standard_grad_x"],
        g_sy=open_maps["standard_grad_y"],
        rng=_spawn_rng(rng),
        sample_stride=int(sample_stride),
        patch_radius=int(patch_radius),
        basis_sigma_px=float(basis_sigma_px),
        candidate_blur_sigma_px=float(candidate_blur_sigma_px),
    )

    phase_scale = _phase_scale(cfg)
    test_phase = np.asarray(test, dtype=np.float64) * phase_scale
    t_y, t_x = np.gradient(test_phase, float(cfg.simulation.dx))
    truth_x = _binary_sign(t_x)
    truth_y = _binary_sign(t_y)

    aperture = aperture_mask(
        int(cfg.simulation.grid_size),
        radius_fraction=float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0),
    )
    local = defect_local_mask(defect, cfg)
    valid_truth_x = _valid_gradient_mask(t_x)
    valid_truth_y = _valid_gradient_mask(t_y)
    global_base_x = aperture & valid_truth_x & np.asarray(raw_maps["sample_mask_x"], dtype=bool)
    global_base_y = aperture & valid_truth_y & np.asarray(raw_maps["sample_mask_y"], dtype=bool)
    local_base_x = local & valid_truth_x & np.asarray(raw_maps["sample_mask_x"], dtype=bool)
    local_base_y = local & valid_truth_y & np.asarray(raw_maps["sample_mask_y"], dtype=bool)

    open_method = {
        "x": AxisMethodMetrics(
            global_metrics=_evaluate_sign_accuracy(pred_sign=open_maps["x"], true_sign=truth_x, eval_mask=global_base_x),
            defect_local_metrics=_evaluate_sign_accuracy(pred_sign=open_maps["x"], true_sign=truth_x, eval_mask=local_base_x),
        ),
        "y": AxisMethodMetrics(
            global_metrics=_evaluate_sign_accuracy(pred_sign=open_maps["y"], true_sign=truth_y, eval_mask=global_base_y),
            defect_local_metrics=_evaluate_sign_accuracy(pred_sign=open_maps["y"], true_sign=truth_y, eval_mask=local_base_y),
        ),
    }
    physical_leakage = {
        "x": AxisMethodMetrics(
            global_metrics=_evaluate_sign_accuracy(pred_sign=raw_maps["x"], true_sign=truth_x, eval_mask=global_base_x),
            defect_local_metrics=_evaluate_sign_accuracy(pred_sign=raw_maps["x"], true_sign=truth_x, eval_mask=local_base_x),
        ),
        "y": AxisMethodMetrics(
            global_metrics=_evaluate_sign_accuracy(pred_sign=raw_maps["y"], true_sign=truth_y, eval_mask=global_base_y),
            defect_local_metrics=_evaluate_sign_accuracy(pred_sign=raw_maps["y"], true_sign=truth_y, eval_mask=local_base_y),
        ),
    }

    return SampleSignComparison(
        index=int(index),
        defect_peak_abs=float(np.max(np.abs(defect))),
        open_method=open_method,
        physical_leakage=physical_leakage,
    )


def run_sign_method_comparison(
    cfg: ExperimentConfig,
    *,
    samples: int,
    seed: int,
    sample_stride: int = 4,
    patch_radius: int = 3,
    basis_sigma_px: float = 1.5,
    candidate_blur_sigma_px: float = 0.6,
) -> list[SampleSignComparison]:
    physics = create_forward_model(cfg.simulation, cfg.training, device="cpu", freeze=True)
    engine = create_simulation_engine(cfg.simulation)
    rng = np.random.default_rng(int(seed))

    if str(getattr(cfg.simulation, "scene", "legacy")) in {"microlens_srt", "microlens_spherical_cap"}:
        expected_standard = microlens_standard(cfg.simulation).astype(np.float32)
    else:
        expected_standard = None

    records: list[SampleSignComparison] = []
    for idx in range(int(samples)):
        triplet = random_triplet(cfg.simulation, rng)
        standard = np.asarray(triplet["standard"].height_map, dtype=np.float32)
        if expected_standard is not None and not np.allclose(standard, expected_standard):
            raise AssertionError("microlens standard should be deterministic across random samples")
        records.append(
            compare_sign_methods_on_sample(
                cfg=cfg,
                physics=physics,
                engine=engine,
                standard=standard,
                defect=np.asarray(triplet["defect"].height_map, dtype=np.float32),
                rng=_spawn_rng(rng),
                index=idx,
                sample_stride=int(sample_stride),
                patch_radius=int(patch_radius),
                basis_sigma_px=float(basis_sigma_px),
                candidate_blur_sigma_px=float(candidate_blur_sigma_px),
            )
        )
    return records


def summarize_sign_method_comparison(records: list[SampleSignComparison]) -> dict[str, Any]:
    def summarize_acc(parts: list[SignAccuracy]) -> dict[str, Any]:
        total = int(sum(int(p.count) for p in parts))
        correct = int(sum(int(p.correct) for p in parts))
        accuracy = float(correct / total) if total > 0 else float("nan")
        return {"count": total, "correct": correct, "accuracy": accuracy}

    out: dict[str, Any] = {}
    for method_name in ("open_method", "physical_leakage"):
        method_payload: dict[str, Any] = {}
        for axis in ("x", "y"):
            metrics = [getattr(r, method_name)[axis] for r in records]
            method_payload[axis] = {
                "global": summarize_acc([m.global_metrics for m in metrics]),
                "defect_local": summarize_acc([m.defect_local_metrics for m in metrics]),
            }
        out[method_name] = method_payload
    return out


__all__ = [
    "AxisMethodMetrics",
    "SampleSignComparison",
    "SignAccuracy",
    "aperture_mask",
    "compare_sign_methods_on_sample",
    "defect_local_mask",
    "extract_test_gradient_sign_map_first_order",
    "extract_test_gradient_sign_map_raw_branch",
    "run_sign_method_comparison",
    "summarize_sign_method_comparison",
]
