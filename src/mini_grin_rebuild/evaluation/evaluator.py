from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional
from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import DataLoader

from mini_grin_rebuild.core.configs import ExperimentConfig
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.data.datasets import DefectDataset
from mini_grin_rebuild.models.checkpoint import infer_checkpoint_info, load_checkpoint
from mini_grin_rebuild.models.unetpp import UNetPP
from mini_grin_rebuild.physics.factory import create_forward_model, freeze_forward_model
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer
from mini_grin_rebuild.reconstruction import (
    reconstruct_defect_coarse_prior,
    reconstruct_defect_first_order_poisson,
    reconstruct_defect_first_order_sign_quadratic_poisson,
    reconstruct_defect_oracle_poisson,
    reconstruct_defect_pseudo_poisson,
)
from mini_grin_rebuild.training.inputs import append_coord_channels, build_inputs
from mini_grin_rebuild.visualization.plots import plot_defect_and_intensity
from mini_grin_rebuild.evaluation.gating import GateThresholds, gate_decision, radial_masks
from mini_grin_rebuild.evaluation.metrics import (
    binary_auprc,
    binary_auroc,
    binary_f1,
    binary_iou,
    binary_precision,
    binary_recall,
    correlation,
    defect_mask,
    masked_abs_quantile,
    masked_correlation,
    masked_mean_abs,
    masked_mean_abs_ratio,
    masked_peak_ratio,
    masked_psnr,
    masked_rmse,
    masked_slope,
    masked_volume_rel_error,
    peak_rel_error,
    psnr,
    rmse,
    slope,
    ssim,
    volume_rel_error,
)


def _safe_nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _safe_quantile(values: list[float], q: float) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.quantile(np.asarray(vals, dtype=float), float(q)))


def _masked_rmse_hw(values_hw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    if values_hw.ndim != 2:
        raise ValueError(f"Expected values [H,W], got {tuple(values_hw.shape)}")
    m = mask_hw.to(dtype=torch.bool)
    if torch.sum(m) == 0:
        return torch.tensor(float("nan"), device=values_hw.device, dtype=torch.float32)
    return torch.sqrt(torch.mean(values_hw[m] ** 2))


def _select_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _freeze_physics_to_ideal(physics: DifferentiableGradientLayer) -> None:
    freeze_forward_model(physics)


def _training_cfg_from_model_meta(cfg: ExperimentConfig, model_meta: dict[str, Any]):
    if "forward_model" not in model_meta and "forward_model_params" not in model_meta:
        return cfg.training
    return replace(
        cfg.training,
        forward_model=str(model_meta.get("forward_model", cfg.training.forward_model)),
        forward_model_params=dict(model_meta.get("forward_model_params", cfg.training.forward_model_params) or {}),
    )


def _forward_model(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = model(inputs)
    if isinstance(out, dict):
        return out["defect"], out.get("logvar")
    return out, None


def _prepare_inputs(
    batch: Dict[str, torch.Tensor],
    physics: DifferentiableGradientLayer,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    train_cfg = cfg.training
    diff_ts = {"I_x": batch["inputs"][:, 0], "I_y": batch["inputs"][:, 1]}
    diff_sr = None
    if "inputs_sr" in batch:
        diff_sr = {"I_x": batch["inputs_sr"][:, 0], "I_y": batch["inputs_sr"][:, 1]}

    intensity_inputs = {}
    for name in ("standard", "reference", "test"):
        key = f"intensity_{name}"
        if key in batch:
            intensity_inputs[name] = {"I_x": batch[key][:, 0], "I_y": batch[key][:, 1]}
        raw_key = f"raw_{name}"
        if raw_key in batch:
            intensity_inputs.setdefault(name, {})
            intensity_inputs[name]["I_raw"] = batch[raw_key][:, 0]

    standard = batch["standard"]
    inputs = build_inputs(
        train_cfg,
        diff_ts,
        physics,
        standard,
        diff_sr=diff_sr,
        intensity_inputs=intensity_inputs if intensity_inputs else None,
    )
    inputs = inputs * float(train_cfg.input_scale)
    if bool(getattr(train_cfg, "use_coord_inputs", False)):
        ap = float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0)
        inputs = append_coord_channels(inputs, aperture_radius=ap)
    return inputs


def _wrap_height(cfg) -> float:
    import math

    phase_scale = (2.0 * math.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air)
    return float((math.pi * cfg.wrap_safety) / max(phase_scale, 1e-12))


def _radial_grid(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lin_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    lin_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing="ij")
    return torch.sqrt(xx**2 + yy**2)


def _region_masks(cfg: ExperimentConfig, *, h: int, w: int, device: torch.device) -> dict[str, torch.Tensor]:
    dtype = torch.float32
    rr = _radial_grid(h, w, device=device, dtype=dtype)
    aperture_r = float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0)
    if aperture_r <= 0:
        aperture_r = 1.0
    edge_start_frac = float(getattr(cfg.training, "eval_edge_band_start_frac", 0.9))
    edge_start_frac = max(0.0, min(edge_start_frac, 1.0))
    center_frac = float(getattr(cfg.training, "eval_center_radius_frac", 0.1))
    center_frac = max(0.0, min(center_frac, 1.0))

    aperture = rr <= aperture_r
    edge = (rr > (edge_start_frac * aperture_r)) & aperture
    center = rr <= (center_frac * aperture_r)
    return {"aperture": aperture, "edge": edge, "center": center}


def _init_region_metrics() -> dict[str, dict[str, list[float]]]:
    keys = (
        "rmse",
        "psnr",
        "auprc",
        "auroc",
        "iou",
        "f1",
        "precision",
        "recall",
        "support_frac",
    )
    return {name: {k: [] for k in keys} for name in ("aperture", "edge", "center")}


def _init_sign_metrics() -> dict[str, dict[str, dict[str, int]]]:
    return {
        axis: {region: {"correct": 0, "count": 0} for region in ("global", "defect_local", "branch_flip")}
        for axis in ("x", "y")
    }


def _binary_sign_t(values: torch.Tensor, fallback: torch.Tensor | float = 1.0) -> torch.Tensor:
    sign = torch.sign(values)
    if not torch.is_tensor(fallback):
        fallback = torch.full_like(values, float(fallback))
    fb = torch.sign(torch.where(fallback == 0, torch.ones_like(fallback), fallback))
    return torch.where(sign == 0, fb, sign)


def _phase_gradients_hw(cfg: ExperimentConfig, height_chw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height = height_chw.squeeze(0)
    phase_scale = (2.0 * np.pi / float(cfg.simulation.wavelength)) * (
        float(cfg.simulation.n_object) - float(cfg.simulation.n_air)
    )
    phase = height * float(phase_scale)
    grad_y, grad_x = torch.gradient(
        phase,
        spacing=(float(cfg.simulation.dx), float(cfg.simulation.dx)),
        dim=(-2, -1),
        edge_order=1,
    )
    return grad_y, grad_x


def _valid_gradient_mask_t(values: torch.Tensor, *, rel_floor: float = 1e-4, abs_floor: float = 1e-8) -> torch.Tensor:
    peak = torch.max(torch.abs(values))
    threshold = max(float(abs_floor), float(rel_floor) * float(peak.detach().cpu()))
    return torch.abs(values) > threshold


def _update_sign_metrics(
    sign_metrics: dict[str, dict[str, dict[str, int]]],
    *,
    cfg: ExperimentConfig,
    standard: torch.Tensor,
    defect_pred: torch.Tensor,
    defect_true: torch.Tensor,
    defect_region: torch.Tensor,
    aperture_region: torch.Tensor,
) -> None:
    std_gy, std_gx = _phase_gradients_hw(cfg, standard)
    true_gy, true_gx = _phase_gradients_hw(cfg, standard + defect_true)
    pred_gy, pred_gx = _phase_gradients_hw(cfg, standard + defect_pred)

    for axis, std_grad, true_grad, pred_grad in (
        ("x", std_gx, true_gx, pred_gx),
        ("y", std_gy, true_gy, pred_gy),
    ):
        true_sign = _binary_sign_t(true_grad)
        pred_sign = _binary_sign_t(pred_grad)
        std_sign = _binary_sign_t(std_grad)
        valid = _valid_gradient_mask_t(true_grad)
        masks = {
            "global": aperture_region.to(dtype=torch.bool) & valid,
            "defect_local": aperture_region.to(dtype=torch.bool) & defect_region.to(dtype=torch.bool) & valid,
            "branch_flip": aperture_region.to(dtype=torch.bool) & valid & (std_sign != true_sign),
        }
        for region, region_mask in masks.items():
            count = int(torch.sum(region_mask).detach().cpu().item())
            if count <= 0:
                continue
            correct = int(torch.sum(pred_sign[region_mask] == true_sign[region_mask]).detach().cpu().item())
            sign_metrics[axis][region]["correct"] += correct
            sign_metrics[axis][region]["count"] += count


def _summarize_sign_metrics(sign_metrics: dict[str, dict[str, dict[str, int]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for axis, regions in sign_metrics.items():
        out[axis] = {}
        for region, vals in regions.items():
            count = int(vals["count"])
            correct = int(vals["correct"])
            out[axis][region] = {
                "count": count,
                "correct": correct,
                "accuracy": float(correct / count) if count > 0 else float("nan"),
            }
    return out


def _init_defect_metrics() -> dict[str, list[float]]:
    return {
        "rmse": [],
        "psnr": [],
        "corr": [],
        "slope": [],
        "peak_ratio": [],
        "mean_abs_ratio": [],
        "volume_rel_error": [],
        "auprc": [],
        "auroc": [],
        "iou": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "support_frac": [],
    }


def _init_wrap_group_metrics() -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "rmse": [],
            "corr": [],
            "slope": [],
            "peak_ratio": [],
            "mean_abs_ratio": [],
            "estimated_wrap_stress_level": [],
            "defect_wrap_target": [],
            "standard_wrap_frac": [],
        }
        for name in ("in_wrap", "cross_wrap")
    }


def _update_wrap_group_metrics(
    wrap_group_metrics: dict[str, dict[str, list[float]]],
    *,
    group_name: str,
    defect_pred: torch.Tensor,
    defect_true: torch.Tensor,
    mask: torch.Tensor,
    estimated_wrap_stress_level: float | None,
    defect_wrap_target: float | None,
    standard_wrap_frac: float | None,
) -> None:
    if group_name not in wrap_group_metrics:
        return
    gm = wrap_group_metrics[group_name]
    gm["rmse"].append(float(masked_rmse(defect_pred, defect_true, mask).cpu()))
    gm["corr"].append(float(masked_correlation(defect_pred, defect_true, mask).cpu()))
    gm["slope"].append(float(masked_slope(defect_pred, defect_true, mask).cpu()))
    gm["peak_ratio"].append(float(masked_peak_ratio(defect_pred, defect_true, mask).cpu()))
    gm["mean_abs_ratio"].append(float(masked_mean_abs_ratio(defect_pred, defect_true, mask).cpu()))
    gm["estimated_wrap_stress_level"].append(float(estimated_wrap_stress_level) if estimated_wrap_stress_level is not None else float("nan"))
    gm["defect_wrap_target"].append(float(defect_wrap_target) if defect_wrap_target is not None else float("nan"))
    gm["standard_wrap_frac"].append(float(standard_wrap_frac) if standard_wrap_frac is not None else float("nan"))


def _summarize_wrap_group_metrics(wrap_group_metrics: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    return {
        group: {
            "count": len(vals["rmse"]),
            "summary_defect": {k: _safe_nanmean(v) for k, v in vals.items() if k in {"rmse", "corr", "slope", "peak_ratio", "mean_abs_ratio"}},
            "wrap_meta": {
                "estimated_wrap_stress_level": _safe_nanmean(vals["estimated_wrap_stress_level"]),
                "defect_wrap_target": _safe_nanmean(vals["defect_wrap_target"]),
                "standard_wrap_frac": _safe_nanmean(vals["standard_wrap_frac"]),
            },
        }
        for group, vals in wrap_group_metrics.items()
    }


def _update_defect_metrics(
    defect_metrics: dict[str, list[float]],
    *,
    defect_pred: torch.Tensor,
    defect_true: torch.Tensor,
    mask: torch.Tensor,
    scores: torch.Tensor,
    pred_mask: torch.Tensor,
    regions: dict[str, torch.Tensor] | None,
) -> None:
    defect_metrics["rmse"].append(float(masked_rmse(defect_pred, defect_true, mask).cpu()))
    defect_metrics["psnr"].append(float(masked_psnr(defect_pred, defect_true, mask).cpu()))
    defect_metrics["corr"].append(float(masked_correlation(defect_pred, defect_true, mask).cpu()))
    defect_metrics["slope"].append(float(masked_slope(defect_pred, defect_true, mask).cpu()))
    defect_metrics["peak_ratio"].append(float(masked_peak_ratio(defect_pred, defect_true, mask).cpu()))
    defect_metrics["mean_abs_ratio"].append(float(masked_mean_abs_ratio(defect_pred, defect_true, mask).cpu()))
    defect_metrics["volume_rel_error"].append(float(masked_volume_rel_error(defect_pred, defect_true, mask).cpu()))
    if regions is not None:
        ap = regions["aperture"].to(dtype=torch.bool)
        defect_metrics["auprc"].append(float(binary_auprc(scores[ap], mask[ap]).cpu()))
        defect_metrics["auroc"].append(float(binary_auroc(scores[ap], mask[ap]).cpu()))
    defect_metrics["iou"].append(float(binary_iou(pred_mask, mask).cpu()))
    defect_metrics["f1"].append(float(binary_f1(pred_mask, mask).cpu()))
    defect_metrics["precision"].append(float(binary_precision(pred_mask, mask).cpu()))
    defect_metrics["recall"].append(float(binary_recall(pred_mask, mask).cpu()))
    defect_metrics["support_frac"].append(float(torch.mean(mask.to(dtype=torch.float32)).cpu()))


def _update_region_metrics(
    region_metrics: dict[str, dict[str, list[float]]],
    *,
    scores: torch.Tensor,
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    defect_pred: torch.Tensor,
    defect_true: torch.Tensor,
    regions: dict[str, torch.Tensor],
) -> None:
    for region_name, region_mask in regions.items():
        rm = region_mask.to(dtype=torch.bool)
        tm = (true_mask.to(dtype=torch.bool) & rm)
        pm = (pred_mask.to(dtype=torch.bool) & rm)

        region_metrics[region_name]["rmse"].append(float(masked_rmse(defect_pred, defect_true, rm).cpu()))
        region_metrics[region_name]["psnr"].append(float(masked_psnr(defect_pred, defect_true, rm).cpu()))

        region_metrics[region_name]["auprc"].append(float(binary_auprc(scores[rm], tm[rm]).cpu()))
        region_metrics[region_name]["auroc"].append(float(binary_auroc(scores[rm], tm[rm]).cpu()))
        region_metrics[region_name]["iou"].append(float(binary_iou(pm, tm).cpu()))
        region_metrics[region_name]["f1"].append(float(binary_f1(pm, tm).cpu()))
        region_metrics[region_name]["precision"].append(float(binary_precision(pm, tm).cpu()))
        region_metrics[region_name]["recall"].append(float(binary_recall(pm, tm).cpu()))
        denom = torch.mean(rm.to(dtype=torch.float32)).cpu().item()
        frac = float(torch.mean(tm.to(dtype=torch.float32)).cpu().item())
        if denom > 1e-12:
            frac = frac / denom
        region_metrics[region_name]["support_frac"].append(frac)


def evaluate_checkpoint(
    cfg: ExperimentConfig,
    *,
    data_root: str | Path,
    split: str,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    num_plots: int = 3,
) -> Dict[str, Any]:
    device = _select_device(cfg.training.device)
    dataset = DefectDataset(Path(data_root), split)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_meta = ckpt.get("model_meta", {}) if isinstance(ckpt, dict) else {}
    use_prior = bool(model_meta.get("use_pseudo_poisson_prior", False))
    prior_method = str(model_meta.get("pseudo_poisson_prior_method", "pseudo_poisson") or "pseudo_poisson")
    prior_as_input = bool(model_meta.get("pseudo_poisson_prior_as_input", False))
    prior_scale = float(model_meta.get("pseudo_poisson_prior_scale", 1.0))
    prior_pad = int(model_meta.get("pseudo_poisson_poisson_pad", getattr(cfg.training, "pseudo_poisson_poisson_pad", 0) or 0))
    prior_pad_mode = str(model_meta.get("pseudo_poisson_pad_mode", getattr(cfg.training, "pseudo_poisson_pad_mode", "reflect")))
    prior_apply_taper = bool(
        model_meta.get("pseudo_poisson_apply_edge_taper", getattr(cfg.training, "pseudo_poisson_apply_edge_taper", False))
    )
    prior_taper_margin = float(
        model_meta.get("pseudo_poisson_taper_margin", getattr(cfg.training, "pseudo_poisson_taper_margin", 0.25))
    )
    padding_mode = str(model_meta.get("model_padding_mode", getattr(cfg.training, "model_padding_mode", "zeros")))
    residual_scale = model_meta.get("pseudo_poisson_residual_scale", None)
    output_scale = None
    if use_prior and residual_scale is not None:
        output_scale = float(residual_scale)

    model_state = ckpt["model"]
    info = infer_checkpoint_info(model_state)
    model = UNetPP(
        in_channels=info.in_channels,
        out_channels=1,
        predict_logvar=info.predict_logvar,
        padding_mode=padding_mode,
        output_scale=output_scale,
    ).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    physics_train_cfg = _training_cfg_from_model_meta(cfg, model_meta)
    physics = create_forward_model(cfg.simulation, physics_train_cfg, device=device, freeze=True)
    if "physics" in ckpt:
        physics.load_state_dict(ckpt["physics"], strict=False)

    wrap_h = _wrap_height(cfg.simulation)
    gate_thresholds = GateThresholds.from_training(cfg.training)
    gate_enabled = any(v is not None for v in gate_thresholds.to_dict().values())
    qc_metrics: dict[str, list[float]] = {
        "physics_rmse": [],
        "physics_p95_abs": [],
        "edge_mean_abs": [],
        "edge_p95_abs": [],
        "outside_mean_abs": [],
        "outside_p95_abs": [],
        "logvar_mean": [],
    }
    qc_pass: list[bool] = []
    qc_fail_counts: dict[str, int] = {}
    qc_fail_samples: list[dict[str, Any]] = []

    metrics = {
        "rmse": [],
        "psnr": [],
        "ssim": [],
        "corr": [],
        "slope": [],
        "peak_rel_error": [],
        "volume_rel_error": [],
    }

    defect_metrics = _init_defect_metrics()
    sign_metrics = _init_sign_metrics()
    wrap_group_metrics = _init_wrap_group_metrics()
    region_metrics = _init_region_metrics()
    artifact_metrics: dict[str, list[float]] = {
        "edge_mean_abs": [],
        "edge_p95_abs": [],
    }
    pp_pad = int(getattr(cfg.training, "pseudo_poisson_poisson_pad", 0) or 0)
    pp_pad_mode = str(getattr(cfg.training, "pseudo_poisson_pad_mode", "reflect"))
    pp_apply_taper = bool(getattr(cfg.training, "pseudo_poisson_apply_edge_taper", False))
    pp_taper_margin = float(getattr(cfg.training, "pseudo_poisson_taper_margin", 0.25))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plotted = 0
    regions: dict[str, torch.Tensor] | None = None
    qc_masks: dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            diff_ts = {"I_x": batch["inputs"][:, 0], "I_y": batch["inputs"][:, 1]}
            prior_defect = None
            if use_prior:
                prior_defect = reconstruct_defect_coarse_prior(
                    method=prior_method,
                    physics=physics,
                    standard_height=batch["standard"],
                    diff_ts=diff_ts,
                    defect_roi_radius=float(cfg.training.defect_roi_radius),
                    apply_edge_offset=True,
                    poisson_pad=prior_pad,
                    pad_mode=prior_pad_mode,
                    apply_edge_taper=prior_apply_taper,
                    taper_margin=prior_taper_margin,
                )

            inputs = _prepare_inputs(batch, physics, cfg)
            if use_prior and prior_as_input:
                if prior_defect is None:
                    raise RuntimeError("prior_defect missing while use_pseudo_poisson_prior=True")
                prior_chan = (prior_defect / max(wrap_h, 1e-12)).clamp(min=-1.0, max=1.0)
                inputs = torch.cat([inputs, prior_chan], dim=1)

            residual_pred, logvar_pred = _forward_model(model, inputs)
            defect_pred = residual_pred
            if use_prior:
                if prior_defect is None:
                    raise RuntimeError("prior_defect missing while use_pseudo_poisson_prior=True")
                defect_pred = prior_defect * prior_scale + defect_pred
            defect_true = batch["defect"]
            if regions is None:
                h = int(defect_true.shape[-2])
                w = int(defect_true.shape[-1])
                regions = _region_masks(cfg, h=h, w=w, device=device)
                if gate_enabled:
                    qc_masks = radial_masks(cfg, h=h, w=w, device=device)

            wrap_class_ids = batch.get("wrap_class_id")
            wrap_stress = batch.get("estimated_wrap_stress_level")
            defect_wrap_target = batch.get("defect_wrap_target")
            standard_wrap_frac = batch.get("standard_wrap_frac")
            for sample_i, (dp, dt) in enumerate(zip(defect_pred, defect_true)):
                metrics["rmse"].append(float(rmse(dp, dt).cpu()))
                metrics["psnr"].append(float(psnr(dp, dt).cpu()))
                metrics["ssim"].append(float(ssim(dp, dt).cpu()))
                metrics["corr"].append(float(correlation(dp, dt).cpu()))
                metrics["slope"].append(float(slope(dp, dt).cpu()))
                metrics["peak_rel_error"].append(float(peak_rel_error(dp, dt).cpu()))
                metrics["volume_rel_error"].append(float(volume_rel_error(dp, dt).cpu()))

                abs_thr = float(cfg.training.eval_defect_abs_threshold)
                rel_thr = float(cfg.training.eval_defect_rel_threshold)
                dilate_px = int(cfg.training.eval_defect_dilate_px)
                mask = defect_mask(dt, abs_threshold=abs_thr, rel_threshold=rel_thr, dilate_px=dilate_px)
                peak = float(torch.max(torch.abs(dt)).cpu())
                thr = max(abs_thr, rel_thr * peak)
                scores = torch.abs(dp.squeeze())
                pred_mask = scores > thr

                if regions is not None:
                    _update_sign_metrics(
                        sign_metrics,
                        cfg=cfg,
                        standard=batch["standard"][sample_i],
                        defect_pred=dp,
                        defect_true=dt,
                        defect_region=mask,
                        aperture_region=regions["aperture"],
                    )

                _update_defect_metrics(
                    defect_metrics,
                    defect_pred=dp,
                    defect_true=dt,
                    mask=mask,
                    scores=scores,
                    pred_mask=pred_mask,
                    regions=regions,
                )
                if wrap_class_ids is not None:
                    group_name = "cross_wrap" if int(wrap_class_ids[sample_i].item()) == 1 else "in_wrap"
                    est_wrap = float(wrap_stress[sample_i].item()) if wrap_stress is not None else None
                    defect_wrap = float(defect_wrap_target[sample_i].item()) if defect_wrap_target is not None else None
                    std_wrap = float(standard_wrap_frac[sample_i].item()) if standard_wrap_frac is not None else None
                    _update_wrap_group_metrics(
                        wrap_group_metrics,
                        group_name=group_name,
                        defect_pred=dp,
                        defect_true=dt,
                        mask=mask,
                        estimated_wrap_stress_level=est_wrap,
                        defect_wrap_target=defect_wrap,
                        standard_wrap_frac=std_wrap,
                    )

                if regions is not None:
                    _update_region_metrics(
                        region_metrics,
                        scores=scores,
                        pred_mask=pred_mask,
                        true_mask=mask,
                        defect_pred=dp,
                        defect_true=dt,
                        regions=regions,
                    )
                    edge = regions["edge"].to(dtype=torch.bool)
                    edge_art = edge & (~mask.to(dtype=torch.bool))
                    artifact_metrics["edge_mean_abs"].append(float(masked_mean_abs(scores, edge_art).cpu()))
                    artifact_metrics["edge_p95_abs"].append(float(masked_abs_quantile(scores, edge_art, q=0.95).cpu()))

            if gate_enabled and qc_masks is not None:
                height = batch["standard"] + defect_pred
                phys_out = physics(height)
                std_phys = physics(batch["standard"])
                pred_ix = (phys_out["I_x"] - std_phys["I_x"]).squeeze(1)
                pred_iy = (phys_out["I_y"] - std_phys["I_y"]).squeeze(1)
                rx = pred_ix - diff_ts["I_x"]
                ry = pred_iy - diff_ts["I_y"]
                rmag = torch.sqrt(rx**2 + ry**2 + 1e-12)

                ap = qc_masks["aperture"].to(dtype=torch.bool)
                edge = qc_masks["edge"].to(dtype=torch.bool)
                outside = qc_masks["outside"].to(dtype=torch.bool)
                for i in range(defect_pred.shape[0]):
                    m: dict[str, float] = {}
                    rmse_i = _masked_rmse_hw(rmag[i], ap)
                    p95_i = masked_abs_quantile(rmag[i], ap, q=0.95)
                    m["physics_rmse"] = float(rmse_i.detach().cpu().item())
                    m["physics_p95_abs"] = float(p95_i.detach().cpu().item())

                    scores = torch.abs(defect_pred[i].squeeze(0))
                    m["edge_mean_abs"] = float(masked_mean_abs(scores, edge).cpu())
                    m["edge_p95_abs"] = float(masked_abs_quantile(scores, edge, q=0.95).cpu())
                    m["outside_mean_abs"] = float(masked_mean_abs(scores, outside).cpu())
                    m["outside_p95_abs"] = float(masked_abs_quantile(scores, outside, q=0.95).cpu())

                    if logvar_pred is not None:
                        lv = logvar_pred[i].squeeze(0)
                        if lv.ndim == 2 and torch.sum(ap) > 0:
                            m["logvar_mean"] = float(torch.mean(lv[ap]).detach().cpu().item())

                    dec = gate_decision(m, gate_thresholds)
                    qc_pass.append(bool(dec["pass"]))
                    if not dec["pass"]:
                        for f in dec.get("fails", []):
                            qc_fail_counts[f] = qc_fail_counts.get(f, 0) + 1
                        if len(qc_fail_samples) < 50:
                            idx = int(batch_idx * int(cfg.training.batch_size) + i)
                            qc_fail_samples.append(
                                {
                                    "idx": idx,
                                    "fails": list(dec.get("fails", [])),
                                    "metrics": {k: m.get(k) for k in dec.get("fails", [])},
                                }
                            )

                    for k, v in m.items():
                        qc_metrics.setdefault(k, []).append(float(v))

            if plotted < num_plots:
                for i in range(min(num_plots - plotted, defect_pred.shape[0])):
                    std = batch["standard"][i : i + 1]
                    height = std + defect_pred[i : i + 1]
                    phys_out = physics(height)
                    std_phys = physics(std)
                    diff_pred = {
                        "I_x": (phys_out["I_x"] - std_phys["I_x"]).squeeze(0).squeeze(0).cpu().numpy(),
                        "I_y": (phys_out["I_y"] - std_phys["I_y"]).squeeze(0).squeeze(0).cpu().numpy(),
                    }
                    diff_true = {
                        "I_x": batch["inputs"][i, 0].cpu().numpy(),
                        "I_y": batch["inputs"][i, 1].cpu().numpy(),
                    }
                    plot_defect_and_intensity(
                        defect_true=defect_true[i].squeeze(0).cpu().numpy(),
                        defect_pred=defect_pred[i].squeeze(0).cpu().numpy(),
                        diff_true=diff_true,
                        diff_pred=diff_pred,
                        output_path=plots_dir / f"sample_{batch_idx:04d}_{i}.png",
                        title=f"{split} sample {batch_idx}:{i}",
                    )
                    plotted += 1

    summary = {k: _safe_nanmean(v) for k, v in metrics.items()}
    defect_summary = {k: _safe_nanmean(v) for k, v in defect_metrics.items()}
    sign_summary = _summarize_sign_metrics(sign_metrics)
    wrap_group_summary = _summarize_wrap_group_metrics(wrap_group_metrics)
    region_summary = {
        region: {k: _safe_nanmean(v) for k, v in vals.items()}
        for region, vals in region_metrics.items()
    }
    artifact_summary = {k: _safe_nanmean(v) for k, v in artifact_metrics.items()}
    result = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "count": len(metrics["rmse"]),
        "summary": summary,
        "defect_masking": {
            "abs_threshold": float(cfg.training.eval_defect_abs_threshold),
            "rel_threshold": float(cfg.training.eval_defect_rel_threshold),
            "dilate_px": int(cfg.training.eval_defect_dilate_px),
        },
        "summary_defect": defect_summary,
        "summary_sign_gradient": sign_summary,
        "summary_wrap_groups": wrap_group_summary,
        "region_defs": {
            "aperture_radius": float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0),
            "edge_band_start_frac": float(getattr(cfg.training, "eval_edge_band_start_frac", 0.9)),
            "center_radius_frac": float(getattr(cfg.training, "eval_center_radius_frac", 0.1)),
        },
        "summary_regions": region_summary,
        "artifacts": artifact_summary,
    }
    if gate_enabled:
        qc_summary = {k + "_mean": _safe_nanmean(v) for k, v in qc_metrics.items()}
        qc_summary.update({k + "_p95": _safe_quantile(v, 0.95) for k, v in qc_metrics.items()})
        result["qc"] = {
            "enabled": True,
            "thresholds": gate_thresholds.to_dict(),
            "pass_rate": float(np.mean([1.0 if p else 0.0 for p in qc_pass])) if qc_pass else float("nan"),
            "fail_counts": qc_fail_counts,
            "summary": qc_summary,
            "fail_samples": qc_fail_samples,
        }
    write_json(out_dir / "eval_metrics.json", result)
    return result


def evaluate_pseudo_poisson(
    cfg: ExperimentConfig,
    *,
    data_root: str | Path,
    split: str,
    out_dir: str | Path,
    num_plots: int = 3,
) -> Dict[str, Any]:
    return _evaluate_baseline_reconstruction(
        cfg,
        data_root=data_root,
        split=split,
        out_dir=out_dir,
        num_plots=num_plots,
        method="pseudo_poisson",
        plot_label="pseudo_poisson",
        reconstructor=reconstruct_defect_pseudo_poisson,
        needs_defect_true=False,
    )


def _evaluate_baseline_reconstruction(
    cfg: ExperimentConfig,
    *,
    data_root: str | Path,
    split: str,
    out_dir: str | Path,
    num_plots: int,
    method: str,
    plot_label: str,
    reconstructor: Callable[..., torch.Tensor],
    needs_defect_true: bool,
) -> Dict[str, Any]:
    device = _select_device(cfg.training.device)
    dataset = DefectDataset(Path(data_root), split)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

    physics = create_forward_model(cfg.simulation, cfg.training, device=device, freeze=True)
    gate_thresholds = GateThresholds.from_training(cfg.training)
    gate_enabled = any(v is not None for v in gate_thresholds.to_dict().values())
    qc_metrics: dict[str, list[float]] = {
        "physics_rmse": [],
        "physics_p95_abs": [],
        "edge_mean_abs": [],
        "edge_p95_abs": [],
        "outside_mean_abs": [],
        "outside_p95_abs": [],
    }
    qc_pass: list[bool] = []
    qc_fail_counts: dict[str, int] = {}
    qc_fail_samples: list[dict[str, Any]] = []

    metrics = {
        "rmse": [],
        "psnr": [],
        "ssim": [],
        "corr": [],
        "slope": [],
        "peak_rel_error": [],
        "volume_rel_error": [],
    }

    defect_metrics = _init_defect_metrics()
    sign_metrics = _init_sign_metrics()
    wrap_group_metrics = _init_wrap_group_metrics()
    region_metrics = _init_region_metrics()
    artifact_metrics: dict[str, list[float]] = {
        "edge_mean_abs": [],
        "edge_p95_abs": [],
    }
    pp_pad = int(getattr(cfg.training, "pseudo_poisson_poisson_pad", 0) or 0)
    pp_pad_mode = str(getattr(cfg.training, "pseudo_poisson_pad_mode", "reflect"))
    pp_apply_taper = bool(getattr(cfg.training, "pseudo_poisson_apply_edge_taper", False))
    pp_taper_margin = float(getattr(cfg.training, "pseudo_poisson_taper_margin", 0.25))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plotted = 0
    regions: dict[str, torch.Tensor] | None = None
    qc_masks: dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            diff_ts = {"I_x": batch["inputs"][:, 0], "I_y": batch["inputs"][:, 1]}
            defect_true = batch["defect"]
            recon_kwargs = {
                "physics": physics,
                "standard_height": batch["standard"],
                "diff_ts": diff_ts,
                "defect_roi_radius": float(cfg.training.defect_roi_radius),
                "apply_edge_offset": True,
                "poisson_pad": pp_pad,
                "pad_mode": pp_pad_mode,
                "apply_edge_taper": pp_apply_taper,
                "taper_margin": pp_taper_margin,
            }
            if needs_defect_true:
                recon_kwargs["defect_true"] = defect_true
            defect_pred = reconstructor(**recon_kwargs)
            if regions is None:
                h = int(defect_true.shape[-2])
                w = int(defect_true.shape[-1])
                regions = _region_masks(cfg, h=h, w=w, device=device)
                if gate_enabled:
                    qc_masks = radial_masks(cfg, h=h, w=w, device=device)

            wrap_class_ids = batch.get("wrap_class_id")
            wrap_stress = batch.get("estimated_wrap_stress_level")
            defect_wrap_target = batch.get("defect_wrap_target")
            standard_wrap_frac = batch.get("standard_wrap_frac")
            for sample_i, (dp, dt) in enumerate(zip(defect_pred, defect_true)):
                metrics["rmse"].append(float(rmse(dp, dt).cpu()))
                metrics["psnr"].append(float(psnr(dp, dt).cpu()))
                metrics["ssim"].append(float(ssim(dp, dt).cpu()))
                metrics["corr"].append(float(correlation(dp, dt).cpu()))
                metrics["slope"].append(float(slope(dp, dt).cpu()))
                metrics["peak_rel_error"].append(float(peak_rel_error(dp, dt).cpu()))
                metrics["volume_rel_error"].append(float(volume_rel_error(dp, dt).cpu()))

                abs_thr = float(cfg.training.eval_defect_abs_threshold)
                rel_thr = float(cfg.training.eval_defect_rel_threshold)
                dilate_px = int(cfg.training.eval_defect_dilate_px)
                mask = defect_mask(dt, abs_threshold=abs_thr, rel_threshold=rel_thr, dilate_px=dilate_px)
                peak = float(torch.max(torch.abs(dt)).cpu())
                thr = max(abs_thr, rel_thr * peak)
                scores = torch.abs(dp.squeeze())
                pred_mask = scores > thr

                if regions is not None:
                    _update_sign_metrics(
                        sign_metrics,
                        cfg=cfg,
                        standard=batch["standard"][sample_i],
                        defect_pred=dp,
                        defect_true=dt,
                        defect_region=mask,
                        aperture_region=regions["aperture"],
                    )

                _update_defect_metrics(
                    defect_metrics,
                    defect_pred=dp,
                    defect_true=dt,
                    mask=mask,
                    scores=scores,
                    pred_mask=pred_mask,
                    regions=regions,
                )
                if wrap_class_ids is not None:
                    group_name = "cross_wrap" if int(wrap_class_ids[sample_i].item()) == 1 else "in_wrap"
                    est_wrap = float(wrap_stress[sample_i].item()) if wrap_stress is not None else None
                    defect_wrap = float(defect_wrap_target[sample_i].item()) if defect_wrap_target is not None else None
                    std_wrap = float(standard_wrap_frac[sample_i].item()) if standard_wrap_frac is not None else None
                    _update_wrap_group_metrics(
                        wrap_group_metrics,
                        group_name=group_name,
                        defect_pred=dp,
                        defect_true=dt,
                        mask=mask,
                        estimated_wrap_stress_level=est_wrap,
                        defect_wrap_target=defect_wrap,
                        standard_wrap_frac=std_wrap,
                    )

                if regions is not None:
                    _update_region_metrics(
                        region_metrics,
                        scores=scores,
                        pred_mask=pred_mask,
                        true_mask=mask,
                        defect_pred=dp,
                        defect_true=dt,
                        regions=regions,
                    )
                    edge = regions["edge"].to(dtype=torch.bool)
                    edge_art = edge & (~mask.to(dtype=torch.bool))
                    artifact_metrics["edge_mean_abs"].append(float(masked_mean_abs(scores, edge_art).cpu()))
                    artifact_metrics["edge_p95_abs"].append(float(masked_abs_quantile(scores, edge_art, q=0.95).cpu()))

            if gate_enabled and qc_masks is not None:
                height = batch["standard"] + defect_pred
                phys_out = physics(height)
                std_phys = physics(batch["standard"])
                pred_ix = (phys_out["I_x"] - std_phys["I_x"]).squeeze(1)
                pred_iy = (phys_out["I_y"] - std_phys["I_y"]).squeeze(1)
                rx = pred_ix - diff_ts["I_x"]
                ry = pred_iy - diff_ts["I_y"]
                rmag = torch.sqrt(rx**2 + ry**2 + 1e-12)

                ap = qc_masks["aperture"].to(dtype=torch.bool)
                edge = qc_masks["edge"].to(dtype=torch.bool)
                outside = qc_masks["outside"].to(dtype=torch.bool)
                for i in range(defect_pred.shape[0]):
                    m: dict[str, float] = {}
                    rmse_i = _masked_rmse_hw(rmag[i], ap)
                    p95_i = masked_abs_quantile(rmag[i], ap, q=0.95)
                    m["physics_rmse"] = float(rmse_i.detach().cpu().item())
                    m["physics_p95_abs"] = float(p95_i.detach().cpu().item())

                    scores = torch.abs(defect_pred[i].squeeze(0))
                    m["edge_mean_abs"] = float(masked_mean_abs(scores, edge).cpu())
                    m["edge_p95_abs"] = float(masked_abs_quantile(scores, edge, q=0.95).cpu())
                    m["outside_mean_abs"] = float(masked_mean_abs(scores, outside).cpu())
                    m["outside_p95_abs"] = float(masked_abs_quantile(scores, outside, q=0.95).cpu())

                    dec = gate_decision(m, gate_thresholds)
                    qc_pass.append(bool(dec["pass"]))
                    if not dec["pass"]:
                        for f in dec.get("fails", []):
                            qc_fail_counts[f] = qc_fail_counts.get(f, 0) + 1
                        if len(qc_fail_samples) < 50:
                            idx = int(batch_idx * int(cfg.training.batch_size) + i)
                            qc_fail_samples.append(
                                {
                                    "idx": idx,
                                    "fails": list(dec.get("fails", [])),
                                    "metrics": {k: m.get(k) for k in dec.get("fails", [])},
                                }
                            )

                    for k, v in m.items():
                        qc_metrics.setdefault(k, []).append(float(v))

            if plotted < num_plots:
                for i in range(min(num_plots - plotted, defect_pred.shape[0])):
                    std = batch["standard"][i : i + 1]
                    height = std + defect_pred[i : i + 1]
                    phys_out = physics(height)
                    std_phys = physics(std)
                    diff_pred = {
                        "I_x": (phys_out["I_x"] - std_phys["I_x"]).squeeze(0).squeeze(0).cpu().numpy(),
                        "I_y": (phys_out["I_y"] - std_phys["I_y"]).squeeze(0).squeeze(0).cpu().numpy(),
                    }
                    diff_true = {
                        "I_x": batch["inputs"][i, 0].cpu().numpy(),
                        "I_y": batch["inputs"][i, 1].cpu().numpy(),
                    }
                    plot_defect_and_intensity(
                        defect_true=defect_true[i].squeeze(0).cpu().numpy(),
                        defect_pred=defect_pred[i].squeeze(0).cpu().numpy(),
                        diff_true=diff_true,
                        diff_pred=diff_pred,
                        output_path=plots_dir / f"sample_{batch_idx:04d}_{i}.png",
                        title=f"{split} sample {batch_idx}:{i} ({plot_label})",
                    )
                    plotted += 1

    summary = {k: _safe_nanmean(v) for k, v in metrics.items()}
    defect_summary = {k: _safe_nanmean(v) for k, v in defect_metrics.items()}
    sign_summary = _summarize_sign_metrics(sign_metrics)
    wrap_group_summary = _summarize_wrap_group_metrics(wrap_group_metrics)
    region_summary = {
        region: {k: _safe_nanmean(v) for k, v in vals.items()}
        for region, vals in region_metrics.items()
    }
    artifact_summary = {k: _safe_nanmean(v) for k, v in artifact_metrics.items()}
    result = {
        "split": split,
        "method": method,
        "count": len(metrics["rmse"]),
        "summary": summary,
        "defect_masking": {
            "abs_threshold": float(cfg.training.eval_defect_abs_threshold),
            "rel_threshold": float(cfg.training.eval_defect_rel_threshold),
            "dilate_px": int(cfg.training.eval_defect_dilate_px),
        },
        "summary_defect": defect_summary,
        "summary_sign_gradient": sign_summary,
        "summary_wrap_groups": wrap_group_summary,
        "region_defs": {
            "aperture_radius": float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0),
            "edge_band_start_frac": float(getattr(cfg.training, "eval_edge_band_start_frac", 0.9)),
            "center_radius_frac": float(getattr(cfg.training, "eval_center_radius_frac", 0.1)),
        },
        "summary_regions": region_summary,
        "artifacts": artifact_summary,
    }
    if gate_enabled:
        qc_summary = {k + "_mean": _safe_nanmean(v) for k, v in qc_metrics.items()}
        qc_summary.update({k + "_p95": _safe_quantile(v, 0.95) for k, v in qc_metrics.items()})
        result["qc"] = {
            "enabled": True,
            "thresholds": gate_thresholds.to_dict(),
            "pass_rate": float(np.mean([1.0 if p else 0.0 for p in qc_pass])) if qc_pass else float("nan"),
            "fail_counts": qc_fail_counts,
            "summary": qc_summary,
            "fail_samples": qc_fail_samples,
        }
    write_json(out_dir / "eval_metrics.json", result)
    return result


def evaluate_first_order_poisson(
    cfg: ExperimentConfig,
    *,
    data_root: str | Path,
    split: str,
    out_dir: str | Path,
    num_plots: int = 3,
) -> Dict[str, Any]:
    return _evaluate_baseline_reconstruction(
        cfg,
        data_root=data_root,
        split=split,
        out_dir=out_dir,
        num_plots=num_plots,
        method="first_order_poisson",
        plot_label="first_order_poisson",
        reconstructor=reconstruct_defect_first_order_poisson,
        needs_defect_true=False,
    )


def evaluate_first_order_sign_quadratic_poisson(
    cfg: ExperimentConfig,
    *,
    data_root: str | Path,
    split: str,
    out_dir: str | Path,
    num_plots: int = 3,
) -> Dict[str, Any]:
    return _evaluate_baseline_reconstruction(
        cfg,
        data_root=data_root,
        split=split,
        out_dir=out_dir,
        num_plots=num_plots,
        method="first_order_sign_quadratic_poisson",
        plot_label="first_order_sign_quadratic_poisson",
        reconstructor=reconstruct_defect_first_order_sign_quadratic_poisson,
        needs_defect_true=False,
    )


def evaluate_oracle_poisson(
    cfg: ExperimentConfig,
    *,
    data_root: str | Path,
    split: str,
    out_dir: str | Path,
    num_plots: int = 3,
) -> Dict[str, Any]:
    return _evaluate_baseline_reconstruction(
        cfg,
        data_root=data_root,
        split=split,
        out_dir=out_dir,
        num_plots=num_plots,
        method="oracle_poisson",
        plot_label="oracle_poisson",
        reconstructor=reconstruct_defect_oracle_poisson,
        needs_defect_true=True,
    )


__all__ = [
    "evaluate_checkpoint",
    "evaluate_first_order_poisson",
    "evaluate_first_order_sign_quadratic_poisson",
    "evaluate_oracle_poisson",
    "evaluate_pseudo_poisson",
]
