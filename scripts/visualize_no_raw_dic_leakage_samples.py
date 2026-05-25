from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from mini_grin_rebuild.core.configs import ExperimentConfig, load_experiment_config  # noqa: E402
from mini_grin_rebuild.core.json_io import write_json  # noqa: E402
from mini_grin_rebuild.data.datasets import DefectDataset  # noqa: E402
from mini_grin_rebuild.evaluation.evaluator import (  # noqa: E402
    _binary_sign_t,
    _forward_model,
    _phase_gradients_hw,
    _prepare_inputs,
    _region_masks,
    _training_cfg_from_model_meta,
    _valid_gradient_mask_t,
    _wrap_height,
)
from mini_grin_rebuild.evaluation.metrics import defect_mask, masked_correlation, masked_rmse  # noqa: E402
from mini_grin_rebuild.models.checkpoint import infer_checkpoint_info, load_checkpoint  # noqa: E402
from mini_grin_rebuild.models.unetpp import UNetPP  # noqa: E402
from mini_grin_rebuild.physics.factory import create_forward_model  # noqa: E402
from mini_grin_rebuild.reconstruction import (  # noqa: E402
    reconstruct_defect_first_order_poisson,
    reconstruct_defect_first_order_sign_quadratic_poisson,
    reconstruct_defect_pseudo_poisson,
)


OPEN_METHODS: dict[str, Callable[..., torch.Tensor]] = {
    "first_order_sign_quadratic_poisson": reconstruct_defect_first_order_sign_quadratic_poisson,
    "first_order_poisson": reconstruct_defect_first_order_poisson,
    "pseudo_poisson": reconstruct_defect_pseudo_poisson,
}


def _select_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _as_batch(sample: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0).to(device)
    return out


def _as_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _as_hw_np(t: torch.Tensor) -> np.ndarray:
    arr = _as_np(t)
    return np.squeeze(arr)


def _nan_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _parse_indices(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _sign_context(
    cfg: ExperimentConfig,
    *,
    standard: torch.Tensor,
    defect_true: torch.Tensor,
    defect_pred: torch.Tensor,
    defect_region: torch.Tensor,
    aperture_region: torch.Tensor,
) -> dict[str, Any]:
    std_gy, std_gx = _phase_gradients_hw(cfg, standard)
    true_gy, true_gx = _phase_gradients_hw(cfg, standard + defect_true)
    pred_gy, pred_gx = _phase_gradients_hw(cfg, standard + defect_pred)

    out: dict[str, Any] = {}
    for axis, std_grad, true_grad, pred_grad in (
        ("x", std_gx, true_gx, pred_gx),
        ("y", std_gy, true_gy, pred_gy),
    ):
        true_sign = _binary_sign_t(true_grad)
        pred_sign = _binary_sign_t(pred_grad)
        std_sign = _binary_sign_t(std_grad)
        valid = _valid_gradient_mask_t(true_grad)
        aperture = aperture_region.to(dtype=torch.bool)
        masks = {
            "global": aperture & valid,
            "defect_local": aperture & defect_region.to(dtype=torch.bool) & valid,
            "branch_flip": aperture & valid & (std_sign != true_sign),
        }
        stats: dict[str, dict[str, float | int]] = {}
        for name, mask in masks.items():
            count = int(torch.sum(mask).detach().cpu().item())
            correct = int(torch.sum((pred_sign == true_sign) & mask).detach().cpu().item())
            stats[name] = {
                "count": count,
                "correct": correct,
                "accuracy": float(correct / count) if count > 0 else float("nan"),
            }
        out[axis] = {
            "true_sign": true_sign,
            "pred_sign": pred_sign,
            "std_sign": std_sign,
            "valid": valid,
            "branch_flip": masks["branch_flip"],
            "mismatch": masks["global"] & (pred_sign != true_sign),
            "stats": stats,
        }
    return out


def _rank_sample_indices(
    dataset: DefectDataset,
    cfg: ExperimentConfig,
    *,
    device: torch.device,
    top_k: int,
) -> list[int]:
    scores: list[tuple[int, int, float, int]] = []
    for idx in range(len(dataset)):
        batch = _as_batch(dataset[idx], device)
        standard = batch["standard"][0]
        defect_true = batch["defect"][0]
        h, w = int(defect_true.shape[-2]), int(defect_true.shape[-1])
        regions = _region_masks(cfg, h=h, w=w, device=device)
        defect_region = defect_mask(
            defect_true,
            abs_threshold=float(cfg.training.eval_defect_abs_threshold),
            rel_threshold=float(cfg.training.eval_defect_rel_threshold),
            dilate_px=int(cfg.training.eval_defect_dilate_px),
        ).to(device)
        ctx = _sign_context(
            cfg,
            standard=standard,
            defect_true=defect_true,
            defect_pred=defect_true,
            defect_region=defect_region,
            aperture_region=regions["aperture"],
        )
        flips = int(ctx["x"]["stats"]["branch_flip"]["count"]) + int(ctx["y"]["stats"]["branch_flip"]["count"])
        defect_px = int(torch.sum(defect_region).detach().cpu().item())
        peak = float(torch.max(torch.abs(defect_true)).detach().cpu().item())
        scores.append((flips, defect_px, peak, idx))
    scores.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [idx for *_rest, idx in scores[: max(0, int(top_k))]]


def _load_nn(
    cfg: ExperimentConfig,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_meta = ckpt.get("model_meta", {}) if isinstance(ckpt, dict) else {}
    padding_mode = str(model_meta.get("model_padding_mode", getattr(cfg.training, "model_padding_mode", "zeros")))
    residual_scale = model_meta.get("pseudo_poisson_residual_scale", None)
    use_prior = bool(model_meta.get("use_pseudo_poisson_prior", False))
    output_scale = float(residual_scale) if use_prior and residual_scale is not None else None

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
    physics.eval()
    return model, physics, model_meta


def _predict_nn(
    cfg: ExperimentConfig,
    *,
    model: torch.nn.Module,
    physics: torch.nn.Module,
    model_meta: dict[str, Any],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
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

    diff_ts = {"I_x": batch["inputs"][:, 0], "I_y": batch["inputs"][:, 1]}
    prior_defect = None
    if use_prior:
        reconstructor = OPEN_METHODS.get(prior_method)
        if reconstructor is None:
            raise ValueError(f"Unsupported pseudo_poisson_prior_method={prior_method!r}")
        prior_defect = reconstructor(
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
            raise RuntimeError("prior_defect missing while pseudo_poisson_prior_as_input=True")
        wrap_h = _wrap_height(cfg.simulation)
        prior_chan = (prior_defect / max(wrap_h, 1e-12)).clamp(min=-1.0, max=1.0)
        inputs = torch.cat([inputs, prior_chan], dim=1)

    residual_pred, _logvar = _forward_model(model, inputs)
    if use_prior:
        if prior_defect is None:
            raise RuntimeError("prior_defect missing while use_pseudo_poisson_prior=True")
        return prior_defect * prior_scale + residual_pred
    return residual_pred


def _predict_open(
    cfg: ExperimentConfig,
    *,
    physics: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    method: str,
) -> torch.Tensor:
    reconstructor = OPEN_METHODS[method]
    return reconstructor(
        physics=physics,
        standard_height=batch["standard"],
        diff_ts={"I_x": batch["inputs"][:, 0], "I_y": batch["inputs"][:, 1]},
        defect_roi_radius=float(cfg.training.defect_roi_radius),
        apply_edge_offset=True,
        poisson_pad=int(getattr(cfg.training, "pseudo_poisson_poisson_pad", 0) or 0),
        pad_mode=str(getattr(cfg.training, "pseudo_poisson_pad_mode", "reflect")),
        apply_edge_taper=bool(getattr(cfg.training, "pseudo_poisson_apply_edge_taper", False)),
        taper_margin=float(getattr(cfg.training, "pseudo_poisson_taper_margin", 0.25)),
    )


def _predict_diff(
    *,
    physics: torch.nn.Module,
    standard: torch.Tensor,
    defect: torch.Tensor,
) -> dict[str, torch.Tensor]:
    height = standard + defect
    phys_out = physics(height)
    std_out = physics(standard)
    return {
        "I_x": (phys_out["I_x"] - std_out["I_x"]).squeeze(1),
        "I_y": (phys_out["I_y"] - std_out["I_y"]).squeeze(1),
    }


def _safe_corr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return _nan_float(masked_correlation(pred, target, mask))


def _safe_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return _nan_float(masked_rmse(pred, target, mask))


def _image_title(title: str, arr: np.ndarray | None = None) -> str:
    if arr is None:
        return title
    finite = np.asarray(arr)[np.isfinite(arr)]
    if finite.size == 0:
        return title
    return f"{title}\n[{finite.min():.2g}, {finite.max():.2g}]"


def _imshow(
    ax: plt.Axes,
    arr: np.ndarray,
    *,
    title: str,
    cmap: str | ListedColormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate_range: bool = True,
) -> None:
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(_image_title(title, arr) if annotate_range else title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _symmetric_limit(*arrays: np.ndarray, floor: float = 1e-12) -> float:
    vmax = floor
    for arr in arrays:
        if arr is None:
            continue
        finite = np.asarray(arr)[np.isfinite(arr)]
        if finite.size:
            vmax = max(vmax, float(np.max(np.abs(finite))))
    return vmax


def _load_ideal_diff(
    ideal_data_root: Path | None,
    *,
    split: str,
    sample_file: Path,
) -> dict[str, np.ndarray] | None:
    if ideal_data_root is None:
        return None
    ideal_path = ideal_data_root / split / sample_file.name
    if not ideal_path.is_file():
        return None
    data = np.load(ideal_path)
    if "diff_ix_st" not in data or "diff_iy_st" not in data:
        return None
    return {
        "I_x": np.asarray(data["diff_ix_st"], dtype=np.float32),
        "I_y": np.asarray(data["diff_iy_st"], dtype=np.float32),
    }


def _plot_prediction_panel(
    *,
    output_path: Path,
    sample_label: str,
    batch: dict[str, torch.Tensor],
    open_pred: torch.Tensor,
    nn_pred: torch.Tensor | None,
    open_sign: dict[str, Any],
    nn_sign: dict[str, Any] | None,
    defect_region: torch.Tensor,
    ideal_diff: dict[str, np.ndarray] | None,
) -> None:
    defect_true = batch["defect"][0]
    open_hw = open_pred[0]
    nn_hw = nn_pred[0] if nn_pred is not None else torch.zeros_like(open_hw)

    obs_ix = _as_hw_np(batch["inputs"][0, 0])
    obs_iy = _as_hw_np(batch["inputs"][0, 1])
    obs_srix = _as_hw_np(batch["inputs_sr"][0, 0]) if "inputs_sr" in batch else np.zeros_like(obs_ix)
    obs_sriy = _as_hw_np(batch["inputs_sr"][0, 1]) if "inputs_sr" in batch else np.zeros_like(obs_iy)

    true_np = _as_hw_np(defect_true)
    open_np = _as_hw_np(open_hw)
    nn_np = _as_hw_np(nn_hw)
    open_err = open_np - true_np
    nn_err = nn_np - true_np if nn_pred is not None else np.full_like(true_np, np.nan)
    v_defect = _symmetric_limit(true_np, open_np, nn_np)
    v_err = _symmetric_limit(open_err, nn_err)
    v_obs = _symmetric_limit(obs_ix, obs_iy, obs_srix, obs_sriy)

    sign_cmap = ListedColormap(["#2f6fb0", "#f3f3f3", "#c7432b"])
    mask_cmap = ListedColormap(["#f4f4f4", "#1d1d1d"])

    fig, axes = plt.subplots(5, 5, figsize=(17, 17), constrained_layout=True)

    _imshow(axes[0, 0], obs_ix, title="Observed dIx test-standard", cmap="coolwarm", vmin=-v_obs, vmax=v_obs)
    _imshow(axes[0, 1], obs_iy, title="Observed dIy test-standard", cmap="coolwarm", vmin=-v_obs, vmax=v_obs)
    _imshow(axes[0, 2], obs_srix, title="Observed dIx standard-ref", cmap="coolwarm", vmin=-v_obs, vmax=v_obs)
    _imshow(axes[0, 3], obs_sriy, title="Observed dIy standard-ref", cmap="coolwarm", vmin=-v_obs, vmax=v_obs)
    if ideal_diff is not None:
        leak_mag = np.sqrt((obs_ix - ideal_diff["I_x"]) ** 2 + (obs_iy - ideal_diff["I_y"]) ** 2)
        _imshow(axes[0, 4], leak_mag, title="|nonideal - ideal| dI", cmap="magma")
    else:
        _imshow(axes[0, 4], _as_hw_np(defect_region.to(torch.float32)), title="Defect support mask", cmap=mask_cmap, vmin=0, vmax=1, annotate_range=False)

    _imshow(axes[1, 0], true_np, title="True defect height", cmap="coolwarm", vmin=-v_defect, vmax=v_defect)
    _imshow(axes[1, 1], open_np, title="Open defect pred", cmap="coolwarm", vmin=-v_defect, vmax=v_defect)
    _imshow(axes[1, 2], open_err, title="Open defect error", cmap="RdBu_r", vmin=-v_err, vmax=v_err)
    if nn_pred is not None:
        _imshow(axes[1, 3], nn_np, title="NN defect pred", cmap="coolwarm", vmin=-v_defect, vmax=v_defect)
        _imshow(axes[1, 4], nn_err, title="NN defect error", cmap="RdBu_r", vmin=-v_err, vmax=v_err)
    else:
        for ax, title in ((axes[1, 3], "NN defect pred"), (axes[1, 4], "NN defect error")):
            _imshow(ax, np.full_like(true_np, np.nan), title=f"{title}\nnot provided", cmap="viridis")

    for row, axis_name in ((2, "x"), (3, "y")):
        true_sign = _as_hw_np(open_sign[axis_name]["true_sign"])
        open_pred_sign = _as_hw_np(open_sign[axis_name]["pred_sign"])
        open_mismatch = _as_hw_np(open_sign[axis_name]["mismatch"].to(torch.float32))
        _imshow(
            axes[row, 0],
            true_sign,
            title=f"True test sign grad_{axis_name}",
            cmap=sign_cmap,
            vmin=-1,
            vmax=1,
            annotate_range=False,
        )
        _imshow(
            axes[row, 1],
            open_pred_sign,
            title=f"Open pred sign grad_{axis_name}",
            cmap=sign_cmap,
            vmin=-1,
            vmax=1,
            annotate_range=False,
        )
        _imshow(
            axes[row, 2],
            open_mismatch,
            title=f"Open sign mismatch grad_{axis_name}",
            cmap=mask_cmap,
            vmin=0,
            vmax=1,
            annotate_range=False,
        )
        if nn_sign is not None:
            nn_pred_sign = _as_hw_np(nn_sign[axis_name]["pred_sign"])
            nn_mismatch = _as_hw_np(nn_sign[axis_name]["mismatch"].to(torch.float32))
            _imshow(
                axes[row, 3],
                nn_pred_sign,
                title=f"NN pred sign grad_{axis_name}",
                cmap=sign_cmap,
                vmin=-1,
                vmax=1,
                annotate_range=False,
            )
            _imshow(
                axes[row, 4],
                nn_mismatch,
                title=f"NN sign mismatch grad_{axis_name}",
                cmap=mask_cmap,
                vmin=0,
                vmax=1,
                annotate_range=False,
            )
        else:
            _imshow(axes[row, 3], np.zeros_like(true_sign), title=f"NN pred sign grad_{axis_name}\nnot provided", cmap=mask_cmap, vmin=0, vmax=1, annotate_range=False)
            _imshow(axes[row, 4], np.zeros_like(true_sign), title=f"NN sign mismatch grad_{axis_name}\nnot provided", cmap=mask_cmap, vmin=0, vmax=1, annotate_range=False)

    _imshow(axes[4, 0], _as_hw_np(defect_region.to(torch.float32)), title="Defect support mask", cmap=mask_cmap, vmin=0, vmax=1, annotate_range=False)
    _imshow(axes[4, 1], _as_hw_np(open_sign["x"]["branch_flip"].to(torch.float32)), title="True branch flip grad_x", cmap=mask_cmap, vmin=0, vmax=1, annotate_range=False)
    _imshow(axes[4, 2], _as_hw_np(open_sign["y"]["branch_flip"].to(torch.float32)), title="True branch flip grad_y", cmap=mask_cmap, vmin=0, vmax=1, annotate_range=False)
    if ideal_diff is not None:
        leak_x = obs_ix - ideal_diff["I_x"]
        leak_y = obs_iy - ideal_diff["I_y"]
        v_leak = _symmetric_limit(leak_x, leak_y)
        _imshow(axes[4, 3], leak_x, title="nonideal - ideal dIx", cmap="coolwarm", vmin=-v_leak, vmax=v_leak)
        _imshow(axes[4, 4], leak_y, title="nonideal - ideal dIy", cmap="coolwarm", vmin=-v_leak, vmax=v_leak)
    else:
        _imshow(axes[4, 3], np.abs(open_err), title="|Open defect error|", cmap="magma")
        _imshow(axes[4, 4], np.abs(nn_err), title="|NN defect error|", cmap="magma")

    fig.suptitle(sample_label, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_dic_fit_panel(
    *,
    output_path: Path,
    sample_label: str,
    batch: dict[str, torch.Tensor],
    true_diff_pred: dict[str, torch.Tensor],
    open_diff_pred: dict[str, torch.Tensor],
    nn_diff_pred: dict[str, torch.Tensor] | None,
) -> None:
    obs = {
        "I_x": batch["inputs"][:, 0],
        "I_y": batch["inputs"][:, 1],
    }
    fig, axes = plt.subplots(3, 5, figsize=(17, 10), constrained_layout=True)
    row_defs = [
        ("I_x", "dIx"),
        ("I_y", "dIy"),
        ("mag", "|dI|"),
    ]
    obs_mag = torch.sqrt(obs["I_x"] ** 2 + obs["I_y"] ** 2)
    true_mag = torch.sqrt(true_diff_pred["I_x"] ** 2 + true_diff_pred["I_y"] ** 2)
    open_mag = torch.sqrt(open_diff_pred["I_x"] ** 2 + open_diff_pred["I_y"] ** 2)
    if nn_diff_pred is not None:
        nn_mag = torch.sqrt(nn_diff_pred["I_x"] ** 2 + nn_diff_pred["I_y"] ** 2)
    else:
        nn_mag = torch.full_like(open_mag, float("nan"))

    tensors = {
        "I_x": (obs["I_x"], true_diff_pred["I_x"], open_diff_pred["I_x"], nn_diff_pred["I_x"] if nn_diff_pred is not None else None),
        "I_y": (obs["I_y"], true_diff_pred["I_y"], open_diff_pred["I_y"], nn_diff_pred["I_y"] if nn_diff_pred is not None else None),
        "mag": (obs_mag, true_mag, open_mag, nn_mag),
    }
    for row, (key, label) in enumerate(row_defs):
        obs_arr = _as_hw_np(tensors[key][0])
        true_arr = _as_hw_np(tensors[key][1])
        open_arr = _as_hw_np(tensors[key][2])
        nn_arr = _as_hw_np(tensors[key][3]) if tensors[key][3] is not None else np.full_like(obs_arr, np.nan)
        residual = nn_arr - obs_arr if tensors[key][3] is not None else np.full_like(obs_arr, np.nan)
        if key == "mag":
            vmax = max(_symmetric_limit(obs_arr, true_arr, open_arr, nn_arr), 1e-12)
            cmap = "magma"
            vmin = 0.0
            resid_v = _symmetric_limit(residual)
            resid_cmap = "RdBu_r"
        else:
            vmax = _symmetric_limit(obs_arr, true_arr, open_arr, nn_arr)
            cmap = "coolwarm"
            vmin = -vmax
            resid_v = _symmetric_limit(residual)
            resid_cmap = "RdBu_r"
        _imshow(axes[row, 0], obs_arr, title=f"Observed nonideal {label}", cmap=cmap, vmin=vmin, vmax=vmax)
        _imshow(axes[row, 1], true_arr, title=f"Ideal f(true defect) {label}", cmap=cmap, vmin=vmin, vmax=vmax)
        _imshow(axes[row, 2], open_arr, title=f"Ideal f(open pred) {label}", cmap=cmap, vmin=vmin, vmax=vmax)
        _imshow(axes[row, 3], nn_arr, title=f"Ideal f(NN pred) {label}", cmap=cmap, vmin=vmin, vmax=vmax)
        _imshow(axes[row, 4], residual, title=f"NN ideal-fit residual {label}", cmap=resid_cmap, vmin=-resid_v, vmax=resid_v)
    fig.suptitle(f"{sample_label} DIC fit through ideal forward model", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _sample_metrics(
    cfg: ExperimentConfig,
    *,
    batch: dict[str, torch.Tensor],
    open_pred: torch.Tensor,
    nn_pred: torch.Tensor | None,
    open_sign: dict[str, Any],
    nn_sign: dict[str, Any] | None,
    defect_region: torch.Tensor,
    ideal_diff: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    defect_true = batch["defect"][0]
    result: dict[str, Any] = {
        "defect_peak_abs": float(torch.max(torch.abs(defect_true)).detach().cpu().item()),
        "defect_support_px": int(torch.sum(defect_region).detach().cpu().item()),
        "open": {
            "defect_rmse_local": _safe_rmse(open_pred[0], defect_true, defect_region),
            "defect_corr_local": _safe_corr(open_pred[0], defect_true, defect_region),
            "sign": {axis: open_sign[axis]["stats"] for axis in ("x", "y")},
        },
        "nn": None,
        "nonideal_vs_ideal_observation": None,
    }
    if nn_pred is not None and nn_sign is not None:
        result["nn"] = {
            "defect_rmse_local": _safe_rmse(nn_pred[0], defect_true, defect_region),
            "defect_corr_local": _safe_corr(nn_pred[0], defect_true, defect_region),
            "sign": {axis: nn_sign[axis]["stats"] for axis in ("x", "y")},
        }
    if ideal_diff is not None:
        obs_ix = _as_hw_np(batch["inputs"][0, 0])
        obs_iy = _as_hw_np(batch["inputs"][0, 1])
        dx = obs_ix - ideal_diff["I_x"]
        dy = obs_iy - ideal_diff["I_y"]
        denom = np.sqrt(ideal_diff["I_x"] ** 2 + ideal_diff["I_y"] ** 2)
        numer = np.sqrt(dx**2 + dy**2)
        result["nonideal_vs_ideal_observation"] = {
            "mean_abs_delta": float(np.mean(numer)),
            "p95_abs_delta": float(np.quantile(numer, 0.95)),
            "relative_mean_delta": float(np.mean(numer) / max(float(np.mean(denom)), 1e-12)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize no-raw DIC physical leakage samples.")
    parser.add_argument("--config", default="mini_grin_rebuild/configs/nonideal_dic_no_raw_smoke64.json")
    parser.add_argument("--data-root", default="/tmp/no_raw_dic_leakage_ablation/data/nonideal_dic_no_raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--ideal-data-root", default=None)
    parser.add_argument("--out-dir", default="/tmp/no_raw_dic_leakage_ablation/visualizations")
    parser.add_argument("--sample-indices", default=None, help="Comma-separated dataset indices. If omitted, branch-flip-heavy samples are selected.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--open-method", choices=sorted(OPEN_METHODS), default="first_order_sign_quadratic_poisson")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    ideal_data_root = Path(args.ideal_data_root).expanduser().resolve() if args.ideal_data_root else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None

    cfg = load_experiment_config(cfg_path)
    device = _select_device(args.device or cfg.training.device)
    dataset = DefectDataset(data_root, args.split)
    sample_indices = _parse_indices(args.sample_indices)
    if sample_indices is None:
        sample_indices = _rank_sample_indices(dataset, cfg, device=device, top_k=int(args.top_k))

    open_physics = create_forward_model(cfg.simulation, cfg.training, device=device, freeze=True)
    open_physics.eval()
    nn_model = None
    nn_physics = None
    nn_meta: dict[str, Any] | None = None
    if checkpoint_path is not None:
        nn_model, nn_physics, nn_meta = _load_nn(cfg, checkpoint_path, device=device)

    summary: dict[str, Any] = {
        "config": str(cfg_path),
        "data_root": str(data_root),
        "split": str(args.split),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "ideal_data_root": str(ideal_data_root) if ideal_data_root is not None else None,
        "open_method": str(args.open_method),
        "selected_indices": [int(i) for i in sample_indices],
        "samples": [],
    }

    with torch.no_grad():
        for idx in sample_indices:
            if idx < 0 or idx >= len(dataset):
                raise IndexError(f"sample index {idx} out of range for dataset length {len(dataset)}")
            sample = dataset[idx]
            batch = _as_batch(sample, device)
            defect_true = batch["defect"][0]
            h, w = int(defect_true.shape[-2]), int(defect_true.shape[-1])
            regions = _region_masks(cfg, h=h, w=w, device=device)
            defect_region = defect_mask(
                defect_true,
                abs_threshold=float(cfg.training.eval_defect_abs_threshold),
                rel_threshold=float(cfg.training.eval_defect_rel_threshold),
                dilate_px=int(cfg.training.eval_defect_dilate_px),
            ).to(device)

            open_pred = _predict_open(cfg, physics=open_physics, batch=batch, method=args.open_method)
            nn_pred = None
            if nn_model is not None and nn_physics is not None and nn_meta is not None:
                nn_pred = _predict_nn(cfg, model=nn_model, physics=nn_physics, model_meta=nn_meta, batch=batch)

            open_sign = _sign_context(
                cfg,
                standard=batch["standard"][0],
                defect_true=defect_true,
                defect_pred=open_pred[0],
                defect_region=defect_region,
                aperture_region=regions["aperture"],
            )
            nn_sign = None
            if nn_pred is not None:
                nn_sign = _sign_context(
                    cfg,
                    standard=batch["standard"][0],
                    defect_true=defect_true,
                    defect_pred=nn_pred[0],
                    defect_region=defect_region,
                    aperture_region=regions["aperture"],
                )

            sample_file = dataset.files[idx]
            ideal_diff = _load_ideal_diff(ideal_data_root, split=args.split, sample_file=sample_file)
            label = f"{args.split} sample {idx:04d} ({sample_file.name})"
            prediction_path = out_dir / f"sample_{idx:04d}_prediction_panel.png"
            dic_fit_path = out_dir / f"sample_{idx:04d}_dic_fit_panel.png"
            _plot_prediction_panel(
                output_path=prediction_path,
                sample_label=label,
                batch=batch,
                open_pred=open_pred,
                nn_pred=nn_pred,
                open_sign=open_sign,
                nn_sign=nn_sign,
                defect_region=defect_region,
                ideal_diff=ideal_diff,
            )

            true_diff_pred = _predict_diff(physics=open_physics, standard=batch["standard"], defect=batch["defect"])
            open_diff_pred = _predict_diff(physics=open_physics, standard=batch["standard"], defect=open_pred)
            nn_diff_pred = _predict_diff(physics=open_physics, standard=batch["standard"], defect=nn_pred) if nn_pred is not None else None
            _plot_dic_fit_panel(
                output_path=dic_fit_path,
                sample_label=label,
                batch=batch,
                true_diff_pred=true_diff_pred,
                open_diff_pred=open_diff_pred,
                nn_diff_pred=nn_diff_pred,
            )

            item = {
                "index": int(idx),
                "file": str(sample_file),
                "prediction_panel": str(prediction_path),
                "dic_fit_panel": str(dic_fit_path),
                "metrics": _sample_metrics(
                    cfg,
                    batch=batch,
                    open_pred=open_pred,
                    nn_pred=nn_pred,
                    open_sign=open_sign,
                    nn_sign=nn_sign,
                    defect_region=defect_region,
                    ideal_diff=ideal_diff,
                ),
            }
            summary["samples"].append(item)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "visualization_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "selected_indices": summary["selected_indices"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
