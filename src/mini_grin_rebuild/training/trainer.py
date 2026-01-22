from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from mini_grin_rebuild.core.configs import ExperimentConfig, SimulationConfig, TrainingConfig
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.core.seed import set_global_seed
from mini_grin_rebuild.core.runs import RunPaths
from mini_grin_rebuild.data.datasets import DefectDataset
from mini_grin_rebuild.models.checkpoint import save_checkpoint
from mini_grin_rebuild.models.unetpp import UNetPP
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer
from mini_grin_rebuild.reconstruction import reconstruct_defect_pseudo_poisson
from mini_grin_rebuild.training.inputs import build_inputs
from mini_grin_rebuild.training.losses import (
    curl_loss,
    diff_loss,
    edge_band_suppress_loss,
    edge_suppress_loss,
    sparsity_loss,
    sr_diff_loss,
    total_loss,
)
from mini_grin_rebuild.training.inputs import append_coord_channels


def _select_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _freeze_physics_to_ideal(physics: DifferentiableGradientLayer) -> None:
    with torch.no_grad():
        physics.log_sigma.fill_(-10.0)
        physics.log_gain.fill_(0.0)
        physics.bias.fill_(0.0)
        physics.shifts.fill_(0.0)
        physics.lfields.fill_(0.0)
    for p in physics.parameters():
        p.requires_grad_(False)


def _wrap_height(cfg: SimulationConfig) -> float:
    phase_scale = (2.0 * math.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air)
    return float((math.pi * cfg.wrap_safety) / max(phase_scale, 1e-12))


def _forward_model(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
    out = model(inputs)
    if isinstance(out, dict):
        return out["defect"], out.get("logvar")
    return out, None


def _crop_batch(t: torch.Tensor, y0: torch.Tensor, x0: torch.Tensor, size: int) -> torch.Tensor:
    if t.ndim == 3:
        t = t.unsqueeze(1)
        squeezed = True
    else:
        squeezed = False
    if t.ndim != 4:
        raise ValueError(f"Expected tensor [B,C,H,W] or [B,H,W], got {tuple(t.shape)}")
    b, c, h, w = t.shape
    if size <= 0 or size > h or size > w:
        raise ValueError(f"Invalid crop_size={size} for shape {tuple(t.shape)}")
    crops = []
    for i in range(b):
        yi = int(y0[i].item())
        xi = int(x0[i].item())
        crops.append(t[i : i + 1, :, yi : yi + size, xi : xi + size])
    out = torch.cat(crops, dim=0)
    if squeezed:
        out = out.squeeze(1)
    return out


def _choose_crop_anchors(diff_ts: Dict[str, torch.Tensor], *, crop_size: int, strategy: str) -> Tuple[torch.Tensor, torch.Tensor]:
    tx = diff_ts["I_x"]
    ty = diff_ts["I_y"]
    if tx.ndim != 3 or ty.ndim != 3:
        raise ValueError(f"Expected diff_ts tensors [B,H,W], got {tuple(tx.shape)} and {tuple(ty.shape)}")
    if tx.shape != ty.shape:
        raise ValueError(f"diff_ts shape mismatch: {tuple(tx.shape)} vs {tuple(ty.shape)}")
    b, h, w = tx.shape
    if crop_size <= 0 or crop_size > h or crop_size > w:
        raise ValueError(f"Invalid crop_size={crop_size} for diff_ts shape {tuple(tx.shape)}")

    if strategy == "center":
        y = torch.full((b,), int((h - crop_size) // 2), device=tx.device, dtype=torch.long)
        x = torch.full((b,), int((w - crop_size) // 2), device=tx.device, dtype=torch.long)
        return y, x
    if strategy != "activity":
        raise ValueError(f"Unknown crop_strategy={strategy!r} (expected 'activity' or 'center')")

    activity = torch.abs(tx) + torch.abs(ty)  # [B,H,W]
    flat = activity.view(b, -1)
    idx = torch.argmax(flat, dim=1)
    y_peak = idx // w
    x_peak = idx % w
    y0 = torch.clamp(y_peak - crop_size // 2, min=0, max=h - crop_size)
    x0 = torch.clamp(x_peak - crop_size // 2, min=0, max=w - crop_size)
    return y0.to(dtype=torch.long), x0.to(dtype=torch.long)


def _teacher_loss(pred: torch.Tensor, target: torch.Tensor, *, kind: str) -> torch.Tensor:
    if kind == "l1":
        return torch.mean(torch.abs(pred - target))
    if kind == "l2":
        return torch.mean((pred - target) ** 2)
    raise ValueError(f"Unknown teacher_loss_type={kind!r} (expected 'l1' or 'l2')")


def _prepare_batch_inputs(
    batch: Dict[str, torch.Tensor],
    physics: DifferentiableGradientLayer,
    cfg: TrainingConfig,
) -> Tuple[torch.Tensor, dict]:
    diff_ts = {"I_x": batch["inputs"][:, 0], "I_y": batch["inputs"][:, 1]}

    diff_sr = None
    if "inputs_sr" in batch:
        diff_sr = {"I_x": batch["inputs_sr"][:, 0], "I_y": batch["inputs_sr"][:, 1]}

    intensity_inputs = {}
    for name in ("standard", "reference", "test"):
        key = f"intensity_{name}"
        if key in batch:
            intensity_inputs[name] = {"I_x": batch[key][:, 0], "I_y": batch[key][:, 1]}

    standard = batch["standard"]
    reference = batch.get("reference")

    pseudo_poisson_defect = None
    need_pseudo_poisson = cfg.use_pseudo_poisson_prior or cfg.teacher_loss_weight > 0
    if need_pseudo_poisson:
        with torch.no_grad():
            pseudo_poisson_defect = reconstruct_defect_pseudo_poisson(
                physics=physics,
                standard_height=standard,
                diff_ts=diff_ts,
                defect_roi_radius=float(cfg.defect_roi_radius),
                apply_edge_offset=True,
            )

    crop_size = int(getattr(cfg, "crop_size", 0) or 0)
    if crop_size > 0:
        y0, x0 = _choose_crop_anchors(diff_ts, crop_size=crop_size, strategy=str(getattr(cfg, "crop_strategy", "activity")))
        standard = _crop_batch(standard, y0, x0, crop_size)
        diff_ts = {k: _crop_batch(v, y0, x0, crop_size) for k, v in diff_ts.items()}
        if diff_sr is not None:
            diff_sr = {k: _crop_batch(v, y0, x0, crop_size) for k, v in diff_sr.items()}
        if reference is not None:
            reference = _crop_batch(reference, y0, x0, crop_size)
        if intensity_inputs:
            intensity_inputs = {
                name: {k: _crop_batch(v, y0, x0, crop_size) for k, v in tens.items()}
                for name, tens in intensity_inputs.items()
            }
        if pseudo_poisson_defect is not None:
            pseudo_poisson_defect = _crop_batch(pseudo_poisson_defect, y0, x0, crop_size)

    inputs = build_inputs(
        cfg,
        diff_ts,
        physics,
        standard,
        diff_sr=diff_sr,
        intensity_inputs=intensity_inputs if intensity_inputs else None,
    )
    inputs = inputs * float(cfg.input_scale)
    if bool(getattr(cfg, "use_coord_inputs", False)):
        ap = float(getattr(physics.cfg, "lens_radius_fraction", 1.0) or 1.0)
        inputs = append_coord_channels(inputs, aperture_radius=ap)

    if cfg.use_pseudo_poisson_prior and cfg.pseudo_poisson_prior_as_input:
        if pseudo_poisson_defect is None:
            raise RuntimeError("pseudo_poisson_defect missing while use_pseudo_poisson_prior=True")
        wrap_h = _wrap_height(physics.cfg)
        prior_chan = (pseudo_poisson_defect / max(wrap_h, 1e-12)).clamp(min=-1.0, max=1.0)
        inputs = torch.cat([inputs, prior_chan], dim=1)

    aux = {
        "standard": standard,
        "reference": reference,
        "diff_ts": diff_ts,
        "diff_sr": diff_sr,
        "pseudo_poisson_defect": pseudo_poisson_defect,
    }
    return inputs, aux


def _accum_add(acc: Dict[str, float], terms: Dict[str, torch.Tensor], batch_size: int) -> None:
    for k, v in terms.items():
        acc[k] = acc.get(k, 0.0) + float(v.detach().cpu().item()) * batch_size


def _finalize_mean(acc: Dict[str, float], n: int) -> Dict[str, float]:
    return {k: (v / max(1, n)) for k, v in acc.items()}


def _epoch_pass(
    model: torch.nn.Module,
    physics: DifferentiableGradientLayer,
    loader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    totals: Dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        inputs, aux = _prepare_batch_inputs(batch, physics, cfg)
        defect_pred, logvar = _forward_model(model, inputs)

        defect_for_loss = defect_pred
        if cfg.use_pseudo_poisson_prior:
            base = aux.get("pseudo_poisson_defect")
            if base is None:
                raise RuntimeError("use_pseudo_poisson_prior=True but pseudo_poisson_defect is missing")
            defect_for_loss = base * float(cfg.pseudo_poisson_prior_scale) + defect_pred

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        if not cfg.use_pseudo_poisson_prior:
            terms = total_loss(
                cfg,
                physics,
                standard_height=aux["standard"],
                defect=defect_for_loss,
                diff_ts_target=aux["diff_ts"],
                reference_height=aux.get("reference"),
                diff_sr_target=aux.get("diff_sr"),
                logvar=logvar,
            )
        else:
            terms = {}
            terms["diff"] = diff_loss(
                cfg,
                physics,
                standard_height=aux["standard"],
                defect=defect_for_loss,
                diff_ts_target=aux["diff_ts"],
                logvar=logvar,
            )
            if cfg.sr_diff_weight > 0 and aux.get("reference") is not None and aux.get("diff_sr") is not None:
                terms["sr_diff"] = cfg.sr_diff_weight * sr_diff_loss(
                    cfg,
                    physics,
                    standard_height=aux["standard"],
                    reference_height=aux["reference"],
                    diff_sr_target=aux["diff_sr"],
                )

            # Regularize the residual branch so we don't penalize the fixed teacher prior.
            if cfg.curl_weight > 0:
                terms["curl"] = cfg.curl_weight * curl_loss(defect_pred, physics)
            if cfg.sparsity_weight > 0:
                terms["sparsity"] = cfg.sparsity_weight * sparsity_loss(defect_pred)
            if cfg.edge_suppress_weight > 0:
                terms["edge_suppress"] = cfg.edge_suppress_weight * edge_suppress_loss(
                    defect_for_loss,
                    cfg.defect_roi_radius,
                )
            if getattr(cfg, "edge_band_suppress_weight", 0.0) > 0:
                ap = float(getattr(physics.cfg, "lens_radius_fraction", 1.0) or 1.0)
                edge_start = float(getattr(cfg, "eval_edge_band_start_frac", 0.9))
                terms["edge_band_suppress"] = float(getattr(cfg, "edge_band_suppress_weight")) * edge_band_suppress_loss(
                    defect_for_loss,
                    aperture_radius=ap,
                    edge_band_start_frac=edge_start,
                )

        if cfg.teacher_loss_weight > 0:
            base = aux.get("pseudo_poisson_defect")
            if base is None:
                raise RuntimeError("teacher_loss_weight>0 but pseudo_poisson_defect is missing")
            target = base
            if cfg.use_pseudo_poisson_prior:
                target = target * float(cfg.pseudo_poisson_prior_scale)
            terms["teacher"] = float(cfg.teacher_loss_weight) * _teacher_loss(
                defect_for_loss,
                target,
                kind=str(getattr(cfg, "teacher_loss_type", "l1")),
            )
        loss = sum(terms.values())
        terms = dict(terms)
        terms["total"] = loss

        if train_mode:
            loss.backward()
            optimizer.step()

        bs = int(inputs.shape[0])
        _accum_add(totals, terms, bs)
        n += bs

    return _finalize_mean(totals, n)


def _infer_in_channels(
    dataset: DefectDataset,
    sim_cfg: SimulationConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> int:
    sample = dataset[0]
    sample_batched = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in sample.items()}
    tmp_physics = DifferentiableGradientLayer(sim_cfg).to(device)
    _freeze_physics_to_ideal(tmp_physics)
    inputs, _ = _prepare_batch_inputs(
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample_batched.items()},
        tmp_physics,
        train_cfg,
    )
    return int(inputs.shape[1])


@dataclass(frozen=True)
class TrainOutputs:
    best_checkpoint: Path
    last_checkpoint: Path


def train_dataset(
    cfg: ExperimentConfig,
    *,
    data_root: str | Path,
    run: RunPaths,
    resume: str | Path | None = None,
) -> TrainOutputs:
    """
    Train on the legacy-compatible `.npz` dataset format.
    Writes `metrics.json` and checkpoints into the provided run directory.
    """
    set_global_seed(cfg.training.seed)
    device = _select_device(cfg.training.device)

    sim_cfg = cfg.simulation
    train_cfg = cfg.training

    train_ds = DefectDataset(Path(data_root), "train")
    val_ds = DefectDataset(Path(data_root), "val")
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)

    in_channels = _infer_in_channels(train_ds, sim_cfg, train_cfg, device)
    residual_scale = float(getattr(train_cfg, "pseudo_poisson_residual_scale", 0.0) or 0.0)
    if train_cfg.use_pseudo_poisson_prior and residual_scale <= 0:
        residual_scale = 0.25 * _wrap_height(sim_cfg)
    output_scale = residual_scale if (train_cfg.use_pseudo_poisson_prior and residual_scale > 0) else None

    model = UNetPP(
        in_channels=in_channels,
        out_channels=1,
        predict_logvar=train_cfg.predict_logvar,
        padding_mode=str(getattr(train_cfg, "model_padding_mode", "zeros")),
        output_scale=output_scale,
    ).to(device)

    physics = DifferentiableGradientLayer(sim_cfg).to(device)
    _freeze_physics_to_ideal(physics)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    start_epoch = 1
    best_val = float("inf")

    if resume is not None:
        ckpt = torch.load(Path(resume), map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))

    history = []
    last_path = run.checkpoints / "last.pt"
    best_path = run.checkpoints / "best.pt"

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        train_stats = _epoch_pass(model, physics, train_loader, train_cfg, device, optimizer=optimizer)
        with torch.no_grad():
            val_stats = _epoch_pass(model, physics, val_loader, train_cfg, device, optimizer=None)

        val_total = float(val_stats.get("total", 0.0))
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)

        payload = {
            "epoch": epoch,
            "best_val": best_val,
            "model": model.state_dict(),
            "physics": physics.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_meta": {
                "use_pseudo_poisson_prior": bool(train_cfg.use_pseudo_poisson_prior),
                "pseudo_poisson_prior_as_input": bool(train_cfg.pseudo_poisson_prior_as_input),
                "pseudo_poisson_prior_scale": float(train_cfg.pseudo_poisson_prior_scale),
                "pseudo_poisson_residual_scale": float(residual_scale),
                "crop_size": int(train_cfg.crop_size),
                "crop_strategy": str(train_cfg.crop_strategy),
                "teacher_loss_weight": float(train_cfg.teacher_loss_weight),
                "teacher_loss_type": str(train_cfg.teacher_loss_type),
                "model_padding_mode": str(getattr(train_cfg, "model_padding_mode", "zeros")),
            },
        }
        save_checkpoint(last_path, payload)
        val_total_cmp = val_total if math.isfinite(val_total) else float("inf")
        if (not best_path.exists()) or (val_total_cmp < best_val):
            if val_total_cmp < best_val:
                best_val = val_total_cmp
            payload["best_val"] = best_val
            save_checkpoint(best_path, payload)

        write_json(run.root / "train_metrics.json", {"history": history, "best_val_total": best_val})

        if epoch % train_cfg.log_interval == 0 or epoch == start_epoch:
            print(
                f"epoch={epoch} train_total={train_stats['total']:.6f} val_total={val_total:.6f}",
                file=sys.stderr,
            )

    return TrainOutputs(best_checkpoint=best_path, last_checkpoint=last_path)


__all__ = ["TrainOutputs", "train_dataset"]
