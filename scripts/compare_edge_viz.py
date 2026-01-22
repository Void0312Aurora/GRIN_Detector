from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


_bootstrap_src()

import argparse
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

from mini_grin_rebuild.core.configs import ExperimentConfig, load_experiment_config
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.core.runs import create_run
from mini_grin_rebuild.data.datasets import DefectDataset
from mini_grin_rebuild.evaluation.metrics import defect_mask, masked_abs_quantile, masked_mean_abs
from mini_grin_rebuild.models.checkpoint import infer_checkpoint_info, load_checkpoint
from mini_grin_rebuild.models.unetpp import UNetPP
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer
from mini_grin_rebuild.reconstruction import reconstruct_defect_oracle_poisson, reconstruct_defect_pseudo_poisson
from mini_grin_rebuild.training.inputs import append_coord_channels, build_inputs


def _infer_project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


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


def _wrap_height(cfg) -> float:
    import math

    phase_scale = (2.0 * math.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air)
    return float((math.pi * cfg.wrap_safety) / max(phase_scale, 1e-12))


def _radial_grid(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lin_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    lin_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing="ij")
    return torch.sqrt(xx**2 + yy**2)


def _edge_artifact_mask(
    cfg: ExperimentConfig,
    *,
    defect_true_hw: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (edge_mask, edge_art_mask) for [H,W] tensors.

    edge_mask: edge band inside aperture.
    edge_art_mask: edge_mask excluding the defect support mask (based on ground truth).
    """
    if defect_true_hw.ndim != 2:
        raise ValueError(f"Expected defect_true_hw [H,W], got {tuple(defect_true_hw.shape)}")
    h, w = int(defect_true_hw.shape[-2]), int(defect_true_hw.shape[-1])
    rr = _radial_grid(h, w, device=device, dtype=torch.float32)

    aperture_r = float(getattr(cfg.simulation, "lens_radius_fraction", 1.0) or 1.0)
    edge_start_frac = float(getattr(cfg.training, "eval_edge_band_start_frac", 0.9))
    edge_start_frac = max(0.0, min(edge_start_frac, 1.0))

    aperture = rr <= aperture_r
    edge = (rr > (edge_start_frac * aperture_r)) & aperture

    m = defect_mask(
        defect_true_hw,
        abs_threshold=float(cfg.training.eval_defect_abs_threshold),
        rel_threshold=float(cfg.training.eval_defect_rel_threshold),
        dilate_px=int(cfg.training.eval_defect_dilate_px),
    ).to(device=device)
    edge_art = edge & (~m.to(dtype=torch.bool))
    return edge.to(device=device), edge_art.to(device=device)


def _load_nn(checkpoint_path: Path, *, device: torch.device) -> tuple[UNetPP, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    state = ckpt["model"]
    info = infer_checkpoint_info(state)
    meta = ckpt.get("model_meta", {}) if isinstance(ckpt, dict) else {}

    use_prior = bool(meta.get("use_pseudo_poisson_prior", False))
    residual_scale = meta.get("pseudo_poisson_residual_scale", None)
    output_scale = None
    if use_prior and residual_scale is not None:
        output_scale = float(residual_scale)

    padding_mode = str(meta.get("model_padding_mode", "zeros"))

    model = UNetPP(
        in_channels=info.in_channels,
        out_channels=1,
        predict_logvar=info.predict_logvar,
        padding_mode=padding_mode,
        output_scale=output_scale,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, meta


def _forward_model(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    out = model(inputs)
    if isinstance(out, dict):
        return out["defect"]
    return out


def _build_nn_inputs(
    cfg: ExperimentConfig,
    physics: DifferentiableGradientLayer,
    *,
    batch: dict[str, torch.Tensor],
    prior_defect: torch.Tensor | None,
    prior_as_input: bool,
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

    if prior_as_input:
        if prior_defect is None:
            raise RuntimeError("prior_as_input=True but prior_defect is None")
        wrap_h = _wrap_height(cfg.simulation)
        prior_chan = (prior_defect / max(wrap_h, 1e-12)).clamp(min=-1.0, max=1.0)
        inputs = torch.cat([inputs, prior_chan], dim=1)
    return inputs


def _predict_methods(
    cfg: ExperimentConfig,
    *,
    physics: DifferentiableGradientLayer,
    sample: dict[str, torch.Tensor],
    device: torch.device,
    nn_models: list[tuple[str, torch.nn.Module, dict[str, Any]]],
    include_oracle: bool,
) -> dict[str, dict[str, Any]]:
    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    diff_ts = {"I_x": batch["inputs"][:, 0], "I_y": batch["inputs"][:, 1]}

    preds: dict[str, dict[str, Any]] = {}

    defect_true = batch["defect"]
    standard = batch["standard"]

    pseudo = reconstruct_defect_pseudo_poisson(
        physics=physics,
        standard_height=standard,
        diff_ts=diff_ts,
        defect_roi_radius=float(cfg.training.defect_roi_radius),
        apply_edge_offset=True,
    )
    preds["pseudo_poisson"] = {"defect": pseudo}

    if include_oracle:
        oracle = reconstruct_defect_oracle_poisson(
            physics=physics,
            standard_height=standard,
            defect_true=defect_true,
            diff_ts=diff_ts,
            defect_roi_radius=float(cfg.training.defect_roi_radius),
            apply_edge_offset=True,
        )
        preds["oracle_poisson"] = {"defect": oracle}

    for name, model, meta in nn_models:
        use_prior = bool(meta.get("use_pseudo_poisson_prior", False))
        prior_as_input = bool(meta.get("pseudo_poisson_prior_as_input", False))
        prior_scale = float(meta.get("pseudo_poisson_prior_scale", 1.0))

        prior_defect = None
        if use_prior:
            with torch.no_grad():
                prior_defect = reconstruct_defect_pseudo_poisson(
                    physics=physics,
                    standard_height=standard,
                    diff_ts=diff_ts,
                    defect_roi_radius=float(cfg.training.defect_roi_radius),
                    apply_edge_offset=True,
                )

        inputs = _build_nn_inputs(
            cfg,
            physics,
            batch=batch,
            prior_defect=prior_defect,
            prior_as_input=(use_prior and prior_as_input),
        )
        with torch.no_grad():
            defect_pred = _forward_model(model, inputs)
            if use_prior:
                if prior_defect is None:
                    raise RuntimeError("use_prior=True but prior_defect is None")
                defect_pred = prior_defect * prior_scale + defect_pred
        preds[name] = {"defect": defect_pred}

    return preds


def _diff_from_defect(
    physics: DifferentiableGradientLayer,
    *,
    standard: torch.Tensor,
    defect: torch.Tensor,
) -> dict[str, torch.Tensor]:
    height = standard + defect
    phys_out = physics(height)
    std_phys = physics(standard)
    return {
        "I_x": (phys_out["I_x"] - std_phys["I_x"]).squeeze(0).squeeze(0),
        "I_y": (phys_out["I_y"] - std_phys["I_y"]).squeeze(0).squeeze(0),
    }


def _plot_compare(
    *,
    cfg: ExperimentConfig,
    sample_idx: int,
    defect_true: np.ndarray,
    diff_true: dict[str, np.ndarray],
    methods: list[tuple[str, np.ndarray, dict[str, np.ndarray], dict[str, float]]],
    edge_mask: np.ndarray,
    edge_art_mask: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["True"] + [m[0] for m in methods]
    ncols = len(names)
    nrows = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), constrained_layout=True)

    vmax = float(np.max(np.abs(defect_true))) if defect_true.size else 1.0
    if vmax == 0:
        vmax = 1e-6

    # Row 0: defect true + preds
    axes[0, 0].imshow(defect_true, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title("Defect True")
    for col, (name, defect_pred, _diff_pred, stats) in enumerate(methods, start=1):
        axes[0, col].imshow(defect_pred, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(f"{name}\nedge_abs={stats['edge_mean_abs']:.3g}")

    # Row 1: defect error (skip true col)
    axes[1, 0].axis("off")
    for col, (_name, defect_pred, _diff_pred, _stats) in enumerate(methods, start=1):
        axes[1, col].imshow(defect_pred - defect_true, cmap="RdBu", vmin=-vmax, vmax=vmax)
        axes[1, col].set_title("Defect Error")

    # Row 2: edge artifact map
    axes[2, 0].imshow(edge_mask.astype(float), cmap="gray", vmin=0.0, vmax=1.0)
    axes[2, 0].set_title("Edge Band Mask")
    for col, (_name, defect_pred, _diff_pred, stats) in enumerate(methods, start=1):
        art = np.abs(defect_pred) * edge_art_mask.astype(float)
        v = float(np.max(art)) if art.size else 1.0
        if v == 0:
            v = 1e-6
        axes[2, col].imshow(art, cmap="viridis", vmin=0.0, vmax=v)
        axes[2, col].set_title(f"Edge Art\np95={stats['edge_p95_abs']:.3g}")

    # Row 3/4: |diff error|
    for row, key in enumerate(("I_x", "I_y"), start=3):
        axes[row, 0].imshow(diff_true[key], cmap="coolwarm")
        axes[row, 0].set_title(f"{key} True")
        for col, (_name, _defect_pred, diff_pred, _stats) in enumerate(methods, start=1):
            err = np.abs(diff_pred[key] - diff_true[key])
            v = float(np.max(err)) if err.size else 1.0
            if v == 0:
                v = 1e-6
            axes[row, col].imshow(err, cmap="viridis", vmin=0.0, vmax=v)
            axes[row, col].set_title(f"{key} |Error|")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"compare_edge_viz sample={sample_idx}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _parse_ckpt_specs(values: Iterable[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for v in values:
        if "=" not in v:
            raise SystemExit(f"--ckpt must be NAME=PATH, got {v!r}")
        name, path = v.split("=", 1)
        name = name.strip()
        path = Path(path).expanduser()
        if not name:
            raise SystemExit(f"Invalid --ckpt spec (empty name): {v!r}")
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        out.append((name, path))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Generate side-by-side edge-artifact comparison plots.")
    p.add_argument("--config", required=True, help="Experiment config JSON (base).")
    p.add_argument("--data-root", required=True, help="Dataset root.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize (selected by baseline edge artifacts).")
    p.add_argument("--include-oracle", action="store_true", help="Also include oracle_poisson baseline.")
    p.add_argument("--ckpt", action="append", default=[], help="NN checkpoint spec: NAME=PATH (repeatable).")
    p.add_argument("--device", default=None, help="Override device (e.g. cpu/cuda).")
    p.add_argument("--name", default="compare_edge_viz", help="Run name suffix.")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = load_experiment_config(cfg_path)
    project_root = _infer_project_root(cfg_path)
    runs_root = project_root / cfg.paths.runs_dir
    run = create_run(
        runs_root,
        name=str(args.name),
        argv=None,
        config_snapshot=cfg.to_dict(),
    )

    dataset_root = Path(args.data_root).expanduser()
    ds = DefectDataset(dataset_root, str(args.split))

    device = _select_device(args.device or cfg.training.device)
    physics = DifferentiableGradientLayer(cfg.simulation).to(device)
    _freeze_physics_to_ideal(physics)

    ckpts = _parse_ckpt_specs(args.ckpt)
    nn_models: list[tuple[str, torch.nn.Module, dict[str, Any]]] = []
    for name, path in ckpts:
        model, meta = _load_nn(path, device=device)
        nn_models.append((name, model, meta))

    # Score samples by baseline pseudo-poisson edge artifacts.
    scores: list[tuple[float, int]] = []
    for idx in range(len(ds)):
        sample = ds[idx]
        batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
        diff_ts = {"I_x": batch["inputs"][:, 0], "I_y": batch["inputs"][:, 1]}
        defect_true = batch["defect"][0, 0]
        edge_mask, edge_art_mask = _edge_artifact_mask(cfg, defect_true_hw=defect_true, device=device)
        with torch.no_grad():
            pseudo = reconstruct_defect_pseudo_poisson(
                physics=physics,
                standard_height=batch["standard"],
                diff_ts=diff_ts,
                defect_roi_radius=float(cfg.training.defect_roi_radius),
                apply_edge_offset=True,
            )[0, 0]
            edge_mean = float(masked_mean_abs(torch.abs(pseudo), edge_art_mask).cpu().item())
        scores.append((edge_mean, idx))

    scores.sort(reverse=True)
    chosen = [idx for _s, idx in scores[: max(1, int(args.num_samples))]]

    out_plots = run.root / "compare_plots"
    out_plots.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for idx in chosen:
        sample = ds[idx]
        sample_cpu = {k: (v.clone().detach().cpu() if torch.is_tensor(v) else v) for k, v in sample.items()}
        defect_true = sample_cpu["defect"].squeeze(0).numpy()
        diff_true = {
            "I_x": sample_cpu["inputs"][0].numpy(),
            "I_y": sample_cpu["inputs"][1].numpy(),
        }

        preds = _predict_methods(
            cfg,
            physics=physics,
            sample=sample,
            device=device,
            nn_models=nn_models,
            include_oracle=bool(args.include_oracle),
        )

        std_b = sample["standard"].unsqueeze(0).to(device)
        dt_hw = sample["defect"][0].to(device)
        edge_mask, edge_art_mask = _edge_artifact_mask(cfg, defect_true_hw=dt_hw, device=device)
        edge_mask_np = edge_mask.detach().cpu().numpy()
        edge_art_np = edge_art_mask.detach().cpu().numpy()

        methods: list[tuple[str, np.ndarray, dict[str, np.ndarray], dict[str, float]]] = []
        for name, pack in preds.items():
            dp = pack["defect"][0, 0]
            diff_pred_t = _diff_from_defect(physics, standard=std_b, defect=pack["defect"])
            stats = {
                "edge_mean_abs": float(masked_mean_abs(torch.abs(dp), edge_art_mask).detach().cpu().item()),
                "edge_p95_abs": float(masked_abs_quantile(torch.abs(dp), edge_art_mask, q=0.95).detach().cpu().item()),
            }
            methods.append(
                (
                    name,
                    dp.detach().cpu().numpy(),
                    {k: v.detach().cpu().numpy() for k, v in diff_pred_t.items()},
                    stats,
                )
            )
            summary_rows.append({"idx": int(idx), "method": name, **stats})

        # Keep stable column order: baselines first, then NNs.
        methods.sort(key=lambda t: (0 if t[0].startswith("pseudo") else 1 if t[0].startswith("oracle") else 2, t[0]))

        _plot_compare(
            cfg=cfg,
            sample_idx=int(idx),
            defect_true=defect_true,
            diff_true=diff_true,
            methods=methods,
            edge_mask=edge_mask_np,
            edge_art_mask=edge_art_np,
            output_path=out_plots / f"sample_{idx:04d}.png",
        )

    write_json(
        run.root / "compare_summary.json",
        {
            "config": str(cfg_path),
            "data_root": str(dataset_root),
            "split": str(args.split),
            "selected_indices": chosen,
            "selection": {"by": "pseudo_poisson.edge_mean_abs", "top_k": int(args.num_samples)},
            "rows": summary_rows,
        },
    )
    write_json(
        run.root / "artifacts.json",
        {
            "dataset_root": str(dataset_root),
            "split": str(args.split),
            "ckpts": [{k: str(v) for k, v in {"name": n, "path": str(p)}.items()} for n, p in ckpts],
            "include_oracle": bool(args.include_oracle),
            "plots_dir": str(out_plots),
        },
    )

    print(str(run.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
