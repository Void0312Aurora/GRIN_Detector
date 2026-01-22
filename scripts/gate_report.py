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
from mini_grin_rebuild.evaluation.gating import (
    GateThresholds,
    artifact_metrics,
    gate_decision,
    logvar_metrics,
    physics_residual_metrics,
    radial_masks,
    suggest_thresholds,
)
from mini_grin_rebuild.models.checkpoint import infer_checkpoint_info, load_checkpoint
from mini_grin_rebuild.models.unetpp import UNetPP
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer
from mini_grin_rebuild.reconstruction import reconstruct_defect_oracle_poisson, reconstruct_defect_pseudo_poisson
from mini_grin_rebuild.training.inputs import append_coord_channels, build_inputs
from mini_grin_rebuild.visualization.plots import plot_defect_and_intensity


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


def _forward_model(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = model(inputs)
    if isinstance(out, dict):
        return out["defect"], out.get("logvar")
    return out, None


def _build_inputs(
    cfg: ExperimentConfig,
    physics: DifferentiableGradientLayer,
    *,
    sample: dict[str, torch.Tensor],
    prior_defect: torch.Tensor | None,
    prior_as_input: bool,
) -> torch.Tensor:
    train_cfg = cfg.training
    diff_ts = {"I_x": sample["inputs"][0], "I_y": sample["inputs"][1]}

    diff_sr = None
    if "inputs_sr" in sample:
        diff_sr = {"I_x": sample["inputs_sr"][0], "I_y": sample["inputs_sr"][1]}

    intensity_inputs = {}
    for name in ("standard", "reference", "test"):
        key = f"intensity_{name}"
        if key in sample:
            intensity_inputs[name] = {"I_x": sample[key][0], "I_y": sample[key][1]}

    standard = sample["standard"]
    inputs = build_inputs(
        train_cfg,
        diff_ts,
        physics,
        standard,
        diff_sr=diff_sr,
        intensity_inputs=intensity_inputs if intensity_inputs else None,
    )
    inputs = inputs * float(train_cfg.input_scale)  # [1,C,H,W]

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


def _load_nn(ckpt_path: Path, *, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
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


def _compute_for_method(
    cfg: ExperimentConfig,
    *,
    physics: DifferentiableGradientLayer,
    ds: DefectDataset,
    device: torch.device,
    method: str,
    nn: tuple[torch.nn.Module, dict[str, Any]] | None,
    thresholds: GateThresholds,
    num_plots: int,
    plots_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for idx in range(len(ds)):
        sample = ds[idx]
        sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}

        std = sample["standard"].unsqueeze(0).to(device)  # [1,1,H,W]
        defect_true = sample["defect"].unsqueeze(0).to(device)  # [1,1,H,W]
        diff_ts = {"I_x": sample["inputs"][0].to(device), "I_y": sample["inputs"][1].to(device)}  # [H,W]

        if method == "pseudo_poisson":
            pred = reconstruct_defect_pseudo_poisson(
                physics=physics,
                standard_height=std,
                diff_ts={"I_x": diff_ts["I_x"].unsqueeze(0), "I_y": diff_ts["I_y"].unsqueeze(0)},
                defect_roi_radius=float(cfg.training.defect_roi_radius),
                apply_edge_offset=True,
            )
            logvar_hw = None
        elif method == "oracle_poisson":
            pred = reconstruct_defect_oracle_poisson(
                physics=physics,
                standard_height=std,
                defect_true=defect_true,
                diff_ts={"I_x": diff_ts["I_x"].unsqueeze(0), "I_y": diff_ts["I_y"].unsqueeze(0)},
                defect_roi_radius=float(cfg.training.defect_roi_radius),
                apply_edge_offset=True,
            )
            logvar_hw = None
        else:
            if nn is None:
                raise RuntimeError("NN method requested but nn is None")
            model, meta = nn
            use_prior = bool(meta.get("use_pseudo_poisson_prior", False))
            prior_as_input = bool(meta.get("pseudo_poisson_prior_as_input", False))
            prior_scale = float(meta.get("pseudo_poisson_prior_scale", 1.0))
            prior_defect = None
            if use_prior:
                with torch.no_grad():
                    prior_defect = reconstruct_defect_pseudo_poisson(
                        physics=physics,
                        standard_height=std,
                        diff_ts={"I_x": diff_ts["I_x"].unsqueeze(0), "I_y": diff_ts["I_y"].unsqueeze(0)},
                        defect_roi_radius=float(cfg.training.defect_roi_radius),
                        apply_edge_offset=True,
                    )
            inputs = _build_inputs(cfg, physics, sample=sample, prior_defect=prior_defect, prior_as_input=(use_prior and prior_as_input))
            inputs = inputs.to(device)
            with torch.no_grad():
                residual, logvar = _forward_model(model, inputs)
                pred = residual
                if use_prior:
                    if prior_defect is None:
                        raise RuntimeError("use_prior=True but prior_defect is None")
                    pred = prior_defect * prior_scale + residual
                logvar_hw = None
                if logvar is not None:
                    logvar_hw = logvar.squeeze(0).squeeze(0)

        pred_hw = pred.squeeze(0).squeeze(0)
        h, w = int(pred_hw.shape[-2]), int(pred_hw.shape[-1])
        masks = radial_masks(cfg, h=h, w=w, device=device)

        m: dict[str, float] = {}
        m.update(
            physics_residual_metrics(
                cfg,
                physics=physics,
                standard_height=std,
                defect_pred=pred,
                diff_ts_target=diff_ts,
                aperture_mask=masks["aperture"],
            )
        )
        m.update(artifact_metrics(pred_hw, edge_mask=masks["edge"], outside_mask=masks["outside"]))
        if logvar_hw is not None:
            m.update(logvar_metrics(logvar_hw, aperture_mask=masks["aperture"]))

        dec = gate_decision(m, thresholds)
        rows.append({"idx": int(idx), "metrics": m, **dec})

    # summary
    def _col(name: str) -> list[float]:
        out = []
        for r in rows:
            v = r["metrics"].get(name, float("nan"))
            if np.isfinite(v):
                out.append(float(v))
        return out

    summary = {}
    for key in ("physics_rmse", "physics_p95_abs", "edge_mean_abs", "edge_p95_abs", "outside_mean_abs", "outside_p95_abs", "logvar_mean"):
        vals = _col(key)
        summary[key + "_mean"] = float(np.mean(vals)) if vals else float("nan")
        summary[key + "_p95"] = float(np.quantile(np.asarray(vals), 0.95)) if vals else float("nan")

    pass_rate = float(np.mean([1.0 if r["pass"] else 0.0 for r in rows])) if rows else float("nan")
    fail_counts: Dict[str, int] = {}
    for r in rows:
        for f in r.get("fails", []):
            fail_counts[f] = fail_counts.get(f, 0) + 1

    # optional plots: top failing by edge_mean_abs
    if num_plots > 0:
        failing = [r for r in rows if not r["pass"]]
        failing.sort(key=lambda r: float(r["metrics"].get("edge_mean_abs", float("nan"))), reverse=True)
        for r in failing[: int(num_plots)]:
            idx = int(r["idx"])
            sample = ds[idx]
            defect_true = sample["defect"].squeeze(0).cpu().numpy()
            pred = None
            if method in ("pseudo_poisson", "oracle_poisson"):
                # re-run quickly on cpu for plotting via physics (cheap here)
                pred = None
            # For simplicity: plot using existing evaluator-style helper with a re-run on device.
            sample_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
            std = sample_dev["standard"].unsqueeze(0)
            diff_ts = {"I_x": sample_dev["inputs"][0], "I_y": sample_dev["inputs"][1]}
            if method == "pseudo_poisson":
                pred_t = reconstruct_defect_pseudo_poisson(
                    physics=physics,
                    standard_height=std,
                    diff_ts={"I_x": diff_ts["I_x"].unsqueeze(0), "I_y": diff_ts["I_y"].unsqueeze(0)},
                    defect_roi_radius=float(cfg.training.defect_roi_radius),
                    apply_edge_offset=True,
                )
            elif method == "oracle_poisson":
                pred_t = reconstruct_defect_oracle_poisson(
                    physics=physics,
                    standard_height=std,
                    defect_true=sample_dev["defect"].unsqueeze(0),
                    diff_ts={"I_x": diff_ts["I_x"].unsqueeze(0), "I_y": diff_ts["I_y"].unsqueeze(0)},
                    defect_roi_radius=float(cfg.training.defect_roi_radius),
                    apply_edge_offset=True,
                )
            else:
                model, meta = nn  # type: ignore[assignment]
                use_prior = bool(meta.get("use_pseudo_poisson_prior", False))
                prior_as_input = bool(meta.get("pseudo_poisson_prior_as_input", False))
                prior_scale = float(meta.get("pseudo_poisson_prior_scale", 1.0))
                prior_defect = None
                if use_prior:
                    with torch.no_grad():
                        prior_defect = reconstruct_defect_pseudo_poisson(
                            physics=physics,
                            standard_height=std,
                            diff_ts={"I_x": diff_ts["I_x"].unsqueeze(0), "I_y": diff_ts["I_y"].unsqueeze(0)},
                            defect_roi_radius=float(cfg.training.defect_roi_radius),
                            apply_edge_offset=True,
                        )
                inputs = _build_inputs(cfg, physics, sample=sample_dev, prior_defect=prior_defect, prior_as_input=(use_prior and prior_as_input))
                with torch.no_grad():
                    residual, _logvar = _forward_model(model, inputs)
                    pred_t = residual
                    if use_prior and prior_defect is not None:
                        pred_t = prior_defect * prior_scale + residual

            height = std + pred_t
            phys_out = physics(height)
            std_phys = physics(std)
            diff_pred = {
                "I_x": (phys_out["I_x"] - std_phys["I_x"]).squeeze(0).squeeze(0).detach().cpu().numpy(),
                "I_y": (phys_out["I_y"] - std_phys["I_y"]).squeeze(0).squeeze(0).detach().cpu().numpy(),
            }
            diff_true = {
                "I_x": sample_dev["inputs"][0].detach().cpu().numpy(),
                "I_y": sample_dev["inputs"][1].detach().cpu().numpy(),
            }
            plot_defect_and_intensity(
                defect_true=defect_true,
                defect_pred=pred_t.squeeze(0).squeeze(0).detach().cpu().numpy(),
                diff_true=diff_true,
                diff_pred=diff_pred,
                output_path=plots_dir / f"{method}_fail_{idx:04d}.png",
                title=f"gate_fail idx={idx} method={method}",
            )

    return {
        "method": method,
        "count": len(rows),
        "pass_rate": pass_rate,
        "thresholds": thresholds.to_dict(),
        "fail_counts": fail_counts,
        "summary": summary,
        "samples": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Compute publishable-style gating/QC metrics for synthetic runs.")
    p.add_argument("--config", required=True, help="Experiment config JSON (base).")
    p.add_argument("--data-root", required=True, help="Dataset root.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--calibrate-split", default=None, choices=[None, "train", "val", "test"], help="Split to calibrate thresholds on.")
    p.add_argument("--calibrate-method", default=None, help="Method name to calibrate thresholds from (defaults to first NN if provided).")
    p.add_argument("--quantile", type=float, default=0.99, help="Quantile for threshold suggestion when calibrating.")
    p.add_argument("--num-plots", type=int, default=0, help="Number of failing-sample plots per method.")
    p.add_argument("--include-oracle", action="store_true", help="Include oracle_poisson baseline.")
    p.add_argument("--ckpt", action="append", default=[], help="NN checkpoint spec: NAME=PATH (repeatable).")
    p.add_argument("--device", default=None, help="Override device (e.g. cpu/cuda).")
    p.add_argument("--name", default="gate_report", help="Run name suffix.")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = load_experiment_config(cfg_path)
    project_root = _infer_project_root(cfg_path)
    runs_root = project_root / cfg.paths.runs_dir
    run = create_run(runs_root, name=str(args.name), argv=None, config_snapshot=cfg.to_dict())

    dataset_root = Path(args.data_root).expanduser()
    device = _select_device(args.device or cfg.training.device)
    physics = DifferentiableGradientLayer(cfg.simulation).to(device)
    _freeze_physics_to_ideal(physics)

    ckpts = _parse_ckpt_specs(args.ckpt)
    nn_models: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
    for name, path in ckpts:
        nn_models[name] = _load_nn(path, device=device)

    # Determine thresholds.
    thresholds = GateThresholds.from_training(cfg.training)
    calib_info: dict[str, Any] | None = None
    if args.calibrate_split is not None:
        calib_split = str(args.calibrate_split)
        calib_ds = DefectDataset(dataset_root, calib_split)
        # choose method to calibrate from
        calib_name = args.calibrate_method
        if calib_name is None and nn_models:
            calib_name = next(iter(nn_models.keys()))
        if calib_name is None:
            calib_name = "pseudo_poisson"

        # compute rows for calibration method without thresholds
        tmp_thresholds = GateThresholds()
        if calib_name == "pseudo_poisson":
            calib_rows = _compute_for_method(
                cfg,
                physics=physics,
                ds=calib_ds,
                device=device,
                method="pseudo_poisson",
                nn=None,
                thresholds=tmp_thresholds,
                num_plots=0,
                plots_dir=run.root / "plots",
            )["samples"]
        elif calib_name == "oracle_poisson":
            calib_rows = _compute_for_method(
                cfg,
                physics=physics,
                ds=calib_ds,
                device=device,
                method="oracle_poisson",
                nn=None,
                thresholds=tmp_thresholds,
                num_plots=0,
                plots_dir=run.root / "plots",
            )["samples"]
        else:
            if calib_name not in nn_models:
                raise SystemExit(f"Unknown calibrate-method {calib_name!r}; available: {list(nn_models)}")
            calib_rows = _compute_for_method(
                cfg,
                physics=physics,
                ds=calib_ds,
                device=device,
                method=calib_name,
                nn=nn_models[calib_name],
                thresholds=tmp_thresholds,
                num_plots=0,
                plots_dir=run.root / "plots",
            )["samples"]
        flat = [r["metrics"] for r in calib_rows]
        thresholds = suggest_thresholds(flat, q=float(args.quantile), include_logvar=False)
        calib_info = {
            "split": calib_split,
            "method": calib_name,
            "quantile": float(args.quantile),
            "suggested_thresholds": thresholds.to_dict(),
        }

    # Evaluate target split
    ds = DefectDataset(dataset_root, str(args.split))
    plots_dir = run.root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods: list[tuple[str, str, tuple[torch.nn.Module, dict[str, Any]] | None]] = [("pseudo_poisson", "pseudo_poisson", None)]
    if args.include_oracle:
        methods.append(("oracle_poisson", "oracle_poisson", None))
    for name, nn in nn_models.items():
        methods.append((name, name, nn))

    results: dict[str, Any] = {}
    for key, method, nn in methods:
        results[key] = _compute_for_method(
            cfg,
            physics=physics,
            ds=ds,
            device=device,
            method=method,
            nn=nn,
            thresholds=thresholds,
            num_plots=int(args.num_plots),
            plots_dir=plots_dir,
        )

    payload = {
        "config": str(cfg_path),
        "data_root": str(dataset_root),
        "split": str(args.split),
        "device": str(device),
        "calibration": calib_info,
        "results": results,
    }
    write_json(run.root / "gate_report.json", payload)
    write_json(run.root / "artifacts.json", {"dataset_root": str(dataset_root), "split": str(args.split), "plots_dir": str(plots_dir)})
    print(str(run.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
