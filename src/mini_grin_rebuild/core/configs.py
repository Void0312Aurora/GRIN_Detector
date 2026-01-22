from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from mini_grin_rebuild.core.json_io import read_json, write_json


def _validate_no_unknown_keys(
    data: Mapping[str, Any],
    cls: type,
    *,
    where: str,
) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - allowed
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise KeyError(f"Unknown keys in {where}: {keys}")


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data"
    runs_dir: str = "runs"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PathsConfig":
        _validate_no_unknown_keys(data, cls, where="paths")
        return cls(**{k: data[k] for k in data})


@dataclass(frozen=True)
class SimulationConfig:
    grid_size: int = 512
    dx: float = 0.39
    gradient_backend: str = "finite"  # "finite" or "spectral"
    wavelength: float = 0.6328
    n_object: float = 1.52
    n_air: float = 1.0
    noise_level: float = 0.005
    amplitude: float = 1.0
    height_scale: float = 250.0
    wrap_safety: float = 0.8
    # Surface / scene presets.
    scene: str = "legacy"  # "legacy" or "microlens_srt"
    lens_radius_fraction: float = 1.0
    standard_residual_wrap_frac: float = 0.5
    defect_sigma_min_um: float = 1.0
    defect_sigma_max_um: float = 5.0
    defect_support_k: float = 3.0
    scratch_width_min_um: float = 1.0
    scratch_width_max_um: float = 5.0
    scratch_length_min_um: float = 20.0
    scratch_length_max_um: float = 80.0
    # Defect sampling controls (dataset generation only; does not affect physics).
    defect_center_sigma_norm: float = 0.0
    defect_center_max_radius_norm: float = 0.5
    defect_amplitude_wrap_min: float = 0.2
    defect_amplitude_wrap_max: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationConfig":
        _validate_no_unknown_keys(data, cls, where="simulation")
        return cls(**{k: data[k] for k in data})


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda"
    epochs: int = 400
    seed: int | None = 42
    lr: float = 1e-3
    batch_size: int = 80
    log_interval: int = 50

    # Enabled loss terms in the legacy project (keep weights identical).
    sr_diff_weight: float = 1e-6
    curl_weight: float = 5e-4
    sparsity_weight: float = 5e-4
    edge_suppress_weight: float = 0.05

    # Data-term weighting (legacy behavior).
    data_grad_alpha: float = 1.0
    data_weight_min: float = 1e-2
    logvar_min: float = 0.0
    logvar_max: float = 6.0
    predict_logvar: bool = True

    # Geometry priors.
    defect_roi_radius: float = 0.6

    # Input channel switches (legacy behavior).
    use_sr_inputs: bool = True
    use_raw_intensity_inputs: bool = True
    use_phase_inputs: bool = False
    use_coord_inputs: bool = False

    # Input normalization used by the legacy dataset trainer.
    input_scale: float = 1e-4

    # Evaluation masking to avoid background-dominated metrics on sparse defects.
    eval_defect_abs_threshold: float = 1e-4
    eval_defect_rel_threshold: float = 0.05
    eval_defect_dilate_px: int = 0
    # Evaluation regions (fractions of aperture radius) for edge/singularity analysis.
    eval_edge_band_start_frac: float = 0.9
    eval_center_radius_frac: float = 0.1

    # Optional edge-band suppression (inside aperture) to reduce boundary artifacts.
    edge_band_suppress_weight: float = 0.0

    # Gating thresholds (paper/production-style QC). Set to None to disable a gate.
    gate_physics_rmse_max: float | None = None
    gate_physics_p95_abs_max: float | None = None
    gate_edge_mean_abs_max: float | None = None
    gate_edge_p95_abs_max: float | None = None
    gate_outside_mean_abs_max: float | None = None
    gate_outside_p95_abs_max: float | None = None
    gate_logvar_mean_max: float | None = None

    # Model options.
    model_padding_mode: str = "zeros"

    # Training-time cropping to avoid background-dominated gradients on sparse defects.
    crop_size: int = 0
    crop_strategy: str = "activity"  # "activity" or "center"

    # Optional pseudo-poisson prior branch (teacher) for stable reconstruction.
    use_pseudo_poisson_prior: bool = False
    pseudo_poisson_prior_as_input: bool = True
    pseudo_poisson_prior_scale: float = 1.0
    pseudo_poisson_residual_scale: float = 0.0

    # Optional teacher distillation towards pseudo-poisson (kept off by default).
    teacher_loss_weight: float = 0.0
    teacher_loss_type: str = "l1"  # "l1" or "l2"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        _validate_no_unknown_keys(data, cls, where="training")
        return cls(**{k: data[k] for k in data})


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Top-level config snapshot saved into each run directory.
    Keep this schema stable and versioned for auditability.
    """

    schema_version: int = 1
    paths: PathsConfig = field(default_factory=PathsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        _validate_no_unknown_keys(data, cls, where="root")
        schema_version = int(data.get("schema_version", cls.schema_version))
        if schema_version != cls.schema_version:
            raise ValueError(f"Unsupported schema_version={schema_version} (expected {cls.schema_version})")
        paths = PathsConfig.from_dict(data.get("paths", {}))
        simulation = SimulationConfig.from_dict(data.get("simulation", {}))
        training = TrainingConfig.from_dict(data.get("training", {}))
        return cls(
            schema_version=schema_version,
            paths=paths,
            simulation=simulation,
            training=training,
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(read_json(path))


def save_experiment_config(path: str | Path, cfg: ExperimentConfig) -> None:
    write_json(path, cfg.to_dict())


__all__ = [
    "ExperimentConfig",
    "PathsConfig",
    "SimulationConfig",
    "TrainingConfig",
    "load_experiment_config",
    "save_experiment_config",
]
