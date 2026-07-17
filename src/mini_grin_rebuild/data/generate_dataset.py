from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.data.virtual_objects import (
    VirtualObject,
    build_triplet,
    defect_patch,
    scratch_defect,
)
from mini_grin_rebuild.simulation.factory import create_simulation_engine


def _wrap_height(cfg: SimulationConfig) -> float:
    scale = (2.0 * np.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air)
    return float((np.pi * cfg.wrap_safety) / max(scale, 1e-12))


def _sample_defect_center(cfg: SimulationConfig, rng: np.random.Generator) -> tuple[float, float]:
    sigma = float(getattr(cfg, "defect_center_sigma_norm", 0.0) or 0.0)
    if sigma <= 0:
        return (float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5)))

    max_r = float(getattr(cfg, "defect_center_max_radius_norm", 0.5) or 0.5)
    max_r = max(1e-6, min(max_r, 1.0))
    for _ in range(50):
        x = float(rng.normal(0.0, sigma))
        y = float(rng.normal(0.0, sigma))
        r = math.sqrt(x * x + y * y)
        if r <= max_r:
            return (x, y)
    # fallback (rare): clip into the allowed disk
    x = float(np.clip(rng.normal(0.0, sigma), -max_r, max_r))
    y = float(np.clip(rng.normal(0.0, sigma), -max_r, max_r))
    return (x, y)


def _sample_defect_amplitude(cfg: SimulationConfig, rng: np.random.Generator) -> float:
    """
    Sample defect amplitude as a fraction of `height_scale`.

    For microlens SRT scenes we prefer sampling directly in "wrap height" units to
    avoid saturating the clamp (otherwise almost all defects become identical).
    """
    scene = str(getattr(cfg, "scene", "legacy"))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    if scene != "microlens_srt":
        return sign * float(rng.uniform(0.05, 0.3))

    wrap_h = _wrap_height(cfg)
    lo = float(getattr(cfg, "defect_amplitude_wrap_min", 0.2))
    hi = float(getattr(cfg, "defect_amplitude_wrap_max", 1.0))
    lo = max(0.0, min(lo, hi))
    hi = max(lo, hi)
    target_height = float(rng.uniform(lo, hi)) * float(wrap_h)
    if cfg.height_scale <= 0:
        raise ValueError("height_scale must be > 0 for amplitude sampling")
    return sign * (target_height / float(cfg.height_scale))


def _phase_scale(cfg: SimulationConfig) -> float:
    return float((2.0 * np.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air))


def _estimate_wrap_metrics(
    cfg: SimulationConfig,
    *,
    reference_height: np.ndarray,
    standard_height: np.ndarray,
    test_height: np.ndarray,
    defect_height: np.ndarray,
) -> dict[str, Any]:
    wrap_h = _wrap_height(cfg)
    phase_scale = _phase_scale(cfg)
    std_phase = phase_scale * np.asarray(standard_height, dtype=np.float32)
    test_phase = phase_scale * np.asarray(test_height, dtype=np.float32)
    defect_phase = phase_scale * np.asarray(defect_height, dtype=np.float32)

    grad_std_y, grad_std_x = np.gradient(std_phase, cfg.dx)
    grad_test_y, grad_test_x = np.gradient(test_phase, cfg.dx)
    grad_std_peak = float(np.max(np.sqrt(grad_std_x**2 + grad_std_y**2)))
    grad_test_peak = float(np.max(np.sqrt(grad_test_x**2 + grad_test_y**2)))

    defect_peak = float(np.max(np.abs(defect_height)))
    defect_wrap_target = float(defect_peak / max(wrap_h, 1e-12))
    standard_residual = np.asarray(standard_height, dtype=np.float32) - np.asarray(reference_height, dtype=np.float32)
    standard_wrap_frac = float(np.max(np.abs(standard_residual)) / max(wrap_h, 1e-12))
    test_wrap_frac = float(np.max(np.abs(test_phase - std_phase)) / math.pi)

    wrap_class = "in_wrap"
    if defect_wrap_target > 1.0 + 1e-9:
        wrap_class = "cross_wrap"

    return {
        "wrap_height": float(wrap_h),
        "standard_wrap_frac": standard_wrap_frac,
        "defect_wrap_target": defect_wrap_target,
        "estimated_wrap_stress_level": max(standard_wrap_frac, defect_wrap_target),
        "phase_grad_peak_standard": grad_std_peak,
        "phase_grad_peak_test": grad_test_peak,
        "test_phase_jump_peak_pi": test_wrap_frac,
        "wrap_class": wrap_class,
    }


def _sample_large_defect_amplitude(cfg: SimulationConfig, rng: np.random.Generator) -> float:
    sign = -1.0 if rng.random() < 0.5 else 1.0
    wrap_h = _wrap_height(cfg)
    lo = float(getattr(cfg, "large_defect_amplitude_wrap_min", 1.0))
    hi = float(getattr(cfg, "large_defect_amplitude_wrap_max", 2.0))
    lo = max(0.0, min(lo, hi))
    hi = max(lo, hi)
    target_height = float(rng.uniform(lo, hi)) * float(wrap_h)
    if cfg.height_scale <= 0:
        raise ValueError("height_scale must be > 0 for amplitude sampling")
    return sign * (target_height / float(cfg.height_scale))


def random_triplet(cfg: SimulationConfig, rng: np.random.Generator) -> Dict[str, VirtualObject]:
    """
    Sample generator kept compatible with the legacy dataset script:
    - base objects from `build_triplet`
    - 50/50 dot vs scratch defect
    - amplitude and placement randomized
    """
    large_prob = float(getattr(cfg, "large_defect_prob", 0.0) or 0.0)
    large_prob = float(np.clip(large_prob, 0.0, 1.0))
    use_large = bool(rng.random() < large_prob)
    amp = _sample_large_defect_amplitude(cfg, rng) if use_large else _sample_defect_amplitude(cfg, rng)
    center = _sample_defect_center(cfg, rng)
    triplet = build_triplet(cfg)

    scratch_prob = float(getattr(cfg, "defect_scratch_prob", 0.5) or 0.5)
    scratch_prob = float(np.clip(scratch_prob, 0.0, 1.0))
    if rng.random() < scratch_prob:
        if use_large:
            width = float(rng.uniform(cfg.large_scratch_width_min_um, cfg.large_scratch_width_max_um))
            length = float(rng.uniform(cfg.large_scratch_length_min_um, cfg.large_scratch_length_max_um))
        else:
            width = float(rng.uniform(cfg.scratch_width_min_um, cfg.scratch_width_max_um))
            length = float(rng.uniform(cfg.scratch_length_min_um, cfg.scratch_length_max_um))
        angle = float(rng.uniform(0.0, 180.0))
        defect = scratch_defect(
            cfg,
            amplitude=amp,
            center=center,
            width_phys_um=width,
            length_phys_um=length,
            angle_deg=angle,
            support_k=cfg.defect_support_k,
        )
    else:
        if use_large:
            sigma_phys = float(rng.uniform(cfg.large_defect_sigma_min_um, cfg.large_defect_sigma_max_um))
        else:
            sigma_phys = float(rng.uniform(cfg.defect_sigma_min_um, cfg.defect_sigma_max_um))
        defect = defect_patch(
            cfg,
            amplitude=amp,
            center=center,
            sigma_phys_um=sigma_phys,
            support_k=cfg.defect_support_k,
        )

    triplet["test"].height_map = triplet["standard"].height_map + defect
    triplet["defect"].height_map = defect
    return triplet


def _spawn_rng(master: np.random.Generator) -> np.random.Generator:
    # Deterministic "independent" RNG derived from master seed stream.
    seed = int(master.integers(0, 2**32 - 1))
    return np.random.default_rng(seed)


def generate_dataset(
    cfg: SimulationConfig,
    *,
    output_root: str | Path,
    train: int,
    val: int,
    test: int,
    seed: int,
    config_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Generate a dataset compatible with the legacy `.npz` schema.

    Returns the dataset metadata dict written to `dataset_meta.json`.
    """
    root = Path(output_root)
    rng = np.random.default_rng(int(seed))
    splits = {"train": int(train), "val": int(val), "test": int(test)}
    engine = create_simulation_engine(cfg)
    sample_records: list[dict[str, Any]] = []
    wrap_class_counts = {"in_wrap": 0, "cross_wrap": 0}

    for split, count in splits.items():
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            triplet = random_triplet(cfg, rng)
            heights = {name: triplet[name].height_map for name in ("standard", "test", "reference")}
            bundle_rng = _spawn_rng(rng)
            simulation_meta: dict[str, Any]
            if hasattr(engine, "simulate_bundle"):
                bundle = engine.simulate_bundle(heights, rng=bundle_rng)  # type: ignore[attr-defined]
                captures = {name: bundle.captures[name].to_channel_dict() for name in heights}
                simulation_meta = dict(bundle.meta)
            else:
                captures = {}
                frame_meta: dict[str, Any] = {}
                for name in ("standard", "test", "reference"):
                    cap_rng = _spawn_rng(rng) if cfg.noise_level > 0 else None
                    cap = engine.simulate_capture(triplet[name].height_map, rng=cap_rng)
                    captures[name] = cap.to_channel_dict()
                    frame_meta[name] = dict(cap.meta)
                simulation_meta = {"engine": engine.meta(), "frames": frame_meta}

            diff_ix_st = captures["test"]["I_x"] - captures["standard"]["I_x"]
            diff_iy_st = captures["test"]["I_y"] - captures["standard"]["I_y"]
            diff_ix_sr = captures["standard"]["I_x"] - captures["reference"]["I_x"]
            diff_iy_sr = captures["standard"]["I_y"] - captures["reference"]["I_y"]

            file_path = split_dir / f"sample_{idx:04d}.npz"
            payload = {
                "diff_ix_st": diff_ix_st.astype(np.float32),
                "diff_iy_st": diff_iy_st.astype(np.float32),
                "diff_ix_sr": diff_ix_sr.astype(np.float32),
                "diff_iy_sr": diff_iy_sr.astype(np.float32),
                "ix_standard": captures["standard"]["I_x"].astype(np.float32),
                "iy_standard": captures["standard"]["I_y"].astype(np.float32),
                "ix_reference": captures["reference"]["I_x"].astype(np.float32),
                "iy_reference": captures["reference"]["I_y"].astype(np.float32),
                "ix_test": captures["test"]["I_x"].astype(np.float32),
                "iy_test": captures["test"]["I_y"].astype(np.float32),
                "standard": triplet["standard"].height_map.astype(np.float32),
                "reference": triplet["reference"].height_map.astype(np.float32),
                "test": triplet["test"].height_map.astype(np.float32),
                "defect": triplet["defect"].height_map.astype(np.float32),
            }
            if all("I_raw" in captures[name] for name in ("standard", "reference", "test")):
                payload["raw_standard"] = captures["standard"]["I_raw"].astype(np.float32)
                payload["raw_reference"] = captures["reference"]["I_raw"].astype(np.float32)
                payload["raw_test"] = captures["test"]["I_raw"].astype(np.float32)
            np.savez(file_path, **payload)
            wrap_meta = _estimate_wrap_metrics(
                cfg,
                reference_height=triplet["reference"].height_map,
                standard_height=triplet["standard"].height_map,
                test_height=triplet["test"].height_map,
                defect_height=triplet["defect"].height_map,
            )
            wrap_class = str(wrap_meta.get("wrap_class", "in_wrap"))
            if wrap_class in wrap_class_counts:
                wrap_class_counts[wrap_class] += 1
            sample_records.append(
                {
                    "split": split,
                    "index": idx,
                    "file": str(file_path.relative_to(root)),
                    "simulation": simulation_meta,
                    "wrap": wrap_meta,
                }
            )

    meta: Dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "splits": splits,
        "simulation": {
            "grid_size": cfg.grid_size,
            "dx": cfg.dx,
            "wavelength": cfg.wavelength,
            "n_object": cfg.n_object,
            "n_air": cfg.n_air,
            "noise_level": cfg.noise_level,
            "height_scale": cfg.height_scale,
            "wrap_safety": cfg.wrap_safety,
            "defect_sigma_min_um": cfg.defect_sigma_min_um,
            "defect_sigma_max_um": cfg.defect_sigma_max_um,
            "defect_support_k": cfg.defect_support_k,
            "scratch_width_min_um": cfg.scratch_width_min_um,
            "scratch_width_max_um": cfg.scratch_width_max_um,
            "scratch_length_min_um": cfg.scratch_length_min_um,
            "scratch_length_max_um": cfg.scratch_length_max_um,
            "defect_scratch_prob": float(getattr(cfg, "defect_scratch_prob", 0.5) or 0.5),
            "defect_center_sigma_norm": float(getattr(cfg, "defect_center_sigma_norm", 0.0) or 0.0),
            "defect_center_max_radius_norm": float(getattr(cfg, "defect_center_max_radius_norm", 0.5) or 0.5),
            "defect_amplitude_wrap_min": float(getattr(cfg, "defect_amplitude_wrap_min", 0.2) or 0.2),
            "defect_amplitude_wrap_max": float(getattr(cfg, "defect_amplitude_wrap_max", 1.0) or 1.0),
            "allow_defect_wrap_exceed": bool(getattr(cfg, "allow_defect_wrap_exceed", False)),
            "large_defect_prob": float(getattr(cfg, "large_defect_prob", 0.0) or 0.0),
            "large_defect_amplitude_wrap_min": float(getattr(cfg, "large_defect_amplitude_wrap_min", 1.0) or 1.0),
            "large_defect_amplitude_wrap_max": float(getattr(cfg, "large_defect_amplitude_wrap_max", 2.0) or 2.0),
        },
        "capture_engine": engine.meta(),
        "wrap_summary": {
            "counts": wrap_class_counts,
            "cross_wrap_fraction": float(wrap_class_counts["cross_wrap"] / max(sum(wrap_class_counts.values()), 1)),
        },
    }
    if config_snapshot is not None:
        meta["config_snapshot"] = config_snapshot

    write_json(root / "sample_meta.json", {"schema_version": 1, "samples": sample_records})
    write_json(root / "dataset_meta.json", meta)
    return meta


__all__ = ["generate_dataset", "random_triplet"]
