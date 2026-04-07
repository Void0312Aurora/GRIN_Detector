from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.data.external_topography import load_plux_topography, simulate_spdic_from_height


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external_data"
PROCESSED_ROOT = EXTERNAL_ROOT / "processed"
MANIFEST_ROOT = EXTERNAL_ROOT / "manifests"


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p01": float("nan"),
            "p99": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p01": float(np.percentile(finite, 1.0)),
        "p99": float(np.percentile(finite, 99.0)),
    }


def _preview_panel(
    *,
    title: str,
    image: np.ndarray,
    ax: plt.Axes,
    cmap: str,
    percentile: float = 99.0,
) -> None:
    finite = image[np.isfinite(image)]
    if finite.size:
        vmax = float(np.percentile(np.abs(finite), percentile))
        if vmax <= 0:
            vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    else:
        vmax = 1.0
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(image, cmap=cmap, vmin=-vmax if cmap != "viridis" else None, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _write_preview(path: Path, *, height_um: np.ndarray, phase: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    _preview_panel(title="Height (um)", image=height_um, ax=axes[0, 0], cmap="coolwarm")
    _preview_panel(title="Phase (rad)", image=phase, ax=axes[0, 1], cmap="coolwarm")
    _preview_panel(title="I_x", image=ix, ax=axes[1, 0], cmap="viridis")
    _preview_panel(title="I_y", image=iy, ax=axes[1, 1], cmap="viridis")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _infer_output_dir(input_path: Path) -> Path:
    stem = input_path.stem
    parent = input_path.parent.name
    return PROCESSED_ROOT / parent / f"{stem}_spdic_smoke"


def main() -> int:
    p = argparse.ArgumentParser(description="Load a Sensofar .plux topography file and run a height-to-SPDIC smoke pass.")
    p.add_argument("input", type=str, help="Path to a .plux sample")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--manifest-path", type=str, default=None)
    p.add_argument("--fov-unit", type=str, default="mm", help="Interpretation of GENERAL/FOV_X and FOV_Y.")
    p.add_argument("--noise-level", type=float, default=0.0)
    p.add_argument("--wavelength", type=float, default=0.6328)
    p.add_argument("--n-object", type=float, default=1.52)
    p.add_argument("--n-air", type=float, default=1.0)
    p.add_argument(
        "--height-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the imported height map before phase conversion.",
    )
    p.add_argument("--keep-tilt", action="store_true", help="Do not subtract a best-fit plane before simulation.")
    p.add_argument("--keep-offset", action="store_true", help="Do not center the height map to zero mean.")
    p.add_argument("--keep-missing", action="store_true", help="Keep NaNs instead of filling them before simulation.")
    args = p.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _infer_output_dir(input_path)
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path
        else MANIFEST_ROOT / f"{input_path.stem}_spdic_smoke.json"
    )

    topography = load_plux_topography(
        input_path,
        fov_unit=args.fov_unit,
        fill_missing=not args.keep_missing,
        detrend_plane=not args.keep_tilt,
        center_height=not args.keep_offset,
    )
    metadata = topography.metadata
    if metadata.pixel_pitch_x_um is None or metadata.pixel_pitch_y_um is None:
        raise ValueError(f"Missing pixel pitch metadata for {input_path}")
    height_um = (topography.height_um * float(args.height_scale)).astype(np.float32, copy=False)

    cfg = SimulationConfig(
        grid_size=height_um.shape[0],
        dx=float(0.5 * (metadata.pixel_pitch_x_um + metadata.pixel_pitch_y_um)),
        wavelength=float(args.wavelength),
        n_object=float(args.n_object),
        n_air=float(args.n_air),
        noise_level=float(args.noise_level),
    )

    sim = simulate_spdic_from_height(
        cfg,
        height_um=height_um,
        pixel_pitch_x_um=float(metadata.pixel_pitch_x_um),
        pixel_pitch_y_um=float(metadata.pixel_pitch_y_um),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "height_um.npy", height_um)
    np.save(output_dir / "valid_mask.npy", topography.valid_mask)
    np.save(output_dir / "phase.npy", sim["phase"])
    np.save(output_dir / "I_x.npy", sim["I_x"])
    np.save(output_dir / "I_y.npy", sim["I_y"])
    _write_preview(
        output_dir / "smoke_preview.png",
        height_um=height_um,
        phase=sim["phase"],
        ix=sim["I_x"],
        iy=sim["I_y"],
    )

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "metadata": metadata.to_dict(),
        "preprocess": {
            "fill_missing": bool(not args.keep_missing),
            "detrend_plane": bool(not args.keep_tilt),
            "center_height": bool(not args.keep_offset),
            "height_scale": float(args.height_scale),
            "valid_fraction": float(np.mean(topography.valid_mask)),
            "missing_pixels": int(topography.valid_mask.size - int(np.count_nonzero(topography.valid_mask))),
        },
        "height_stats_um": _stats(height_um),
        "phase_stats_rad": _stats(sim["phase"]),
        "intensity_stats": {
            "I_x": _stats(sim["I_x"]),
            "I_y": _stats(sim["I_y"]),
        },
    }
    write_json(manifest_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
