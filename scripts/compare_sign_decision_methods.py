from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

from mini_grin_rebuild.core.configs import load_experiment_config
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.evaluation.sign_decision import run_sign_method_comparison, summarize_sign_method_comparison


def main() -> int:
    p = argparse.ArgumentParser(description="Compare test phase-gradient sign-map extraction methods.")
    p.add_argument("--config", required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-stride", type=int, default=4)
    p.add_argument("--patch-radius", type=int, default=3)
    p.add_argument("--basis-sigma-px", type=float, default=1.5)
    p.add_argument("--candidate-blur-sigma-px", type=float, default=0.6)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_experiment_config(Path(args.config).expanduser())
    records = run_sign_method_comparison(
        cfg,
        samples=int(args.samples),
        seed=int(args.seed),
        sample_stride=int(args.sample_stride),
        patch_radius=int(args.patch_radius),
        basis_sigma_px=float(args.basis_sigma_px),
        candidate_blur_sigma_px=float(args.candidate_blur_sigma_px),
    )
    payload = {
        "config": str(Path(args.config).expanduser()),
        "samples": int(args.samples),
        "seed": int(args.seed),
        "sample_stride": int(args.sample_stride),
        "patch_radius": int(args.patch_radius),
        "basis_sigma_px": float(args.basis_sigma_px),
        "candidate_blur_sigma_px": float(args.candidate_blur_sigma_px),
        "summary": summarize_sign_method_comparison(records),
        "records": [record.to_dict() for record in records],
    }

    out_path = Path(args.out).expanduser() if args.out else (Path("runs") / "compare_sign_decision_methods.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
