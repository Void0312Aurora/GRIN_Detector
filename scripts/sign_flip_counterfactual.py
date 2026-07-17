from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

import numpy as np

from mini_grin_rebuild.core.configs import load_experiment_config
from mini_grin_rebuild.core.json_io import write_json
from mini_grin_rebuild.data.generate_dataset import random_triplet
from mini_grin_rebuild.simulation.factory import create_simulation_engine


def _l2_sep(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)
    return num / den


def _cosine_score(obs: np.ndarray, direction: np.ndarray) -> float:
    num = float(np.sum(obs * direction))
    den = float(np.linalg.norm(obs) * np.linalg.norm(direction) + 1e-12)
    return num / den


def main() -> int:
    p = argparse.ArgumentParser(description="Counterfactual sign-flip audit for physical leakage.")
    p.add_argument("--config", required=True)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-scale", type=float, default=0.2)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_experiment_config(Path(args.config).expanduser())
    engine = create_simulation_engine(cfg.simulation)
    rng = np.random.default_rng(int(args.seed))

    records = []
    for idx in range(int(args.samples)):
        triplet = random_triplet(cfg.simulation, rng)
        standard = np.asarray(triplet["standard"].height_map, dtype=np.float32)
        defect = np.asarray(triplet["defect"].height_map, dtype=np.float32)
        mag = np.abs(defect)
        sign_true = 1.0 if float(np.sum(defect)) >= 0.0 else -1.0
        test_plus = standard + mag
        test_minus = standard - mag

        pair = engine.simulate_bundle(
            {"standard": standard, "reference": triplet["reference"].height_map, "plus": test_plus, "minus": test_minus},
            rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
        )
        probe = engine.simulate_bundle(
            {
                "standard": standard,
                "reference": triplet["reference"].height_map,
                "plus_probe": standard + float(args.probe_scale) * mag,
                "minus_probe": standard - float(args.probe_scale) * mag,
            },
            rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
        )

        standard_ch = pair.captures["standard"].channels
        plus_test = pair.captures["plus"].channels
        minus_test = pair.captures["minus"].channels
        plus_probe = probe.captures["plus_probe"].channels
        minus_probe = probe.captures["minus_probe"].channels

        raw_direction_score_plus = float("nan")
        raw_direction_score_minus = float("nan")
        raw_direction_correct_plus = None
        raw_direction_correct_minus = None
        if "I_raw" in plus_test and "I_raw" in minus_test and "I_raw" in standard_ch:
            raw_diff_plus = plus_test["I_raw"] - standard_ch["I_raw"]
            raw_diff_minus = minus_test["I_raw"] - standard_ch["I_raw"]
            raw_probe_direction = (plus_probe["I_raw"] - probe.captures["standard"].channels["I_raw"]) - (
                minus_probe["I_raw"] - probe.captures["standard"].channels["I_raw"]
            )
            raw_direction_score_plus = _cosine_score(raw_diff_plus, raw_probe_direction)
            raw_direction_score_minus = _cosine_score(raw_diff_minus, raw_probe_direction)
            raw_direction_correct_plus = bool(raw_direction_score_plus > 0.0)
            raw_direction_correct_minus = bool(raw_direction_score_minus < 0.0)

        dic_diff_plus = np.concatenate(
            [
                (plus_test["I_x"] - standard_ch["I_x"]).ravel(),
                (plus_test["I_y"] - standard_ch["I_y"]).ravel(),
            ]
        )
        dic_diff_minus = np.concatenate(
            [
                (minus_test["I_x"] - standard_ch["I_x"]).ravel(),
                (minus_test["I_y"] - standard_ch["I_y"]).ravel(),
            ]
        )
        dic_probe_direction = np.concatenate(
            [
                ((plus_probe["I_x"] - probe.captures["standard"].channels["I_x"]) - (minus_probe["I_x"] - probe.captures["standard"].channels["I_x"])).ravel(),
                ((plus_probe["I_y"] - probe.captures["standard"].channels["I_y"]) - (minus_probe["I_y"] - probe.captures["standard"].channels["I_y"])).ravel(),
            ]
        )
        dic_direction_score_plus = _cosine_score(dic_diff_plus, dic_probe_direction)
        dic_direction_score_minus = _cosine_score(dic_diff_minus, dic_probe_direction)

        rec = {
            "index": idx,
            "lambda_dic": 0.5 * (_l2_sep(plus_test["I_x"], minus_test["I_x"]) + _l2_sep(plus_test["I_y"], minus_test["I_y"])),
            "lambda_raw": _l2_sep(plus_test["I_raw"], minus_test["I_raw"]) if "I_raw" in plus_test and "I_raw" in minus_test else float("nan"),
            "defect_peak_abs": float(np.max(np.abs(defect))),
            "sign_true": sign_true,
            "raw_direction_score_plus": raw_direction_score_plus,
            "raw_direction_score_minus": raw_direction_score_minus,
            "raw_direction_correct_plus": raw_direction_correct_plus,
            "raw_direction_correct_minus": raw_direction_correct_minus,
            "dic_direction_score_plus": dic_direction_score_plus,
            "dic_direction_score_minus": dic_direction_score_minus,
            "dic_direction_correct_plus": bool(dic_direction_score_plus > 0.0),
            "dic_direction_correct_minus": bool(dic_direction_score_minus < 0.0),
        }
        records.append(rec)

    lambda_dic = [float(r["lambda_dic"]) for r in records]
    lambda_raw = [float(r["lambda_raw"]) for r in records if np.isfinite(float(r["lambda_raw"]))]
    payload = {
        "config": str(Path(args.config).expanduser()),
        "samples": int(args.samples),
        "probe_scale": float(args.probe_scale),
        "engine_name": getattr(engine, "name", "unknown"),
        "summary": {
            "lambda_dic_mean": float(np.mean(lambda_dic)) if lambda_dic else float("nan"),
            "lambda_raw_mean": float(np.mean(lambda_raw)) if lambda_raw else float("nan"),
            "raw_better_than_dic_fraction": float(
                np.mean([float(r["lambda_raw"] > r["lambda_dic"]) for r in records if np.isfinite(float(r["lambda_raw"]))])
            )
            if lambda_raw
            else float("nan"),
            "raw_direction_accuracy": float(
                np.mean(
                    [
                        float(bool(r["raw_direction_correct_plus"]) and bool(r["raw_direction_correct_minus"]))
                        for r in records
                        if r["raw_direction_correct_plus"] is not None and r["raw_direction_correct_minus"] is not None
                    ]
                )
            )
            if any(r["raw_direction_correct_plus"] is not None for r in records)
            else float("nan"),
            "dic_direction_accuracy": float(
                np.mean([float(bool(r["dic_direction_correct_plus"]) and bool(r["dic_direction_correct_minus"])) for r in records])
            )
            if records
            else float("nan"),
            "raw_direction_score_plus_mean": float(
                np.mean([float(r["raw_direction_score_plus"]) for r in records if np.isfinite(float(r["raw_direction_score_plus"]))])
            )
            if any(np.isfinite(float(r["raw_direction_score_plus"])) for r in records)
            else float("nan"),
            "raw_direction_score_minus_mean": float(
                np.mean([float(r["raw_direction_score_minus"]) for r in records if np.isfinite(float(r["raw_direction_score_minus"]))])
            )
            if any(np.isfinite(float(r["raw_direction_score_minus"])) for r in records)
            else float("nan"),
        },
        "records": records,
    }

    out_path = Path(args.out).expanduser() if args.out else (Path("runs") / "sign_flip_counterfactual.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
