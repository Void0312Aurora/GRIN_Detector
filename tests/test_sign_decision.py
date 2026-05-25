from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

from mini_grin_rebuild.core.configs import load_experiment_config  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import microlens_standard  # noqa: E402
from mini_grin_rebuild.evaluation.sign_decision import (  # noqa: E402
    SampleSignComparison,
    aperture_mask,
    extract_test_gradient_sign_map_first_order,
    extract_test_gradient_sign_map_raw_branch,
    run_sign_method_comparison,
    summarize_sign_method_comparison,
)
from mini_grin_rebuild.physics.factory import create_forward_model  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402


class TestSignDecisionHelpers(unittest.TestCase):
    def test_aperture_mask_has_center_and_excludes_corner(self) -> None:
        mask = aperture_mask(9, radius_fraction=1.0)
        self.assertTrue(bool(mask[4, 4]))
        self.assertFalse(bool(mask[0, 0]))

    def test_first_order_sign_map_matches_closed_form(self) -> None:
        cfg = load_experiment_config(
            Path("/home/void0312/PINNs/mini_grin_rebuild/configs/optical_leakage_dic_nominal.json")
        )
        physics = create_forward_model(cfg.simulation, cfg.training, device="cpu", freeze=True)
        standard = microlens_standard(cfg.simulation).astype(np.float32)

        std = np.asarray(standard, dtype=np.float32)
        std_t = physics._phase(
            __import__("torch").from_numpy(std).unsqueeze(0).unsqueeze(0)  # type: ignore[attr-defined]
        ).squeeze(1)
        g_y_t, g_x_t = physics._gradient(std_t)
        g_x = g_x_t.squeeze(0).cpu().numpy().astype(np.float64)
        g_y = g_y_t.squeeze(0).cpu().numpy().astype(np.float64)

        diff_ix = np.zeros_like(g_x, dtype=np.float32)
        diff_iy = np.zeros_like(g_y, dtype=np.float32)
        diff_ix[32, 40] = float(2.0 * g_x[32, 40] * 0.5)
        diff_iy[28, 24] = float(2.0 * g_y[28, 24] * -0.5)

        signs = extract_test_gradient_sign_map_first_order(
            physics=physics,
            standard_height=standard,
            diff_ix=diff_ix,
            diff_iy=diff_iy,
        )
        self.assertEqual(int(signs["x"][32, 40]), int(np.sign(g_x[32, 40] + 0.5) or np.sign(g_x[32, 40]) or 1))
        self.assertEqual(int(signs["y"][28, 24]), int(np.sign(g_y[28, 24] - 0.5) or np.sign(g_y[28, 24]) or 1))


class TestSignDecisionComparison(unittest.TestCase):
    def test_raw_branch_sign_map_smoke(self) -> None:
        cfg = load_experiment_config(
            Path("/home/void0312/PINNs/mini_grin_rebuild/configs/optical_leakage_dic_nominal.json")
        )
        physics = create_forward_model(cfg.simulation, cfg.training, device="cpu", freeze=True)
        engine = create_simulation_engine(cfg.simulation)
        rng = np.random.default_rng(7)

        standard = microlens_standard(cfg.simulation).astype(np.float32)
        yy, xx = np.indices(standard.shape, dtype=np.float32)
        defect = (0.05 * np.exp(-((xx - 32.0) ** 2 + (yy - 30.0) ** 2) / (2.0 * 2.0**2))).astype(np.float32)
        test = standard + defect

        bundle = engine.simulate_bundle({"standard": standard, "test": test}, rng=rng)
        std = bundle.captures["standard"].channels
        tst = bundle.captures["test"].channels
        diff_ix = np.asarray(tst["I_x"] - std["I_x"], dtype=np.float32)
        diff_iy = np.asarray(tst["I_y"] - std["I_y"], dtype=np.float32)
        raw_diff = np.asarray(tst["I_raw"] - std["I_raw"], dtype=np.float32)

        open_maps = extract_test_gradient_sign_map_first_order(
            physics=physics,
            standard_height=standard,
            diff_ix=diff_ix,
            diff_iy=diff_iy,
        )
        raw_maps = extract_test_gradient_sign_map_raw_branch(
            cfg=cfg,
            engine=engine,
            standard_height=standard,
            observed_raw_diff=raw_diff,
            diff_ix=diff_ix,
            diff_iy=diff_iy,
            g_sx=open_maps["standard_grad_x"],
            g_sy=open_maps["standard_grad_y"],
            rng=np.random.default_rng(11),
            sample_stride=8,
            patch_radius=2,
            basis_sigma_px=1.4,
            candidate_blur_sigma_px=0.5,
        )

        self.assertGreater(int(np.sum(raw_maps["sample_mask_x"])), 0)
        self.assertGreater(int(np.sum(raw_maps["sample_mask_y"])), 0)
        self.assertTrue(np.all(np.isin(raw_maps["x"][raw_maps["sample_mask_x"]], (-1, 1))))
        self.assertTrue(np.all(np.isin(raw_maps["y"][raw_maps["sample_mask_y"]], (-1, 1))))

    def test_nominal_sign_map_comparison(self) -> None:
        cfg = load_experiment_config(
            Path("/home/void0312/PINNs/mini_grin_rebuild/configs/optical_leakage_dic_nominal.json")
        )
        records = run_sign_method_comparison(
            cfg,
            samples=8,
            seed=123,
            sample_stride=4,
            patch_radius=3,
            basis_sigma_px=1.5,
            candidate_blur_sigma_px=0.6,
        )
        self.assertEqual(len(records), 8)
        self.assertIsInstance(records[0], SampleSignComparison)

        summary = summarize_sign_method_comparison(records)
        open_x = float(summary["open_method"]["x"]["global"]["accuracy"])
        open_y = float(summary["open_method"]["y"]["global"]["accuracy"])
        raw_x = float(summary["physical_leakage"]["x"]["global"]["accuracy"])
        raw_y = float(summary["physical_leakage"]["y"]["global"]["accuracy"])
        raw_local_x = float(summary["physical_leakage"]["x"]["defect_local"]["accuracy"])
        raw_local_y = float(summary["physical_leakage"]["y"]["defect_local"]["accuracy"])

        self.assertGreaterEqual(open_x, 0.95)
        self.assertGreaterEqual(open_y, 0.95)
        self.assertGreaterEqual(raw_x, 0.95)
        self.assertGreaterEqual(raw_y, 0.95)
        self.assertGreaterEqual(raw_local_x, 0.90)
        self.assertGreaterEqual(raw_local_y, 0.85)


if __name__ == "__main__":
    unittest.main()
