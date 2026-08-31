from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
src_path = str(ROOT / "src")
scripts_path = str(ROOT / "scripts")
original_sys_path = list(sys.path)
try:
    sys.path.insert(0, src_path)
    sys.path.insert(0, scripts_path)
    from evaluate_reflection_sim2real import (  # noqa: E402
        _angular_harmonic_metrics,
        _evaluate_fold,
        _interior_texture_metrics,
        _validate_split_manifest,
    )
    from search_reflection_sim2real import _deep_set  # noqa: E402
finally:
    # Imported scripts bootstrap their sibling directory themselves.  Restore
    # the complete pre-import path, rather than removing a single occurrence,
    # so test collection cannot make scripts/mini_grin.py shadow the optional
    # legacy mini_grin package used by test_compat_legacy.py.
    sys.path[:] = original_sys_path


class TestReflectionSim2RealTools(unittest.TestCase):
    def test_ordered_harmonics_distinguish_a_four_lobe_profile_from_a_ring(self) -> None:
        bins = 144
        theta = 2.0 * np.pi * (np.arange(bins, dtype=np.float64) + 0.5) / bins
        ring = np.full(bins, 10.0, dtype=np.float64)
        four_lobe = 10.0 * (1.0 + 0.45 * np.cos(4.0 * (theta - np.deg2rad(7.0))))

        ring_metrics = _angular_harmonic_metrics(ring, baseline=0.0)
        lobe_metrics = _angular_harmonic_metrics(four_lobe, baseline=0.0)

        self.assertLess(ring_metrics["inner_edge_h4_relative_amplitude"], 1e-6)
        self.assertAlmostEqual(lobe_metrics["inner_edge_h4_relative_amplitude"], 0.45, places=5)
        self.assertAlmostEqual(lobe_metrics["inner_edge_h4_peak_phase_deg"], 7.0, places=5)

    def test_interior_texture_metrics_reject_sparse_point_morphology(self) -> None:
        size = 192
        yy = np.arange(size, dtype=np.float32) - 0.5 * (size - 1)
        xx = np.arange(size, dtype=np.float32) - 0.5 * (size - 1)
        y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
        rho = np.sqrt(x_grid**2 + y_grid**2) / 78.0
        mask = rho <= 0.72
        rng = np.random.default_rng(11)
        dense = np.full((size, size), 3.0, dtype=np.float32)
        dense[mask] += rng.normal(0.0, 1.5, int(np.sum(mask))).astype(np.float32)
        sparse = np.full((size, size), 3.0, dtype=np.float32)
        indices = rng.choice(np.flatnonzero(mask), size=90, replace=False)
        sparse.flat[indices] += 24.0

        dense_metrics = _interior_texture_metrics(dense, rho)
        sparse_metrics = _interior_texture_metrics(sparse, rho)

        self.assertGreater(
            dense_metrics["interior_fine_texture_coverage_1dn"],
            4.0 * sparse_metrics["interior_fine_texture_coverage_1dn"],
        )
        self.assertGreater(
            sparse_metrics["interior_fine_texture_kurtosis"],
            10.0 * dense_metrics["interior_fine_texture_kurtosis"],
        )

    def test_split_manifest_requires_an_exact_disjoint_partition(self) -> None:
        manifest = {
            "calibration_blocks": [["1.bmp", "2.bmp"], ["3.bmp"]],
            "temporal_test": ["4.bmp"],
        }
        _validate_split_manifest(manifest, ["1.bmp", "2.bmp", "3.bmp", "4.bmp"])

        overlapping = {
            "calibration_blocks": [["1.bmp", "2.bmp"], ["2.bmp", "3.bmp"]],
            "temporal_test": ["4.bmp"],
        }
        with self.assertRaisesRegex(ValueError, "disjoint"):
            _validate_split_manifest(overlapping, ["1.bmp", "2.bmp", "3.bmp", "4.bmp"])

        with self.assertRaisesRegex(ValueError, "mismatch"):
            _validate_split_manifest(manifest, ["1.bmp", "2.bmp", "3.bmp", "4.bmp", "5.bmp"])

    def test_radiometric_fit_does_not_read_eval_images(self) -> None:
        size = 160
        yy = np.arange(size, dtype=np.float32) - 0.5 * (size - 1)
        xx = np.arange(size, dtype=np.float32) - 0.5 * (size - 1)
        y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
        rho = np.sqrt(x_grid**2 + y_grid**2) / 58.0
        base = (
            0.03
            + 0.25 * (1.0 + x_grid / size)
            + 2.0 * np.exp(-0.5 * ((rho - 1.0) / 0.08) ** 2)
        ).astype(np.float32)
        sim_stack = np.stack([base, 1.04 * base], axis=0)
        real_fit = np.stack([np.clip(42.0 * base + offset, 0.0, 255.0) for offset in (3.0, 4.0)])
        eval_a = np.stack([np.clip(38.0 * base + 2.0, 0.0, 255.0)] * 2)
        eval_b = np.stack([np.clip(70.0 * np.flipud(base) + 9.0, 0.0, 255.0)] * 2)

        first, _ = _evaluate_fold(
            sim_stack,
            rho,
            real_fit=real_fit,
            real_eval=eval_a,
            real_rho=rho,
            radiometry_mode="legacy",
        )
        second, _ = _evaluate_fold(
            sim_stack,
            rho,
            real_fit=real_fit,
            real_eval=eval_b,
            real_rho=rho,
            radiometry_mode="legacy",
        )
        self.assertEqual(first["radiometry"], second["radiometry"])
        self.assertNotEqual(first["score"], second["score"])

    def test_dn_mode_never_refits_exposure(self) -> None:
        size = 160
        yy = np.arange(size, dtype=np.float32) - 0.5 * (size - 1)
        xx = np.arange(size, dtype=np.float32) - 0.5 * (size - 1)
        y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
        rho = np.sqrt(x_grid**2 + y_grid**2) / 58.0
        image = (3.0 + 80.0 * np.exp(-0.5 * ((rho - 1.0) / 0.08) ** 2)).astype(np.float32)
        stack = np.stack([image, image], axis=0)
        result, _ = _evaluate_fold(
            stack,
            rho,
            real_fit=2.0 * stack,
            real_eval=stack,
            real_rho=rho,
            radiometry_mode="dn",
        )
        self.assertEqual(result["radiometry"]["scale"], 1.0)
        self.assertEqual(result["radiometry"]["offset"], 0.0)
        self.assertIsNone(result["radiometry"]["fit_profile_rmse_dn"])

    def test_search_override_updates_only_the_requested_leaf(self) -> None:
        data = {"simulation": {"capture_engine_params": {"reflectance": {"rim_amplitude": 3.0}}}}
        _deep_set(data, "simulation.capture_engine_params.reflectance.rim_amplitude", 0.5)
        self.assertEqual(data["simulation"]["capture_engine_params"]["reflectance"]["rim_amplitude"], 0.5)
        with self.assertRaisesRegex(KeyError, "does not resolve"):
            _deep_set(data, "simulation.missing.value", 1.0)


if __name__ == "__main__":
    unittest.main()
