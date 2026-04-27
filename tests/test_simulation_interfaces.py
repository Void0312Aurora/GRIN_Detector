from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


_bootstrap_src()

from mini_grin_rebuild.core.configs import ExperimentConfig, SimulationConfig, TrainingConfig  # noqa: E402
from mini_grin_rebuild.data.generate_dataset import generate_dataset  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import VirtualObject  # noqa: E402
from mini_grin_rebuild.physics.factory import create_forward_model, forward_model_meta  # noqa: E402
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer  # noqa: E402
from mini_grin_rebuild.physics.simulator import simulate_capture  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402
from mini_grin_rebuild.simulation.types import Capture  # noqa: E402


class TestSimulationInterfaces(unittest.TestCase):
    def test_default_capture_engine_matches_legacy_wrapper(self) -> None:
        cfg = SimulationConfig(grid_size=16, dx=0.39, noise_level=0.0)
        height = np.linspace(0.0, 1.0, cfg.grid_size * cfg.grid_size, dtype=np.float32).reshape(
            cfg.grid_size, cfg.grid_size
        )
        engine = create_simulation_engine(cfg)
        capture = engine.simulate_capture(height)
        wrapped = simulate_capture(cfg, VirtualObject(cfg, height))

        self.assertIsInstance(capture, Capture)
        self.assertEqual(capture.meta["engine_name"], "ideal_gradient")
        for key in ("I_x", "I_y"):
            self.assertTrue(np.allclose(capture.channels[key], wrapped[key], rtol=0.0, atol=1e-7), msg=key)

    def test_default_forward_model_factory_returns_frozen_ideal_layer(self) -> None:
        sim_cfg = SimulationConfig(grid_size=16, dx=0.39, noise_level=0.0)
        train_cfg = TrainingConfig(forward_model="ideal_gradient")
        model = create_forward_model(sim_cfg, train_cfg, device="cpu", freeze=True)

        self.assertIsInstance(model, DifferentiableGradientLayer)
        self.assertEqual(forward_model_meta(train_cfg)["forward_model"], "ideal_gradient")
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

        height = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
        out = model(height)
        self.assertEqual(set(out.keys()), {"I_x", "I_y"})
        self.assertEqual(tuple(out["I_x"].shape), (1, 1, 16, 16))

    def test_config_accepts_engine_fields(self) -> None:
        cfg = ExperimentConfig.from_dict(
            {
                "schema_version": 1,
                "simulation": {"capture_engine": "ideal_gradient", "capture_engine_params": {"a": 1}},
                "training": {"forward_model": "ideal_gradient", "forward_model_params": {"b": 2}},
            }
        )
        self.assertEqual(cfg.simulation.capture_engine, "ideal_gradient")
        self.assertEqual(cfg.simulation.capture_engine_params["a"], 1)
        self.assertEqual(cfg.training.forward_model, "ideal_gradient")
        self.assertEqual(cfg.training.forward_model_params["b"], 2)

class TestInstrumentLiteEngine(unittest.TestCase):
    def _cfg(self) -> SimulationConfig:
        return SimulationConfig(
            grid_size=16,
            dx=0.39,
            noise_level=0.0,
            capture_engine="instrument_lite",
            capture_engine_params={
                "pipeline": ["response", "optics", "illumination", "geometry", "camera"],
                "response": {"gain": [0.8, 1.2], "bias": [0.0, 0.01], "saturation": [5.0, 10.0], "cross_talk": [0.0, 0.05]},
                "optics": {"psf_sigma_px": [0.1, 0.5], "edge_falloff_width_px": [0.0, 2.0], "outside_leakage": [0.0, 0.02]},
                "illumination": {"field_strength": [0.0, 0.05], "bias": [0.0, 0.01], "lowres_size": 3},
                "geometry": {"capture_shift_px": [-0.2, 0.2], "channel_shift_px": [-0.1, 0.1], "rotation_deg": [-0.05, 0.05], "scale": [0.999, 1.001]},
                "camera": {"shot_noise": True, "photon_gain": 1000.0, "read_noise_std": [0.0, 0.001], "saturation_level": 20.0, "bit_depth": 12, "bad_pixel_fraction": 0.0},
            },
        )

    def test_instrument_lite_bundle_is_finite_and_reproducible(self) -> None:
        cfg = self._cfg()
        engine = create_simulation_engine(cfg)
        base = np.linspace(0.0, 0.1, cfg.grid_size * cfg.grid_size, dtype=np.float32).reshape(cfg.grid_size, cfg.grid_size)
        heights = {
            "standard": base,
            "reference": base * 0.9,
            "test": base + np.eye(cfg.grid_size, dtype=np.float32) * 0.01,
        }

        bundle_a = engine.simulate_bundle(heights, rng=np.random.default_rng(123))  # type: ignore[attr-defined]
        bundle_b = engine.simulate_bundle(heights, rng=np.random.default_rng(123))  # type: ignore[attr-defined]
        self.assertEqual(bundle_a.meta["engine_name"], "instrument_lite")
        self.assertEqual(bundle_a.meta["pipeline"], ["response", "optics", "illumination", "geometry", "camera"])
        for frame in heights:
            cap_a = bundle_a.captures[frame]
            cap_b = bundle_b.captures[frame]
            for key in ("I_x", "I_y"):
                arr = cap_a.channels[key]
                self.assertEqual(arr.shape, heights[frame].shape)
                self.assertTrue(np.isfinite(arr).all())
                self.assertGreaterEqual(float(np.min(arr)), 0.0)
                self.assertTrue(np.allclose(arr, cap_b.channels[key], rtol=0.0, atol=1e-7), msg=f"{frame}:{key}")

    def test_instrument_lite_pipeline_can_disable_modules(self) -> None:
        cfg = SimulationConfig(
            grid_size=8,
            dx=0.39,
            noise_level=0.0,
            capture_engine="instrument_lite",
            capture_engine_params={"pipeline": ["response"], "response": {"gain": 1.0, "bias": 0.0, "saturation": 1e9}},
        )
        engine = create_simulation_engine(cfg)
        height = np.ones((8, 8), dtype=np.float32)
        capture = engine.simulate_capture(height, rng=np.random.default_rng(1))
        self.assertEqual(capture.meta["pipeline"], ["response"])
        self.assertEqual(set(capture.channels.keys()), {"I_x", "I_y"})

    def test_generate_dataset_writes_sample_simulation_meta(self) -> None:
        cfg = self._cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            meta = generate_dataset(cfg, output_root=root, train=1, val=1, test=1, seed=123)
            self.assertEqual(meta["capture_engine"]["engine_name"], "instrument_lite")
            sample_meta_path = root / "sample_meta.json"
            self.assertTrue(sample_meta_path.is_file())
            sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
            self.assertEqual(len(sample_meta["samples"]), 3)
            first = sample_meta["samples"][0]
            self.assertEqual(first["simulation"]["engine_name"], "instrument_lite")
            self.assertIn("sampled_bundle_params", first["simulation"])
            self.assertIn("response", first["simulation"]["sampled_bundle_params"])
            self.assertEqual(
                first["simulation"]["sampled_bundle_params"]["illumination"]["field"],
                {"omitted": "field"},
            )
            self.assertLess(sample_meta_path.stat().st_size, 500_000)


if __name__ == "__main__":
    unittest.main()
