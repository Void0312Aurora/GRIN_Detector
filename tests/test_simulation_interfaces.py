from __future__ import annotations

import json
from dataclasses import replace
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
from mini_grin_rebuild.data.generate_dataset import (  # noqa: E402
    _sample_defect_scattering_modifier,
    generate_dataset,
)
from mini_grin_rebuild.data.virtual_objects import VirtualObject, spherical_cap  # noqa: E402
from mini_grin_rebuild.physics.factory import create_forward_model, forward_model_meta  # noqa: E402
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer  # noqa: E402
from mini_grin_rebuild.physics.phase import phase_scale  # noqa: E402
from mini_grin_rebuild.physics.simulator import simulate_capture  # noqa: E402
from mini_grin_rebuild.simulation.factory import create_simulation_engine  # noqa: E402
from mini_grin_rebuild.simulation.engines.optical_leakage_lite import (  # noqa: E402
    OpticalLeakageLiteEngine,
)
from mini_grin_rebuild.simulation.types import Capture  # noqa: E402


class TestSimulationInterfaces(unittest.TestCase):
    def test_virtual_object_rejects_mismatched_field_modifier(self) -> None:
        cfg = SimulationConfig(grid_size=8)
        with self.assertRaisesRegex(ValueError, "field_modifier"):
            VirtualObject(
                cfg,
                np.zeros((8, 8), dtype=np.float32),
                np.ones((7, 8), dtype=np.complex64),
            )

    def test_scattering_modifier_is_local_and_reproducible(self) -> None:
        cfg = SimulationConfig(
            grid_size=32,
            defect_scattering={
                "enabled": True,
                "probability": 1.0,
                "amplitude_factor": 8.0,
                "rough_phase_rad": 0.8,
                "texture_sigma_px": 1.2,
            },
        )
        defect = np.zeros((32, 32), dtype=np.float32)
        defect[13:19, 13:19] = 1.0
        first, first_meta = _sample_defect_scattering_modifier(
            cfg, defect, rng=np.random.default_rng(41)
        )
        repeated, repeated_meta = _sample_defect_scattering_modifier(
            cfg, defect, rng=np.random.default_rng(41)
        )
        assert first is not None and repeated is not None
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(first[:10, :10], np.ones((10, 10), dtype=np.complex64))
        self.assertTrue(first_meta["applied"])
        self.assertEqual(first_meta, repeated_meta)

    def test_reflection_phase_scale_ignores_sample_index(self) -> None:
        cfg_a = SimulationConfig(phase_mode="reflection", wavelength=0.520, n_object=1.2, n_air=1.0)
        cfg_b = SimulationConfig(phase_mode="reflection", wavelength=0.520, n_object=2.1, n_air=1.0)
        expected = 4.0 * np.pi / 0.520
        self.assertAlmostEqual(phase_scale(cfg_a), expected, places=12)
        self.assertAlmostEqual(phase_scale(cfg_b), expected, places=12)

    def test_physical_spherical_cap_matches_configured_geometry(self) -> None:
        cfg = SimulationConfig(
            scene="microlens_spherical_cap",
            grid_size=513,
            dx=240.0 / 513.0,
            lens_curvature_radius_um=400.0,
            lens_sag_um=15.34,
        )
        height = spherical_cap(cfg)
        center = cfg.grid_size // 2
        self.assertEqual(height.shape, (cfg.grid_size, cfg.grid_size))
        self.assertAlmostEqual(float(height[center, center]), 15.34, places=5)
        self.assertEqual(float(height[0, 0]), 0.0)
        self.assertGreater(int(np.count_nonzero(height)), 0)

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
            self.assertIn("wrap", first)
            self.assertIn(first["wrap"]["wrap_class"], {"in_wrap", "cross_wrap"})
            self.assertIn("estimated_wrap_stress_level", first["wrap"])
            self.assertIn("cross_wrap_fraction", meta["wrap_summary"])
            self.assertLess(sample_meta_path.stat().st_size, 500_000)


class TestOpticalLeakageLiteEngine(unittest.TestCase):
    def _cfg(self) -> SimulationConfig:
        return SimulationConfig(
            grid_size=32,
            dx=0.39,
            noise_level=0.0,
            capture_engine="optical_leakage_lite",
            capture_engine_params={
                "defocus_strength": 600.0,
                "aperture_sigma_freq": 0.08,
                "aberration_strength": 0.05,
                "raw_blur_sigma_px": 0.3,
                "dic_blur_sigma_px": 0.2,
                "raw_gain": 1.0,
                "raw_bias": 0.0,
                "shear_px": 1.5,
            },
        )

    def test_optical_leakage_engine_emits_raw_channel(self) -> None:
        cfg = self._cfg()
        engine = create_simulation_engine(cfg)
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        capture = engine.simulate_capture(height, rng=np.random.default_rng(7))
        self.assertEqual(capture.meta["engine_name"], "optical_leakage_lite")
        self.assertEqual(set(capture.channels.keys()), {"I_x", "I_y", "I_raw"})
        self.assertTrue(np.isfinite(capture.channels["I_raw"]).all())
        self.assertGreaterEqual(float(np.min(capture.channels["I_raw"])), 0.0)

    def test_optical_leakage_can_disable_raw_channel(self) -> None:
        cfg = self._cfg()
        params = dict(cfg.capture_engine_params)
        params["emit_raw"] = False
        cfg = SimulationConfig(
            grid_size=cfg.grid_size,
            dx=cfg.dx,
            noise_level=cfg.noise_level,
            capture_engine=cfg.capture_engine,
            capture_engine_params=params,
        )
        engine = create_simulation_engine(cfg)
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        capture = engine.simulate_capture(height, rng=np.random.default_rng(7))
        self.assertEqual(set(capture.channels.keys()), {"I_x", "I_y"})

    def test_optical_leakage_bundle_is_reproducible(self) -> None:
        cfg = self._cfg()
        engine = create_simulation_engine(cfg)
        base = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        test = base.copy()
        test[cfg.grid_size // 2, cfg.grid_size // 2] = 0.01
        bundle_a = engine.simulate_bundle(
            {"standard": base, "reference": base, "test": test},
            rng=np.random.default_rng(11),
        )  # type: ignore[attr-defined]
        bundle_b = engine.simulate_bundle(
            {"standard": base, "reference": base, "test": test},
            rng=np.random.default_rng(11),
        )  # type: ignore[attr-defined]
        for frame in ("standard", "reference", "test"):
            for key in ("I_x", "I_y", "I_raw"):
                self.assertTrue(
                    np.allclose(bundle_a.captures[frame].channels[key], bundle_b.captures[frame].channels[key], rtol=0.0, atol=1e-7),
                    msg=f"{frame}:{key}",
                )

    def test_optical_field_texture_and_camera_noise_are_reproducible(self) -> None:
        clean_cfg = self._cfg()
        noisy_params = dict(clean_cfg.capture_engine_params)
        noisy_params["field_texture"] = {
            "amplitude_strength": 0.15,
            "phase_strength_rad": 0.2,
            "correlation_sigma_px": 1.5,
            "illumination_strength": 0.05,
            "illumination_sigma_px": 8.0,
            "shared_fraction": 0.0,
        }
        noisy_params["camera"] = {
            "shot_noise": True,
            "photon_gain": 500.0,
            "read_noise_std": 0.002,
            "saturation_level": 10.0,
            "bit_depth": 12,
        }
        noisy_cfg = replace(clean_cfg, capture_engine_params=noisy_params)
        height = np.zeros((clean_cfg.grid_size, clean_cfg.grid_size), dtype=np.float32)

        clean = create_simulation_engine(clean_cfg).simulate_capture(height, rng=np.random.default_rng(23))
        noisy_a = create_simulation_engine(noisy_cfg).simulate_capture(height, rng=np.random.default_rng(23))
        noisy_b = create_simulation_engine(noisy_cfg).simulate_capture(height, rng=np.random.default_rng(23))

        self.assertTrue(np.allclose(noisy_a.channels["I_raw"], noisy_b.channels["I_raw"], rtol=0.0, atol=0.0))
        self.assertGreater(float(np.linalg.norm(noisy_a.channels["I_raw"] - clean.channels["I_raw"])), 1e-4)
        sampled = noisy_a.meta["sampled_params"]
        self.assertIn("field_texture", sampled)
        self.assertIn("camera", sampled)

    def test_localized_coherent_ghost_is_spatially_confined(self) -> None:
        base_cfg = self._cfg()
        clean_params = dict(base_cfg.capture_engine_params)
        clean_params.update(
            {
                "defocus_strength": 0.0,
                "aberration_strength": 0.0,
                "raw_blur_sigma_px": 0.0,
                "dic_blur_sigma_px": 0.0,
            }
        )
        clean_cfg = replace(base_cfg, capture_engine_params=clean_params)
        ghost_params = dict(clean_params)
        ghost_params["coherent_ghost"] = {
            "amplitude": 0.25,
            "tilt_cycles_per_frame": 8.0,
            "tilt_angle_deg": 20.0,
            "defocus_delta": 0.0,
            "phase_offset_rad": 0.3,
            "visibility_envelope": {
                "enabled": True,
                "center_x_norm": -0.35,
                "center_y_norm": -0.25,
                "radius_x_norm": 0.22,
                "radius_y_norm": 0.28,
                "rotation_deg": 15.0,
                "order": 4.0,
            },
        }
        localized_cfg = replace(base_cfg, capture_engine_params=ghost_params)
        height = np.zeros((base_cfg.grid_size, base_cfg.grid_size), dtype=np.float32)

        clean = create_simulation_engine(clean_cfg).simulate_capture(height, rng=np.random.default_rng(31))
        localized = create_simulation_engine(localized_cfg).simulate_capture(height, rng=np.random.default_rng(31))
        residual = np.abs(localized.channels["I_raw"] - clean.channels["I_raw"])

        coord = np.linspace(-1.0, 1.0, base_cfg.grid_size, dtype=np.float32)
        yy, xx = np.meshgrid(coord, coord, indexing="ij")
        theta = np.deg2rad(15.0)
        x_rel = xx + 0.35
        y_rel = yy + 0.25
        x_rot = np.cos(theta) * x_rel + np.sin(theta) * y_rel
        y_rot = -np.sin(theta) * x_rel + np.cos(theta) * y_rel
        distance = (np.abs(x_rot) / 0.22) ** 4 + (np.abs(y_rot) / 0.28) ** 4
        inside = distance <= 1.0
        outside = distance >= 16.0

        self.assertGreater(float(np.mean(residual[inside])), 20.0 * float(np.mean(residual[outside]) + 1e-12))
        sampled = localized.meta["sampled_params"]["coherent_ghost"]
        self.assertTrue(sampled["visibility_envelope"]["enabled"])

    def test_multiple_local_ghost_components_are_reproducible(self) -> None:
        base_cfg = self._cfg()
        params = dict(base_cfg.capture_engine_params)
        params["coherent_ghost"] = {
            "components": [
                {
                    "amplitude": 0.08,
                    "tilt_cycles_per_frame": 5.0,
                    "tilt_angle_deg": 0.0,
                    "source_support": {
                        "center_x_norm": -0.2,
                        "center_y_norm": -0.1,
                        "radius_x_norm": 0.4,
                        "radius_y_norm": 0.3,
                        "order": 2.0,
                    },
                    "source_texture": {
                        "amplitude_strength": 0.2,
                        "phase_strength_rad": 0.35,
                        "correlation_sigma_x_px": 0.8,
                        "correlation_sigma_y_px": 2.0,
                        "correlation_angle_deg": 25.0,
                    },
                },
                {
                    "amplitude": 0.05,
                    "tilt_cycles_per_frame": 7.0,
                    "tilt_angle_deg": 55.0,
                    "visibility_envelope": {
                        "center_x_norm": 0.2,
                        "center_y_norm": 0.1,
                        "radius_x_norm": 0.3,
                        "radius_y_norm": 0.25,
                        "order": 4.0,
                    },
                },
            ]
        }
        cfg = replace(base_cfg, capture_engine_params=params)
        height = np.zeros((base_cfg.grid_size, base_cfg.grid_size), dtype=np.float32)

        capture_a = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(37))
        capture_b = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(37))

        self.assertTrue(np.allclose(capture_a.channels["I_raw"], capture_b.channels["I_raw"], rtol=0.0, atol=0.0))
        sampled = capture_a.meta["sampled_params"]["coherent_ghost"]
        self.assertEqual(len(sampled["components"]), 2)
        self.assertTrue(sampled["components"][0]["source_support"]["enabled"])
        self.assertTrue(sampled["components"][0]["source_texture"]["enabled"])
        self.assertTrue(sampled["components"][1]["visibility_envelope"]["enabled"])

    def test_bundle_shared_texture_correlation_is_configurable(self) -> None:
        base_cfg = self._cfg()
        height = np.zeros((base_cfg.grid_size, base_cfg.grid_size), dtype=np.float32)

        def _cfg(shared_fraction: float) -> SimulationConfig:
            params = dict(base_cfg.capture_engine_params)
            params["field_texture"] = {
                "amplitude_strength": 0.2,
                "phase_strength_rad": 0.2,
                "correlation_sigma_px": 1.0,
                "illumination_strength": 0.0,
                "shared_fraction": shared_fraction,
            }
            return replace(base_cfg, capture_engine_params=params)

        shared = create_simulation_engine(_cfg(1.0)).simulate_bundle(
            {"standard": height, "test": height},
            rng=np.random.default_rng(29),
        )
        independent = create_simulation_engine(_cfg(0.0)).simulate_bundle(
            {"standard": height, "test": height},
            rng=np.random.default_rng(29),
        )

        self.assertTrue(
            np.allclose(
                shared.captures["standard"].channels["I_raw"],
                shared.captures["test"].channels["I_raw"],
                rtol=0.0,
                atol=1e-7,
            )
        )
        self.assertGreater(
            float(
                np.linalg.norm(
                    independent.captures["standard"].channels["I_raw"]
                    - independent.captures["test"].channels["I_raw"]
                )
            ),
            1e-4,
        )

    def test_optical_leakage_raw_channel_separates_sign_flips(self) -> None:
        cfg = self._cfg()
        engine = create_simulation_engine(cfg)
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, cfg.grid_size, dtype=np.float32),
            np.linspace(-1.0, 1.0, cfg.grid_size, dtype=np.float32),
            indexing="ij",
        )
        standard = (0.12 * (xx**2 + yy**2)).astype(np.float32)
        defect = (0.01 * np.exp(-((xx / 0.22) ** 2 + (yy / 0.18) ** 2))).astype(np.float32)

        plus = engine.simulate_capture(standard + defect, rng=np.random.default_rng(3))
        minus = engine.simulate_capture(standard - defect, rng=np.random.default_rng(3))
        raw_sep = float(np.linalg.norm(plus.channels["I_raw"] - minus.channels["I_raw"]))
        dic_sep = 0.5 * (
            float(np.linalg.norm(plus.channels["I_x"] - minus.channels["I_x"]))
            + float(np.linalg.norm(plus.channels["I_y"] - minus.channels["I_y"]))
        )
        self.assertGreater(raw_sep, 1e-5)
        self.assertGreater(raw_sep, 0.25 * dic_sep)

    def test_optical_leakage_aux_sign_direction_flips_with_sign(self) -> None:
        cfg = self._cfg()
        engine = create_simulation_engine(cfg)
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, cfg.grid_size, dtype=np.float32),
            np.linspace(-1.0, 1.0, cfg.grid_size, dtype=np.float32),
            indexing="ij",
        )
        standard = (0.12 * (xx**2 + yy**2)).astype(np.float32)
        magnitude = (0.01 * np.exp(-((xx / 0.22) ** 2 + (yy / 0.18) ** 2))).astype(np.float32)
        probe_scale = 0.2

        pair = engine.simulate_bundle(
            {
                "standard": standard,
                "reference": standard,
                "plus": standard + magnitude,
                "minus": standard - magnitude,
            },
            rng=np.random.default_rng(13),
        )  # type: ignore[attr-defined]
        probe = engine.simulate_bundle(
            {
                "standard": standard,
                "reference": standard,
                "plus_probe": standard + probe_scale * magnitude,
                "minus_probe": standard - probe_scale * magnitude,
            },
            rng=np.random.default_rng(17),
        )  # type: ignore[attr-defined]

        std_raw = pair.captures["standard"].channels["I_raw"]
        raw_plus = pair.captures["plus"].channels["I_raw"] - std_raw
        raw_minus = pair.captures["minus"].channels["I_raw"] - std_raw
        raw_direction = (probe.captures["plus_probe"].channels["I_raw"] - probe.captures["standard"].channels["I_raw"]) - (
            probe.captures["minus_probe"].channels["I_raw"] - probe.captures["standard"].channels["I_raw"]
        )

        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        score_plus = cosine(raw_plus, raw_direction)
        score_minus = cosine(raw_minus, raw_direction)
        self.assertGreater(score_plus, 0.0)
        self.assertLess(score_minus, 0.0)

    def test_dark_port_image_shear_without_leak_matches_legacy(self) -> None:
        cfg = self._cfg()
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        height[10:20, 12:22] = 0.02
        legacy = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(3))
        params = dict(cfg.capture_engine_params)
        params["dark_port"] = {"enabled": True, "mode": "image_shear", "leak_amplitude": 0.0}
        dark_cfg = SimulationConfig(
            grid_size=cfg.grid_size,
            dx=cfg.dx,
            noise_level=cfg.noise_level,
            capture_engine=cfg.capture_engine,
            capture_engine_params=params,
        )
        dark = create_simulation_engine(dark_cfg).simulate_capture(height, rng=np.random.default_rng(3))
        for key in ("I_x", "I_y", "I_raw"):
            np.testing.assert_allclose(dark.channels[key], legacy.channels[key], rtol=1e-5, atol=1e-7)
        self.assertEqual(dark.meta["sampled_params"]["dark_port"]["mode"], "image_shear")

    def test_dark_port_leak_raises_smooth_region_floor(self) -> None:
        cfg = self._cfg()
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)

        def _capture(leak: float):
            params = dict(cfg.capture_engine_params)
            params["dark_port"] = {
                "enabled": True,
                "mode": "image_shear",
                "leak_amplitude": leak,
                "leak_phase_rad": 0.0,
            }
            leak_cfg = SimulationConfig(
                grid_size=cfg.grid_size,
                dx=cfg.dx,
                noise_level=cfg.noise_level,
                capture_engine=cfg.capture_engine,
                capture_engine_params=params,
            )
            return create_simulation_engine(leak_cfg).simulate_capture(height, rng=np.random.default_rng(5))

        without = _capture(0.0)
        with_leak = _capture(0.3)
        self.assertGreater(
            float(np.mean(with_leak.channels["I_x"])),
            float(np.mean(without.channels["I_x"])),
        )

    def test_dark_port_fourier_tilt_modulates_field(self) -> None:
        cfg = self._cfg()
        params = dict(cfg.capture_engine_params)
        params["dark_port"] = {
            "enabled": True,
            "mode": "fourier_tilt",
            "fringe_cycles_per_frame": 4.0,
            "fringe_phase_rad": 0.0,
            "fringe_angle_deg": 0.0,
        }
        tilt_cfg = SimulationConfig(
            grid_size=cfg.grid_size,
            dx=cfg.dx,
            noise_level=cfg.noise_level,
            capture_engine=cfg.capture_engine,
            capture_engine_params=params,
        )
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        capture = create_simulation_engine(tilt_cfg).simulate_capture(height, rng=np.random.default_rng(5))
        column_means = np.mean(capture.channels["I_x"], axis=0)
        self.assertGreater(float(np.std(column_means)), 1e-6)
        self.assertTrue(np.isfinite(capture.channels["I_x"]).all())

    def test_speckle_realizations_reduce_fixture_contrast(self) -> None:
        def _capture(realizations: int):
            cfg = SimulationConfig(
                grid_size=64,
                dx=0.39,
                noise_level=0.0,
                lens_radius_fraction=0.6,
                capture_engine="optical_leakage_lite",
                capture_engine_params={
                    "defocus_strength": 0.0,
                    "aperture_sigma_freq": 0.08,
                    "raw_blur_sigma_px": 0.0,
                    "dic_blur_sigma_px": 0.0,
                    "shear_px": 1.0,
                    "reflectance": {
                        "enabled": True,
                        "lens_amplitude": 0.1,
                        "background_amplitude": 2.0,
                        "background_phase_rough_rad": 2.0,
                        "background_texture_sigma_px": 2.0,
                        "speckle_realizations": realizations,
                    },
                },
            )
            engine = create_simulation_engine(cfg)
            height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
            return create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(11))

        def _corner_contrast(capture) -> float:
            corner = np.asarray(capture.channels["I_raw"][:12, :12], dtype=np.float64)
            return float(np.std(corner) / max(np.mean(corner), 1e-9))

        single = _capture(1)
        averaged = _capture(12)
        self.assertLess(_corner_contrast(averaged), 0.7 * _corner_contrast(single))
        self.assertEqual(
            averaged.meta["sampled_params"]["reflectance"]["speckle_realizations"], 12
        )

    def test_scatter_defect_modifier_lights_up_dark_port(self) -> None:
        cfg = SimulationConfig(
            grid_size=64,
            dx=0.39,
            noise_level=0.0,
            lens_radius_fraction=0.9,
            capture_engine="optical_leakage_lite",
            capture_engine_params={
                "defocus_strength": 0.0,
                "aperture_sigma_freq": 0.08,
                "raw_blur_sigma_px": 0.0,
                "dic_blur_sigma_px": 0.0,
                "shear_px": 1.0,
                "dark_port": {"enabled": True, "mode": "image_shear", "leak_amplitude": 0.0},
            },
        )
        engine = create_simulation_engine(cfg)
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)

        rough = np.random.default_rng(3).uniform(-1.5, 1.5, (cfg.grid_size, cfg.grid_size)).astype(np.float32)
        support = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        support[28:36, 28:36] = 1.0
        scatter = np.exp(1j * support * rough).astype(np.complex64)

        clean = engine.simulate_capture(height, rng=np.random.default_rng(4))
        bundle = engine.simulate_bundle(
            {"standard": height, "test": height},
            rng=np.random.default_rng(4),
            extra_field_modifiers={"test": scatter},
        )
        test_ix = np.asarray(bundle.captures["test"].channels["I_x"], dtype=np.float64)
        std_ix = np.asarray(bundle.captures["standard"].channels["I_x"], dtype=np.float64)
        patch_energy = float(np.mean((test_ix - std_ix)[28:36, 28:36] ** 2))
        outside_energy = float(np.mean((test_ix - std_ix)[:16, :16] ** 2))
        self.assertGreater(patch_energy, 100.0 * max(outside_energy, 1e-12))
        np.testing.assert_allclose(
            std_ix,
            np.asarray(clean.channels["I_x"], dtype=np.float64),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_reflectance_map_separates_lens_and_background(self) -> None:
        cfg = SimulationConfig(
            grid_size=64,
            dx=0.39,
            noise_level=0.0,
            lens_radius_fraction=0.6,
            capture_engine="optical_leakage_lite",
            capture_engine_params={
                "defocus_strength": 0.0,
                "aperture_sigma_freq": 0.08,
                "raw_blur_sigma_px": 0.0,
                "dic_blur_sigma_px": 0.0,
                "shear_px": 1.0,
                "reflectance": {
                    "enabled": True,
                    "lens_amplitude": 1.0,
                    "background_amplitude": 0.0,
                    "background_phase_rough_rad": 0.0,
                },
            },
        )
        engine = create_simulation_engine(cfg)
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        capture = engine.simulate_capture(height, rng=np.random.default_rng(9))
        raw = capture.channels["I_raw"]
        center = float(raw[32, 32])
        corner = float(np.mean(raw[:6, :6]))
        self.assertGreater(center, 10.0 * max(corner, 1e-9))

    def test_sim2real_reflectance_terms_are_default_off(self) -> None:
        base_reflectance = {
            "enabled": True,
            "lens_amplitude": 0.05,
            "background_amplitude": 2.0,
            "background_phase_rough_rad": 1.5,
            "background_texture_sigma_px": 3.0,
            "rim_amplitude": 1.0,
            "rim_width_px": 2.0,
            "speckle_realizations": 2,
        }

        def _capture(reflectance: dict[str, float | int | bool]):
            cfg = SimulationConfig(
                grid_size=48,
                dx=0.39,
                noise_level=0.0,
                lens_radius_fraction=0.7,
                capture_engine="optical_leakage_lite",
                capture_engine_params={
                    "defocus_strength": 0.0,
                    "aperture_sigma_freq": 0.08,
                    "raw_blur_sigma_px": 0.0,
                    "dic_blur_sigma_px": 0.0,
                    "shear_px": 1.0,
                    "reflectance": reflectance,
                    "dark_port": {"enabled": True, "mode": "image_shear"},
                },
            )
            height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
            return create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(17))

        legacy = _capture(base_reflectance)
        explicit_off = _capture(
            {
                **base_reflectance,
                "background_phase_fine_fraction": 0.0,
                "background_amplitude_texture_strength": 0.0,
                "rim_angular_modulation_strength": 0.0,
            }
        )
        for key in ("I_raw", "I_x", "I_y"):
            np.testing.assert_array_equal(explicit_off.channels[key], legacy.channels[key])

    def test_sim2real_reflectance_terms_are_reproducible_and_effective(self) -> None:
        def _capture(*, enhanced: bool, seed: int):
            reflectance: dict[str, float | int | bool] = {
                "enabled": True,
                "lens_amplitude": 0.05,
                "background_amplitude": 2.0,
                "background_phase_rough_rad": 1.5,
                "background_texture_sigma_px": 4.0,
                "rim_amplitude": 1.5,
                "rim_width_px": 2.0,
                "speckle_realizations": 3,
            }
            if enhanced:
                reflectance.update(
                    {
                        "background_phase_fine_fraction": 0.45,
                        "background_phase_fine_sigma_px": 0.8,
                        "background_amplitude_texture_strength": 0.3,
                        "background_amplitude_texture_sigma_px": 8.0,
                        "rim_texture_sigma_px": 1.0,
                        "rim_angular_modulation_strength": 0.5,
                        "rim_angular_correlation_deg": 18.0,
                    }
                )
            cfg = SimulationConfig(
                grid_size=64,
                dx=0.39,
                noise_level=0.0,
                lens_radius_fraction=0.7,
                capture_engine="optical_leakage_lite",
                capture_engine_params={
                    "defocus_strength": 0.0,
                    "aperture_sigma_freq": 0.08,
                    "raw_blur_sigma_px": 0.0,
                    "dic_blur_sigma_px": 0.0,
                    "shear_px": 1.0,
                    "reflectance": reflectance,
                    "dark_port": {"enabled": True, "mode": "image_shear"},
                },
            )
            height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
            return create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(seed))

        first = _capture(enhanced=True, seed=23)
        repeated = _capture(enhanced=True, seed=23)
        baseline = _capture(enhanced=False, seed=23)
        np.testing.assert_array_equal(first.channels["I_x"], repeated.channels["I_x"])
        self.assertGreater(
            float(np.mean(np.abs(first.channels["I_x"] - baseline.channels["I_x"]))),
            1e-6,
        )
        sampled = first.meta["sampled_params"]["reflectance"]
        self.assertEqual(sampled["background_phase_fine_fraction"], 0.45)
        self.assertEqual(sampled["rim_angular_modulation_strength"], 0.5)

    def test_reflectance_rim_sigma_reuses_sampled_background_sigma(self) -> None:
        cfg = SimulationConfig(
            grid_size=32,
            dx=0.39,
            noise_level=0.0,
            lens_radius_fraction=0.7,
            capture_engine="optical_leakage_lite",
            capture_engine_params={
                "reflectance": {
                    "enabled": True,
                    "background_phase_rough_rad": 1.0,
                    "background_texture_sigma_px": [1.0, 5.0],
                    "rim_amplitude": 1.0,
                    "rim_phase_rough_rad": 1.0,
                }
            },
        )
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        capture = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(31))
        sampled = capture.meta["sampled_params"]["reflectance"]
        self.assertEqual(sampled["rim_texture_sigma_px"], sampled["background_texture_sigma_px"])

    def test_reflectance_rim_roughness_reuses_sampled_background_roughness(self) -> None:
        cfg = SimulationConfig(
            grid_size=32,
            dx=0.39,
            noise_level=0.0,
            lens_radius_fraction=0.7,
            capture_engine="optical_leakage_lite",
            capture_engine_params={
                "reflectance": {
                    "enabled": True,
                    "background_phase_rough_rad": [1.0, 2.0],
                    "background_texture_sigma_px": 3.0,
                    "rim_amplitude": 1.0,
                }
            },
        )
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        capture = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(33))
        sampled = capture.meta["sampled_params"]["reflectance"]
        self.assertEqual(sampled["rim_phase_rough_rad"], sampled["background_phase_rough_rad"])

    def test_rim_angular_field_can_remove_random_dipole(self) -> None:
        field = OpticalLeakageLiteEngine._periodic_angular_random_field(
            np.random.default_rng(41),
            (192, 192),
            correlation_deg=18.0,
            remove_dipole=True,
        )
        yy = np.arange(field.shape[0], dtype=np.float32) - 0.5 * (field.shape[0] - 1)
        xx = np.arange(field.shape[1], dtype=np.float32) - 0.5 * (field.shape[1] - 1)
        y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
        radius = np.sqrt(x_grid**2 + y_grid**2)
        annulus = (radius >= 60.0) & (radius <= 70.0)
        theta = np.arctan2(y_grid[annulus], x_grid[annulus])
        values = field[annulus] - float(np.mean(field[annulus]))
        cosine = abs(float(np.mean(values * np.cos(theta)))) / max(float(np.std(values)), 1e-6)
        sine = abs(float(np.mean(values * np.sin(theta)))) / max(float(np.std(values)), 1e-6)
        self.assertLess(cosine, 0.02)
        self.assertLess(sine, 0.02)
        self.assertGreater(float(np.std(values)), 0.5)

    def test_ordered_rim_focus_has_four_axis_lobes(self) -> None:
        cfg = SimulationConfig(
            grid_size=192,
            lens_radius_fraction=0.7,
            capture_engine="optical_leakage_lite",
        )
        engine = OpticalLeakageLiteEngine(cfg)
        field = engine._rim_focus_intensity(
            (cfg.grid_size, cfg.grid_size),
            {
                "rim_focus_amplitude": 1.0,
                "rim_focus_order": 4,
                "rim_focus_phase_deg": 0.0,
                "rim_focus_radial_offset_px": 0.0,
                "rim_focus_width_px": 2.0,
                "rim_focus_angular_power": 2.0,
            },
        )
        assert field is not None
        radius = int(round(0.5 * cfg.grid_size * cfg.lens_radius_fraction))
        center = (cfg.grid_size - 1) / 2.0

        def _sample(angle_deg: float) -> float:
            angle = np.deg2rad(angle_deg)
            y = int(round(center + radius * np.sin(angle)))
            x = int(round(center + radius * np.cos(angle)))
            return float(field[y, x])

        axis_mean = float(np.mean([_sample(angle) for angle in (0.0, 90.0, 180.0, 270.0)]))
        diagonal_mean = float(np.mean([_sample(angle) for angle in (45.0, 135.0, 225.0, 315.0)]))
        self.assertGreater(axis_mean, 20.0 * max(diagonal_mean, 1e-9))

    def test_lens_scatter_common_basis_is_coordinate_locked(self) -> None:
        cfg = SimulationConfig(grid_size=64, capture_engine="optical_leakage_lite")
        engine = OpticalLeakageLiteEngine(cfg)
        params = {
            "lens_scatter_common_fraction": 0.6,
            "lens_scatter_capture_fraction": 0.4,
            "lens_scatter_common_seed": 123,
            "lens_texture_sigma_px": 4.0,
            "lens_scatter_fine_fraction": 0.7,
            "lens_scatter_fine_sigma_px": 0.7,
            "background_amplitude_texture_strength": 0.0,
            "rim_angular_modulation_strength": 0.0,
            "rim_angular_harmonics": [],
        }
        first = engine._sample_reflectance_context(np.random.default_rng(1), (64, 64), params)
        second = engine._sample_reflectance_context(np.random.default_rng(2), (64, 64), params)
        np.testing.assert_array_equal(first["lens_scatter_common"], second["lens_scatter_common"])
        self.assertGreater(
            float(np.mean(np.abs(first["lens_scatter_capture"] - second["lens_scatter_capture"]))),
            1e-3,
        )

    def test_camera_common_fixed_pattern_survives_capture_seed(self) -> None:
        cfg = SimulationConfig(
            grid_size=64,
            dx=0.39,
            noise_level=0.0,
            lens_radius_fraction=0.7,
            capture_engine="optical_leakage_lite",
            capture_engine_params={
                "defocus_strength": 0.0,
                "aperture_sigma_freq": 0.08,
                "raw_blur_sigma_px": 0.0,
                "dic_blur_sigma_px": 0.0,
                "camera": {
                    "output_mode": "dn",
                    "exposure_scale": 1.0,
                    "black_level": 10.0,
                    "shot_noise": False,
                    "read_noise_std": 0.0,
                    "fixed_pattern_std": 2.0,
                    "fixed_pattern_correlation_sigma_px": 3.0,
                    "fixed_pattern_fine_fraction": 0.7,
                    "fixed_pattern_fine_sigma_px": 0.7,
                    "fixed_pattern_common_fraction": 1.0,
                    "fixed_pattern_common_seed": 456,
                    "fixed_pattern_lens_only": True,
                    "saturation_level": 255.0,
                    "bit_depth": 8,
                },
            },
        )
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        first = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(1))
        second = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(2))
        np.testing.assert_array_equal(first.channels["I_x"], second.channels["I_x"])
        yy = np.arange(cfg.grid_size, dtype=np.float32) - 0.5 * (cfg.grid_size - 1)
        xx = np.arange(cfg.grid_size, dtype=np.float32) - 0.5 * (cfg.grid_size - 1)
        y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
        radius = np.sqrt(x_grid**2 + y_grid**2)
        lens = radius <= 0.45 * cfg.grid_size * cfg.lens_radius_fraction
        outside = radius >= 0.55 * cfg.grid_size * cfg.lens_radius_fraction
        self.assertGreater(float(np.std(first.channels["I_x"][lens])), 0.5)
        self.assertLess(float(np.std(first.channels["I_x"][outside])), 1e-6)

    def test_camera_dn_mode_applies_exposure_black_level_and_adc(self) -> None:
        cfg = SimulationConfig(
            grid_size=32,
            dx=0.39,
            noise_level=0.0,
            capture_engine="optical_leakage_lite",
            capture_engine_params={
                "raw_blur_sigma_px": 0.0,
                "dic_blur_sigma_px": 0.0,
                "camera": {
                    "output_mode": "dn",
                    "exposure_scale": 50.0,
                    "black_level": 3.0,
                    "shot_noise": False,
                    "read_noise_std": 0.0,
                    "saturation_level": 255.0,
                    "bit_depth": 8,
                    "pre_adc_blur_sigma_px": 0.7,
                    "post_blur_sigma_px": 0.0,
                },
            },
        )
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        height[13:19, 13:19] = 0.02
        capture = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(37))
        for channel in capture.channels.values():
            self.assertGreaterEqual(float(np.min(channel)), 3.0)
            self.assertLessEqual(float(np.max(channel)), 255.0)
            np.testing.assert_array_equal(channel, np.round(channel))
        sampled = capture.meta["sampled_params"]["camera"]
        self.assertEqual(sampled["output_mode"], "dn")
        self.assertEqual(sampled["exposure_scale"], 50.0)
        self.assertEqual(sampled["black_level"], 3.0)

    def test_camera_correlated_read_noise_is_reproducible(self) -> None:
        cfg = SimulationConfig(
            grid_size=64,
            dx=0.39,
            noise_level=0.0,
            capture_engine="optical_leakage_lite",
            capture_engine_params={
                "raw_blur_sigma_px": 0.0,
                "dic_blur_sigma_px": 0.0,
                "camera": {
                    "output_mode": "dn",
                    "exposure_scale": 0.0,
                    "black_level": 20.0,
                    "shot_noise": False,
                    "read_noise_std": 2.0,
                    "read_noise_correlation_sigma_px": 1.0,
                    "saturation_level": 255.0,
                    "bit_depth": 0,
                },
            },
        )
        height = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
        first = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(43))
        repeated = create_simulation_engine(cfg).simulate_capture(height, rng=np.random.default_rng(43))
        np.testing.assert_array_equal(first.channels["I_x"], repeated.channels["I_x"])
        noise = first.channels["I_x"] - 20.0
        self.assertAlmostEqual(float(np.std(noise)), 2.0, delta=0.05)
        horizontal_corr = float(np.corrcoef(noise[:, :-1].ravel(), noise[:, 1:].ravel())[0, 1])
        self.assertGreater(horizontal_corr, 0.25)
        sampled = first.meta["sampled_params"]["camera"]
        self.assertEqual(sampled["read_noise_correlation_sigma_px"], 1.0)

    def test_generate_dataset_stores_raw_observations(self) -> None:
        cfg = self._cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            generate_dataset(cfg, output_root=root, train=1, val=0, test=0, seed=5)
            with np.load(root / "train" / "sample_0000.npz") as sample:
                self.assertIn("raw_standard", sample)
                self.assertIn("raw_reference", sample)
                self.assertIn("raw_test", sample)
                self.assertEqual(sample["raw_standard"].shape, sample["ix_standard"].shape)

    def test_generate_dataset_applies_scattering_to_test_only(self) -> None:
        base_cfg = SimulationConfig(
            grid_size=48,
            dx=0.8,
            scene="microlens_srt",
            lens_radius_fraction=0.7,
            noise_level=0.0,
            capture_engine="optical_leakage_lite",
            capture_engine_params={
                "defocus_strength": 0.0,
                "aperture_sigma_freq": 0.08,
                "raw_blur_sigma_px": 0.0,
                "dic_blur_sigma_px": 0.0,
                "shear_px": 1.0,
                "dark_port": {"enabled": True, "mode": "image_shear"},
            },
        )
        scatter_cfg = replace(
            base_cfg,
            defect_scattering={
                "enabled": True,
                "probability": 1.0,
                "amplitude_factor": 10.0,
                "rough_phase_rad": 1.0,
                "texture_sigma_px": 1.0,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_root = Path(tmpdir) / "baseline"
            scatter_root = Path(tmpdir) / "scatter"
            generate_dataset(base_cfg, output_root=baseline_root, train=1, val=0, test=0, seed=43)
            generate_dataset(scatter_cfg, output_root=scatter_root, train=1, val=0, test=0, seed=43)
            with np.load(baseline_root / "train" / "sample_0000.npz") as baseline, np.load(
                scatter_root / "train" / "sample_0000.npz"
            ) as scattered:
                for key in ("standard", "reference", "test", "defect", "ix_standard", "iy_standard"):
                    np.testing.assert_array_equal(scattered[key], baseline[key])
                self.assertGreater(
                    float(np.mean(np.abs(scattered["diff_ix_st"] - baseline["diff_ix_st"]))),
                    1e-6,
                )
            sample_meta = json.loads((scatter_root / "sample_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(sample_meta["samples"][0]["defect_scattering"]["applied"])

    def test_generate_dataset_rejects_scattering_for_unsupported_engine(self) -> None:
        cfg = SimulationConfig(
            grid_size=32,
            dx=0.39,
            capture_engine="ideal_gradient",
            defect_scattering={
                "enabled": True,
                "probability": 1.0,
                "amplitude_factor": 8.0,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "does not support simulation.defect_scattering"):
                generate_dataset(
                    cfg,
                    output_root=Path(tmpdir) / "dataset",
                    train=1,
                    val=0,
                    test=0,
                    seed=47,
                )


if __name__ == "__main__":
    unittest.main()
