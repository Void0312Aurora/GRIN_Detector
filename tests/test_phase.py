from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    import sys

    sys.path.insert(0, str(src))


_bootstrap_src()

from mini_grin_rebuild.core.configs import SimulationConfig  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import spherical_cap  # noqa: E402
from mini_grin_rebuild.physics.phase import phase_model_meta, phase_scale  # noqa: E402


class TestPhaseScale(unittest.TestCase):
    def test_transmission_matches_legacy_formula_bitwise(self) -> None:
        cfg = SimulationConfig(grid_size=16, dx=0.39, wavelength=0.633, n_object=1.52, n_air=1.0)
        legacy = (2.0 * math.pi / cfg.wavelength) * (cfg.n_object - cfg.n_air)
        self.assertEqual(phase_scale(cfg), legacy)

    def test_reflection_normal_incidence(self) -> None:
        cfg = SimulationConfig(
            grid_size=16,
            dx=0.39,
            wavelength=0.52,
            n_object=1.0,
            n_air=1.0,
            phase_mode="reflection",
        )
        self.assertAlmostEqual(phase_scale(cfg), 4.0 * math.pi / 0.52, places=12)

    def test_reflection_oblique_incidence_uses_cosine(self) -> None:
        cfg = SimulationConfig(
            grid_size=16,
            dx=0.39,
            wavelength=0.52,
            phase_mode="reflection",
            reflection_incidence_angle_deg=30.0,
        )
        expected = (4.0 * math.pi / 0.52) * math.cos(math.radians(30.0))
        self.assertAlmostEqual(phase_scale(cfg), expected, places=12)

    def test_unknown_mode_raises(self) -> None:
        cfg = SimulationConfig(grid_size=16, dx=0.39, phase_mode="holography")
        with self.assertRaises(ValueError):
            phase_scale(cfg)

    def test_grazing_incidence_raises(self) -> None:
        cfg = SimulationConfig(
            grid_size=16,
            dx=0.39,
            phase_mode="reflection",
            reflection_incidence_angle_deg=90.0,
        )
        with self.assertRaises(ValueError):
            phase_scale(cfg)

    def test_nonpositive_ambient_index_raises(self) -> None:
        cfg = SimulationConfig(grid_size=16, dx=0.39, phase_mode="reflection", n_air=-1.0)
        with self.assertRaises(ValueError):
            phase_scale(cfg)

    def test_meta_reports_mode_and_ambient_index(self) -> None:
        cfg = SimulationConfig(grid_size=16, dx=0.39, phase_mode="reflection", n_air=1.0)
        meta = phase_model_meta(cfg)
        self.assertEqual(meta["phase_mode"], "reflection")
        self.assertEqual(meta["n_air"], 1.0)
        self.assertAlmostEqual(meta["phase_scale_rad_per_height_unit"], phase_scale(cfg), places=12)


class TestSphericalCapGeometry(unittest.TestCase):
    def _cfg(self, **overrides) -> SimulationConfig:
        # 256 px at 0.9375 um/px gives a 240 um field of view, comfortably
        # holding the 219 um clear aperture of the R=400/sag=15.34 cap.
        base = dict(
            grid_size=256,
            dx=0.9375,
            scene="microlens_spherical_cap",
            lens_curvature_radius_um=400.0,
            lens_sag_um=15.34,
        )
        base.update(overrides)
        return SimulationConfig(**base)

    def test_cap_apex_and_edge(self) -> None:
        cfg = self._cfg()
        height = spherical_cap(cfg)
        centre = height[cfg.grid_size // 2, cfg.grid_size // 2]
        self.assertAlmostEqual(float(centre), 15.34, delta=1e-3)
        self.assertEqual(float(height[0, 0]), 0.0)
        aperture_radius = math.sqrt(2.0 * 400.0 * 15.34 - 15.34**2)
        self.assertLess(aperture_radius, 0.5 * cfg.grid_size * cfg.dx)

    def test_zero_fillet_is_bit_identical_to_sharp_corner(self) -> None:
        sharp = spherical_cap(self._cfg())
        explicit_zero = spherical_cap(self._cfg(seam_fillet_width_um=0.0))
        np.testing.assert_array_equal(sharp, explicit_zero)

    def test_fillet_keeps_apex_and_far_substrate(self) -> None:
        sharp = spherical_cap(self._cfg())
        filleted = spherical_cap(self._cfg(seam_fillet_width_um=3.0))
        centre = cfg_centre = sharp.shape[0] // 2
        self.assertAlmostEqual(float(filleted[centre, cfg_centre]), float(sharp[centre, cfg_centre]), delta=1e-3)
        self.assertAlmostEqual(float(filleted[0, 0]), 0.0, delta=1e-6)
        self.assertFalse(np.array_equal(filleted, sharp))

    def test_shoulder_offset_zero_is_respected(self) -> None:
        with_zero_offset = spherical_cap(
            self._cfg(seam_shoulder_height_um=0.2, seam_shoulder_offset_um=0.0)
        )
        with_default_offset = spherical_cap(
            self._cfg(seam_shoulder_height_um=0.2, seam_shoulder_offset_um=5.0)
        )
        self.assertFalse(np.array_equal(with_zero_offset, with_default_offset))

    def test_fov_violation_raises(self) -> None:
        cfg = self._cfg(grid_size=64, dx=0.46875)
        with self.assertRaises(ValueError):
            spherical_cap(cfg)


if __name__ == "__main__":
    unittest.main()
