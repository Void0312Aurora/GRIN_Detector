from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


_bootstrap_src()

from mini_grin_rebuild.core.configs import SimulationConfig  # noqa: E402
from mini_grin_rebuild.data.external_topography import (  # noqa: E402
    load_plux_topography,
    read_plux_metadata,
    simulate_spdic_from_height,
)


def _write_fake_plux(path: Path) -> np.ndarray:
    height = np.array(
        [
            [np.nan, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ],
        dtype=np.float32,
    )
    xml = """<?xml version="1.0" encoding="utf-8"?>
<xml>
  <GENERAL>
    <FOV_X>1.0</FOV_X>
    <FOV_Y>0.6</FOV_Y>
    <IMAGE_SIZE_X>4</IMAGE_SIZE_X>
    <IMAGE_SIZE_Y>3</IMAGE_SIZE_Y>
  </GENERAL>
  <Instrument>
    <Manufacturer>UnitTest</Manufacturer>
    <Model>FakeScope</Model>
  </Instrument>
  <INFO>
    <ITEM_0>
      <NAME>Technique</NAME>
      <VALUE>Interferometric</VALUE>
    </ITEM_0>
    <ITEM_1>
      <NAME>Measure type</NAME>
      <VALUE>Topography</VALUE>
    </ITEM_1>
    <ITEM_2>
      <NAME>Algorithm</NAME>
      <VALUE>CSI</VALUE>
    </ITEM_2>
    <ITEM_3>
      <NAME>Measured</NAME>
      <VALUE>95%</VALUE>
    </ITEM_3>
  </INFO>
  <LAYER_0>
    <FILENAME_Z>LAYER_0.raw</FILENAME_Z>
  </LAYER_0>
</xml>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("index.xml", xml)
        zf.writestr("LAYER_0.raw", height.astype("<f4").tobytes())
    return height


class TestPluxImport(unittest.TestCase):
    def test_read_plux_metadata_and_load_height(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fake.plux"
            raw = _write_fake_plux(path)

            metadata = read_plux_metadata(path)
            self.assertEqual(metadata.image_size_x, 4)
            self.assertEqual(metadata.image_size_y, 3)
            self.assertEqual(metadata.technique, "Interferometric")
            self.assertEqual(metadata.measure_type, "Topography")
            self.assertAlmostEqual(float(metadata.measured_fraction), 0.95)
            self.assertAlmostEqual(float(metadata.fov_x_um), 1000.0)
            self.assertAlmostEqual(float(metadata.fov_y_um), 600.0)
            self.assertAlmostEqual(float(metadata.pixel_pitch_x_um), 250.0)
            self.assertAlmostEqual(float(metadata.pixel_pitch_y_um), 200.0)

            loaded = load_plux_topography(path)
            self.assertEqual(loaded.height_um.shape, (3, 4))
            self.assertEqual(loaded.valid_mask.shape, (3, 4))
            self.assertEqual(int(np.count_nonzero(loaded.valid_mask)), raw.size - 1)
            self.assertTrue(np.isfinite(loaded.height_um).all())
            self.assertAlmostEqual(float(np.mean(loaded.height_um[loaded.valid_mask])), 0.0, places=5)

    def test_simulate_spdic_from_loaded_topography(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fake.plux"
            _write_fake_plux(path)
            loaded = load_plux_topography(path)
            cfg = SimulationConfig(dx=225.0, noise_level=0.0, grid_size=loaded.height_um.shape[0])
            sim = simulate_spdic_from_height(
                cfg,
                height_um=loaded.height_um,
                pixel_pitch_x_um=float(loaded.metadata.pixel_pitch_x_um),
                pixel_pitch_y_um=float(loaded.metadata.pixel_pitch_y_um),
            )
            self.assertEqual(sim["phase"].shape, loaded.height_um.shape)
            self.assertEqual(sim["I_x"].shape, loaded.height_um.shape)
            self.assertEqual(sim["I_y"].shape, loaded.height_um.shape)
            self.assertTrue(np.isfinite(sim["phase"]).all())
            self.assertTrue(np.isfinite(sim["I_x"]).all())
            self.assertTrue(np.isfinite(sim["I_y"]).all())
            self.assertGreaterEqual(float(np.min(sim["I_x"])), 0.0)
            self.assertGreaterEqual(float(np.min(sim["I_y"])), 0.0)


class TestRealSubsetSmoke(unittest.TestCase):
    SAMPLE = (
        Path(__file__).resolve().parents[1]
        / "external_data"
        / "raw"
        / "zenodo_10365872_subset"
        / "Fig3_CSI_50x_AM.plux"
    )

    @unittest.skipUnless(SAMPLE.exists(), "local external sample not available")
    def test_real_subset_loads_and_simulates(self) -> None:
        loaded = load_plux_topography(self.SAMPLE)
        self.assertEqual(loaded.height_um.shape, (1024, 1224))
        self.assertGreater(float(np.mean(loaded.valid_mask)), 0.95)
        cfg = SimulationConfig(
            dx=float(
                0.5 * (float(loaded.metadata.pixel_pitch_x_um) + float(loaded.metadata.pixel_pitch_y_um))
            ),
            noise_level=0.0,
            grid_size=loaded.height_um.shape[0],
        )
        sim = simulate_spdic_from_height(
            cfg,
            height_um=loaded.height_um,
            pixel_pitch_x_um=float(loaded.metadata.pixel_pitch_x_um),
            pixel_pitch_y_um=float(loaded.metadata.pixel_pitch_y_um),
        )
        self.assertTrue(np.isfinite(sim["phase"]).all())
        self.assertTrue(np.isfinite(sim["I_x"]).all())
        self.assertTrue(np.isfinite(sim["I_y"]).all())
        self.assertGreater(float(np.max(sim["I_x"])), 0.0)
        self.assertGreater(float(np.max(sim["I_y"])), 0.0)


if __name__ == "__main__":
    unittest.main()
