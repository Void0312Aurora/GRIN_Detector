from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def _bootstrap_src() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    src = project_root / "src"
    import sys

    for path in (src, workspace_root / "Archive", workspace_root):
        if path.exists():
            sys.path.insert(0, str(path))


_bootstrap_src()

from mini_grin_rebuild.core.configs import SimulationConfig as NewSimCfg  # noqa: E402
from mini_grin_rebuild.core.configs import TrainingConfig as NewTrainCfg  # noqa: E402
from mini_grin_rebuild.data.datasets import DefectDataset as NewDataset  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import VirtualObject as NewObj  # noqa: E402
from mini_grin_rebuild.data.virtual_objects import build_triplet as new_build_triplet  # noqa: E402
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer as NewLayer  # noqa: E402
from mini_grin_rebuild.physics.simulator import simulate_capture as new_simulate_capture  # noqa: E402
from mini_grin_rebuild.training.inputs import build_inputs as new_build_inputs  # noqa: E402
from mini_grin_rebuild.training.losses import sr_diff_loss as new_sr_diff_loss  # noqa: E402


class TestLegacyCompatibility(unittest.TestCase):
    def test_virtual_objects_build_triplet_matches_legacy(self) -> None:
        from mini_grin.core.configs import SimulationConfig as OldSimCfg
        from mini_grin.data.virtual_objects import build_triplet as old_build_triplet

        old_cfg = OldSimCfg(grid_size=32, noise_level=0.0, height_scale=250.0, dx=0.39)
        new_cfg = NewSimCfg(grid_size=32, noise_level=0.0, height_scale=250.0, dx=0.39)

        old_triplet = old_build_triplet(old_cfg)
        new_triplet = new_build_triplet(new_cfg)

        for key in ("standard", "reference", "test", "defect"):
            a = old_triplet[key].height_map
            b = new_triplet[key].height_map
            self.assertTrue(np.allclose(a, b, rtol=0.0, atol=1e-7), msg=f"mismatch: {key}")

    def test_simulator_capture_matches_legacy_no_noise(self) -> None:
        from mini_grin.core.configs import SimulationConfig as OldSimCfg
        from mini_grin.data.virtual_objects import VirtualObject as OldObj
        from mini_grin.data.virtual_objects import build_triplet as old_build_triplet
        from mini_grin.physics.simulator import simulate_capture as old_simulate_capture

        old_cfg = OldSimCfg(grid_size=32, noise_level=0.0, height_scale=250.0, dx=0.39)
        new_cfg = NewSimCfg(grid_size=32, noise_level=0.0, height_scale=250.0, dx=0.39)

        old_triplet = old_build_triplet(old_cfg)
        new_triplet = new_build_triplet(new_cfg)

        for name in ("standard", "reference", "test"):
            old_obj = OldObj(old_cfg, old_triplet[name].height_map)
            new_obj = NewObj(new_cfg, new_triplet[name].height_map)
            old_cap = old_simulate_capture(old_cfg, old_obj)
            new_cap = new_simulate_capture(new_cfg, new_obj)
            for k in ("I_x", "I_y"):
                self.assertTrue(np.allclose(old_cap[k], new_cap[k], rtol=0.0, atol=1e-7), msg=f"{name}:{k}")

    def test_differentiable_layer_matches_legacy_defaults(self) -> None:
        from mini_grin.core.configs import SimulationConfig as OldSimCfg
        from mini_grin.physics.layer import DifferentiableGradientLayer as OldLayer

        old_cfg = OldSimCfg(grid_size=32, noise_level=0.0, height_scale=250.0, dx=0.39)
        new_cfg = NewSimCfg(grid_size=32, noise_level=0.0, height_scale=250.0, dx=0.39)

        height = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        old_layer = OldLayer(old_cfg).cpu()
        new_layer = NewLayer(new_cfg).cpu()
        new_layer.load_state_dict(old_layer.state_dict(), strict=True)

        old_out = old_layer(height)
        new_out = new_layer(height)
        for k in ("I_x", "I_y"):
            self.assertTrue(torch.allclose(old_out[k], new_out[k], rtol=0.0, atol=1e-6), msg=k)

    def test_build_inputs_matches_legacy(self) -> None:
        from mini_grin.core.configs import SimulationConfig as OldSimCfg
        from mini_grin.core.configs import TrainingConfig as OldTrainCfg
        from mini_grin.physics.layer import DifferentiableGradientLayer as OldLayer
        from mini_grin.training.trainer import build_inputs as old_build_inputs

        h, w = 16, 16
        old_sim = OldSimCfg(grid_size=h, noise_level=0.0, height_scale=250.0, dx=0.39)
        new_sim = NewSimCfg(grid_size=h, noise_level=0.0, height_scale=250.0, dx=0.39, gradient_backend="spectral")

        old_train = OldTrainCfg(use_sr_inputs=True, use_raw_intensity_inputs=True, use_phase_inputs=True)
        new_train = NewTrainCfg(use_sr_inputs=True, use_raw_intensity_inputs=True, use_phase_inputs=True)

        old_phys = OldLayer(old_sim).cpu()
        new_phys = NewLayer(new_sim).cpu()
        new_phys.load_state_dict(old_phys.state_dict(), strict=True)

        diff_st = {"I_x": torch.randn(h, w), "I_y": torch.randn(h, w)}
        diff_sr = {"I_x": torch.randn(h, w), "I_y": torch.randn(h, w)}
        intensity_inputs = {
            "standard": {"I_x": torch.randn(h, w), "I_y": torch.randn(h, w)},
            "reference": {"I_x": torch.randn(h, w), "I_y": torch.randn(h, w)},
            "test": {"I_x": torch.randn(h, w), "I_y": torch.randn(h, w)},
        }
        standard_height = torch.randn(h, w)

        old_x = old_build_inputs(
            old_train,
            diff_st,
            old_phys,
            standard_height,
            diff_sr=diff_sr,
            intensity_inputs=intensity_inputs,
        )
        new_x = new_build_inputs(
            new_train,
            diff_st,
            new_phys,
            standard_height,
            diff_sr=diff_sr,
            intensity_inputs=intensity_inputs,
        )
        self.assertEqual(tuple(old_x.shape), tuple(new_x.shape))
        self.assertTrue(torch.allclose(old_x, new_x, rtol=0.0, atol=1e-6))

    def test_dataset_reader_matches_legacy(self) -> None:
        from mini_grin.data.datasets import DefectDataset as OldDataset

        sample = {
            "diff_ix_st": np.zeros((8, 8), dtype=np.float32),
            "diff_iy_st": np.ones((8, 8), dtype=np.float32),
            "diff_ix_sr": np.full((8, 8), 2.0, dtype=np.float32),
            "diff_iy_sr": np.full((8, 8), 3.0, dtype=np.float32),
            "ix_standard": np.zeros((8, 8), dtype=np.float32),
            "iy_standard": np.zeros((8, 8), dtype=np.float32),
            "ix_reference": np.zeros((8, 8), dtype=np.float32),
            "iy_reference": np.zeros((8, 8), dtype=np.float32),
            "ix_test": np.zeros((8, 8), dtype=np.float32),
            "iy_test": np.zeros((8, 8), dtype=np.float32),
            "standard": np.zeros((8, 8), dtype=np.float32),
            "reference": np.zeros((8, 8), dtype=np.float32),
            "test": np.zeros((8, 8), dtype=np.float32),
            "defect": np.zeros((8, 8), dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "ds"
            (root / "train").mkdir(parents=True, exist_ok=True)
            np.savez(root / "train" / "sample_0000.npz", **sample)

            old_ds = OldDataset(root, "train")
            new_ds = NewDataset(root, "train")
            old_item = old_ds[0]
            new_item = new_ds[0]
            self.assertEqual(set(old_item.keys()), set(new_item.keys()))
            for k in old_item.keys():
                self.assertTrue(torch.allclose(old_item[k], new_item[k], rtol=0.0, atol=0.0), msg=k)

    def test_sr_diff_loss_matches_legacy_dataset_trainer(self) -> None:
        from mini_grin.core.configs import SimulationConfig as OldSimCfg
        from mini_grin.core.configs import TrainingConfig as OldTrainCfg
        from mini_grin.physics.layer import DifferentiableGradientLayer as OldLayer
        from mini_grin.scripts.train_dataset import _sr_diff_loss as old_sr_diff_loss

        old_sim = OldSimCfg(grid_size=16, noise_level=0.0, height_scale=250.0, dx=0.39)
        new_sim = NewSimCfg(grid_size=16, noise_level=0.0, height_scale=250.0, dx=0.39, gradient_backend="spectral")
        old_train = OldTrainCfg(sr_diff_weight=1e-6)
        new_train = NewTrainCfg(sr_diff_weight=1e-6)

        old_phys = OldLayer(old_sim).cpu()
        new_phys = NewLayer(new_sim).cpu()
        new_phys.load_state_dict(old_phys.state_dict(), strict=True)

        b, h, w = 2, 16, 16
        standard = torch.randn((b, 1, h, w), dtype=torch.float32)
        reference = torch.randn((b, 1, h, w), dtype=torch.float32)
        std_phys = old_phys(standard)
        ref_phys = old_phys(reference)
        target_sr = torch.randn((b, 2, h, w), dtype=torch.float32)

        old_val = old_sr_diff_loss(std_phys, ref_phys, target_sr, old_train)
        diff_sr = {"I_x": target_sr[:, 0], "I_y": target_sr[:, 1]}
        new_val = new_sr_diff_loss(
            new_train,
            new_phys,
            standard_height=standard,
            reference_height=reference,
            diff_sr_target=diff_sr,
        )
        self.assertTrue(torch.allclose(old_val, new_val, rtol=0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
