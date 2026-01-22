from __future__ import annotations

import unittest
from pathlib import Path

import torch


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    import sys

    sys.path.insert(0, str(src))


_bootstrap_src()

from mini_grin_rebuild.core.configs import ExperimentConfig, SimulationConfig, TrainingConfig  # noqa: E402
from mini_grin_rebuild.physics.layer import DifferentiableGradientLayer  # noqa: E402
from mini_grin_rebuild.training.losses import total_loss  # noqa: E402


class TestLossScope(unittest.TestCase):
    def test_unused_loss_keys_are_rejected(self) -> None:
        data = {
            "schema_version": 1,
            "paths": {"data_dir": "data", "runs_dir": "runs"},
            "simulation": {"grid_size": 16},
            "training": {"device": "cpu", "epochs": 1, "tv_weight": 0.1},
        }
        with self.assertRaises(KeyError):
            ExperimentConfig.from_dict(data)

    def test_total_loss_only_contains_legacy_enabled_terms(self) -> None:
        sim_cfg = SimulationConfig(grid_size=16, noise_level=0.0)
        cfg = TrainingConfig(device="cpu", epochs=1)
        physics = DifferentiableGradientLayer(sim_cfg).cpu()

        h = sim_cfg.grid_size
        standard = torch.zeros((1, 1, h, h), dtype=torch.float32)
        reference = torch.zeros((1, 1, h, h), dtype=torch.float32)
        defect = torch.zeros((1, 1, h, h), dtype=torch.float32)

        diff_ts = {"I_x": torch.zeros((h, h), dtype=torch.float32), "I_y": torch.zeros((h, h), dtype=torch.float32)}
        diff_sr = {"I_x": torch.zeros((h, h), dtype=torch.float32), "I_y": torch.zeros((h, h), dtype=torch.float32)}

        terms = total_loss(
            cfg,
            physics,
            standard_height=standard,
            defect=defect,
            diff_ts_target=diff_ts,
            reference_height=reference,
            diff_sr_target=diff_sr,
            logvar=None,
        )
        self.assertEqual(
            set(terms.keys()),
            {"diff", "sr_diff", "curl", "sparsity", "edge_suppress"},
        )


if __name__ == "__main__":
    unittest.main()

