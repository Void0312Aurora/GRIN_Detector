from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


_bootstrap_src()

from mini_grin_rebuild.core.configs import SimulationConfig, TrainingConfig  # noqa: E402
from mini_grin_rebuild.evaluation.evaluator import evaluate_residue_cut_poisson  # noqa: E402
from mini_grin_rebuild.physics.factory import create_forward_model  # noqa: E402
from mini_grin_rebuild.reconstruction.pseudo_poisson import (  # noqa: E402
    reconstruct_defect_residue_cut_poisson,
)


class TestWrapBaselineAPI(unittest.TestCase):
    def test_residue_cut_evaluator_is_public(self) -> None:
        self.assertTrue(callable(evaluate_residue_cut_poisson))

    def test_residue_cut_zero_input_is_finite(self) -> None:
        sim_cfg = SimulationConfig(grid_size=16, dx=0.39, noise_level=0.0)
        train_cfg = TrainingConfig(forward_model="ideal_gradient")
        physics = create_forward_model(sim_cfg, train_cfg, device="cpu", freeze=True)

        standard = torch.zeros((1, 1, sim_cfg.grid_size, sim_cfg.grid_size), dtype=torch.float32)
        diff_ts = {
            "I_x": torch.zeros((1, sim_cfg.grid_size, sim_cfg.grid_size), dtype=torch.float32),
            "I_y": torch.zeros((1, sim_cfg.grid_size, sim_cfg.grid_size), dtype=torch.float32),
        }
        pred = reconstruct_defect_residue_cut_poisson(
            physics=physics,
            standard_height=standard,
            diff_ts=diff_ts,
        )

        self.assertEqual(tuple(pred.shape), tuple(standard.shape))
        self.assertTrue(torch.isfinite(pred).all().item())


if __name__ == "__main__":
    unittest.main()
