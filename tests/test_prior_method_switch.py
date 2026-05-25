from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


_bootstrap_src()

from mini_grin_rebuild.core.configs import SimulationConfig, TrainingConfig  # noqa: E402
from mini_grin_rebuild.physics.factory import create_forward_model  # noqa: E402
from mini_grin_rebuild.training.trainer import _prepare_batch_inputs  # noqa: E402


class TestPriorMethodSwitch(unittest.TestCase):
    def _batch(self, h: int, w: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": torch.zeros((1, 2, h, w), dtype=torch.float32),
            "inputs_sr": torch.zeros((1, 2, h, w), dtype=torch.float32),
            "intensity_standard": torch.zeros((1, 2, h, w), dtype=torch.float32),
            "intensity_reference": torch.zeros((1, 2, h, w), dtype=torch.float32),
            "intensity_test": torch.zeros((1, 2, h, w), dtype=torch.float32),
            "standard": torch.zeros((1, 1, h, w), dtype=torch.float32),
            "reference": torch.zeros((1, 1, h, w), dtype=torch.float32),
        }

    def test_first_order_prior_method_flows_into_aux_and_input_channels(self) -> None:
        sim_cfg = SimulationConfig(grid_size=16, dx=0.39, noise_level=0.0)
        train_cfg = TrainingConfig(
            forward_model="ideal_gradient",
            use_pseudo_poisson_prior=True,
            pseudo_poisson_prior_method="first_order_poisson",
            pseudo_poisson_prior_as_input=True,
            use_sr_inputs=True,
            use_raw_intensity_inputs=True,
            use_phase_inputs=False,
        )
        physics = create_forward_model(sim_cfg, train_cfg, device="cpu", freeze=True)
        batch = self._batch(sim_cfg.grid_size, sim_cfg.grid_size)

        inputs, aux = _prepare_batch_inputs(batch, physics, train_cfg)
        self.assertEqual(aux["coarse_prior_method"], "first_order_poisson")
        self.assertIn("coarse_prior_defect", aux)
        self.assertIsNotNone(aux["coarse_prior_defect"])
        self.assertEqual(tuple(aux["coarse_prior_defect"].shape), (1, 1, sim_cfg.grid_size, sim_cfg.grid_size))
        self.assertEqual(inputs.shape[1], 11)

    def test_hybrid_prior_method_flows_into_aux_and_input_channels(self) -> None:
        sim_cfg = SimulationConfig(grid_size=16, dx=0.39, noise_level=0.0)
        train_cfg = TrainingConfig(
            forward_model="ideal_gradient",
            use_pseudo_poisson_prior=True,
            pseudo_poisson_prior_method="first_order_sign_quadratic_poisson",
            pseudo_poisson_prior_as_input=True,
            use_sr_inputs=True,
            use_raw_intensity_inputs=True,
            use_phase_inputs=False,
        )
        physics = create_forward_model(sim_cfg, train_cfg, device="cpu", freeze=True)
        batch = self._batch(sim_cfg.grid_size, sim_cfg.grid_size)

        inputs, aux = _prepare_batch_inputs(batch, physics, train_cfg)
        self.assertEqual(aux["coarse_prior_method"], "first_order_sign_quadratic_poisson")
        self.assertIn("coarse_prior_defect", aux)
        self.assertIsNotNone(aux["coarse_prior_defect"])
        self.assertEqual(tuple(aux["coarse_prior_defect"].shape), (1, 1, sim_cfg.grid_size, sim_cfg.grid_size))
        self.assertEqual(inputs.shape[1], 11)

if __name__ == "__main__":
    unittest.main()
