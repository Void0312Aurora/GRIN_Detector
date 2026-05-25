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
from mini_grin_rebuild.reconstruction import reconstruct_defect_first_order_poisson  # noqa: E402


class TestFirstOrderPoisson(unittest.TestCase):
    def test_linearized_gradient_reconstruction_recovers_integrable_defect(self) -> None:
        sim_cfg = SimulationConfig(grid_size=32, dx=0.39, noise_level=0.0)
        train_cfg = TrainingConfig(forward_model="ideal_gradient")
        physics = create_forward_model(sim_cfg, train_cfg, device="cpu", freeze=True)

        h = w = sim_cfg.grid_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, w, dtype=torch.float32),
            indexing="ij",
        )

        standard = (0.15 * xx + 0.09 * yy).unsqueeze(0).unsqueeze(0)
        defect = (0.01 * torch.sin(torch.pi * xx) * torch.cos(0.5 * torch.pi * yy)).unsqueeze(0).unsqueeze(0)

        std_phase = physics._phase(standard).squeeze(1)
        defect_phase = physics._phase(defect).squeeze(1)
        g_y, g_x = physics._gradient(std_phase)
        d_y, d_x = physics._gradient(defect_phase)

        diff_ts = {
            "I_x": 2.0 * g_x * d_x,
            "I_y": 2.0 * g_y * d_y,
        }
        pred = reconstruct_defect_first_order_poisson(
            physics=physics,
            standard_height=standard,
            diff_ts=diff_ts,
            apply_edge_offset=False,
        )

        err = torch.sqrt(torch.mean((pred - defect) ** 2)).item()
        self.assertLess(err, 5e-4)

    def test_zero_standard_gradient_is_stabilized(self) -> None:
        sim_cfg = SimulationConfig(grid_size=16, dx=0.39, noise_level=0.0)
        train_cfg = TrainingConfig(forward_model="ideal_gradient")
        physics = create_forward_model(sim_cfg, train_cfg, device="cpu", freeze=True)

        standard = torch.zeros((1, 1, sim_cfg.grid_size, sim_cfg.grid_size), dtype=torch.float32)
        diff_ts = {
            "I_x": torch.zeros((1, sim_cfg.grid_size, sim_cfg.grid_size), dtype=torch.float32),
            "I_y": torch.zeros((1, sim_cfg.grid_size, sim_cfg.grid_size), dtype=torch.float32),
        }
        pred = reconstruct_defect_first_order_poisson(
            physics=physics,
            standard_height=standard,
            diff_ts=diff_ts,
        )
        self.assertTrue(torch.isfinite(pred).all().item())


if __name__ == "__main__":
    unittest.main()
