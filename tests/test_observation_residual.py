from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mini_grin_rebuild.models.observation_residual import ObservationResidualVAE  # noqa: E402


class TestObservationResidualVAE(unittest.TestCase):
    def test_forward_backward_and_eval_shape(self) -> None:
        torch.manual_seed(7)
        model = ObservationResidualVAE(
            image_size=32,
            latent_dim=4,
            base_channels=4,
        )
        inputs = torch.randn(2, 1, 32, 32)
        model.train()
        output = model(inputs)
        self.assertEqual(tuple(output.residual.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(output.mean.shape), (2, 4))
        self.assertEqual(tuple(output.logvar.shape), (2, 4))
        loss = (output.residual - inputs).abs().mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

        model.eval()
        with torch.no_grad():
            first = model(inputs).residual
            second = model(inputs).residual
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_seeded_prior_sampling_is_reproducible(self) -> None:
        model = ObservationResidualVAE(image_size=32, latent_dim=3, base_channels=4)
        first_generator = torch.Generator().manual_seed(19)
        second_generator = torch.Generator().manual_seed(19)
        with torch.no_grad():
            first = model.sample(3, generator=first_generator)
            second = model.sample(3, generator=second_generator)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(first.shape), (3, 1, 32, 32))

    def test_invalid_spatial_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            ObservationResidualVAE(image_size=40)
        model = ObservationResidualVAE(image_size=32, latent_dim=2, base_channels=4)
        with self.assertRaisesRegex(ValueError, "residual must have shape"):
            model(torch.zeros(1, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
