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

from mini_grin_rebuild.models.checkpoint import infer_checkpoint_info  # noqa: E402
from mini_grin_rebuild.models.unetpp import UNetPP  # noqa: E402


class TestUNetPP(unittest.TestCase):
    def test_forward_tensor(self) -> None:
        model = UNetPP(in_channels=10, out_channels=1, predict_logvar=False)
        x = torch.randn((2, 10, 32, 32), dtype=torch.float32)
        y = model(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(tuple(y.shape), (2, 1, 32, 32))

    def test_forward_with_logvar(self) -> None:
        model = UNetPP(in_channels=6, out_channels=1, predict_logvar=True)
        x = torch.randn((1, 6, 16, 16), dtype=torch.float32)
        out = model(x)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out.keys()), {"defect", "logvar"})
        self.assertEqual(tuple(out["defect"].shape), (1, 1, 16, 16))
        self.assertEqual(tuple(out["logvar"].shape), (1, 1, 16, 16))

    def test_infer_checkpoint_info(self) -> None:
        model = UNetPP(in_channels=7, out_channels=1, predict_logvar=True)
        info = infer_checkpoint_info(model.state_dict())
        self.assertEqual(info.in_channels, 7)
        self.assertTrue(info.predict_logvar)


if __name__ == "__main__":
    unittest.main()

