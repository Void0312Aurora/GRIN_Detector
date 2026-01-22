from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    import sys

    sys.path.insert(0, str(src))


_bootstrap_src()

from mini_grin_rebuild.cli.main import main as cli_main  # noqa: E402


class TestCLIPipeline(unittest.TestCase):
    def test_generate_train_eval_minimal(self) -> None:
        root_repo = Path(__file__).resolve().parents[1]
        base_cfg = json.loads((root_repo / "configs" / "default.json").read_text(encoding="utf-8"))
        base_cfg["simulation"]["grid_size"] = 16
        base_cfg["simulation"]["noise_level"] = 0.0
        base_cfg["training"]["device"] = "cpu"
        base_cfg["training"]["epochs"] = 1
        base_cfg["training"]["batch_size"] = 1
        base_cfg["training"]["log_interval"] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "proj"
            (project / "configs").mkdir(parents=True, exist_ok=True)
            cfg_path = project / "configs" / "default.json"
            cfg_path.write_text(json.dumps(base_cfg, indent=2, sort_keys=True), encoding="utf-8")

            dataset_root = project / "data" / "ds"

            # generate dataset
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(
                    [
                        "generate-dataset",
                        "--config",
                        str(cfg_path),
                        "--output",
                        str(dataset_root),
                        "--train",
                        "1",
                        "--val",
                        "1",
                        "--test",
                        "1",
                        "--seed",
                        "123",
                        "--overwrite",
                        "--name",
                        "unittest",
                    ]
                )
            self.assertEqual(rc, 0)
            run_gen = Path(buf.getvalue().strip())
            self.assertTrue((run_gen / "dataset.json").is_file())
            self.assertTrue((dataset_root / "dataset_meta.json").is_file())
            for split in ("train", "val", "test"):
                files = list((dataset_root / split).glob("*.npz"))
                self.assertEqual(len(files), 1)
                sample = np.load(files[0])
                for key in (
                    "diff_ix_st",
                    "diff_iy_st",
                    "diff_ix_sr",
                    "diff_iy_sr",
                    "ix_standard",
                    "iy_standard",
                    "ix_reference",
                    "iy_reference",
                    "ix_test",
                    "iy_test",
                    "standard",
                    "reference",
                    "test",
                    "defect",
                ):
                    self.assertIn(key, sample)

            # train (1 epoch)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(
                    [
                        "train",
                        "--config",
                        str(cfg_path),
                        "--data-root",
                        str(dataset_root),
                        "--name",
                        "unittest",
                    ]
                )
            self.assertEqual(rc, 0)
            run_train = Path(buf.getvalue().strip())
            self.assertTrue((run_train / "checkpoints" / "best.pt").is_file())
            self.assertTrue((run_train / "checkpoints" / "last.pt").is_file())
            self.assertTrue((run_train / "train_metrics.json").is_file())

            # eval (no plots)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(
                    [
                        "eval",
                        "--config",
                        str(cfg_path),
                        "--data-root",
                        str(dataset_root),
                        "--checkpoint",
                        str(run_train / "checkpoints" / "best.pt"),
                        "--split",
                        "val",
                        "--num-plots",
                        "0",
                        "--name",
                        "unittest",
                    ]
                )
            self.assertEqual(rc, 0)
            run_eval = Path(buf.getvalue().strip())
            self.assertTrue((run_eval / "eval_metrics.json").is_file())

            # baseline eval (no plots)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(
                    [
                        "baseline",
                        "--config",
                        str(cfg_path),
                        "--data-root",
                        str(dataset_root),
                        "--split",
                        "val",
                        "--num-plots",
                        "0",
                        "--name",
                        "unittest",
                    ]
                )
            self.assertEqual(rc, 0)
            run_base = Path(buf.getvalue().strip())
            self.assertTrue((run_base / "eval_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
