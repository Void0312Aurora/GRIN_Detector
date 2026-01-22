from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    import sys

    sys.path.insert(0, str(src))


_bootstrap_src()

from mini_grin_rebuild.cli.main import main as cli_main  # noqa: E402


class TestSuiteCommand(unittest.TestCase):
    def test_suite_minimal(self) -> None:
        root_repo = Path(__file__).resolve().parents[1]
        base_cfg = json.loads((root_repo / "configs" / "default.json").read_text(encoding="utf-8"))
        base_cfg["simulation"]["grid_size"] = 16
        base_cfg["simulation"]["noise_level"] = 0.0
        base_cfg["training"]["device"] = "cpu"
        base_cfg["training"]["epochs"] = 1
        base_cfg["training"]["batch_size"] = 1
        base_cfg["training"]["log_interval"] = 1

        suite_spec = {
            "name": "unittest_suite",
            "seeds": [123],
            "baselines": [{"name": "pseudo_poisson", "method": "pseudo_poisson"}],
            "experiments": [
                {
                    "name": "nn_default",
                    "overrides": {"training": {"use_pseudo_poisson_prior": False, "teacher_loss_weight": 0.0}},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "proj"
            (project / "configs").mkdir(parents=True, exist_ok=True)
            cfg_path = project / "configs" / "default.json"
            cfg_path.write_text(json.dumps(base_cfg, indent=2, sort_keys=True), encoding="utf-8")

            suite_path = project / "configs" / "suite.json"
            suite_path.write_text(json.dumps(suite_spec, indent=2, sort_keys=True), encoding="utf-8")

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

            # suite (1 experiment + 1 baseline)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(
                    [
                        "suite",
                        "--config",
                        str(cfg_path),
                        "--suite",
                        str(suite_path),
                        "--data-root",
                        str(dataset_root),
                        "--eval-split",
                        "val",
                        "--num-plots",
                        "0",
                        "--name",
                        "unittest",
                    ]
                )
            self.assertEqual(rc, 0)
            suite_run = Path(buf.getvalue().strip())
            self.assertTrue((suite_run / "suite_summary.json").is_file())
            self.assertTrue((suite_run / "suite_table.csv").is_file())
            self.assertTrue((suite_run / "suite_table.md").is_file())


if __name__ == "__main__":
    unittest.main()

