from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


_bootstrap_src()

from mini_grin_rebuild.cli.main import main as cli_main  # noqa: E402
from mini_grin_rebuild.core.configs import ExperimentConfig, load_experiment_config  # noqa: E402
from mini_grin_rebuild.core.runs import create_run  # noqa: E402


class TestConfigs(unittest.TestCase):
    def test_load_default_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg_path = root / "configs" / "default.json"
        cfg = load_experiment_config(cfg_path)
        self.assertIsInstance(cfg, ExperimentConfig)
        self.assertEqual(cfg.schema_version, 1)
        self.assertGreater(cfg.simulation.grid_size, 0)

    def test_unknown_keys_raise(self) -> None:
        with self.assertRaises(KeyError):
            ExperimentConfig.from_dict({"schema_version": 1, "unknown": 123})


class TestRuns(unittest.TestCase):
    def test_create_run_writes_audit_files(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = load_experiment_config(root / "configs" / "default.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / "runs"
            run = create_run(
                runs_root,
                name="unittest",
                argv=["unit", "test"],
                config_snapshot=cfg.to_dict(),
            )
            self.assertTrue(run.root.exists())
            self.assertTrue((run.root / "checkpoints").is_dir())
            self.assertTrue((run.root / "plots").is_dir())
            self.assertTrue((run.root / "config.json").is_file())
            self.assertTrue((run.root / "meta.json").is_file())

            meta = json.loads((run.root / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("utc_time", meta)
            self.assertIn("python", meta)


class TestCLI(unittest.TestCase):
    def test_init_run_creates_directory(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg_src = root / "configs" / "default.json"
        cfg_data = json.loads(cfg_src.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            (project / "configs").mkdir(parents=True, exist_ok=True)
            cfg_path = project / "configs" / "default.json"
            cfg_path.write_text(json.dumps(cfg_data, indent=2, sort_keys=True), encoding="utf-8")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(["init-run", "--config", str(cfg_path), "--name", "cli"])
            self.assertEqual(rc, 0)
            run_path = Path(buf.getvalue().strip())
            self.assertTrue(run_path.exists())
            self.assertTrue((run_path / "config.json").is_file())
            self.assertTrue((run_path / "meta.json").is_file())


if __name__ == "__main__":
    unittest.main()

