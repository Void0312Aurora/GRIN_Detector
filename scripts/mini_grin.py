from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


def main() -> int:
    _bootstrap_src()
    from mini_grin_rebuild.cli.main import main as cli_main

    return int(cli_main())


if __name__ == "__main__":
    raise SystemExit(main())

