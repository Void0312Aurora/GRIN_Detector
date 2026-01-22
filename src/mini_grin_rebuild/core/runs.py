from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import platform
from pathlib import Path
import socket
import subprocess
from typing import Any, Sequence
import uuid

from mini_grin_rebuild.core.json_io import write_json


def _find_git_root(start: Path) -> Path | None:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _safe_run(cmd: Sequence[str], *, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(list(cmd), cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def collect_run_meta(*, argv: Sequence[str] | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    meta: dict[str, Any] = {
        "utc_time": now,
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "argv": list(argv) if argv is not None else None,
    }

    root = repo_root or _find_git_root(Path.cwd())
    if root is not None:
        meta["git_root"] = str(root)
        meta["git_commit"] = _safe_run(["git", "rev-parse", "HEAD"], cwd=root)
        meta["git_branch"] = _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
        meta["git_dirty"] = (_safe_run(["git", "status", "--porcelain"], cwd=root) or "") != ""

    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        meta["torch"] = None

    return meta


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    plots: Path


def create_run(
    runs_root: str | Path,
    *,
    name: str | None = None,
    argv: Sequence[str] | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> RunPaths:
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid.uuid4().hex[:8]
    slug = f"{stamp}_{token}"
    if name:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name).strip("-")
        if safe:
            slug = f"{slug}_{safe}"

    root = runs_root / slug
    checkpoints = root / "checkpoints"
    plots = root / "plots"
    checkpoints.mkdir(parents=True, exist_ok=False)
    plots.mkdir(parents=True, exist_ok=False)

    if config_snapshot is not None:
        write_json(root / "config.json", config_snapshot)
    meta = collect_run_meta(argv=argv)
    write_json(root / "meta.json", meta)
    return RunPaths(root=root, checkpoints=checkpoints, plots=plots)


__all__ = ["RunPaths", "collect_run_meta", "create_run"]

