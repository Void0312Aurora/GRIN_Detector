from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"
ARCHIVE_ROOT = ROOT / "runs_archive"
REGISTRY_PATH = ROOT / "run_registry.json"
DOC_PATH = ROOT / "docs" / "RUNS.md"
REFS_ROOT = ROOT / "run_refs"
MANIFEST_PATH = ROOT / "archive_manifest.json"


@dataclass(frozen=True)
class RunInfo:
    name: str
    path: Path
    stamp: str | None
    kind: str
    size_bytes: int
    status: str
    alias: str | None
    note: str | None
    storage: str

    @property
    def size_human(self) -> str:
        return format_bytes(self.size_bytes)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_stamp(name: str) -> str | None:
    token = name[:15]
    try:
        dt = datetime.strptime(token, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def infer_kind(name: str) -> str:
    parts = name.split("_")
    suffix = "_".join(parts[3:]) if len(parts) >= 4 else name
    if suffix.startswith("20x_"):
        return "20x"
    if suffix.startswith("pre_real_"):
        return "pre_real"
    if suffix.startswith("qc_"):
        return "qc"
    if suffix.startswith("gate_"):
        return "gate"
    if suffix.startswith("scan_"):
        return "scan"
    if suffix.startswith("smoke"):
        return "smoke"
    if suffix.startswith("suite_"):
        return "suite_root"
    if "_suite_" in name:
        return "suite_member"
    return "other"


def dir_size_bytes(root: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _run_info_for_path(path: Path, registry: dict[str, Any], *, storage: str) -> RunInfo:
    default_status = str(registry.get("default_status", "archive_candidate"))
    run_entries = registry.get("runs", {})
    meta = run_entries.get(path.name, {})
    return RunInfo(
        name=path.name,
        path=path,
        stamp=infer_stamp(path.name),
        kind=infer_kind(path.name),
        size_bytes=dir_size_bytes(path),
        status=str(meta.get("status", default_status)),
        alias=meta.get("alias"),
        note=meta.get("note"),
        storage=storage,
    )


def scan_live_runs(runs_root: Path, registry: dict[str, Any]) -> list[RunInfo]:
    infos: list[RunInfo] = []
    if not runs_root.exists():
        return infos
    for path in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        infos.append(_run_info_for_path(path, registry, storage="live"))
    return infos


def scan_archived_runs(archive_root: Path, registry: dict[str, Any]) -> list[RunInfo]:
    infos: list[RunInfo] = []
    if not archive_root.exists():
        return infos
    for group_dir in sorted(p for p in archive_root.iterdir() if p.is_dir()):
        for path in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            infos.append(_run_info_for_path(path, registry, storage="archived"))
    return infos


def scan_runs(runs_root: Path, archive_root: Path, registry: dict[str, Any]) -> list[RunInfo]:
    return scan_live_runs(runs_root, registry) + scan_archived_runs(archive_root, registry)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"moved_runs": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_refs(runs: list[RunInfo]) -> None:
    REFS_ROOT.mkdir(exist_ok=True)
    desired: dict[str, str] = {}
    for run in runs:
        if run.status not in {"active", "reference"} or not run.alias:
            continue
        link_name = f"{run.status}_{run.alias}"
        desired[link_name] = os.path.relpath(run.path, REFS_ROOT)

    for path in REFS_ROOT.iterdir():
        if path.name == "README.md":
            continue
        if path.name not in desired:
            if path.is_symlink() or path.is_file():
                path.unlink()

    for link_name, target in desired.items():
        link_path = REFS_ROOT / link_name
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(target)


def _table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["| None |", "| --- |"]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    header = render(rows[0])
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = [render(row) for row in rows[1:]]
    return [header, sep, *body]


def write_report(path: Path, runs: list[RunInfo], registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_size = sum(run.size_bytes for run in runs)
    status_counts = Counter(run.status for run in runs)
    storage_counts = Counter(run.storage for run in runs)

    by_status: dict[str, list[RunInfo]] = defaultdict(list)
    for run in runs:
        by_status[run.status].append(run)

    archive_by_kind: dict[str, list[RunInfo]] = defaultdict(list)
    for run in by_status.get("archive_candidate", []):
        archive_by_kind[run.kind].append(run)

    top_runs = sorted(runs, key=lambda run: run.size_bytes, reverse=True)[:10]
    top_archived_runs = sorted((run for run in runs if run.storage == "archived"), key=lambda run: run.size_bytes, reverse=True)[:10]
    lines: list[str] = []
    lines.append("# Run Management")
    lines.append("")
    lines.append("该文件可通过 `python scripts/manage_runs.py refresh` 重新生成。")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total runs: `{len(runs)}`")
    lines.append(f"- Total size: `{format_bytes(total_size)}`")
    lines.append(f"- Live runs in `runs/`: `{storage_counts.get('live', 0)}`")
    lines.append(f"- Archived runs in `runs_archive/`: `{storage_counts.get('archived', 0)}`")
    lines.append(f"- Active runs: `{status_counts.get('active', 0)}`")
    lines.append(f"- Reference runs: `{status_counts.get('reference', 0)}`")
    lines.append(f"- Archive candidates: `{status_counts.get('archive_candidate', 0)}`")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append("- `runs/` only keeps curated `active` runs plus future new outputs.")
    lines.append("- Historical exploratory runs are physically moved to `runs_archive/<kind>/`.")
    lines.append("- `reference` runs are also archived, but still exposed through `run_refs/`.")
    lines.append("- Unlisted runs default to `archive_candidate`; they are the first targets for migration.")
    lines.append("- Shortcut symlinks for curated runs are created in `run_refs/`, regardless of live/archive storage.")
    lines.append("")

    for status in ("active", "reference"):
        title = "Active Runs" if status == "active" else "Reference Runs"
        lines.append(f"## {title}")
        lines.append("")
        rows = [["Alias", "Run", "Storage", "Kind", "Size", "Timestamp", "Note"]]
        for run in sorted(by_status.get(status, []), key=lambda item: (item.stamp or "", item.name), reverse=True):
            rows.append(
                [
                    run.alias or "",
                    run.name,
                    run.storage,
                    run.kind,
                    run.size_human,
                    run.stamp or "",
                    run.note or "",
                ]
            )
        lines.extend(_table(rows))
        lines.append("")

    lines.append("## Largest Runs")
    lines.append("")
    rows = [["Run", "Status", "Storage", "Kind", "Size", "Timestamp"]]
    for run in top_runs:
        rows.append([run.name, run.status, run.storage, run.kind, run.size_human, run.stamp or ""])
    lines.extend(_table(rows))
    lines.append("")

    lines.append("## Largest Archived Runs")
    lines.append("")
    rows = [["Run", "Kind", "Size", "Timestamp"]]
    for run in top_archived_runs:
        rows.append([run.name, run.kind, run.size_human, run.stamp or ""])
    lines.extend(_table(rows))
    lines.append("")

    lines.append("## Archived Runs By Kind")
    lines.append("")
    rows = [["Kind", "Count", "Combined Size"]]
    for kind, kind_runs in sorted(
        ((kind, [run for run in kind_runs if run.storage == "archived"]) for kind, kind_runs in archive_by_kind.items()),
        key=lambda item: (-len(item[1]), -sum(run.size_bytes for run in item[1]), item[0]),
    ):
        if not kind_runs:
            continue
        rows.append([kind, str(len(kind_runs)), format_bytes(sum(run.size_bytes for run in kind_runs))])
    lines.extend(_table(rows))
    lines.append("")

    lines.append("## Live Archive Candidates By Kind")
    lines.append("")
    rows = [["Kind", "Count", "Combined Size"]]
    for kind, kind_runs in sorted(
        ((kind, [run for run in kind_runs if run.storage == "live"]) for kind, kind_runs in archive_by_kind.items()),
        key=lambda item: (-len(item[1]), -sum(run.size_bytes for run in item[1]), item[0]),
    ):
        if not kind_runs:
            continue
        rows.append([kind, str(len(kind_runs)), format_bytes(sum(run.size_bytes for run in kind_runs))])
    lines.extend(_table(rows))
    lines.append("")

    lines.append("## Registry Statuses")
    lines.append("")
    rows = [["Status", "Meaning"]]
    for key, value in registry.get("statuses", {}).items():
        rows.append([str(key), str(value)])
    lines.extend(_table(rows))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage and summarize mini_grin_rebuild runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_refresh = sub.add_parser("refresh", help="Regenerate run report and shortcut refs.")
    p_refresh.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p_refresh.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    p_refresh.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    p_refresh.add_argument("--report", type=Path, default=DOC_PATH)

    p_summary = sub.add_parser("summary", help="Print a short textual summary.")
    p_summary.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p_summary.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    p_summary.add_argument("--registry", type=Path, default=REGISTRY_PATH)

    p_archive = sub.add_parser("archive", help="Move live archive_candidate runs into runs_archive/<kind>/.")
    p_archive.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p_archive.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    p_archive.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    p_archive.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p_archive.add_argument(
        "--move-statuses",
        type=str,
        default="archive_candidate",
        help="Comma-separated statuses to move from runs/ into runs_archive/.",
    )
    p_archive.add_argument("--dry-run", action="store_true")
    return parser


def cmd_refresh(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    runs = scan_runs(args.runs_root, args.archive_root, registry)
    write_refs(runs)
    write_report(args.report, runs, registry)
    print(f"refreshed report: {args.report}")
    print(f"curated refs: {REFS_ROOT}")
    print(f"total runs: {len(runs)}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    runs = scan_runs(args.runs_root, args.archive_root, registry)
    counts = Counter(run.status for run in runs)
    storage = Counter(run.storage for run in runs)
    print(f"runs={len(runs)} total_size={format_bytes(sum(run.size_bytes for run in runs))}")
    print(f"live={storage.get('live', 0)} archived={storage.get('archived', 0)}")
    for status in ("active", "reference", "archive_candidate"):
        print(f"{status}={counts.get(status, 0)}")
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    manifest = load_manifest(args.manifest)
    manifest.setdefault("moved_runs", {})
    statuses = {item.strip() for item in str(args.move_statuses).split(",") if item.strip()}
    if not statuses:
        raise SystemExit("No statuses selected for archiving.")

    live_runs = scan_live_runs(args.runs_root, registry)
    to_move = [run for run in live_runs if run.status in statuses]
    if not to_move:
        print(f"no live runs found for statuses={sorted(statuses)}")
        return 0

    args.archive_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    total_bytes = 0
    now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    for run in to_move:
        target_dir = args.archive_root / run.kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / run.name
        if target.exists():
            raise SystemExit(f"archive target already exists: {target}")
        print(f"[{run.status}] {run.path} -> {target}")
        if args.dry_run:
            continue
        shutil.move(str(run.path), str(target))
        manifest["moved_runs"][run.name] = {
            "archived_at": now,
            "kind": run.kind,
            "status": run.status,
            "original_path": str(run.path.relative_to(ROOT)),
            "archived_path": str(target.relative_to(ROOT)),
            "size_bytes": run.size_bytes,
        }
        moved += 1
        total_bytes += run.size_bytes

    if args.dry_run:
        print(f"dry-run only: {len(to_move)} runs would be moved")
        return 0

    write_manifest(args.manifest, manifest)
    refreshed_args = argparse.Namespace(
        runs_root=args.runs_root,
        archive_root=args.archive_root,
        registry=args.registry,
        report=DOC_PATH,
    )
    cmd_refresh(refreshed_args)
    print(f"moved_runs={moved} moved_size={format_bytes(total_bytes)} manifest={args.manifest}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "refresh":
        return cmd_refresh(args)
    if args.cmd == "summary":
        return cmd_summary(args)
    if args.cmd == "archive":
        return cmd_archive(args)
    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
