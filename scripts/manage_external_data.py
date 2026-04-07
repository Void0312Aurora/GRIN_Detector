from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external_data"
RAW_ROOT = EXTERNAL_ROOT / "raw"
PROCESSED_ROOT = EXTERNAL_ROOT / "processed"
MANIFEST_ROOT = EXTERNAL_ROOT / "manifests"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    source: str
    archive_name: str | None
    archive_url: str | None
    note: str
    license_note: str
    intended_use: str


DATASETS: dict[str, DatasetSpec] = {
    "zenodo_10365872": DatasetSpec(
        name="zenodo_10365872",
        title="CSI/FV/Confocal topography comparison",
        source="https://zenodo.org/records/10365872",
        archive_name="10365872.zip",
        archive_url="https://zenodo.org/api/records/10365872/files-archive",
        note="Small topography dataset for quick height-to-SPDIC pipeline validation.",
        license_note="CC BY 4.0",
        intended_use="first_import_smoke",
    ),
    "zenodo_18014400": DatasetSpec(
        name="zenodo_18014400",
        title="Micro-milled textures and friction topography",
        source="https://zenodo.org/records/18014400",
        archive_name="18014400.zip",
        archive_url="https://zenodo.org/api/records/18014400/files-archive",
        note="Texture-heavy surface dataset for hybrid defect/background priors.",
        license_note="CC BY 4.0",
        intended_use="texture_library",
    ),
    "daks_33": DatasetSpec(
        name="daks_33",
        title="Three-Dimensional Transfer Functions of Interference Microscopes",
        source="https://doi.org/10.48662/daks-33",
        archive_name=None,
        archive_url=None,
        note="Recommended for later because it exposes raw interference measurements, but the direct download path still needs a stable resolver.",
        license_note="CC BY 4.0 (per report)",
        intended_use="future_physics_calibration",
    ),
}


def _dataset_dir(name: str) -> Path:
    return RAW_ROOT / name


def _archive_path(spec: DatasetSpec) -> Path | None:
    if spec.archive_name is None:
        return None
    return _dataset_dir(spec.name) / spec.archive_name


def _download(spec: DatasetSpec, *, overwrite: bool) -> Path:
    if spec.archive_url is None or spec.archive_name is None:
        raise RuntimeError(f"{spec.name} has no direct archive_url configured")
    target_dir = _dataset_dir(spec.name)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / spec.archive_name
    if out_path.exists() and not overwrite and out_path.stat().st_size > 0:
        return out_path

    with urllib.request.urlopen(spec.archive_url) as resp, out_path.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    return out_path


def _extract(spec: DatasetSpec) -> Path:
    archive = _archive_path(spec)
    if archive is None or not archive.exists():
        raise RuntimeError(f"Archive missing for {spec.name}")
    out_dir = PROCESSED_ROOT / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)
    return out_dir


def _scan_tree(root: Path) -> dict[str, Any]:
    suffix_counts: Counter[str] = Counter()
    file_count = 0
    total_bytes = 0
    sample_files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
        suffix = path.suffix.lower() or "<no_suffix>"
        suffix_counts[suffix] += 1
        if len(sample_files) < 20:
            sample_files.append(str(path.relative_to(root)))
    top_level = sorted(p.name for p in root.iterdir()) if root.exists() else []
    return {
        "root": str(root),
        "exists": root.exists(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "top_level_entries": top_level,
        "sample_files": sample_files,
    }


def _scan_archive(spec: DatasetSpec) -> dict[str, Any]:
    archive = _archive_path(spec)
    if archive is None:
        return {"configured": False}
    if not archive.exists():
        return {"configured": True, "exists": False}
    info: dict[str, Any] = {
        "configured": True,
        "exists": True,
        "path": str(archive),
        "size_bytes": archive.stat().st_size,
    }
    try:
        with zipfile.ZipFile(archive) as zf:
            members = [zi.filename for zi in zf.infolist()]
            info["member_count"] = len(members)
            info["sample_members"] = members[:20]
            info["is_zip_valid"] = True
    except zipfile.BadZipFile:
        info["is_zip_valid"] = False
        info["error"] = "bad_zip_or_incomplete_download"
    return info


def _scan_dataset(spec: DatasetSpec) -> dict[str, Any]:
    processed_dir = PROCESSED_ROOT / spec.name
    manifest = {
        "name": spec.name,
        "title": spec.title,
        "source": spec.source,
        "license_note": spec.license_note,
        "note": spec.note,
        "intended_use": spec.intended_use,
        "archive": _scan_archive(spec),
        "processed": _scan_tree(processed_dir),
    }
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_ROOT / f"{spec.name}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def _write_status_report() -> Path:
    lines = [
        "# External Dataset Status",
        "",
        "| dataset | archive | zip_valid | processed_files | intended_use | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for spec in DATASETS.values():
        manifest_path = MANIFEST_ROOT / f"{spec.name}.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            archive = data.get("archive", {})
            processed = data.get("processed", {})
            archive_state = "missing"
            if archive.get("exists"):
                archive_state = f"{archive.get('size_bytes', 0)} B"
            zip_valid = ""
            if archive.get("configured"):
                zip_valid = str(archive.get("is_zip_valid", ""))
            lines.append(
                "| "
                + " | ".join(
                    [
                        spec.name,
                        archive_state,
                        zip_valid,
                        str(processed.get("file_count", 0)),
                        spec.intended_use,
                        spec.note,
                    ]
                )
                + " |"
            )
        else:
            lines.append(
                "| "
                + " | ".join([spec.name, "unscanned", "", "0", spec.intended_use, spec.note])
                + " |"
            )
    out = MANIFEST_ROOT / "STATUS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _resolve_dataset_names(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(DATASETS.keys())
    if args.dataset is None:
        raise SystemExit("Specify --dataset <name> or --all")
    if args.dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset: {args.dataset}")
    return [args.dataset]


def cmd_list(_: argparse.Namespace) -> int:
    for spec in DATASETS.values():
        archive = spec.archive_url if spec.archive_url is not None else "manual"
        print(f"{spec.name}\t{spec.intended_use}\t{archive}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    for name in _resolve_dataset_names(args):
        spec = DATASETS[name]
        if spec.archive_url is None:
            print(f"{name}: no direct archive_url configured; skipping")
            continue
        path = _download(spec, overwrite=bool(args.overwrite))
        print(str(path))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    for name in _resolve_dataset_names(args):
        spec = DATASETS[name]
        if spec.archive_url is None:
            print(f"{name}: no direct archive configured; skipping")
            continue
        out = _extract(spec)
        print(str(out))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    for name in _resolve_dataset_names(args):
        spec = DATASETS[name]
        manifest = _scan_dataset(spec)
        print(json.dumps(manifest, indent=2, ensure_ascii=True))
    print(str(_write_status_report()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manage external datasets used for hybrid / pre-real testing.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List configured external datasets")
    p_list.set_defaults(func=cmd_list)

    for name in ("download", "extract", "scan"):
        sp = sub.add_parser(name)
        sp.add_argument("--dataset", type=str, default=None)
        sp.add_argument("--all", action="store_true")
        if name == "download":
            sp.add_argument("--overwrite", action="store_true")
        sp.set_defaults(func={"download": cmd_download, "extract": cmd_extract, "scan": cmd_scan}[name])

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
