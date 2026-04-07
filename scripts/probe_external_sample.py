from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile


def _md5(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hex_head(path: Path, *, n: int = 32) -> str:
    with path.open("rb") as fh:
        return fh.read(n).hex()


def _probe_text(path: Path) -> dict[str, Any]:
    rows = 0
    header: list[str] = []
    sample_rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for idx, row in enumerate(reader):
            if idx == 0:
                header = row
            else:
                rows += 1
                if len(sample_rows) < 5:
                    sample_rows.append(row)
    return {
        "kind": "tabular_text",
        "header": header,
        "row_count_excluding_header": rows,
        "sample_rows": sample_rows,
    }


def _probe_zip_like(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": "binary"}
    if not zipfile.is_zipfile(path):
        out["zip_valid"] = False
        out["hex_head"] = _hex_head(path)
        return out
    with zipfile.ZipFile(path) as zf:
        members = [zi.filename for zi in zf.infolist()]
    out["zip_valid"] = True
    out["kind"] = "zip_container"
    out["member_count"] = len(members)
    out["sample_members"] = members[:20]
    return out


def probe(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
        "md5": _md5(path),
    }
    if path.suffix.lower() in {".txt", ".tsv", ".csv"}:
        out.update(_probe_text(path))
    else:
        out.update(_probe_zip_like(path))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Probe an external sample file and emit a small JSON summary.")
    p.add_argument("path", type=str)
    args = p.parse_args()
    payload = probe(Path(args.path).expanduser())
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
