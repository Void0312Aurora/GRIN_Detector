from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import torch
from torch.utils.data import Dataset


def as_torch_images(images: Dict[str, torch.Tensor], device: str = "cpu") -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in images.items()}


class DefectDataset(Dataset):
    """
    Dataset reader for the existing `.npz` sample format produced by the current project.
    The schema is intentionally kept compatible to avoid changing "待测样件" storage.
    """

    def __init__(self, root: Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.files: List[Path] = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No samples found in {self.root / split}")
        self._sample_meta_by_file: dict[str, dict[str, Any]] = {}
        sample_meta_path = self.root / "sample_meta.json"
        if sample_meta_path.is_file():
            payload = json.loads(sample_meta_path.read_text(encoding="utf-8"))
            for item in payload.get("samples", []):
                rel = str(item.get("file", ""))
                if rel:
                    self._sample_meta_by_file[rel] = dict(item)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        diff_ix_st = self._as_tensor(data, "diff_ix_st", fallback="diff_ix")
        diff_iy_st = self._as_tensor(data, "diff_iy_st", fallback="diff_iy")
        inputs = torch.stack([diff_ix_st, diff_iy_st], dim=0)

        sample: Dict[str, torch.Tensor] = {
            "inputs": inputs,
            "standard": self._as_tensor(data, "standard").unsqueeze(0),
            "defect": self._as_tensor(data, "defect").unsqueeze(0),
        }

        diff_ix_sr = self._optional_tensor(data, "diff_ix_sr")
        diff_iy_sr = self._optional_tensor(data, "diff_iy_sr")
        if diff_ix_sr is not None and diff_iy_sr is not None:
            sample["inputs_sr"] = torch.stack([diff_ix_sr, diff_iy_sr], dim=0)

        reference_height = self._optional_tensor(data, "reference")
        if reference_height is not None:
            sample["reference"] = reference_height.unsqueeze(0)
        test_height = self._optional_tensor(data, "test")
        if test_height is not None:
            sample["test"] = test_height.unsqueeze(0)

        for prefix in ("standard", "reference", "test"):
            ix = self._optional_tensor(data, f"ix_{prefix}")
            iy = self._optional_tensor(data, f"iy_{prefix}")
            if ix is not None and iy is not None:
                sample[f"intensity_{prefix}"] = torch.stack([ix, iy], dim=0)
            raw = self._optional_tensor(data, f"raw_{prefix}")
            if raw is not None:
                sample[f"raw_{prefix}"] = raw.unsqueeze(0)

        rel = str(self.files[idx].relative_to(self.root))
        meta = self._sample_meta_by_file.get(rel)
        if meta is not None:
            wrap = dict(meta.get("wrap", {}) or {})
            sample["sample_index"] = torch.tensor(int(meta.get("index", idx)), dtype=torch.int64)
            sample["wrap_class_id"] = torch.tensor(1 if str(wrap.get("wrap_class", "in_wrap")) == "cross_wrap" else 0, dtype=torch.int64)
            sample["estimated_wrap_stress_level"] = torch.tensor(float(wrap.get("estimated_wrap_stress_level", float("nan"))), dtype=torch.float32)
            sample["defect_wrap_target"] = torch.tensor(float(wrap.get("defect_wrap_target", float("nan"))), dtype=torch.float32)
            sample["standard_wrap_frac"] = torch.tensor(float(wrap.get("standard_wrap_frac", float("nan"))), dtype=torch.float32)

        return sample

    @staticmethod
    def _as_tensor(data, key: str, fallback: str | None = None) -> torch.Tensor:
        if key in data:
            return torch.as_tensor(data[key], dtype=torch.float32)
        if fallback and fallback in data:
            return torch.as_tensor(data[fallback], dtype=torch.float32)
        raise KeyError(f"Key {key} not found in dataset sample and no fallback provided")

    @staticmethod
    def _optional_tensor(data, key: str) -> torch.Tensor | None:
        if key in data:
            return torch.as_tensor(data[key], dtype=torch.float32)
        return None


__all__ = ["as_torch_images", "DefectDataset"]
