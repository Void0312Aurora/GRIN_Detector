from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: Optional[int], *, deterministic_torch: bool = True) -> None:
    """
    Set seeds for Python/NumPy (and Torch if available) for run-to-run reproducibility.
    This function is intentionally side-effectful and should be called once per run.
    """
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch may be absent in minimal environments; keep NumPy/Python deterministic.
        return


__all__ = ["set_global_seed"]

