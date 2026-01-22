from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def plot_defect_and_intensity(
    *,
    defect_true: np.ndarray,
    defect_pred: np.ndarray,
    diff_true: Dict[str, np.ndarray],
    diff_pred: Dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "mini_grin_eval",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)

    vmax = float(np.max(np.abs(defect_true))) if defect_true.size else 1.0
    if vmax == 0:
        vmax = 1e-6
    axes[0, 0].imshow(defect_true, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title("Defect True")
    axes[0, 1].imshow(defect_pred, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title("Defect Pred")
    axes[0, 2].imshow(defect_pred - defect_true, cmap="RdBu")
    axes[0, 2].set_title("Defect Error")

    for row, key in enumerate(["I_x", "I_y"], start=1):
        true_img = diff_true[key]
        pred_img = diff_pred[key]
        v = float(np.max(np.abs(true_img))) if true_img.size else 1.0
        if v == 0:
            v = 1e-6
        axes[row, 0].imshow(true_img, cmap="coolwarm", vmin=-v, vmax=v)
        axes[row, 0].set_title(f"{key} True")
        axes[row, 1].imshow(pred_img, cmap="coolwarm", vmin=-v, vmax=v)
        axes[row, 1].set_title(f"{key} Pred")
        axes[row, 2].imshow(np.abs(pred_img - true_img), cmap="viridis")
        axes[row, 2].set_title(f"{key} |Error|")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


__all__ = ["plot_defect_and_intensity"]
