from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def _as_hw(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[0] == 1:
        return img.squeeze(0)
    if img.ndim == 4 and img.shape[0] == 1 and img.shape[1] == 1:
        return img.squeeze(0).squeeze(0)
    raise ValueError(f"Expected [H,W], [1,H,W], or [1,1,H,W]; got {tuple(img.shape)}")


def defect_mask(
    target: torch.Tensor,
    *,
    abs_threshold: float = 1e-4,
    rel_threshold: float = 0.05,
    dilate_px: int = 0,
) -> torch.Tensor:
    """
    Build a boolean defect support mask from ground-truth defect height.

    Threshold policy:
      |target| > max(abs_threshold, rel_threshold * max(|target|)).
    """
    t = _as_hw(target)
    peak = torch.max(torch.abs(t))
    thr = max(float(abs_threshold), float(rel_threshold) * float(peak))
    mask = torch.abs(t) > thr
    if dilate_px <= 0:
        return mask
    d = int(dilate_px)
    kernel = 2 * d + 1
    m = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    m = F.max_pool2d(m, kernel_size=kernel, stride=1, padding=d)
    return (m.squeeze(0).squeeze(0) > 0.5)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    v = _as_hw(values)
    m = _as_hw(mask).to(dtype=torch.bool)
    if v.shape != m.shape:
        raise ValueError(f"mask shape mismatch: values={tuple(v.shape)} mask={tuple(m.shape)}")
    if torch.sum(m) == 0:
        return torch.tensor(float("nan"), device=v.device, dtype=v.dtype)
    return torch.mean(v[m])


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = _as_hw(pred)
    t = _as_hw(target)
    m = _as_hw(mask).to(dtype=torch.bool)
    if torch.sum(m) == 0:
        return torch.tensor(float("nan"), device=p.device, dtype=p.dtype)
    return torch.sqrt(torch.mean((p[m] - t[m]) ** 2))


def masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    data_range: float | None = None,
) -> torch.Tensor:
    p = _as_hw(pred)
    t = _as_hw(target)
    m = _as_hw(mask).to(dtype=torch.bool)
    if torch.sum(m) == 0:
        return torch.tensor(float("nan"), device=p.device, dtype=p.dtype)

    if data_range is None:
        tt = t[m]
        data_range = float(torch.max(tt) - torch.min(tt))
        if data_range == 0:
            data_range = 1.0

    mse = torch.mean((p[m] - t[m]) ** 2)
    if torch.allclose(mse, torch.zeros_like(mse)):
        return torch.tensor(float("inf"), device=p.device)
    return 20.0 * torch.log10(torch.tensor(float(data_range), device=p.device)) - 10.0 * torch.log10(mse)


def masked_correlation(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = _as_hw(pred)
    t = _as_hw(target)
    m = _as_hw(mask).to(dtype=torch.bool)
    if torch.sum(m) < 2:
        return torch.tensor(float("nan"), device=p.device, dtype=p.dtype)
    pv = p[m].flatten()
    tv = t[m].flatten()
    p_mean = torch.mean(pv)
    t_mean = torch.mean(tv)
    num = torch.sum((pv - p_mean) * (tv - t_mean))
    den = torch.sqrt(torch.sum((pv - p_mean) ** 2)) * torch.sqrt(torch.sum((tv - t_mean) ** 2))
    if torch.allclose(den, torch.zeros_like(den)):
        return torch.zeros((), device=p.device)
    return num / den


def masked_volume_rel_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = _as_hw(pred)
    t = _as_hw(target)
    m = _as_hw(mask).to(dtype=torch.bool)
    if torch.sum(m) == 0:
        return torch.tensor(float("nan"), device=p.device, dtype=p.dtype)
    denom = torch.sum(torch.abs(t[m])) + 1e-8
    num = torch.abs(torch.sum(p[m]) - torch.sum(t[m]))
    return num / denom


def binary_precision(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    p = _as_hw(pred_mask).to(dtype=torch.bool)
    t = _as_hw(true_mask).to(dtype=torch.bool)
    if p.shape != t.shape:
        raise ValueError(f"mask shape mismatch: pred={tuple(p.shape)} true={tuple(t.shape)}")
    tp = torch.sum(p & t).to(dtype=torch.float32)
    fp = torch.sum(p & ~t).to(dtype=torch.float32)
    denom = tp + fp
    if denom < 1e-8:
        return torch.tensor(float("nan"), device=p.device)
    return tp / denom


def binary_recall(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    p = _as_hw(pred_mask).to(dtype=torch.bool)
    t = _as_hw(true_mask).to(dtype=torch.bool)
    if p.shape != t.shape:
        raise ValueError(f"mask shape mismatch: pred={tuple(p.shape)} true={tuple(t.shape)}")
    tp = torch.sum(p & t).to(dtype=torch.float32)
    fn = torch.sum(~p & t).to(dtype=torch.float32)
    denom = tp + fn
    if denom < 1e-8:
        return torch.tensor(float("nan"), device=p.device)
    return tp / denom


def binary_f1(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    precision = binary_precision(pred_mask, true_mask)
    recall = binary_recall(pred_mask, true_mask)
    if torch.isnan(precision) or torch.isnan(recall):
        return torch.tensor(float("nan"), device=precision.device)
    denom = precision + recall
    if denom < 1e-8:
        return torch.tensor(0.0, device=precision.device)
    return 2.0 * precision * recall / denom


def binary_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    p = _as_hw(pred_mask).to(dtype=torch.bool)
    t = _as_hw(true_mask).to(dtype=torch.bool)
    if p.shape != t.shape:
        raise ValueError(f"mask shape mismatch: pred={tuple(p.shape)} true={tuple(t.shape)}")
    inter = torch.sum(p & t).to(dtype=torch.float32)
    union = torch.sum(p | t).to(dtype=torch.float32)
    if union < 1e-8:
        return torch.tensor(float("nan"), device=p.device)
    return inter / union


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    AUROC for a binary pixel classifier.

    `scores`: higher means more likely positive (e.g., |pred|).
    `labels`: boolean mask (True = positive).
    """
    if scores.ndim == 1:
        s = scores
    else:
        s = _as_hw(scores).flatten()
    if labels.ndim == 1:
        y = labels.to(dtype=torch.bool)
    else:
        y = _as_hw(labels).to(dtype=torch.bool).flatten()
    pos = int(torch.sum(y).item())
    neg = int(y.numel() - pos)
    if pos == 0 or neg == 0:
        return torch.tensor(float("nan"), device=s.device, dtype=torch.float32)

    idx = torch.argsort(s, descending=True)
    y_sorted = y[idx].to(dtype=torch.float32)
    tp = torch.cumsum(y_sorted, dim=0)
    fp = torch.cumsum(1.0 - y_sorted, dim=0)

    tpr = tp / float(pos)
    fpr = fp / float(neg)

    # prepend (0,0)
    z = torch.zeros((1,), device=s.device, dtype=torch.float32)
    tpr = torch.cat([z, tpr.to(dtype=torch.float32)], dim=0)
    fpr = torch.cat([z, fpr.to(dtype=torch.float32)], dim=0)
    # trapezoidal integration
    dfpr = fpr[1:] - fpr[:-1]
    auc = torch.sum(dfpr * (tpr[1:] + tpr[:-1]) * 0.5)
    return auc


def binary_auprc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    AUPRC (Average Precision) for a binary pixel classifier.

    Implements the standard average-precision definition:
      AP = (1 / #pos) * sum_k precision@k for each true positive k
    after sorting by score descending.
    """
    if scores.ndim == 1:
        s = scores
    else:
        s = _as_hw(scores).flatten()
    if labels.ndim == 1:
        y = labels.to(dtype=torch.bool)
    else:
        y = _as_hw(labels).to(dtype=torch.bool).flatten()
    pos = int(torch.sum(y).item())
    if pos == 0:
        return torch.tensor(float("nan"), device=s.device, dtype=torch.float32)

    idx = torch.argsort(s, descending=True)
    y_sorted = y[idx].to(dtype=torch.float32)
    tp = torch.cumsum(y_sorted, dim=0)
    denom = torch.arange(1, y_sorted.numel() + 1, device=s.device, dtype=torch.float32)
    precision = tp / denom
    ap = torch.sum(precision[y_sorted > 0.5]) / float(pos)
    return ap


def psnr(pred: torch.Tensor, target: torch.Tensor, *, data_range: float | None = None) -> torch.Tensor:
    if data_range is None:
        data_range = float(target.max() - target.min())
        if data_range == 0:
            data_range = 1.0
    mse = torch.mean((pred - target) ** 2)
    if torch.allclose(mse, torch.zeros_like(mse)):
        return torch.tensor(float("inf"), device=pred.device)
    return 20.0 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10.0 * torch.log10(mse)


def correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred.flatten()
    t = target.flatten()
    p_mean = torch.mean(p)
    t_mean = torch.mean(t)
    num = torch.sum((p - p_mean) * (t - t_mean))
    den = torch.sqrt(torch.sum((p - p_mean) ** 2)) * torch.sqrt(torch.sum((t - t_mean) ** 2))
    if torch.allclose(den, torch.zeros_like(den)):
        return torch.zeros((), device=pred.device)
    return num / den


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g /= g.sum()
    return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)


def ssim(img1: torch.Tensor, img2: torch.Tensor, *, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if img1.ndim == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
    if img2.ndim == 2:
        img2 = img2.unsqueeze(0).unsqueeze(0)
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)
    channel = img1.size(1)
    window = _gaussian_window(window_size, sigma, img1.device).expand(channel, 1, window_size, window_size)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def ssim_adaptive(img1: torch.Tensor, img2: torch.Tensor, *, max_window: int = 11) -> torch.Tensor:
    a = _as_hw(img1)
    b = _as_hw(img2)
    h, w = int(a.shape[-2]), int(a.shape[-1])
    win = min(int(max_window), h, w)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return torch.tensor(float("nan"), device=a.device, dtype=a.dtype)
    sigma = 1.5 if win >= 11 else max(0.5, 1.5 * win / 11.0)
    return ssim(a, b, window_size=win, sigma=float(sigma))


def slope(pred: torch.Tensor, target: torch.Tensor, *, threshold: float = 1e-4) -> torch.Tensor:
    mask = torch.abs(target) > threshold
    if torch.sum(mask) == 0:
        return torch.tensor(float("nan"), device=pred.device)
    yt = target[mask]
    yp = pred[mask]
    num = torch.sum(yt * yp)
    den = torch.sum(yt * yt) + 1e-8
    return num / den


def peak_rel_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    t_peak = torch.max(torch.abs(target))
    if t_peak < 1e-8:
        return torch.tensor(float("nan"), device=pred.device)
    p_peak = torch.max(torch.abs(pred))
    return torch.abs(p_peak - t_peak) / t_peak


def volume_rel_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(torch.abs(target)) + 1e-8
    if denom < 1e-8:
        return torch.tensor(float("nan"), device=pred.device)
    num = torch.abs(torch.sum(pred) - torch.sum(target))
    return num / denom


def masked_mean_abs(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    v = _as_hw(values)
    m = _as_hw(mask).to(dtype=torch.bool)
    if torch.sum(m) == 0:
        return torch.tensor(float("nan"), device=v.device, dtype=v.dtype)
    return torch.mean(torch.abs(v[m]))


def masked_abs_quantile(values: torch.Tensor, mask: torch.Tensor, *, q: float) -> torch.Tensor:
    v = _as_hw(values)
    m = _as_hw(mask).to(dtype=torch.bool)
    if torch.sum(m) == 0:
        return torch.tensor(float("nan"), device=v.device, dtype=v.dtype)
    q = float(q)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0,1], got {q}")
    vals = torch.abs(v[m]).flatten()
    n = int(vals.numel())
    if n == 0:
        return torch.tensor(float("nan"), device=v.device, dtype=v.dtype)
    k = int(round(q * (n - 1)))
    sorted_vals, _ = torch.sort(vals)
    return sorted_vals[k]


def batch_defect_metrics(defect_pred: torch.Tensor, defect_true: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return (
        rmse(defect_pred, defect_true),
        psnr(defect_pred, defect_true),
        ssim(defect_pred, defect_true),
        correlation(defect_pred, defect_true),
        slope(defect_pred, defect_true),
        peak_rel_error(defect_pred, defect_true),
        volume_rel_error(defect_pred, defect_true),
    )


__all__ = [
    "batch_defect_metrics",
    "binary_auprc",
    "binary_auroc",
    "binary_f1",
    "binary_iou",
    "binary_precision",
    "binary_recall",
    "correlation",
    "defect_mask",
    "masked_correlation",
    "masked_abs_quantile",
    "masked_mean_abs",
    "masked_psnr",
    "masked_rmse",
    "masked_volume_rel_error",
    "peak_rel_error",
    "psnr",
    "rmse",
    "slope",
    "ssim",
    "ssim_adaptive",
    "volume_rel_error",
]
