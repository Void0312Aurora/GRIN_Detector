from __future__ import annotations

"""Identify the physical origin of the common concentric arcs near the lens edge.

Three discriminating tests on the native full-resolution frames:

1. Centre locking: cross-frame coherence of the arc profile computed around each
   frame's own detected lens centre versus around the session-mean centre
   (frame-fixed). A frame-fixed instrument artifact aligns better in fixed
   coordinates; a structure attached to the lens aligns better in lens
   coordinates.
2. Two-sidedness: does a mirrored arc system exist just outside the seam, on the
   fixture side? Surface structure on the lens exists only inside; edge-wave
   interference from the seam exists on both sides.
3. Spacing law: equally spaced peaks indicate a periodic surface structure;
   x_m ~ sqrt(m * lambda * dz) chirp indicates defocused edge diffraction.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "external_data" / "raw" / "wechat_2026-07_15-34" / "extracted" / "15.34"
DETECTIONS = ROOT / "external_data" / "processed" / "wechat_2026-07_15-34" / "valid_sample_detections.json"
UM_PER_PX = 109.71 / 620.0  # native scale from the cap-geometry self-calibration


def radial_profile(
    frame: np.ndarray,
    cy: float,
    cx: float,
    radii_px: np.ndarray,
    width_px: float = 1.0,
) -> np.ndarray:
    h, w = frame.shape
    size = int(radii_px[-1] + 8)
    y0, y1 = max(0, int(cy) - size), min(h, int(cy) + size + 1)
    x0, x1 = max(0, int(cx) - size), min(w, int(cx) + size + 1)
    sub = frame[y0:y1, x0:x1].astype(np.float32)
    yy = np.arange(y0, y1, dtype=np.float32) - cy
    xx = np.arange(x0, x1, dtype=np.float32) - cx
    rr = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    profile = np.empty(radii_px.shape, dtype=np.float64)
    for index, radius in enumerate(radii_px):
        ring = (rr >= radius - width_px) & (rr < radius + width_px)
        profile[index] = float(np.mean(sub[ring])) if np.any(ring) else np.nan
    return profile


def detrend(profile: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    filled = np.where(np.isfinite(profile), profile, np.nanmean(profile))
    return filled - gaussian_filter1d(filled, sigma)


def main() -> int:
    detections = json.loads(DETECTIONS.read_text(encoding="utf-8"))
    mean_cy = float(np.mean([d["center_y"] for d in detections]))
    mean_cx = float(np.mean([d["center_x"] for d in detections]))
    scatter = np.sqrt(
        np.var([d["center_x"] for d in detections]) + np.var([d["center_y"] for d in detections])
    )
    print(f"lens-centre scatter across frames: {scatter:.1f} px ({scatter * UM_PER_PX:.2f} um)")

    # Arc zone: rho 0.70-1.00 -> in native px around the ~620 px radius.
    radii_px = np.arange(0.70 * 620.0, 0.995 * 620.0, 1.0, dtype=np.float64)
    outside_px = np.arange(1.03 * 620.0, 1.30 * 620.0, 1.0, dtype=np.float64)

    lens_profiles: list[np.ndarray] = []
    fixed_profiles: list[np.ndarray] = []
    outside_profiles: list[np.ndarray] = []
    for record in detections:
        with Image.open(RAW_DIR / str(record["file"])) as source:
            frame = np.asarray(source.convert("L"), dtype=np.float32)
        cy, cx = float(record["center_y"]), float(record["center_x"])
        radius = float(record["radius"])
        scale = radius / 620.0  # normalise each lens to its own detected radius
        lens_profiles.append(detrend(radial_profile(frame, cy, cx, radii_px * scale)))
        fixed_profiles.append(detrend(radial_profile(frame, mean_cy, mean_cx, radii_px)))
        outside_profiles.append(detrend(radial_profile(frame, cy, cx, outside_px * scale)))

    lens_stack = np.stack(lens_profiles)
    fixed_stack = np.stack(fixed_profiles)

    def mean_pairwise_corr(stack: np.ndarray) -> float:
        n = stack.shape[0]
        cors = [
            float(np.corrcoef(stack[i], stack[j])[0, 1]) for i in range(n) for j in range(i + 1, n)
        ]
        return float(np.mean(cors))

    corr_lens = mean_pairwise_corr(lens_stack)
    corr_fixed = mean_pairwise_corr(fixed_stack)
    print(f"cross-frame arc coherence, lens-centred:  {corr_lens:.3f}")
    print(f"cross-frame arc coherence, frame-fixed:   {corr_fixed:.3f}")

    # Two-sidedness.
    outside_stack = np.stack(outside_profiles)
    outside_median = np.median(outside_stack, axis=0)
    inside_median = np.median(lens_stack, axis=0)
    print(f"arc std inside  (median profile): {np.std(inside_median):.3f} DN")
    print(f"arc std outside (median profile): {np.std(outside_median):.3f} DN")
    corr_outside = mean_pairwise_corr(outside_stack)
    print(f"cross-frame coherence outside:    {corr_outside:.3f}")

    # Spacing law on the inside median arcs.
    r_um = radii_px * UM_PER_PX
    peaks, _ = find_peaks(inside_median, prominence=0.8)
    peak_r = r_um[peaks]
    seam_um = 109.71
    x_from_seam = seam_um - peak_r
    order = np.argsort(x_from_seam)
    x_sorted = x_from_seam[order]
    print("peak distances from seam (um):", np.round(x_sorted, 2))
    if x_sorted.size >= 4:
        m_index = np.arange(1, x_sorted.size + 1, dtype=np.float64)
        # Fresnel law: x_m^2 = m * lambda * dz + c
        coeffs = np.polyfit(m_index, x_sorted**2, 1)
        pred = np.polyval(coeffs, m_index)
        residual = x_sorted**2 - pred
        r2_fresnel = 1.0 - float(np.sum(residual**2)) / max(
            float(np.sum((x_sorted**2 - np.mean(x_sorted**2)) ** 2)), 1e-9
        )
        dz = coeffs[0] / 0.52
        # Equal-spacing law: x_m = m * d + c
        coeffs_lin = np.polyfit(m_index, x_sorted, 1)
        pred_lin = np.polyval(coeffs_lin, m_index)
        residual_lin = x_sorted - pred_lin
        r2_equal = 1.0 - float(np.sum(residual_lin**2)) / max(
            float(np.sum((x_sorted - np.mean(x_sorted)) ** 2)), 1e-9
        )
        print(f"Fresnel-law fit:  x^2 = m*lambda*dz + c -> dz = {dz:.0f} um, R^2 = {r2_fresnel:.4f}")
        print(f"Equal-spacing fit: x = m*d + c -> d = {coeffs_lin[0]:.2f} um, R^2 = {r2_equal:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
