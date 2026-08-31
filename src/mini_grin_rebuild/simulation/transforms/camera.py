from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mini_grin_rebuild.simulation.transforms.base import BaseTransform, TransformContext
from mini_grin_rebuild.simulation.transforms.utils import gaussian_blur, sample_range, sanitize_channels


class CameraTransform(BaseTransform):
    name = "camera"

    def sample_bundle_params(self, context: TransformContext) -> dict[str, Any]:
        bit_depth = int(sample_range(context.rng, self.cfg.get("bit_depth"), 0))
        output_mode = str(self.cfg.get("output_mode", "legacy") or "legacy").lower()
        if output_mode not in {"legacy", "dn"}:
            raise ValueError("camera.output_mode must be 'legacy' or 'dn'")
        default_saturation = float((2**bit_depth) - 1) if output_mode == "dn" and bit_depth > 0 else 1e9
        return {
            "output_mode": output_mode,
            # Radiometric calibration from optical intensity to the camera's
            # output unit.  In ``dn`` mode this is DN per optical-intensity
            # unit and photon_gain remains electrons per output unit.
            "exposure_scale": sample_range(context.rng, self.cfg.get("exposure_scale"), 1.0),
            "black_level": sample_range(context.rng, self.cfg.get("black_level"), 0.0),
            "shot_noise": bool(self.cfg.get("shot_noise", False)),
            "photon_gain": sample_range(context.rng, self.cfg.get("photon_gain"), 100.0),
            "read_noise_std": sample_range(context.rng, self.cfg.get("read_noise_std"), 0.0),
            "read_noise_correlation_sigma_px": (
                sample_range(context.rng, self.cfg.get("read_noise_correlation_sigma_px"), 0.0)
                if "read_noise_correlation_sigma_px" in self.cfg
                else 0.0
            ),
            # Empirical fixed spatial residual of the acquisition path.  This
            # is separate from stochastic read noise: a configurable fraction
            # is coordinate-locked across captures, so a shared acquisition
            # base survives an ensemble median.
            "fixed_pattern_std": max(
                0.0,
                sample_range(context.rng, self.cfg.get("fixed_pattern_std"), 0.0),
            ),
            "fixed_pattern_correlation_sigma_px": max(
                0.0,
                sample_range(
                    context.rng,
                    self.cfg.get("fixed_pattern_correlation_sigma_px"),
                    3.0,
                ),
            ),
            "fixed_pattern_fine_fraction": float(
                np.clip(
                    sample_range(context.rng, self.cfg.get("fixed_pattern_fine_fraction"), 0.0),
                    0.0,
                    1.0,
                )
            ),
            "fixed_pattern_fine_sigma_px": max(
                0.0,
                sample_range(context.rng, self.cfg.get("fixed_pattern_fine_sigma_px"), 0.7),
            ),
            "fixed_pattern_common_fraction": float(
                np.clip(
                    sample_range(context.rng, self.cfg.get("fixed_pattern_common_fraction"), 0.0),
                    0.0,
                    1.0,
                )
            ),
            "fixed_pattern_common_seed": int(
                round(sample_range(context.rng, self.cfg.get("fixed_pattern_common_seed"), 0.0))
            ),
            "fixed_pattern_lens_only": bool(self.cfg.get("fixed_pattern_lens_only", False)),
            "saturation_level": sample_range(
                context.rng,
                self.cfg.get("saturation_level"),
                default_saturation,
            ),
            "bit_depth": bit_depth,
            "bad_pixel_fraction": sample_range(context.rng, self.cfg.get("bad_pixel_fraction"), 0.0),
            "hot_pixel_value": sample_range(context.rng, self.cfg.get("hot_pixel_value"), 1e9),
        }

    @staticmethod
    def _normalised_pattern(
        rng: np.random.Generator,
        shape: tuple[int, int],
        *,
        coarse_sigma_px: float,
        fine_fraction: float,
        fine_sigma_px: float,
    ) -> np.ndarray:
        coarse = rng.normal(0.0, 1.0, shape).astype(np.float32)
        if coarse_sigma_px > 0.0:
            coarse = gaussian_blur(coarse, coarse_sigma_px)
        coarse = coarse - float(np.mean(coarse))
        coarse_std = float(np.std(coarse))
        coarse = coarse / max(coarse_std, 1e-9)
        if fine_fraction <= 0.0:
            return coarse.astype(np.float32)
        fine = rng.normal(0.0, 1.0, shape).astype(np.float32)
        if fine_sigma_px > 0.0:
            fine = gaussian_blur(fine, fine_sigma_px)
        fine = fine - float(np.mean(fine))
        fine_std = float(np.std(fine))
        fine = fine / max(fine_std, 1e-9)
        coarse_weight = 1.0 - fine_fraction
        norm = max(float(np.hypot(coarse_weight, fine_fraction)), 1e-9)
        return ((coarse_weight * coarse + fine_fraction * fine) / norm).astype(np.float32)

    def apply(
        self,
        channels: Mapping[str, np.ndarray],
        *,
        context: TransformContext,
        params: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        out = sanitize_channels(channels)
        output_mode = str(params.get("output_mode", "legacy") or "legacy").lower()
        if output_mode not in {"legacy", "dn"}:
            raise ValueError("camera output_mode must be 'legacy' or 'dn'")
        exposure_scale = max(float(params.get("exposure_scale", 1.0)), 0.0)
        black_level = float(params.get("black_level", 0.0))
        shot_noise = bool(params.get("shot_noise", False))
        photon_gain = max(float(params.get("photon_gain", 100.0)), 1e-6)
        read_noise_std = max(float(params.get("read_noise_std", 0.0)), 0.0)
        read_noise_correlation = max(
            float(params.get("read_noise_correlation_sigma_px", 0.0)),
            0.0,
        )
        fixed_pattern_std = max(float(params.get("fixed_pattern_std", 0.0)), 0.0)
        saturation_level = max(float(params.get("saturation_level", 1e9)), 1e-6)
        bit_depth = int(params.get("bit_depth", 0) or 0)
        bad_pixel_fraction = max(float(params.get("bad_pixel_fraction", 0.0)), 0.0)
        hot_pixel_value = float(params.get("hot_pixel_value", saturation_level))

        fixed_pattern: np.ndarray | None = None
        if fixed_pattern_std > 0.0:
            pattern_kwargs = {
                "coarse_sigma_px": max(
                    float(params.get("fixed_pattern_correlation_sigma_px", 3.0)),
                    0.0,
                ),
                "fine_fraction": float(
                    np.clip(params.get("fixed_pattern_fine_fraction", 0.0), 0.0, 1.0)
                ),
                "fine_sigma_px": max(
                    float(params.get("fixed_pattern_fine_sigma_px", 0.7)),
                    0.0,
                ),
            }
            common_fraction = float(
                np.clip(params.get("fixed_pattern_common_fraction", 0.0), 0.0, 1.0)
            )
            capture_pattern = self._normalised_pattern(
                context.rng,
                context.shape,
                **pattern_kwargs,
            )
            if common_fraction > 0.0:
                common_rng = np.random.default_rng(int(params.get("fixed_pattern_common_seed", 0)))
                common_pattern = self._normalised_pattern(
                    common_rng,
                    context.shape,
                    **pattern_kwargs,
                )
                fixed_pattern = (
                    np.sqrt(common_fraction) * common_pattern
                    + np.sqrt(1.0 - common_fraction) * capture_pattern
                )
            else:
                fixed_pattern = capture_pattern
            fixed_pattern = fixed_pattern_std * np.asarray(fixed_pattern, dtype=np.float32)
            if bool(params.get("fixed_pattern_lens_only", False)):
                h, w = context.shape
                yy = np.arange(h, dtype=np.float32) - 0.5 * (h - 1)
                xx = np.arange(w, dtype=np.float32) - 0.5 * (w - 1)
                y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
                radius_px = (
                    float(getattr(context.cfg, "lens_radius_fraction", 1.0) or 1.0)
                    * 0.5
                    * min(h, w)
                )
                radius = np.sqrt(x_grid**2 + y_grid**2)
                lens_mask = np.clip((radius_px - radius) / 2.0, 0.0, 1.0)
                fixed_pattern = fixed_pattern * lens_mask

        for name in list(out.keys()):
            image = np.clip(exposure_scale * out[name], a_min=0.0, a_max=None).astype(np.float32)
            if shot_noise:
                lam = np.clip(image * photon_gain, 0.0, 1e7)
                image = (context.rng.poisson(lam).astype(np.float32) / photon_gain).astype(np.float32)
            if read_noise_std > 0.0:
                if read_noise_correlation > 0.0:
                    read_noise = context.rng.normal(0.0, 1.0, image.shape).astype(np.float32)
                    read_noise = gaussian_blur(read_noise, read_noise_correlation)
                    read_noise = read_noise - float(np.mean(read_noise))
                    noise_std = float(np.std(read_noise))
                    if noise_std > 1e-9:
                        read_noise = read_noise * (read_noise_std / noise_std)
                    else:
                        read_noise = np.zeros_like(read_noise)
                    image = image + read_noise
                else:
                    # Keep the legacy independent-noise path byte-compatible
                    # when the new correlation option is absent or disabled.
                    image = image + context.rng.normal(0.0, read_noise_std, image.shape).astype(np.float32)
            if fixed_pattern is not None:
                image = image + fixed_pattern
            if black_level != 0.0:
                image = image + black_level
            image = np.clip(image, 0.0, saturation_level).astype(np.float32)
            if bit_depth > 0:
                levels = float((2**bit_depth) - 1)
                step = saturation_level / max(levels, 1.0)
                image = (np.round(image / step) * step).astype(np.float32)
            if bad_pixel_fraction > 0.0:
                mask = context.rng.random(image.shape) < bad_pixel_fraction
                if np.any(mask):
                    dead = context.rng.random(image.shape) < 0.5
                    image = image.copy()
                    image[mask & dead] = 0.0
                    image[mask & ~dead] = hot_pixel_value
            out[name] = np.clip(image, 0.0, saturation_level).astype(np.float32)
        return out, {"params": dict(params)}


__all__ = ["CameraTransform"]
