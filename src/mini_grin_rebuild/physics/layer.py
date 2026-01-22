from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_grin_rebuild.core.configs import SimulationConfig


class DifferentiableGradientLayer(nn.Module):
    """
    Torch differentiable forward model consistent with the NumPy simulator.

    It produces two intensity channels (x/y) derived from squared phase gradients,
    with learnable per-axis gain/bias, PSF blur, low-frequency illumination, and sub-pixel shifts.
    """

    def __init__(self, cfg: SimulationConfig, *, kernel_size: int = 9, lowres_size: int = 32) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        self.cfg = cfg
        self.kernel_size = kernel_size
        self.lowres_size = lowres_size

        radius = kernel_size // 2
        coords = torch.linspace(-radius, radius, steps=kernel_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("psf_grid", xx**2 + yy**2)
        self.register_buffer("eps", torch.tensor(1e-6))

        self.log_gain = nn.Parameter(torch.zeros(2))
        self.bias = nn.Parameter(torch.zeros(2))
        self.log_sigma = nn.Parameter(torch.zeros(2))
        self.lfields = nn.Parameter(torch.zeros(2, 1, lowres_size, lowres_size))
        self.shifts = nn.Parameter(torch.zeros(2, 2))

    def forward(self, height: torch.Tensor) -> Dict[str, torch.Tensor]:
        if height.ndim == 2:
            height = height.unsqueeze(0)
        if height.ndim == 3:
            height = height.unsqueeze(1)
        phase = self._phase(height).squeeze(1)
        grad_y, grad_x = self._gradient(phase)
        ix = self._apply_response(grad_x, 0)
        iy = self._apply_response(grad_y, 1)
        return {"I_x": ix, "I_y": iy}

    def _phase(self, height: torch.Tensor) -> torch.Tensor:
        scale = (2.0 * math.pi / self.cfg.wavelength) * (self.cfg.n_object - self.cfg.n_air)
        return scale * height

    def _gradient(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backend = getattr(self.cfg, "gradient_backend", "finite")
        if backend == "finite":
            return self._finite_gradient(phase)
        if backend == "spectral":
            return self._spectral_gradient(phase)
        raise ValueError(f"Unknown gradient_backend={backend!r} (expected 'finite' or 'spectral')")

    def _finite_gradient(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finite-difference gradient (matches dataset generation using `np.gradient`).
        Expects `phase` shape [B,H,W].
        """
        grad_y, grad_x = torch.gradient(
            phase,
            spacing=(self.cfg.dx, self.cfg.dx),
            dim=(-2, -1),
            edge_order=1,
        )
        return grad_y, grad_x

    def _spectral_gradient(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, height, width = phase.shape
        device = phase.device
        dtype = phase.dtype
        freq_y = torch.fft.fftfreq(height, d=self.cfg.dx, device=device).view(1, height, 1)
        freq_x = torch.fft.fftfreq(width, d=self.cfg.dx, device=device).view(1, 1, width)
        phase_fft = torch.fft.fft2(phase)
        grad_x_fft = phase_fft * (2j * math.pi * freq_x)
        grad_y_fft = phase_fft * (2j * math.pi * freq_y)
        grad_x = torch.fft.ifft2(grad_x_fft).real
        grad_y = torch.fft.ifft2(grad_y_fft).real
        return grad_y.to(dtype), grad_x.to(dtype)

    def _apply_response(self, grad_component: torch.Tensor, idx: int) -> torch.Tensor:
        power = grad_component.pow(2).unsqueeze(1)
        filtered = self._apply_psf(power, idx)
        filtered = self._apply_lowfreq(filtered, idx)
        gain = torch.exp(self.log_gain[idx])
        out = gain * filtered + self.bias[idx]
        out = self._apply_shift(out, idx)
        return out

    def _apply_psf(self, image: torch.Tensor, idx: int) -> torch.Tensor:
        kernel = self._gaussian_kernel(idx, image.device, image.dtype)
        padding = self.kernel_size // 2
        return F.conv2d(image, kernel, padding=padding)

    def _gaussian_kernel(self, idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma[idx]) + self.eps
        grid = self.psf_grid.to(device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * grid / (sigma**2))
        kernel = kernel / (kernel.sum() + self.eps)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def _apply_lowfreq(self, image: torch.Tensor, idx: int) -> torch.Tensor:
        field = F.interpolate(
            self.lfields[idx : idx + 1],
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return image * (1.0 + field)

    def _apply_shift(self, image: torch.Tensor, idx: int) -> torch.Tensor:
        shift = self.shifts[idx]
        if torch.allclose(shift, torch.zeros_like(shift)):
            return image
        n, _, h, w = image.shape
        shift_y, shift_x = shift[0], shift[1]
        norm_y = 2.0 * shift_y / max(h, 1)
        norm_x = 2.0 * shift_x / max(w, 1)
        theta = torch.eye(2, 3, device=image.device, dtype=image.dtype).unsqueeze(0).repeat(n, 1, 1)
        theta[:, 0, 2] = norm_x
        theta[:, 1, 2] = norm_y
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        return F.grid_sample(image, grid, align_corners=False, padding_mode="border")

    def illumination_tv(self) -> torch.Tensor:
        total = 0.0
        for idx in range(2):
            field = self.lfields[idx : idx + 1]
            dx = field[:, :, 1:, :] - field[:, :, :-1, :]
            dy = field[:, :, :, 1:] - field[:, :, :, :-1]
            total = total + torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        return total

    def shift_penalty(self) -> torch.Tensor:
        return torch.sum(self.shifts**2)


__all__ = ["DifferentiableGradientLayer"]
