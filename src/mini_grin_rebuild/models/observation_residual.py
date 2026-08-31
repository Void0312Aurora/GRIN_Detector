from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _DownBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class _UpBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


@dataclass(frozen=True)
class ObservationResidualOutput:
    residual: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor


class ObservationResidualVAE(nn.Module):
    """Small VAE for an appearance-only residual around a physics render.

    This module deliberately does not predict defect height.  It learns a
    low-dimensional distribution of log-intensity residuals after real crops
    have been registered to the physics-render coordinate system.  The caller
    remains responsible for keeping the model out of causal/height claims when
    no paired topography is available.
    """

    def __init__(
        self,
        *,
        image_size: int = 256,
        latent_dim: int = 8,
        base_channels: int = 8,
        max_abs_residual: float = 1.0,
    ) -> None:
        super().__init__()
        if image_size < 32 or image_size % 16 != 0:
            raise ValueError("image_size must be >= 32 and divisible by 16")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if base_channels < 2:
            raise ValueError("base_channels must be >= 2")
        if max_abs_residual <= 0:
            raise ValueError("max_abs_residual must be positive")

        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.max_abs_residual = float(max_abs_residual)

        channels = (
            self.base_channels,
            2 * self.base_channels,
            4 * self.base_channels,
            8 * self.base_channels,
        )
        encoder: list[nn.Module] = []
        current = 1
        for out_channels in channels:
            encoder.append(_DownBlock(current, out_channels))
            current = out_channels
        self.encoder = nn.Sequential(*encoder)

        spatial = self.image_size // 16
        self._encoded_shape = (channels[-1], spatial, spatial)
        flattened = channels[-1] * spatial * spatial
        self.to_mean = nn.Linear(flattened, self.latent_dim)
        self.to_logvar = nn.Linear(flattened, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, flattened)

        decoder_channels = (channels[-2], channels[-3], channels[-4])
        decoder: list[nn.Module] = []
        current = channels[-1]
        for out_channels in decoder_channels:
            decoder.append(_UpBlock(current, out_channels))
            current = out_channels
        self.decoder = nn.Sequential(*decoder)
        self.output_head = nn.ConvTranspose2d(current, 1, kernel_size=4, stride=2, padding=1)

    def encode(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_image(residual)
        encoded = self.encoder(residual).flatten(1)
        mean = self.to_mean(encoded)
        logvar = torch.clamp(self.to_logvar(encoded), min=-12.0, max=8.0)
        return mean, logvar

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must have shape [B,{self.latent_dim}], got {tuple(latent.shape)}"
            )
        decoded = self.from_latent(latent).reshape(latent.shape[0], *self._encoded_shape)
        decoded = self.decoder(decoded)
        return torch.tanh(self.output_head(decoded)) * self.max_abs_residual

    def forward(self, residual: torch.Tensor) -> ObservationResidualOutput:
        mean, logvar = self.encode(residual)
        latent = self.reparameterize(mean, logvar) if self.training else mean
        return ObservationResidualOutput(
            residual=self.decode(latent),
            mean=mean,
            logvar=logvar,
        )

    def sample(
        self,
        count: int,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        if device is None:
            device = next(self.parameters()).device
        latent = torch.randn(
            (int(count), self.latent_dim),
            device=device,
            generator=generator,
        )
        return self.decode(latent)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def _validate_image(self, residual: torch.Tensor) -> None:
        expected = (1, self.image_size, self.image_size)
        if residual.ndim != 4 or tuple(residual.shape[1:]) != expected:
            raise ValueError(
                f"residual must have shape [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(residual.shape)}"
            )


__all__ = ["ObservationResidualOutput", "ObservationResidualVAE"]
