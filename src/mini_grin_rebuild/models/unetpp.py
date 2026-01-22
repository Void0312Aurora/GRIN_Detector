from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8, padding_mode: str = "zeros") -> None:
        super().__init__()
        hidden = max(1, channels // max(1, reduction))
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * channel

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * spatial


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_attention: bool = True,
        attention_reduction: int = 8,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.attention = (
            ChannelSpatialAttention(out_channels, reduction=attention_reduction, padding_mode=padding_mode) if use_attention else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


class UNetPP(nn.Module):
    """
    A lightweight UNet++ variant with optional attention and optional log-variance head.

    Returns:
    - `Tensor` of shape [B,1,H,W] when `predict_logvar=False`
    - `dict(defect=Tensor, logvar=Tensor)` when `predict_logvar=True`
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = 1,
        features: Iterable[int] = (64, 128, 256, 512),
        use_attention: bool = True,
        attention_reduction: int = 8,
        padding_mode: str = "zeros",
        output_scale: float | None = None,
        predict_logvar: bool = False,
    ) -> None:
        super().__init__()
        feats = tuple(int(f) for f in features)
        if len(feats) < 2:
            raise ValueError("UNetPP requires at least two feature levels")
        self.output_scale = output_scale
        self.depth = len(feats)
        self.pool = nn.MaxPool2d(2, 2)

        self.encoders = nn.ModuleList()
        cur = in_channels
        for feat in feats:
            self.encoders.append(
                ConvBlock(
                    cur,
                    feat,
                    use_attention=use_attention,
                    attention_reduction=attention_reduction,
                    padding_mode=padding_mode,
                )
            )
            cur = feat

        self.decoders = nn.ModuleDict()
        for level in range(1, self.depth):
            for col in range(self.depth - level):
                in_ch = level * feats[col] + feats[col + 1]
                self.decoders[f"{level}_{col}"] = ConvBlock(
                    in_ch,
                    feats[col],
                    use_attention=use_attention,
                    attention_reduction=attention_reduction,
                    padding_mode=padding_mode,
                )

        self.output_head = nn.Conv2d(feats[0], out_channels, kernel_size=1)
        self.predict_logvar = bool(predict_logvar)
        self.logvar_head = nn.Conv2d(feats[0], 1, kernel_size=1) if self.predict_logvar else None

    def forward(self, x: torch.Tensor):
        nodes: list[list[torch.Tensor | None]] = [[None for _ in range(self.depth)] for _ in range(self.depth)]

        cur = x
        for idx, encoder in enumerate(self.encoders):
            cur = encoder(cur)
            nodes[0][idx] = cur
            if idx != self.depth - 1:
                cur = self.pool(cur)

        for level in range(1, self.depth):
            for col in range(self.depth - level):
                concat_parts = [nodes[row][col] for row in range(level)]
                upsample = nodes[level - 1][col + 1]
                if upsample is None:
                    raise RuntimeError("UNetPP decoder received undefined node")
                if concat_parts[0] is None:
                    raise RuntimeError("UNetPP decoder received undefined skip")
                upsample = F.interpolate(upsample, size=concat_parts[0].shape[2:], mode="bilinear", align_corners=False)
                concat = torch.cat([*concat_parts, upsample], dim=1)  # type: ignore[arg-type]
                nodes[level][col] = self.decoders[f"{level}_{col}"](concat)

        out = nodes[self.depth - 1][0]
        if out is None:
            raise RuntimeError("UNetPP forward produced no output")

        defect = self.output_head(out)
        if self.output_scale is not None:
            defect = torch.tanh(defect) * float(self.output_scale)
        if self.predict_logvar and self.logvar_head is not None:
            return {"defect": defect, "logvar": self.logvar_head(out)}
        return defect


__all__ = ["UNetPP"]
