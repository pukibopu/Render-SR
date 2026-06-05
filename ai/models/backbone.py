"""Shared SR backbone (Phase 3.1).

One small U-Net used for *every* variant in this project — the only thing that
changes between the RGB-only baseline and the rendering-aware model is
``in_channels`` (3 vs 7). Keeping a single backbone is what makes the 3ch↔7ch
comparison fair, so do not fork the architecture per variant; parameterise it.

Shape contract:
    input  (B, in_channels, H, W)
    output (B, out_channels, 2H, 2W)   # fixed 2x upscale

The network runs at low resolution and upsamples once at the end via a
PixelShuffle(2) head, plus a global bilinear-upsample residual on the first
``out_channels`` input channels. Channel order is ``[R, G, B, depth, Nx, Ny, Nz]``
across the project, so ``x[:, :out_channels]`` is always the RGB block — the
residual is identical whether the input is 3ch or 7ch.

Note: H and W should be divisible by 4 (two 2x downsamples in the U-Net). The
dataset's paired crops (e.g. 128x256) satisfy this.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv3 -> BN -> ReLU) x2, spatial size preserved."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """2x downsample (maxpool) then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """2x upsample (bilinear), concat the skip, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, skip_ch, kernel_size=1)
        self.conv = DoubleConv(skip_ch * 2, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SRUNet(nn.Module):
    """Small U-Net SR backbone, parameterised by ``in_channels``.

    ``scale`` is fixed to 2 in v1 (the dataset is exactly 2x); other values are
    rejected so a mismatch fails loudly rather than silently.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base: int = 32, scale: int = 2, residual: bool = True):
        super().__init__()
        if scale != 2:
            raise ValueError(f"v1 backbone supports scale=2 only, got {scale}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.residual = residual

        self.inc = DoubleConv(in_channels, base)        # H,   base
        self.down1 = Down(base, base * 2)               # H/2, 2base
        self.down2 = Down(base * 2, base * 4)           # H/4, 4base (bottleneck)
        self.up1 = Up(base * 4, base * 2, base * 2)     # H/2
        self.up2 = Up(base * 2, base, base)             # H

        # Upsample head: base -> out_channels * scale^2 -> PixelShuffle(scale).
        self.head = nn.Sequential(
            nn.Conv2d(base, out_channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        u1 = self.up1(x2, x1)
        u2 = self.up2(u1, x0)
        out = self.head(u2)                              # (B, out, 2H, 2W)

        if self.residual:
            base_rgb = x[:, : self.out_channels]
            up = F.interpolate(base_rgb, scale_factor=self.scale,
                               mode="bilinear", align_corners=False)
            out = out + up
        return out
