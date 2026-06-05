"""Rendering-aware SR model (Phase 4.1).

Thin wrapper over the *same* shared :class:`~ai.models.backbone.SRUNet` used by
the RGB-only baseline, with ``in_channels=7``. Only the input channel count
differs from :class:`~ai.models.sr_rgb.SRRGB` — architecture, scale, residual,
and width are otherwise identical. That is what keeps the 3ch <-> 7ch comparison
fair, so this file must stay a thin wrapper and never diverge in structure.

Input channel order (fixed):

    [R, G, B, depth_norm, Nx, Ny, Nz]

Shapes:

    input  (B, 7, H, W)   low-res RGB + depth_norm + view-space normal
    output (B, 3, 2H, 2W) high-res RGB
"""

from __future__ import annotations

from .backbone import SRUNet


class SRRenderingAware(SRUNet):
    """SRUNet fixed to 7 input channels ([R,G,B,depth_norm,Nx,Ny,Nz])."""

    def __init__(self, base: int = 32, residual: bool = True):
        super().__init__(in_channels=7, out_channels=3, base=base,
                         scale=2, residual=residual)


def build_sr_rendering_aware(base: int = 32, residual: bool = True) -> SRRenderingAware:
    """Construct the rendering-aware (7ch) model."""
    return SRRenderingAware(base=base, residual=residual)
