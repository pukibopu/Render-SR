"""RGB-only SR baseline (Phase 3.1).

Thin wrapper over the shared :class:`~ai.models.backbone.SRUNet` with
``in_channels=3``. This is the Phase 3 baseline; the Phase 4 rendering-aware
model is the *same* backbone with ``in_channels=7``. Nothing about the
architecture, just the input channel count, may differ between them — that is
what keeps the comparison fair, so this file must stay a thin wrapper and never
diverge in structure.

    input  (B, 3, H, W)   low-res RGB
    output (B, 3, 2H, 2W) high-res RGB
"""

from __future__ import annotations

from .backbone import SRUNet


class SRRGB(SRUNet):
    """SRUNet fixed to 3 input channels (RGB-only baseline)."""

    def __init__(self, base: int = 32, residual: bool = True):
        super().__init__(in_channels=3, out_channels=3, base=base,
                         scale=2, residual=residual)


def build_sr_rgb(base: int = 32, residual: bool = True) -> SRRGB:
    """Construct the RGB-only baseline model."""
    return SRRGB(base=base, residual=residual)
