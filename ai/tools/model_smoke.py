"""Model shape smoke test (Phase 3.1).

Builds the RGB-only baseline and checks the shape contract on random input:

    input  (B, 3, H, W)  ->  output (B, 3, 2H, 2W)

torch is required to actually run a forward pass; if it isn't installed this
exits non-zero with a clear message (the data layer runs without torch, but the
model obviously cannot). Run from the repo root:

    python -m ai.tools.model_smoke
    python -m ai.tools.model_smoke --batch 2 --height 64 --width 128 --base 16
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--height", type=int, default=64, help="low-res H (div. by 4)")
    ap.add_argument("--width", type=int, default=128, help="low-res W (div. by 4)")
    ap.add_argument("--base", type=int, default=32, help="backbone width")
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("model smoke test SKIPPED: torch not installed (install "
              "ai/requirements.txt on a GPU/Colab/Kaggle machine to run it).",
              file=sys.stderr)
        return 1

    from ai.models.sr_rgb import build_sr_rgb
    from ai.utils.device import select_device

    if args.height % 4 or args.width % 4:
        print(f"error: height/width must be divisible by 4 (got {args.height}x{args.width})",
              file=sys.stderr)
        return 1

    name, device = select_device()
    model = build_sr_rgb(base=args.base)
    if device is not None:
        model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(args.batch, 3, args.height, args.width,
                    device=device if device is not None else None)
    with torch.no_grad():
        y = model(x)

    expected = (args.batch, 3, args.height * 2, args.width * 2)
    print("model smoke test (sr_rgb / SRUNet in=3)")
    print(f"  device:   {name}")
    print(f"  params:   {n_params:,}")
    print(f"  input:    {tuple(x.shape)}")
    print(f"  output:   {tuple(y.shape)}")
    print(f"  expected: {expected}")
    if tuple(y.shape) != expected:
        print("FAILED: output shape mismatch", file=sys.stderr)
        return 1
    print("OK: 2x output shape correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
