"""DataLoader smoke test (Phase 3.2).

Builds a ``RenderSRTorchDataset`` + ``DataLoader`` from a config and pulls one
batch, checking the shape contract:

    x  (B, C, H, W)        network input
    y  (B, 3, 2H, 2W)      high-res RGB target

torch is required; if it isn't installed this exits non-zero with a clear
message. Tensors stay on CPU here (the Dataset never moves to device). Run from
the repo root:

    python -m ai.tools.loader_smoke --config ai/configs/dataset_smoke.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="dataset config YAML (see ai/configs/dataset_smoke.yaml)")
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader
    except ImportError:
        print("loader smoke test SKIPPED: torch not installed (install "
              "ai/requirements.txt on a GPU/Colab/Kaggle machine to run it).",
              file=sys.stderr)
        return 1

    from ai.datasets.torch_dataset import RenderSRTorchDataset
    from ai.utils.config import load_config, resolve_path
    from ai.utils.device import select_device

    try:
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        ds = cfg.get("dataset", {})
        smoke = cfg.get("smoke", {})

        data_root = resolve_path(paths.get("data_root", "output_buffers"))
        split = ds.get("split", "train")
        mode = ds.get("input_mode", "rgb_only")
        batch_size = int(smoke.get("batch_size", 2))
        num_workers = int(smoke.get("num_workers", 0))

        device_name, _ = select_device()

        dataset = RenderSRTorchDataset(
            root=data_root,
            split=split,
            input_mode=mode,
            crop_hw=ds.get("crop"),
            hflip=bool(ds.get("hflip", False)),
            transform_seed=int(ds.get("transform_seed", 0)),
            depth_range=ds.get("depth_range"),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

        x, y, meta = next(iter(loader))
    except (FileNotFoundError, ValueError, KeyError, StopIteration) as e:
        print(f"loader smoke test FAILED: {e}", file=sys.stderr)
        return 1

    print("loader smoke test")
    print(f"  data_root:  {data_root.resolve()}")
    print(f"  device:     {device_name} (tensors stay on CPU in Dataset)")
    print(f"  split:      {split}   input_mode: {mode}")
    print(f"  dataset:    {len(dataset)} sample(s)   batch_size={batch_size}")
    print(f"  x: shape={tuple(x.shape)} dtype={x.dtype} "
          f"min={float(x.min()):.4f} max={float(x.max()):.4f}")
    print(f"  y: shape={tuple(y.shape)} dtype={y.dtype} "
          f"min={float(y.min()):.4f} max={float(y.max()):.4f}")
    print(f"  meta keys:  {sorted(meta.keys())}  path_id={meta['path_id'].tolist()}")

    b, c, h, w = x.shape
    if tuple(y.shape) != (b, 3, h * 2, w * 2):
        print(f"FAILED: target shape {tuple(y.shape)} is not (B,3,2H,2W) of input",
              file=sys.stderr)
        return 1
    print("OK: batch shapes consistent (y == 2x input spatially).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
