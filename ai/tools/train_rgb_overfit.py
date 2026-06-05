"""RGB-only tiny-overfit training loop (Phase 3.3a).

A sanity check that the model + data + optimizer wire together and can learn:
overfit ONE fixed batch for a few hundred iterations and confirm the L1 loss
drops. This is deliberately NOT real training — no validation, PSNR, bicubic
reference, scheduling, or logging beyond initial/final loss.

torch is required; without it this exits non-zero with a clear message. Run from
the repo root:

    python -m ai.tools.train_rgb_overfit --config ai/configs/train_rgb_overfit.yaml

Exits 0 if the loss decreased and a checkpoint was written, non-zero otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="overfit config YAML (see ai/configs/train_rgb_overfit.yaml)")
    args = ap.parse_args(argv)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("train_rgb_overfit SKIPPED: torch not installed (install "
              "ai/requirements.txt on a GPU/Colab/Kaggle machine to run it).",
              file=sys.stderr)
        return 1

    from ai.datasets.torch_dataset import RenderSRTorchDataset
    from ai.models.sr_rgb import build_sr_rgb
    from ai.utils.config import load_config, resolve_path
    from ai.utils.device import select_device

    try:
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        ds = cfg.get("dataset", {})
        tr = cfg.get("train", {})

        data_root = resolve_path(paths.get("data_root", "output_buffers"))
        ckpt_dir = resolve_path(paths.get("checkpoint_dir", "checkpoints"))

        seed = int(tr.get("seed", 0))
        batch_size = int(tr.get("batch_size", 2))
        iterations = int(tr.get("iterations", 200))
        lr = float(tr.get("lr", 1e-3))
        weight_decay = float(tr.get("weight_decay", 1e-4))
        base = int(tr.get("base", 16))

        torch.manual_seed(seed)
        name, device = select_device()
        device = device if device is not None else torch.device("cpu")

        # rgb_only is fixed for this trainer regardless of config.
        dataset = RenderSRTorchDataset(
            root=data_root,
            split=ds.get("split", "train"),
            input_mode="rgb_only",
            crop_hw=ds.get("crop"),
            hflip=bool(ds.get("hflip", False)),
            transform_seed=int(ds.get("transform_seed", 0)),
            depth_range=ds.get("depth_range"),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        x, y, _meta = next(iter(loader))             # ONE fixed batch
    except (FileNotFoundError, ValueError, KeyError, StopIteration) as e:
        print(f"train_rgb_overfit FAILED: {e}", file=sys.stderr)
        return 1

    x = x.to(device)
    y = y.to(device)

    model = build_sr_rgb(base=base).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    l1 = nn.L1Loss()

    print("RGB-only tiny overfit (Phase 3.3a)")
    print(f"  device:     {name}")
    print(f"  data_root:  {data_root.resolve()}")
    print(f"  batch:      x={tuple(x.shape)} y={tuple(y.shape)}")
    print(f"  iters:      {iterations}  lr={lr}  wd={weight_decay}  base={base}")

    initial_loss = None
    final_loss = None
    for it in range(iterations):
        optim.zero_grad()
        pred = model(x)
        loss = l1(pred, y)
        loss.backward()
        optim.step()
        lv = float(loss.detach().cpu())
        if it == 0:
            initial_loss = lv
        final_loss = lv
        if it == 0 or it == iterations - 1 or (it + 1) % max(1, iterations // 5) == 0:
            print(f"  iter {it + 1:>4}/{iterations}  L1={lv:.6f}")

    print(f"  initial L1: {initial_loss:.6f}")
    print(f"  final   L1: {final_loss:.6f}")

    if not (final_loss < initial_loss):
        print("FAILED: loss did not decrease on the overfit batch", file=sys.stderr)
        return 1

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "sr_rgb_overfit.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optim.state_dict(),
        "iterations": iterations,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "input_mode": "rgb_only",
        "base": base,
        "config": cfg,
    }, ckpt_path)

    print(f"  checkpoint: {ckpt_path.resolve()}")
    print(f"OK: loss decreased {initial_loss:.6f} -> {final_loss:.6f}; checkpoint saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
