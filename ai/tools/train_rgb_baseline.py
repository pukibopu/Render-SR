"""RGB-only full baseline training (Phase 3.3b).

Real baseline training for the RGB-only model over the manifest's path-ID
train/test split: L1 loss, AdamW, a cosine LR schedule, multi-epoch, with
per-epoch train/val loss logging and last + best checkpoints. Metrics beyond
loss (PSNR/SSIM/LPIPS), the bicubic reference, and the 7ch model are out of
scope here (Phase 5 / Phase 4).

torch is required; without it this exits non-zero with a clear message. Run from
the repo root:

    python -m ai.tools.train_rgb_baseline --config ai/configs/train_rgb_baseline.yaml

All paths (data_root / output_dir / checkpoint_dir) come from the config.

Note: RenderSRTorchDataset seeds its paired transform per sample index, so a
sample's crop/flip is fixed across epochs (deterministic, no epoch-varying
augmentation). That is a known limitation of the 3.2 dataset contract, fine for a
controllable baseline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _run_epoch(model, loader, l1, device, optim=None):
    """One pass; trains if optim is given, else evaluates. Returns mean L1."""
    import torch

    train = optim is not None
    model.train(train)
    total = 0.0
    count = 0
    with torch.set_grad_enabled(train):
        for x, y, _meta in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = l1(pred, y)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            bs = x.shape[0]
            total += float(loss.detach().cpu()) * bs
            count += bs
    return total / max(1, count)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="baseline config YAML (see ai/configs/train_rgb_baseline.yaml)")
    args = ap.parse_args(argv)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("train_rgb_baseline SKIPPED: torch not installed (install "
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
        output_dir = resolve_path(paths.get("output_dir", "results"))
        ckpt_dir = resolve_path(paths.get("checkpoint_dir", "checkpoints"))

        seed = int(tr.get("seed", 0))
        epochs = int(tr.get("epochs", 50))
        batch_size = int(tr.get("batch_size", 8))
        val_batch_size = int(tr.get("val_batch_size", batch_size))
        num_workers = int(tr.get("num_workers", 0))
        lr = float(tr.get("lr", 5e-4))
        weight_decay = float(tr.get("weight_decay", 1e-4))
        eta_min = float(tr.get("eta_min", 0.0))
        base = int(tr.get("base", 32))

        torch.manual_seed(seed)
        name, device = select_device()
        device = device if device is not None else torch.device("cpu")

        common = dict(
            root=data_root,
            input_mode="rgb_only",
            crop_hw=ds.get("crop"),
            transform_seed=int(ds.get("transform_seed", 0)),
            depth_range=ds.get("depth_range"),
        )
        train_ds = RenderSRTorchDataset(split="train",
                                        hflip=bool(ds.get("hflip", False)), **common)
        # Val: held-out test paths, augmentation off.
        val_ds = RenderSRTorchDataset(split="test", hflip=False, **common)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"train_rgb_baseline FAILED: {e}", file=sys.stderr)
        return 1

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g, drop_last=False)
    has_val = len(val_ds) > 0
    val_loader = (DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                             num_workers=num_workers) if has_val else None)

    model = build_sr_rgb(base=base).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=eta_min)
    l1 = nn.L1Loss()

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log_rgb_baseline.csv"
    log_f = log_path.open("w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "train_loss", "val_loss", "lr"])

    print("RGB-only baseline training (Phase 3.3b)")
    print(f"  device:     {name}")
    print(f"  data_root:  {data_root.resolve()}")
    print(f"  train/val:  {len(train_ds)} / {len(val_ds)} sample(s)"
          f"{'  (no val split)' if not has_val else ''}")
    print(f"  epochs={epochs} batch={batch_size} lr={lr} wd={weight_decay} base={base}")

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, l1, device, optim=optim)
        val_loss = _run_epoch(model, val_loader, l1, device) if has_val else float("nan")
        cur_lr = optim.param_groups[0]["lr"]
        sched.step()

        print(f"  epoch {epoch:>3}/{epochs}  train_L1={train_loss:.6f}  "
              f"val_L1={val_loss:.6f}  lr={cur_lr:.2e}")
        log_w.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{cur_lr:.6e}"])
        log_f.flush()

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scheduler_state": sched.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "input_mode": "rgb_only",
            "base": base,
            "config": cfg,
        }
        torch.save(ckpt, ckpt_dir / "sr_rgb_last.pt")

        score = val_loss if has_val else train_loss
        if score < best_val:
            best_val = score
            torch.save(ckpt, ckpt_dir / "sr_rgb_best.pt")

    log_f.close()
    print(f"  log:        {log_path.resolve()}")
    print(f"  checkpoints: {(ckpt_dir / 'sr_rgb_last.pt').resolve()} (last), "
          f"{(ckpt_dir / 'sr_rgb_best.pt').resolve()} (best)")
    print(f"OK: trained {epochs} epoch(s); best {'val' if has_val else 'train'} "
          f"L1={best_val:.6f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
