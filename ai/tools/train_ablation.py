"""Input-channel ablation training (Phase 5.3).

One config-driven trainer for the ablation variants between the RGB-only
baseline and the full 7ch model:

  - rgb_depth   -> in_channels=4  ([R,G,B,depth_norm])
  - rgb_normal  -> in_channels=6  ([R,G,B,Nx,Ny,Nz])

It also accepts `rgb_only` (3) and `rgb_depth_normal` (7), but those have their
own dedicated trainers; this exists for the two ablation points. The model is
the **same** shared `SRUNet` backbone with only `in_channels` changed — identical
loss / optimizer / schedule / seed / split / crop to the RGB baseline so the
four-way comparison (RGB / RGB+depth / RGB+normal / RGB+depth+normal) is fair. A
config-equality guard against the baseline enforces that.

Checkpoints and logs are named from `experiment`:
`sr_<experiment>_last.pt` / `_best.pt` and `train_log_<experiment>.csv`.

torch is required; without it this exits non-zero with a clear message. Run from
the repo root:

    python -m ai.tools.train_ablation --config ai/configs/train_rgb_depth.yaml
    python -m ai.tools.train_ablation --config ai/configs/train_rgb_normal.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Reuse the fairness guard so it cannot drift from the 7ch trainer's.
from ai.tools.train_rendering_aware import _check_fair_against_baseline


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
                    help="ablation config YAML (e.g. ai/configs/train_rgb_depth.yaml)")
    args = ap.parse_args(argv)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("train_ablation SKIPPED: torch not installed (install "
              "ai/requirements.txt on a GPU/Colab/Kaggle machine to run it).",
              file=sys.stderr)
        return 1

    from ai.datasets.render_sr_dataset import INPUT_MODES, _MODE_CHANNELS
    from ai.datasets.torch_dataset import RenderSRTorchDataset
    from ai.models.backbone import SRUNet
    from ai.utils.config import load_config, resolve_path
    from ai.utils.device import select_device

    try:
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        ds = cfg.get("dataset", {})
        tr = cfg.get("train", {})
        experiment = str(cfg.get("experiment", "ablation"))

        input_mode = str(ds.get("input_mode", ""))
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")
        in_channels = _MODE_CHANNELS[input_mode]

        # Fair-comparison guard against the RGB-only baseline config.
        baseline_ref = cfg.get("compare", {}).get("baseline_config")
        if baseline_ref:
            baseline_cfg = load_config(resolve_path(baseline_ref))
            _check_fair_against_baseline(cfg, baseline_cfg)
        else:
            print("warning: no compare.baseline_config set; skipping fairness guard.",
                  file=sys.stderr)

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
            input_mode=input_mode,
            crop_hw=ds.get("crop"),
            transform_seed=int(ds.get("transform_seed", 0)),
            depth_range=ds.get("depth_range"),
        )
        train_ds = RenderSRTorchDataset(split="train",
                                        hflip=bool(ds.get("hflip", False)), **common)
        val_ds = RenderSRTorchDataset(split="test", hflip=False, **common)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"train_ablation FAILED: {e}", file=sys.stderr)
        return 1

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g, drop_last=False)
    has_val = len(val_ds) > 0
    val_loader = (DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                             num_workers=num_workers) if has_val else None)

    # Same shared backbone; only in_channels differs from the baseline.
    model = SRUNet(in_channels=in_channels, out_channels=3, base=base,
                   scale=2, residual=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=eta_min)
    l1 = nn.L1Loss()

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_log_{experiment}.csv"
    last_path = ckpt_dir / f"sr_{experiment}_last.pt"
    best_path = ckpt_dir / f"sr_{experiment}_best.pt"
    log_f = log_path.open("w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "train_loss", "val_loss", "lr"])

    print("Input-channel ablation training (Phase 5.3)")
    print(f"  experiment: {experiment}")
    print(f"  device:     {name}")
    print(f"  data_root:  {data_root.resolve()}")
    print(f"  input_mode: {input_mode}  (in_channels={in_channels})")
    print(f"  train/val:  {len(train_ds)} / {len(val_ds)} sample(s)"
          f"{'  (no val split)' if not has_val else ''}")
    print(f"  epochs={epochs} batch={batch_size} lr={lr} wd={weight_decay} base={base}")
    print("  fairness guard: passed (matches RGB baseline config)")

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
            "input_mode": input_mode,
            "in_channels": in_channels,
            "base": base,
            "config": cfg,
        }
        torch.save(ckpt, last_path)

        score = val_loss if has_val else train_loss
        if score < best_val:
            best_val = score
            torch.save(ckpt, best_path)

    log_f.close()
    print(f"  log:        {log_path.resolve()}")
    print(f"  checkpoints: {last_path.resolve()} (last), {best_path.resolve()} (best)")
    print(f"OK: trained {epochs} epoch(s); best {'val' if has_val else 'train'} "
          f"L1={best_val:.6f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
