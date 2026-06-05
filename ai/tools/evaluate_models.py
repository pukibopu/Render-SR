"""Unified evaluation: RGB-only vs rendering-aware 7ch (Phase 5.1).

Loads BOTH best checkpoints and evaluates them on the **same** held-out manifest
path-ID split, full frame (no crop/flip), with identical dataset preprocessing
and depth normalisation. Only the input differs: the RGB-only model sees
`rgb_only` ([R,G,B]); the rendering-aware model sees `rgb_depth_normal`
([R,G,B,depth_norm,Nx,Ny,Nz]).

Outputs (all under the configured output_dir):
  - eval_summary.csv     per-model mean L1 / PSNR / SSIM over the split
  - eval_per_image.csv   per-image L1 / PSNR / SSIM for each model
  - visuals/sample_*.png 6-tile comparison (low-res in, target, RGB pred,
                         7ch pred, RGB error map, 7ch error map)

Metrics only: L1, PSNR, SSIM. LPIPS, a bicubic reference, and edge-specific
metrics are out of scope here (later Phase 5 sub-steps).

torch and scikit-image are required; without either this exits non-zero with a
clear message. Run from the repo root:

    python -m ai.tools.evaluate_models --config ai/configs/evaluate_models.yaml

All paths come from the config.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _to_hwc(t):
    """(C,H,W) torch tensor -> (H,W,C) float32 numpy, clamped to [0,1]."""
    import numpy as np

    a = t.detach().cpu().numpy()
    a = np.transpose(a, (1, 2, 0))
    return np.clip(a, 0.0, 1.0).astype("float32")


def _metrics(pred_hwc, target_hwc):
    """Return (l1, psnr, ssim) for two (H,W,C) float arrays in [0,1]."""
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    l1 = float(np.abs(pred_hwc - target_hwc).mean())
    psnr = float(peak_signal_noise_ratio(target_hwc, pred_hwc, data_range=1.0))
    ssim = float(structural_similarity(target_hwc, pred_hwc,
                                       channel_axis=2, data_range=1.0))
    return l1, psnr, ssim


def _load_model(path, builder, device):
    """Load a best checkpoint into `builder(base=ckpt.base)` and eval()."""
    import torch

    ckpt = torch.load(path, map_location=device)
    base = int(ckpt.get("base", 32))
    model = builder(base=base)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, base


def _save_visual(out_path, low_in, target, pred_rgb, pred_ra, err_rgb, err_ra,
                 tile_w=480):
    """Compose a 2x3 PNG: [low-in, RGB pred, RGB err] / [target, 7ch pred, 7ch err]."""
    import numpy as np
    from PIL import Image

    def tile(arr, gray=False):
        a = (np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")
        img = Image.fromarray(a, mode="L" if gray else "RGB").convert("RGB")
        h = max(1, round(tile_w * img.height / img.width))
        return img.resize((tile_w, h), Image.BILINEAR)

    tiles = [
        tile(low_in), tile(pred_rgb), tile(err_rgb, gray=True),
        tile(target), tile(pred_ra), tile(err_ra, gray=True),
    ]
    tw, th = tiles[0].size
    grid = Image.new("RGB", (tw * 3, th * 2), (0, 0, 0))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 3) * tw, (i // 3) * th))
    grid.save(out_path)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="evaluation config YAML (see ai/configs/evaluate_models.yaml)")
    args = ap.parse_args(argv)

    try:
        import numpy as np
        import torch
    except ImportError:
        print("evaluate_models SKIPPED: torch/numpy not installed (install "
              "ai/requirements.txt on a GPU/Colab/Kaggle machine to run it).",
              file=sys.stderr)
        return 1
    try:
        import skimage.metrics  # noqa: F401
    except ImportError:
        print("evaluate_models SKIPPED: scikit-image not installed (needed for "
              "PSNR/SSIM; install ai/requirements.txt).", file=sys.stderr)
        return 1

    from ai.datasets.torch_dataset import RenderSRTorchDataset
    from ai.models.sr_rendering_aware import build_sr_rendering_aware
    from ai.models.sr_rgb import build_sr_rgb
    from ai.utils.config import load_config, resolve_path
    from ai.utils.device import select_device

    try:
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        ck = cfg.get("checkpoints", {})
        ev = cfg.get("eval", {})

        data_root = resolve_path(paths.get("data_root", "output_buffers"))
        output_dir = resolve_path(paths.get("output_dir", "results"))
        rgb_ckpt = resolve_path(ck["rgb_checkpoint"])
        ra_ckpt = resolve_path(ck["rendering_aware_checkpoint"])

        split = str(ev.get("split", "test"))
        depth_range = ev.get("depth_range")
        num_visuals = int(ev.get("num_visuals", 4))
        error_scale = float(ev.get("error_scale", 5.0))

        for p in (rgb_ckpt, ra_ckpt):
            if not p.exists():
                raise FileNotFoundError(f"checkpoint not found: {p}")

        name, device = select_device()
        device = device if device is not None else torch.device("cpu")

        # Same split / preprocessing for both; full frame (no crop, no flip).
        common = dict(root=data_root, split=split, crop_hw=None, hflip=False,
                      depth_range=depth_range)
        rgb_ds = RenderSRTorchDataset(input_mode="rgb_only", **common)
        ra_ds = RenderSRTorchDataset(input_mode="rgb_depth_normal", **common)
        if len(rgb_ds) != len(ra_ds):
            raise ValueError("RGB and 7ch datasets disagree on sample count")
        if len(rgb_ds) == 0:
            raise ValueError(f"no samples in split {split!r}")

        rgb_model, rgb_base = _load_model(rgb_ckpt, build_sr_rgb, device)
        ra_model, ra_base = _load_model(ra_ckpt, build_sr_rendering_aware, device)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"evaluate_models FAILED: {e}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "eval_per_image.csv"
    summary_path = output_dir / "eval_summary.csv"

    n = len(rgb_ds)
    print("Unified evaluation (Phase 5.1)")
    print(f"  device:   {name}")
    print(f"  split:    {split}  ({n} sample(s))")
    print(f"  rgb ckpt: {rgb_ckpt}  (base={rgb_base})")
    print(f"  7ch ckpt: {ra_ckpt}  (base={ra_base})")

    acc = {"sr_rgb": [0.0, 0.0, 0.0], "sr_rendering_aware": [0.0, 0.0, 0.0]}
    pf = per_image_path.open("w", newline="")
    pw = csv.writer(pf)
    pw.writerow(["index", "path_id", "frame_in_path", "model", "l1", "psnr", "ssim"])

    with torch.no_grad():
        for i in range(n):
            x_rgb, y, meta = rgb_ds[i]
            x_ra, _y2, _m2 = ra_ds[i]
            target = _to_hwc(y)

            x_rgb = x_rgb.unsqueeze(0).to(device)
            x_ra = x_ra.unsqueeze(0).to(device)
            pred_rgb = _to_hwc(rgb_model(x_rgb)[0])
            pred_ra = _to_hwc(ra_model(x_ra)[0])

            pid = int(meta["path_id"])
            fip = int(meta["frame_in_path"])
            for model_name, pred in (("sr_rgb", pred_rgb),
                                     ("sr_rendering_aware", pred_ra)):
                l1, psnr, ssim = _metrics(pred, target)
                acc[model_name][0] += l1
                acc[model_name][1] += psnr
                acc[model_name][2] += ssim
                pw.writerow([i, pid, fip, model_name,
                             f"{l1:.6f}", f"{psnr:.4f}", f"{ssim:.6f}"])

            if i < num_visuals:
                low_in = _to_hwc(x_rgb[0, :3])           # low-res RGB input
                err_rgb = np.clip(np.abs(pred_rgb - target).mean(axis=2)
                                  * error_scale, 0.0, 1.0)
                err_ra = np.clip(np.abs(pred_ra - target).mean(axis=2)
                                 * error_scale, 0.0, 1.0)
                _save_visual(vis_dir / f"sample_{i:03d}_p{pid}_f{fip}.png",
                             low_in, target, pred_rgb, pred_ra, err_rgb, err_ra)
    pf.close()

    sf = summary_path.open("w", newline="")
    sw = csv.writer(sf)
    sw.writerow(["model", "num_images", "l1_mean", "psnr_mean", "ssim_mean"])
    print(f"  visuals:  {min(num_visuals, n)} written to {vis_dir.resolve()}")
    print("  -- mean over split --")
    for model_name in ("sr_rgb", "sr_rendering_aware"):
        l1, psnr, ssim = (v / n for v in acc[model_name])
        sw.writerow([model_name, n, f"{l1:.6f}", f"{psnr:.4f}", f"{ssim:.6f}"])
        print(f"  {model_name:>20}: L1={l1:.6f}  PSNR={psnr:.4f}  SSIM={ssim:.6f}")
    sf.close()

    print(f"  summary:  {summary_path.resolve()}")
    print(f"  per-image: {per_image_path.resolve()}")
    print("OK: evaluated both models on the same split.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
