"""Ablation evaluation (Phase 5.4): four input variants + bicubic, one split.

Unified test-split evaluation across the full input-channel ablation:

    sr_rgb              rgb_only           (3ch)
    sr_rgb_depth        rgb_depth          (4ch)
    sr_rgb_normal       rgb_normal         (6ch)
    sr_rendering_aware  rgb_depth_normal   (7ch)

plus a **bicubic** 2x anchor. Every model is the SAME shared `SRUNet` backbone
with only `in_channels` changed, evaluated on the SAME held-out path-ID test
split with identical preprocessing (full frame, no crop/flip), best checkpoints
only. Metrics and the edge/non-edge logic are reused verbatim from the fixed
Phase 5.2b extended evaluator so they cannot drift:

    L1, PSNR, SSIM, LPIPS (optional), edge_L1, non_edge_L1, edge_ratio, latency.

The method list, checkpoints, and input modes are config-driven. 7ch input order
stays [R,G,B,depth_norm,Nx,Ny,Nz].

Outputs (under output_dir):
  - eval_ablation_summary.csv     per-method means
  - eval_ablation_per_image.csv   per-image rows
  - visuals/ablation_*.png        target, edge mask, per-method error maps

torch and scikit-image are required (clean exit 1 if absent); LPIPS is optional.
Run from the repo root:

    python -m ai.tools.evaluate_ablation --config ai/configs/evaluate_ablation.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Reuse the FIXED Phase 5.2b metric/edge helpers so this cannot drift from them.
from ai.tools.evaluate_extended import (
    _METRIC_KEYS,
    _edge_mask,
    _img_metrics,
    _save_mask,
    _sync,
    _timed,
    _to_hwc,
)


def _load_ablation_model(path, in_channels, device):
    """Load a shared SRUNet checkpoint with the given in_channels."""
    import torch

    from ai.models.backbone import SRUNet

    ckpt = torch.load(path, map_location=device)
    base = int(ckpt.get("base", 32))
    # Trust the checkpoint's recorded in_channels when present; fall back to the
    # config-derived value (older RGB/7ch checkpoints predate the field).
    ckpt_in = int(ckpt.get("in_channels", in_channels))
    if ckpt_in != in_channels:
        raise ValueError(f"checkpoint {path.name} in_channels={ckpt_in} != "
                         f"config-derived {in_channels}")
    model = SRUNet(in_channels=in_channels, out_channels=3, base=base,
                   scale=2, residual=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, base


def _save_grid(out_path, target, edge_mask, low_in, err_maps, names, tile_w=420):
    """Grid: target, edge mask, low-res in, then one error map per method."""
    import numpy as np
    from PIL import Image
    from PIL import ImageDraw

    def tile(arr, label, gray=False):
        a = (np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")
        img = Image.fromarray(a, mode="L" if gray else "RGB").convert("RGB")
        h = max(1, round(tile_w * img.height / img.width))
        img = img.resize((tile_w, h), Image.BILINEAR)
        ImageDraw.Draw(img).text((4, 4), label, fill=(255, 255, 0))
        return img

    tiles = [
        tile(target, "target"),
        tile(edge_mask.astype("float32"), "edge mask", gray=True),
        tile(low_in, "low-res in"),
    ]
    tiles += [tile(err_maps[n], f"{n} err", gray=True) for n in names]

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    grid = Image.new("RGB", (tw * cols, th * rows), (0, 0, 0))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * tw, (i // cols) * th))
    grid.save(out_path)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="ablation-eval config YAML (see ai/configs/evaluate_ablation.yaml)")
    args = ap.parse_args(argv)

    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("evaluate_ablation SKIPPED: torch/numpy not installed (install "
              "ai/requirements.txt on a GPU/Colab/Kaggle machine to run it).",
              file=sys.stderr)
        return 1
    try:
        import skimage.metrics  # noqa: F401
    except ImportError:
        print("evaluate_ablation SKIPPED: scikit-image not installed (needed for "
              "PSNR/SSIM; install ai/requirements.txt).", file=sys.stderr)
        return 1

    from ai.datasets.render_sr_dataset import INPUT_MODES, _MODE_CHANNELS
    from ai.datasets.torch_dataset import RenderSRTorchDataset
    from ai.utils.config import load_config, resolve_path
    from ai.utils.device import select_device

    try:
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        ev = cfg.get("eval", {})
        method_specs = cfg.get("methods", [])
        if not method_specs:
            raise ValueError("config 'methods' list is empty")

        data_root = resolve_path(paths.get("data_root", "output_buffers"))
        output_dir = resolve_path(paths.get("output_dir", "results"))

        split = str(ev.get("split", "test"))
        depth_range = ev.get("depth_range")
        include_bicubic = bool(ev.get("include_bicubic", True))
        num_visuals = int(ev.get("num_visuals", 4))
        error_scale = float(ev.get("error_scale", 5.0))
        edge_pct = float(ev.get("edge_percentile", 90.0))
        lpips_net = str(ev.get("lpips_net", "alex"))
        warmup = int(ev.get("latency_warmup", 3))

        # Validate + resolve each method spec.
        methods = []  # (name, ckpt_path, input_mode, in_channels)
        for spec in method_specs:
            mname = str(spec["name"])
            mode = str(spec["input_mode"])
            if mode not in INPUT_MODES:
                raise ValueError(f"method {mname!r}: input_mode must be one of "
                                 f"{INPUT_MODES}, got {mode!r}")
            ckpt = resolve_path(spec["checkpoint"])
            if not ckpt.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
            methods.append((mname, ckpt, mode, _MODE_CHANNELS[mode]))

        name, device = select_device()
        device = device if device is not None else torch.device("cpu")

        # One dataset per distinct input_mode (deduped), all on the same split.
        datasets = {}
        for _n, _c, mode, _ic in methods:
            if mode not in datasets:
                datasets[mode] = RenderSRTorchDataset(
                    root=data_root, split=split, input_mode=mode,
                    crop_hw=None, hflip=False, depth_range=depth_range)
        lengths = {len(d) for d in datasets.values()}
        if len(lengths) != 1:
            raise ValueError(f"datasets disagree on sample count: {lengths}")
        n = lengths.pop()
        if n == 0:
            raise ValueError(f"no samples in split {split!r}")

        models = {}  # name -> model
        for mname, ckpt, mode, ic in methods:
            models[mname], _ = _load_ablation_model(ckpt, ic, device)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"evaluate_ablation FAILED: {e}", file=sys.stderr)
        return 1

    # LPIPS is optional.
    lpips_fn = None
    try:
        import lpips as _lpips_mod

        lpips_fn = _lpips_mod.LPIPS(net=lpips_net).to(device).eval()
        print(f"  LPIPS:    enabled (net={lpips_net})")
    except Exception as e:  # noqa: BLE001 - any failure -> skip metric cleanly
        print(f"  LPIPS:    skipped ({type(e).__name__}: {e}); install `lpips` to enable.",
              file=sys.stderr)

    def _bicubic(low):  # low: (1,3,h,w) -> (1,3,2h,2w)
        return F.interpolate(low, scale_factor=2, mode="bicubic",
                             align_corners=False).clamp(0.0, 1.0)

    def _lpips(pred_t, target_t):
        if lpips_fn is None:
            return float("nan")
        with torch.no_grad():
            d = lpips_fn(pred_t * 2 - 1, target_t * 2 - 1)
        return float(d.flatten()[0].cpu())

    # Final ordered method-name list (bicubic anchor appended if requested).
    method_names = [m[0] for m in methods]
    if include_bicubic:
        method_names.append("bicubic")

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "eval_ablation_per_image.csv"
    summary_path = output_dir / "eval_ablation_summary.csv"

    print("Ablation evaluation (Phase 5.4)")
    print(f"  device:   {name}")
    print(f"  split:    {split}  ({n} sample(s))")
    print(f"  methods:  {', '.join(method_names)}")
    print(f"  edge pct: {edge_pct}")

    acc = {m: {k: 0.0 for k in _METRIC_KEYS} for m in method_names}
    mcnt = {m: {k: 0 for k in _METRIC_KEYS} for m in method_names}
    nimg = {m: 0 for m in method_names}

    pf = per_image_path.open("w", newline="")
    pw = csv.writer(pf)
    pw.writerow(["index", "path_id", "frame_in_path", "method", *_METRIC_KEYS])

    with torch.no_grad():
        for i in range(n):
            # Target + meta come from any dataset (aligned across modes).
            ref_mode = methods[0][2]
            _x0, y, meta = datasets[ref_mode][i]
            y_t = y.unsqueeze(0).to(device)
            target = _to_hwc(y)
            mask = _edge_mask(target, edge_pct)

            # Per-mode model inputs for this index (deduped by mode).
            inputs = {}
            for mode, ds in datasets.items():
                xi, _yi, _mi = ds[i]
                inputs[mode] = xi.unsqueeze(0).to(device)
            low = inputs[ref_mode][:, :3]

            # Warmup (does not count toward latency means).
            if i == 0:
                for _ in range(max(0, warmup)):
                    for mname, _c, mode, _ic in methods:
                        _ = models[mname](inputs[mode])
                    if include_bicubic:
                        _ = _bicubic(low)
                _sync(device)

            preds_t, secs = {}, {}
            for mname, _c, mode, _ic in methods:
                preds_t[mname], secs[mname] = _timed(
                    lambda m=mname, mode=mode: models[m](inputs[mode]).clamp(0, 1),
                    device)
            if include_bicubic:
                preds_t["bicubic"], secs["bicubic"] = _timed(
                    lambda: _bicubic(low), device)

            pid, fip = int(meta["path_id"]), int(meta["frame_in_path"])
            for m in method_names:
                pred_hwc = _to_hwc(preds_t[m][0])
                (l1, psnr, ssim, edge_l1, nonedge_l1,
                 edge_ratio) = _img_metrics(pred_hwc, target, mask)
                lp = _lpips(preds_t[m], y_t)
                lat = secs[m] * 1000.0
                row = {"l1": l1, "psnr": psnr, "ssim": ssim, "lpips": lp,
                       "edge_l1": edge_l1, "nonedge_l1": nonedge_l1,
                       "edge_ratio": edge_ratio, "latency_ms": lat}
                for k, v in row.items():
                    if v == v:  # finite only; NaN regions don't dilute the mean
                        acc[m][k] += v
                        mcnt[m][k] += 1
                nimg[m] += 1
                pw.writerow([i, pid, fip, m, *(f"{row[k]:.6f}" for k in _METRIC_KEYS)])

            if i < num_visuals:
                low_in = _to_hwc(low[0])
                emaps = {m: np.clip(np.abs(_to_hwc(preds_t[m][0]) - target).mean(axis=2)
                                    * error_scale, 0.0, 1.0) for m in method_names}
                stem = f"ablation_{i:03d}_p{pid}_f{fip}"
                _save_grid(vis_dir / f"{stem}.png", target, mask, low_in,
                           emaps, method_names)
                _save_mask(vis_dir / f"{stem}_edgemask.png", mask)
    pf.close()

    sf = summary_path.open("w", newline="")
    sw = csv.writer(sf)
    sw.writerow(["method", "num_images", *(f"{k}_mean" for k in _METRIC_KEYS)])
    print("  -- mean over split (NaN regions excluded per metric) --")
    for m in method_names:
        means = {k: (acc[m][k] / mcnt[m][k] if mcnt[m][k] else float("nan"))
                 for k in _METRIC_KEYS}
        sw.writerow([m, nimg[m], *(f"{means[k]:.6f}" for k in _METRIC_KEYS)])
        print(f"  {m:>20}: L1={means['l1']:.6f} PSNR={means['psnr']:.4f} "
              f"SSIM={means['ssim']:.6f} LPIPS={means['lpips']:.6f} "
              f"edgeL1={means['edge_l1']:.6f} nonedgeL1={means['nonedge_l1']:.6f} "
              f"edgeRatio={means['edge_ratio']:.4f} lat={means['latency_ms']:.3f}ms")
    sf.close()

    print(f"  summary:   {summary_path.resolve()}")
    print(f"  per-image: {per_image_path.resolve()}")
    print(f"  visuals:   {min(num_visuals, n)} written to {vis_dir.resolve()}")
    print("OK: ablation evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
