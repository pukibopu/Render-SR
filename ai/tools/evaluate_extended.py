"""Extended evaluation (Phase 5.2): bicubic, LPIPS, edge metrics, latency.

Builds on the unified Phase 5.1 evaluation (same held-out path-ID test split,
full frame, identical preprocessing, best checkpoints only) and adds:

  - a **bicubic** 2x upsampling baseline (anchor point alongside the two models);
  - **LPIPS** (optional — skipped with a warning if `lpips` is not installed);
  - **edge metrics** — an edge mask from the high-res target luminance gradient
    (Sobel-style), reporting L1 on edge pixels and on non-edge pixels separately;
  - **inference latency** (mean ms/image, after warmup) for RGB-only, 7ch, and
    bicubic.

Methods compared: `sr_rgb`, `sr_rendering_aware`, `bicubic`. No training, no
architecture change. The 7ch input order stays [R,G,B,depth_norm,Nx,Ny,Nz].

Outputs (under output_dir):
  - eval_extended_summary.csv     per-method means
  - eval_extended_per_image.csv   per-image rows
  - visuals/ext_*.png             target, edge mask, RGB/7ch/bicubic error maps

torch and scikit-image are required (clean exit 1 if absent); LPIPS is optional.
Run from the repo root:

    python -m ai.tools.evaluate_extended --config ai/configs/evaluate_extended.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_METHODS = ("sr_rgb", "sr_rendering_aware", "bicubic")
_METRIC_KEYS = ("l1", "psnr", "ssim", "lpips", "edge_l1", "nonedge_l1",
                "edge_ratio", "latency_ms")

# Warn at most once per kind of degenerate edge region (avoids per-image spam).
_WARNED = set()


def _to_hwc(t):
    import numpy as np

    a = t.detach().cpu().numpy()
    a = np.transpose(a, (1, 2, 0))
    return np.clip(a, 0.0, 1.0).astype("float32")


def _edge_mask(target_hwc, percentile):
    """Bool (H,W) edge mask from target luminance gradient magnitude.

    Uses a STRICT threshold ``mag > percentile(mag, p)``. Rendered frames have
    large flat regions, so for high ``p`` the percentile can be 0; a strict
    comparison then naturally selects only non-zero-gradient (real edge) pixels
    instead of the whole image (the `>=` bug). If the strict threshold still
    yields no edge pixels (e.g. the percentile sits at the max), fall back to
    ``mag > 0`` (any gradient).
    """
    import numpy as np

    r, g, b = target_hwc[..., 0], target_hwc[..., 1], target_hwc[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(lum)
    mag = np.sqrt(gx * gx + gy * gy)
    thresh = float(np.percentile(mag, percentile))
    mask = mag > thresh
    if not mask.any():
        mask = mag > 0.0
    return mask


def _img_metrics(pred_hwc, target_hwc, edge_mask):
    """L1, PSNR, SSIM, edge-L1, non-edge-L1, edge-ratio for (H,W,C) in [0,1].

    edge_l1 is averaged over edge pixels only, nonedge_l1 over the rest. An empty
    region yields NaN (and a one-time warning) rather than silently folding into
    the global value.
    """
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    abs_err = np.abs(pred_hwc - target_hwc)
    l1 = float(abs_err.mean())
    psnr = float(peak_signal_noise_ratio(target_hwc, pred_hwc, data_range=1.0))
    ssim = float(structural_similarity(target_hwc, pred_hwc,
                                       channel_axis=2, data_range=1.0))
    per_px = abs_err.mean(axis=2)

    n_edge = int(edge_mask.sum())
    n_total = int(edge_mask.size)
    edge_ratio = n_edge / n_total

    if n_edge > 0:
        edge_l1 = float(per_px[edge_mask].mean())
    else:
        edge_l1 = float("nan")
        if "no_edge" not in _WARNED:
            _WARNED.add("no_edge")
            print("  warning: an image has NO edge pixels; edge_l1=NaN for it.",
                  file=sys.stderr)
    if n_edge < n_total:
        nonedge_l1 = float(per_px[~edge_mask].mean())
    else:
        nonedge_l1 = float("nan")
        if "all_edge" not in _WARNED:
            _WARNED.add("all_edge")
            print("  warning: an image is ALL edge pixels; nonedge_l1=NaN for it.",
                  file=sys.stderr)
    return l1, psnr, ssim, edge_l1, nonedge_l1, edge_ratio


def _load_model(path, builder, device):
    import torch

    ckpt = torch.load(path, map_location=device)
    base = int(ckpt.get("base", 32))
    model = builder(base=base)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, base


def _sync(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(fn, device):
    """Run fn() with device sync around it; return (output, seconds)."""
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t0


def _save_visual(out_path, target, edge_mask, low_in, err_rgb, err_ra, err_bic,
                 tile_w=480):
    import numpy as np
    from PIL import Image

    def tile(arr, gray=False):
        a = (np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")
        img = Image.fromarray(a, mode="L" if gray else "RGB").convert("RGB")
        h = max(1, round(tile_w * img.height / img.width))
        return img.resize((tile_w, h), Image.BILINEAR)

    tiles = [
        tile(target), tile(edge_mask.astype("float32"), gray=True), tile(low_in),
        tile(err_rgb, gray=True), tile(err_ra, gray=True), tile(err_bic, gray=True),
    ]
    tw, th = tiles[0].size
    grid = Image.new("RGB", (tw * 3, th * 2), (0, 0, 0))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 3) * tw, (i // 3) * th))
    grid.save(out_path)


def _save_mask(out_path, edge_mask):
    """Save the binary edge mask full-res for inspection (white = edge)."""
    import numpy as np
    from PIL import Image

    a = (edge_mask.astype("uint8") * 255)
    Image.fromarray(a, mode="L").save(out_path)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="extended-eval config YAML (see ai/configs/evaluate_extended.yaml)")
    args = ap.parse_args(argv)

    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("evaluate_extended SKIPPED: torch/numpy not installed (install "
              "ai/requirements.txt on a GPU/Colab/Kaggle machine to run it).",
              file=sys.stderr)
        return 1
    try:
        import skimage.metrics  # noqa: F401
    except ImportError:
        print("evaluate_extended SKIPPED: scikit-image not installed (needed for "
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
        edge_pct = float(ev.get("edge_percentile", 90.0))
        lpips_net = str(ev.get("lpips_net", "alex"))
        warmup = int(ev.get("latency_warmup", 3))

        for p in (rgb_ckpt, ra_ckpt):
            if not p.exists():
                raise FileNotFoundError(f"checkpoint not found: {p}")

        name, device = select_device()
        device = device if device is not None else torch.device("cpu")

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
        print(f"evaluate_extended FAILED: {e}", file=sys.stderr)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "eval_extended_per_image.csv"
    summary_path = output_dir / "eval_extended_summary.csv"

    n = len(rgb_ds)
    print("Extended evaluation (Phase 5.2)")
    print(f"  device:   {name}")
    print(f"  split:    {split}  ({n} sample(s))")
    print(f"  methods:  {', '.join(_METHODS)}")
    print(f"  edge pct: {edge_pct}")

    # Per-metric sums AND counts so NaN regions are excluded from means cleanly
    # (dividing a NaN-skipped sum by the image count was the fake-0 bug).
    acc = {m: {k: 0.0 for k in _METRIC_KEYS} for m in _METHODS}
    mcnt = {m: {k: 0 for k in _METRIC_KEYS} for m in _METHODS}
    nimg = {m: 0 for m in _METHODS}

    pf = per_image_path.open("w", newline="")
    pw = csv.writer(pf)
    pw.writerow(["index", "path_id", "frame_in_path", "method", *_METRIC_KEYS])

    with torch.no_grad():
        for i in range(n):
            x_rgb, y, meta = rgb_ds[i]
            x_ra, _y2, _m2 = ra_ds[i]
            x_rgb = x_rgb.unsqueeze(0).to(device)
            x_ra = x_ra.unsqueeze(0).to(device)
            y_t = y.unsqueeze(0).to(device)
            low = x_rgb[:, :3]

            target = _to_hwc(y)
            mask = _edge_mask(target, edge_pct)

            # Warmup (does not count toward latency means).
            if i == 0:
                for _ in range(max(0, warmup)):
                    _ = rgb_model(x_rgb); _ = ra_model(x_ra); _ = _bicubic(low)
                _sync(device)

            preds_t, secs = {}, {}
            preds_t["sr_rgb"], secs["sr_rgb"] = _timed(lambda: rgb_model(x_rgb).clamp(0, 1), device)
            preds_t["sr_rendering_aware"], secs["sr_rendering_aware"] = _timed(
                lambda: ra_model(x_ra).clamp(0, 1), device)
            preds_t["bicubic"], secs["bicubic"] = _timed(lambda: _bicubic(low), device)

            pid, fip = int(meta["path_id"]), int(meta["frame_in_path"])
            for m in _METHODS:
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
                                    * error_scale, 0.0, 1.0) for m in _METHODS}
                stem = f"ext_{i:03d}_p{pid}_f{fip}"
                _save_visual(vis_dir / f"{stem}.png",
                             target, mask, low_in,
                             emaps["sr_rgb"], emaps["sr_rendering_aware"], emaps["bicubic"])
                _save_mask(vis_dir / f"{stem}_edgemask.png", mask)
    pf.close()

    sf = summary_path.open("w", newline="")
    sw = csv.writer(sf)
    sw.writerow(["method", "num_images", *(f"{k}_mean" for k in _METRIC_KEYS)])
    print("  -- mean over split (NaN regions excluded per metric) --")
    for m in _METHODS:
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
    print("OK: extended evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
