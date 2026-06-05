"""Dataset smoke test (Phase 2.5).

Config-driven check that the dataset layer loads and validates on whatever
machine it runs on (local Mac, Colab/Kaggle, remote GPU). It exercises the full
read path — manifest -> sample load -> depth normalisation -> input assembly ->
paired transform — and prints what it produced. It does NOT train anything.

Run from the repo root:

    python -m ai.tools.dataset_smoke --config ai/configs/dataset_smoke.yaml

Exits 0 on success, non-zero if loading or validation fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai.datasets.render_sr_dataset import (
    RenderSRManifest,
    PairedTransform,
    assemble_input,
    load_sample,
    make_rng,
    INPUT_MODES,
)
from ai.utils.config import load_config, resolve_path
from ai.utils.device import select_device


def _summarise_array(name: str, a) -> str:
    return (f"  {name:<10} shape={tuple(a.shape)} dtype={a.dtype} "
            f"min={float(a.min()):.4f} max={float(a.max()):.4f}")


def run(config_path: str | Path) -> int:
    cfg = load_config(config_path)

    paths = cfg.get("paths", {})
    ds = cfg.get("dataset", {})
    smoke = cfg.get("smoke", {})

    data_root = resolve_path(paths.get("data_root", "output_buffers"))
    split = ds.get("split", "all")
    mode = ds.get("input_mode", "rgb_depth_normal")
    depth_range = ds.get("depth_range")  # [near, far] or None
    crop = ds.get("crop")                # [H, W] or None
    hflip = bool(ds.get("hflip", False))
    transform_seed = int(ds.get("transform_seed", 0))
    num_samples = int(smoke.get("num_samples", 4))

    if mode not in INPUT_MODES:
        raise ValueError(f"input_mode {mode!r} not in {INPUT_MODES}")

    device_name, _ = select_device()

    print("dataset smoke test")
    print(f"  config:      {config_path}")
    print(f"  data_root:   {data_root.resolve()}")
    print(f"  output_dir:  {resolve_path(paths.get('output_dir', 'results')).resolve()}")
    print(f"  ckpt_dir:    {resolve_path(paths.get('checkpoint_dir', 'checkpoints')).resolve()}")
    print(f"  device:      {device_name}")
    print(f"  split:       {split}")
    print(f"  input_mode:  {mode}")
    print(f"  depth_range: {depth_range if depth_range is not None else 'meta near/far'}")
    print(f"  crop:        {crop}   hflip: {hflip}   transform_seed: {transform_seed}")

    man = RenderSRManifest(data_root)  # validates files exist + split-by-path
    if split in (None, "all"):
        samples = man.samples
    else:
        samples = man.by_split(split)
    print(f"\n  manifest:    {len(man)} sample(s) total, {len(samples)} in split '{split}'")

    if not samples:
        raise ValueError(f"no samples in split '{split}'")

    dr = tuple(depth_range) if depth_range is not None else None
    crop_hw = tuple(crop) if crop is not None else None
    tf = PairedTransform(crop_hw=crop_hw, hflip=hflip)

    n = min(num_samples, len(samples))
    print(f"\n  loading {n} sample(s):")
    for i in range(n):
        sample = samples[i]
        s = load_sample(sample, man.low_hw, man.high_hw, depth_range=dr)  # validated
        out, info = tf(s, make_rng(transform_seed + i))
        x = assemble_input(out, mode)        # network input  (C, h, w)
        y = out["rgb_high"]                  # SR target      (3, 2h, 2w)

        print(f"  [{i}] {sample.stem}  split={sample.split} "
              f"crop={info['low_box']} flip={info['flip']}")
        print(_summarise_array("input", x))
        print(_summarise_array("target", y))

    print(f"\nOK: {n} sample(s) loaded, assembled (mode={mode}), and validated.")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="path to a dataset config YAML (see ai/configs/dataset_smoke.yaml)")
    args = ap.parse_args(argv)

    try:
        return run(args.config)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"smoke test FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
