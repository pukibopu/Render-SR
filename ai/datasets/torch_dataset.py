"""PyTorch Dataset wrapper (Phase 3.2).

Thin torch adapter over the numpy sample path in ``render_sr_dataset`` — it adds
no new data logic, only tensor conversion and the ``Dataset`` protocol so a
``DataLoader`` can batch for training. The pipeline per item is exactly:

    manifest sample -> load_sample (+ depth_norm) -> PairedTransform -> assemble_input

This module imports torch at top (it subclasses ``torch.utils.data.Dataset``);
the rest of the data layer stays torch-free, so only training/loader code pulls
torch in. Tensors are returned on CPU — moving to the device is the training
loop's job, not the Dataset's (keeps DataLoader workers picklable/fork-safe).

Per item it returns ``(x, y, meta)``:

    x    float32 (C, H, W)       network input for ``input_mode``
    y    float32 (3, 2H, 2W)     high-res RGB target
    meta {path_id, frame_in_path, split}
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .render_sr_dataset import (
    INPUT_MODES,
    PairedTransform,
    RenderSRManifest,
    assemble_input,
    load_sample,
    make_rng,
)


class RenderSRTorchDataset(Dataset):
    """Batches the renderer dataset for a given split + input mode.

    The paired transform is seeded per item as ``transform_seed + index``, so a
    sample's crop/flip is deterministic and reproducible across runs (handy for
    debugging; epoch-varying augmentation can come with the training loop later).
    """

    def __init__(self, root, split: str = "train", input_mode: str = "rgb_only",
                 crop_hw=None, hflip: bool = False, transform_seed: int = 0,
                 depth_range=None, validate: bool = True):
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode {input_mode!r} not in {INPUT_MODES}")

        self.manifest = RenderSRManifest(root, validate=validate)
        if split in (None, "all"):
            self.samples = self.manifest.samples
        else:
            self.samples = self.manifest.by_split(split)
            if not self.samples:
                raise ValueError(f"no samples in split {split!r}")

        self.split = split
        self.input_mode = input_mode
        self.crop_hw = tuple(crop_hw) if crop_hw is not None else None
        self.hflip = bool(hflip)
        self.transform_seed = int(transform_seed)
        self.depth_range = tuple(depth_range) if depth_range is not None else None
        self.transform = PairedTransform(crop_hw=self.crop_hw, hflip=self.hflip)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        s = load_sample(sample, self.manifest.low_hw, self.manifest.high_hw,
                        depth_range=self.depth_range)
        out, _info = self.transform(s, make_rng(self.transform_seed + idx))

        x = assemble_input(out, self.input_mode)          # (C, H, W) float32
        y = np.ascontiguousarray(out["rgb_high"])         # (3, 2H, 2W) float32

        meta = {
            "path_id": int(sample.path_id),
            "frame_in_path": int(sample.frame_in_path),
            "split": sample.split,
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta
