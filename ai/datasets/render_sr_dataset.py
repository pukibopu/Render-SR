"""Render-SR dataset manifest reader (Phase 2.1).

Reads a renderer output directory's ``manifest.json`` and builds a list of
samples, one per rendered frame. Each sample resolves the five files the
renderer writes for that frame and exposes the frame's identity (path id,
frame-in-path, split, seed) and camera snapshot.

Importing this module and using ``RenderSRManifest`` pulls in no array
dependencies — the manifest/sample layer is pure stdlib. Pixel loading
(``load_sample`` / ``RenderSRManifest.load``) imports numpy + PIL lazily and
returns float32 CHW numpy arrays (no torch dependency, so it stays
device-agnostic; a torch wrapper can come with training). Depth is returned
both raw (``depth``, eye-space metres) and normalised to [0,1] (``depth_norm``,
Phase 2.3 — see normalize_depth). Paired crop/flip and the 3ch/7ch channel
assembly are later sub-steps, deliberately not done here.

Layout it expects (written by the renderer, see renderer/src/io/FrameWriter):

    <root>/manifest.json
    <root>/rgb_low/frame_PP_FFFF.png
    <root>/rgb_high/frame_PP_FFFF.png
    <root>/depth/frame_PP_FFFF.npy
    <root>/normal/frame_PP_FFFF.npy
    <root>/meta/frame_PP_FFFF.json

The train/test split is recorded per frame in the manifest and is by *path id*
(adjacent frames within a path are near-duplicates; splitting on them would
leak). This reader enforces that no path id appears in more than one split.

Debug CLI:

    python -m ai.datasets.render_sr_dataset --root output_buffers
    python -m ai.datasets.render_sr_dataset --root output_buffers --split test
    python -m ai.datasets.render_sr_dataset --root output_buffers --load --index 0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    """One rendered frame: identity, file paths, and camera snapshot.

    Pixel data is intentionally not loaded here — only the paths to it.
    """

    stem: str                 # "frame_PP_FFFF"
    path_id: int
    frame_in_path: int
    split: str                # "train" | "test"
    seed: int                 # the camera path's derived seed
    path_type: str            # "orbit" | "dolly"
    camera: dict              # manifest camera snapshot (target/azimuth/.../eye)

    rgb_low: Path
    rgb_high: Path
    depth: Path
    normal: Path
    meta: Path

    def files(self) -> list[Path]:
        return [self.rgb_low, self.rgb_high, self.depth, self.normal, self.meta]


class RenderSRManifest:
    """Parsed view of a renderer ``manifest.json`` plus its frame samples.

    Metadata only. Loading happens in a later phase; this class is what the
    loader and the split logic key off.
    """

    def __init__(self, root: str | Path, validate: bool = True):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no manifest.json under {self.root}")

        with manifest_path.open() as f:
            man = json.load(f)

        self.seed: int = man["seed"]
        self.frames_per_path: int = man["frames_per_path"]
        self.low_res: tuple[int, int] = tuple(man["low_res"])    # (W, H)
        self.high_res: tuple[int, int] = tuple(man["high_res"])
        self.paths: list[dict] = man.get("paths", [])

        self.samples: list[Sample] = [self._make_sample(fr) for fr in man["frames"]]

        self._check_split_by_path()
        if validate:
            self.validate_files()

    # --- construction helpers ---

    def _make_sample(self, fr: dict) -> Sample:
        stem = fr["frame"]
        return Sample(
            stem=stem,
            path_id=fr["path_id"],
            frame_in_path=fr["frame_in_path"],
            split=fr["split"],
            seed=fr["seed"],
            path_type=fr.get("path_type", "?"),
            camera=fr.get("camera", {}),
            rgb_low=self.root / "rgb_low" / f"{stem}.png",
            rgb_high=self.root / "rgb_high" / f"{stem}.png",
            depth=self.root / "depth" / f"{stem}.npy",
            normal=self.root / "normal" / f"{stem}.npy",
            meta=self.root / "meta" / f"{stem}.json",
        )

    def _check_split_by_path(self) -> None:
        """Fail if a path id appears in more than one split (would leak)."""
        split_by_path: dict[int, set[str]] = {}
        for s in self.samples:
            split_by_path.setdefault(s.path_id, set()).add(s.split)
        offenders = {pid: sorted(sp) for pid, sp in split_by_path.items() if len(sp) > 1}
        if offenders:
            raise ValueError(
                f"split is not by path id — these path ids span multiple splits: "
                f"{offenders}"
            )

    def validate_files(self) -> None:
        """Raise if any sample is missing one of its five files."""
        missing: list[str] = []
        for s in self.samples:
            for p in s.files():
                if not p.exists():
                    missing.append(str(p.relative_to(self.root)))
        if missing:
            preview = "\n  ".join(missing[:20])
            more = "" if len(missing) <= 20 else f"\n  ... (+{len(missing) - 20} more)"
            raise FileNotFoundError(
                f"{len(missing)} referenced file(s) missing under {self.root}:\n  "
                f"{preview}{more}"
            )

    # --- access ---

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Sample:
        return self.samples[i]

    @property
    def low_hw(self) -> tuple[int, int]:
        """Low-res (height, width). manifest low_res is stored (W, H)."""
        w, h = self.low_res
        return (h, w)

    @property
    def high_hw(self) -> tuple[int, int]:
        w, h = self.high_res
        return (h, w)

    def load(self, i: int, validate: bool = True) -> dict:
        """Load one sample's pixels into a sample dict (see load_sample)."""
        return load_sample(self.samples[i], self.low_hw, self.high_hw, validate)

    def by_split(self, split: str) -> list[Sample]:
        return [s for s in self.samples if s.split == split]

    def split_names(self) -> list[str]:
        return sorted({s.split for s in self.samples})

    def path_ids(self) -> list[int]:
        return sorted({s.path_id for s in self.samples})


# --- pixel loading (Phase 2.2) ----------------------------------------------

# Tolerances for the normal unit-length check on foreground pixels.
_NORMAL_MEAN_TOL = 0.05
_NORMAL_MAX_TOL = 0.10


# --- depth normalisation (Phase 2.3) ----------------------------------------
#
# Convention (deterministic, documented once here):
#   depth_norm = clip((depth_raw - near) / (far - near), 0, 1)
# where (near, far) are the camera planes from meta["camera"] by default
# (constant across the v1 dataset — Camera near/far are fixed), or an explicit
# dataset-wide override passed to load_sample().
#
#   * Linear map: near -> 0, far -> 1.
#   * Background pixels carry raw depth 0 (cleared attachment) and therefore map
#     to 0 after clipping. Background and a surface exactly at the near plane are
#     thus indistinguishable in depth_norm; that is acceptable for v1 and noted
#     so downstream code doesn't read 0 as "near surface".
#   * Foreground geometry outside [near, far] is clipped; load_sample records the
#     clipped fraction so out-of-range renders are visible, not silent.
#
# `depth` (raw eye-space, unnormalised) stays the key returned by Phase 2.2.
# `depth_norm` (this normalised [0,1] map) is the network-facing depth used by
# the later 7ch channel assembly. Keep both so nothing downstream re-derives it.


def normalize_depth(depth_raw, near: float, far: float):
    """Map raw linear eye-space depth to [0, 1] via clip((d-near)/(far-near)).

    Returns a float32 array the same shape as ``depth_raw``. See the module
    comment above for the convention. Raises if the range is non-positive.
    """
    import numpy as np

    if not (far > near):
        raise ValueError(f"depth normalisation needs far > near, got near={near}, far={far}")
    norm = (np.asarray(depth_raw, dtype=np.float32) - np.float32(near)) / np.float32(far - near)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _depth_norm_stats(depth_raw, depth_norm, near: float, far: float) -> dict:
    """Foreground-focused stats for debug output (see CLI)."""
    import numpy as np

    raw = depth_raw[0] if depth_raw.ndim == 3 else depth_raw
    norm = depth_norm[0] if depth_norm.ndim == 3 else depth_norm
    fg = raw > 0.0
    clipped_low = int(np.count_nonzero(fg & (raw < near)))
    clipped_high = int(np.count_nonzero(fg & (raw > far)))
    fg_count = int(np.count_nonzero(fg))
    return {
        "near": float(near),
        "far": float(far),
        "raw_min": float(raw[fg].min()) if fg_count else 0.0,
        "raw_max": float(raw[fg].max()) if fg_count else 0.0,
        "norm_min": float(norm[fg].min()) if fg_count else 0.0,
        "norm_max": float(norm[fg].max()) if fg_count else 0.0,
        "fg_count": fg_count,
        "clipped_low": clipped_low,    # foreground nearer than `near`
        "clipped_high": clipped_high,  # foreground farther than `far`
        "clipped": bool(clipped_low or clipped_high),
    }


def _load_png_chw(path: Path, expect_hw: tuple[int, int] | None):
    """Load an RGB PNG as float32 CHW in [0, 1]; drop alpha if present."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0  # HWC
    h, w = arr.shape[:2]
    if expect_hw is not None and (h, w) != expect_hw:
        raise ValueError(f"{path.name}: image is {(h, w)}, expected {expect_hw}")
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))  # CHW


def _load_npy(path: Path):
    import numpy as np
    return np.asarray(np.load(path), dtype=np.float32)


def load_sample(sample: Sample,
                low_hw: tuple[int, int],
                high_hw: tuple[int, int],
                validate: bool = True,
                depth_range: tuple[float, float] | None = None) -> dict:
    """Load one frame's pixels + identity into a sample dict.

    Returns float32 numpy arrays in CHW order:
      rgb_low    (3, lowH,  lowW)  in [0, 1]
      rgb_high   (3, highH, highW) in [0, 1]
      depth      (1, lowH,  lowW)  raw eye-space metres (unnormalised)
      depth_norm (1, lowH,  lowW)  depth normalised to [0, 1] (Phase 2.3)
      normal     (3, lowH,  lowW)  view-space, components in [-1, 1]
    plus path_id, frame_in_path, split, depth_norm_stats, and the parsed meta.

    ``depth`` stays the raw, unnormalised eye-space map (unchanged from Phase
    2.2); ``depth_norm`` is the network-facing [0,1] depth. By default the
    normalisation range is meta["camera"] near/far; pass ``depth_range`` to
    override with an explicit dataset-wide (near, far). See normalize_depth.
    """
    import numpy as np

    rgb_low = _load_png_chw(sample.rgb_low, low_hw)
    rgb_high = _load_png_chw(sample.rgb_high, high_hw)

    depth2d = _load_npy(sample.depth)
    if depth2d.ndim != 2:
        raise ValueError(f"{sample.depth.name}: expected 2-D depth, got {depth2d.shape}")
    depth = depth2d[None, :, :]  # (1, H, W)

    normal_hwc = _load_npy(sample.normal)
    if normal_hwc.ndim != 3 or normal_hwc.shape[2] != 3:
        raise ValueError(f"{sample.normal.name}: expected (H,W,3), got {normal_hwc.shape}")
    normal = np.ascontiguousarray(np.transpose(normal_hwc, (2, 0, 1)))  # (3, H, W)

    with sample.meta.open() as f:
        meta = json.load(f)

    if depth_range is not None:
        near, far = float(depth_range[0]), float(depth_range[1])
    else:
        cam = meta.get("camera", {})
        if "near" not in cam or "far" not in cam:
            raise ValueError(f"{sample.meta.name}: meta.camera has no near/far for depth norm")
        near, far = float(cam["near"]), float(cam["far"])

    depth_norm = normalize_depth(depth, near, far)  # (1, H, W) in [0, 1]
    stats = _depth_norm_stats(depth, depth_norm, near, far)

    out = {
        "rgb_low": rgb_low,
        "rgb_high": rgb_high,
        "depth": depth,
        "depth_norm": depth_norm,
        "normal": normal,
        "path_id": sample.path_id,
        "frame_in_path": sample.frame_in_path,
        "split": sample.split,
        "depth_norm_stats": stats,
        "meta": meta,
    }
    if validate:
        validate_sample(out, low_hw, high_hw)
    return out


def validate_sample(s: dict,
                    low_hw: tuple[int, int],
                    high_hw: tuple[int, int]) -> None:
    """Raise ValueError if shapes/dtypes/ranges are wrong for a sample dict."""
    import numpy as np

    lh, lw = low_hw
    hh, hw = high_hw
    expected = {
        "rgb_low": (3, lh, lw),
        "rgb_high": (3, hh, hw),
        "depth": (1, lh, lw),
        "normal": (3, lh, lw),
    }
    if "depth_norm" in s:
        expected["depth_norm"] = (1, lh, lw)
    for name, shape in expected.items():
        a = s[name]
        if a.dtype != np.float32:
            raise ValueError(f"{name}: dtype {a.dtype}, expected float32")
        if a.shape != shape:
            raise ValueError(f"{name}: shape {a.shape}, expected {shape}")

    # RGB in [0, 1], finite.
    for name in ("rgb_low", "rgb_high"):
        a = s[name]
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: contains non-finite values")
        if a.min() < 0.0 or a.max() > 1.0:
            raise ValueError(f"{name}: out of [0,1] (min={a.min():.4f}, max={a.max():.4f})")

    # Depth (raw): finite, has foreground, non-degenerate range.
    depth = s["depth"][0]
    if not np.all(np.isfinite(depth)):
        raise ValueError("depth: contains non-finite values")
    fg = depth > 0.0
    if not fg.any():
        raise ValueError("depth: no foreground pixels (depth>0 nowhere)")
    if depth[fg].min() == depth[fg].max():
        raise ValueError("depth: foreground degenerate (min == max)")

    # depth_norm: finite and within [0, 1] (clip target of Phase 2.3).
    if "depth_norm" in s:
        dn = s["depth_norm"]
        if not np.all(np.isfinite(dn)):
            raise ValueError("depth_norm: contains non-finite values")
        if dn.min() < 0.0 or dn.max() > 1.0:
            raise ValueError(
                f"depth_norm: out of [0,1] (min={dn.min():.4f}, max={dn.max():.4f})"
            )

    # Normal: finite, has negative components (not [0,1] colour), unit on fg.
    normal = s["normal"]
    if not np.all(np.isfinite(normal)):
        raise ValueError("normal: contains non-finite values")
    if normal.min() >= 0.0:
        raise ValueError(
            "normal: no negative components — likely stored as [0,1] colour "
            "rather than view-space float in [-1,1]"
        )
    lengths = np.sqrt((normal * normal).sum(axis=0))  # (H, W)
    fg_len = lengths[fg]
    if fg_len.size:
        mean_len = float(fg_len.mean())
        max_dev = float(np.max(np.abs(fg_len - 1.0)))
        if abs(mean_len - 1.0) > _NORMAL_MEAN_TOL:
            raise ValueError(
                f"normal: foreground mean length {mean_len:.4f} not within "
                f"{_NORMAL_MEAN_TOL} of 1.0"
            )
        if max_dev > _NORMAL_MAX_TOL:
            raise ValueError(f"normal: max |len-1| {max_dev:.4f} > {_NORMAL_MAX_TOL}")


# --- input channel assembly (Phase 2.4a) ------------------------------------
#
# Build the network input tensor by stacking low-res channels in the *fixed*
# canonical order [R, G, B, depth, Nx, Ny, Nz]. The depth slot always uses
# `depth_norm` (the [0,1] map from Phase 2.3), never the raw eye-space depth —
# raw `depth` stays in the sample dict for inspection/metrics but is never fed.
#
# Each mode is a contiguous-from-the-front subset of that order, so the RGB-only
# (3ch) and rendering-aware (7ch) inputs are byte-identical in their shared
# leading channels — the fair 3ch↔7ch comparison the project hinges on.
#
#   rgb_only         3ch  [R,G,B]
#   rgb_depth        4ch  [R,G,B,depth_norm]
#   rgb_normal       6ch  [R,G,B,Nx,Ny,Nz]
#   rgb_depth_normal 7ch  [R,G,B,depth_norm,Nx,Ny,Nz]
#
# Paired crop/flip are a later sub-step (2.4b) and are deliberately not here.

INPUT_MODES = ("rgb_only", "rgb_depth", "rgb_normal", "rgb_depth_normal")

_MODE_CHANNELS = {
    "rgb_only": 3,
    "rgb_depth": 4,
    "rgb_normal": 6,
    "rgb_depth_normal": 7,
}


def assemble_input(s: dict, mode: str):
    """Stack a loaded sample's low-res channels into the input tensor for `mode`.

    Returns a float32 (C, lowH, lowW) array. C is 3/4/6/7 by mode. Channel order
    is always [R,G,B,(depth_norm),(Nx,Ny,Nz)]; the depth channel is `depth_norm`.
    """
    import numpy as np

    if mode not in INPUT_MODES:
        raise ValueError(f"unknown input mode {mode!r}; expected one of {INPUT_MODES}")

    rgb = s["rgb_low"]            # (3, H, W)
    parts = [rgb]
    if mode in ("rgb_depth", "rgb_depth_normal"):
        if "depth_norm" not in s:
            raise ValueError(f"mode {mode!r} needs depth_norm (run load_sample with normalisation)")
        parts.append(s["depth_norm"])   # (1, H, W)
    if mode in ("rgb_normal", "rgb_depth_normal"):
        parts.append(s["normal"])       # (3, H, W)

    x = np.ascontiguousarray(np.concatenate(parts, axis=0).astype(np.float32))

    expected_c = _MODE_CHANNELS[mode]
    _, h, w = rgb.shape
    if x.shape != (expected_c, h, w):
        raise ValueError(f"mode {mode!r}: assembled {x.shape}, expected {(expected_c, h, w)}")
    return x


# --- paired transforms (Phase 2.4b) -----------------------------------------
#
# Geometric augmentation applied *identically* to every aligned buffer so the
# pixel-alignment invariant survives (misalignment silently destroys the
# experiment). All spatial ops are channel-agnostic except the horizontal flip
# of a view-space normal, which must negate Nx (and only Nx): mirroring x in
# screen space mirrors the x-component of the surface direction. Ny/Nz are
# unchanged. This is the one place normals are not treated like generic data.
#
#   crop:  low-res box (top,left,h,w); the high-res target is cropped at the
#          matching box * scale (default 2) so the pair stays pixel-exact.
#   flip:  optional random horizontal flip; spatial reverse for rgb/depth,
#          spatial reverse + negate Nx for normal.
#
# Determinism: pass a numpy Generator (make_rng(seed)) so a run is reproducible
# and debuggable. Order is fixed as [R,G,B,depth_norm,Nx,Ny,Nz]; for an assembled
# input tensor the Nx channel index is mode-dependent (see _nx_index).

# Which buffers are cropped/flipped as plain spatial data vs. as normals.
_GEOM_PLAIN = ("rgb_low", "depth", "depth_norm")  # cropped at low res, plain flip
_GEOM_NORMAL = "normal"                            # plain crop, Nx-negating flip
_GEOM_HIGH = "rgb_high"                            # cropped at low_box * scale


def make_rng(seed: int | None):
    """A numpy default_rng for deterministic, debuggable transforms."""
    import numpy as np
    return np.random.default_rng(seed)


def random_crop_box(in_hw: tuple[int, int],
                    crop_hw: tuple[int, int],
                    rng) -> tuple[int, int, int, int]:
    """Pick a (top, left, h, w) crop box uniformly inside ``in_hw``."""
    ih, iw = in_hw
    ch, cw = crop_hw
    if ch > ih or cw > iw:
        raise ValueError(f"crop {crop_hw} does not fit inside {in_hw}")
    top = int(rng.integers(0, ih - ch + 1))
    left = int(rng.integers(0, iw - cw + 1))
    return (top, left, ch, cw)


def crop_chw(a, top: int, left: int, h: int, w: int):
    """Crop a (C,H,W) array; returns a contiguous view-copy."""
    import numpy as np
    return np.ascontiguousarray(a[:, top:top + h, left:left + w])


def hflip_chw(a):
    """Horizontal (width-axis) spatial flip of a (C,H,W) array."""
    import numpy as np
    return np.ascontiguousarray(a[:, :, ::-1])


def hflip_normal_chw(normal):
    """Horizontal flip of a view-space normal (3,H,W): mirror + negate Nx only."""
    import numpy as np
    out = np.ascontiguousarray(normal[:, :, ::-1])
    out[0] = -out[0]   # Nx; Ny/Nz unchanged
    return out


def _nx_index(mode: str) -> int | None:
    """Channel index of Nx in an assembled tensor for ``mode`` (None if no normal)."""
    if mode == "rgb_normal":
        return 3           # [R,G,B,Nx,Ny,Nz]
    if mode == "rgb_depth_normal":
        return 4           # [R,G,B,depth_norm,Nx,Ny,Nz]
    return None            # rgb_only / rgb_depth carry no normal


@dataclass(frozen=True)
class PairedTransform:
    """Paired random crop (+ optional random hflip) over aligned buffers.

    ``crop_hw`` is the low-res crop size; the high-res target is cropped at
    ``crop_hw * scale``. ``hflip`` enables a random horizontal flip (p=0.5).
    """

    crop_hw: tuple[int, int] | None = None
    hflip: bool = False
    scale: int = 2

    def plan(self, low_hw: tuple[int, int], rng) -> dict:
        """Decide the crop box + flip for one sample (the random draws)."""
        if self.crop_hw is not None:
            top, left, h, w = random_crop_box(low_hw, self.crop_hw, rng)
        else:
            top, left, h, w = 0, 0, low_hw[0], low_hw[1]
        flip = bool(self.hflip and rng.random() < 0.5)
        s = self.scale
        return {
            "low_box": (top, left, h, w),
            "high_box": (top * s, left * s, h * s, w * s),
            "flip": flip,
            "scale": s,
        }

    def __call__(self, sample: dict, rng) -> tuple[dict, dict]:
        """Return (transformed_sample, info). Non-spatial keys pass through."""
        rgb_low = sample["rgb_low"]
        low_hw = (rgb_low.shape[1], rgb_low.shape[2])
        info = self.plan(low_hw, rng)
        tl, ll, th, tw = info["low_box"]
        ht, hl, hh, hw = info["high_box"]
        flip = info["flip"]

        out = dict(sample)  # shallow copy; replace spatial buffers below
        for key in _GEOM_PLAIN:
            if key in sample:
                a = crop_chw(sample[key], tl, ll, th, tw)
                out[key] = hflip_chw(a) if flip else a
        if _GEOM_NORMAL in sample:
            a = crop_chw(sample[_GEOM_NORMAL], tl, ll, th, tw)
            out[_GEOM_NORMAL] = hflip_normal_chw(a) if flip else a
        if _GEOM_HIGH in sample:
            a = crop_chw(sample[_GEOM_HIGH], ht, hl, hh, hw)
            out[_GEOM_HIGH] = hflip_chw(a) if flip else a

        # depth_norm_stats no longer describes the cropped depth — recompute if
        # we can, else drop it so nothing reads a stale summary.
        if "depth" in out and "depth_norm" in out and "depth_norm_stats" in sample:
            near = sample["depth_norm_stats"]["near"]
            far = sample["depth_norm_stats"]["far"]
            out["depth_norm_stats"] = _depth_norm_stats(out["depth"], out["depth_norm"], near, far)
        else:
            out.pop("depth_norm_stats", None)
        return out, info


def transform_input(x, mode: str, info: dict):
    """Apply the same crop+flip (from PairedTransform.plan/__call__) to an
    already-assembled input tensor, negating Nx for the mode if a flip happened.

    Equivalent to transforming the named buffers then re-assembling.
    """
    tl, ll, th, tw = info["low_box"]
    x = crop_chw(x, tl, ll, th, tw)
    if info["flip"]:
        x = hflip_chw(x)
        nx = _nx_index(mode)
        if nx is not None:
            x[nx] = -x[nx]
    return x


# --- debug CLI --------------------------------------------------------------


def _summary(man: RenderSRManifest, split: str | None) -> str:
    lines: list[str] = []
    lw, lh = man.low_res
    hw, hh = man.high_res
    lines.append(f"root:            {man.root}")
    lines.append(f"seed:            {man.seed}")
    lines.append(f"frames_per_path: {man.frames_per_path}")
    lines.append(f"low_res:         {lw}x{lh}")
    lines.append(f"high_res:        {hw}x{hh}")
    lines.append(f"total frames:    {len(man)}")

    # per-split counts
    counts: dict[str, int] = {}
    for s in man.samples:
        counts[s.split] = counts.get(s.split, 0) + 1
    lines.append("by split:        " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # per-path summary
    lines.append("paths:")
    for p in man.paths:
        lines.append(
            f"  path {p['path_id']:>2}  {p.get('type','?'):<6}  "
            f"split={p.get('split','?'):<5}  seed={p.get('seed')}  "
            f"frames={p.get('frames')}"
        )

    samples = man.by_split(split) if split else man.samples
    if split:
        lines.append(f"\nsplit '{split}': {len(samples)} sample(s)")
    lines.append("\nexamples:")
    for s in samples[:3]:
        cam = s.camera
        az = cam.get("azimuth_rad")
        dist = cam.get("distance")
        lines.append(
            f"  {s.stem}  path={s.path_id} f={s.frame_in_path} split={s.split} "
            f"type={s.path_type} az={az} dist={dist}"
        )
        lines.append(f"      rgb_low={s.rgb_low.relative_to(man.root)} "
                     f"depth={s.depth.relative_to(man.root)} "
                     f"normal={s.normal.relative_to(man.root)}")
    return "\n".join(lines)


def _tensor_summary(s: dict) -> str:
    import numpy as np
    lines: list[str] = []
    for name in ("rgb_low", "rgb_high", "depth", "normal"):
        a = s[name]
        lines.append(f"  {name:<8} shape={tuple(a.shape)} dtype={a.dtype} "
                     f"min={a.min():.4f} max={a.max():.4f}")
    depth = s["depth"][0]
    fg = depth > 0.0
    lines.append(f"  depth fg: cov={fg.mean()*100:.1f}% "
                 f"min={depth[fg].min():.3f} max={depth[fg].max():.3f}")
    st = s.get("depth_norm_stats")
    if st is not None:
        lines.append(
            f"  depth_norm: near={st['near']:.4g} far={st['far']:.4g}  "
            f"raw[fg]=[{st['raw_min']:.3f}, {st['raw_max']:.3f}]  "
            f"norm[fg]=[{st['norm_min']:.4f}, {st['norm_max']:.4f}]")
        lines.append(
            f"  depth_norm clip: clipped={st['clipped']} "
            f"(near={st['clipped_low']}, far={st['clipped_high']} of fg={st['fg_count']})")
    normal = s["normal"]
    lengths = np.sqrt((normal * normal).sum(axis=0))
    lines.append(f"  normal fg: mean_len={float(lengths[fg].mean()):.4f} "
                 f"has_negative={bool(normal.min() < 0.0)}")
    lines.append(f"  path_id={s['path_id']} frame_in_path={s['frame_in_path']} "
                 f"split={s['split']}")
    return "\n".join(lines)


def _transform_summary(s: dict, args) -> str:
    """Apply a paired transform deterministically and describe what it did."""
    import numpy as np

    crop_hw = tuple(args.crop) if args.crop else None
    tf = PairedTransform(crop_hw=crop_hw, hflip=args.hflip)
    rng = make_rng(args.transform_seed)

    nx_before = float(s["normal"][0].mean())
    out, info = tf(s, rng)
    nx_after = float(out["normal"][0].mean())

    tl, ll, th, tw = info["low_box"]
    ht, hl, hh, hw = info["high_box"]
    lines = [
        f"transform (crop={crop_hw}, hflip={args.hflip}, seed={args.transform_seed}):",
        f"  low_box  (top,left,h,w) = ({tl},{ll},{th},{tw})",
        f"  high_box (top,left,h,w) = ({ht},{hl},{hh},{hw})  scale={info['scale']}",
        f"  flip decision          = {info['flip']}",
        f"  shapes: rgb_low={tuple(out['rgb_low'].shape)} "
        f"depth_norm={tuple(out['depth_norm'].shape)} "
        f"normal={tuple(out['normal'].shape)} rgb_high={tuple(out['rgb_high'].shape)}",
        f"  Nx mean: before={nx_before:+.4f} after={nx_after:+.4f} "
        f"({'negated (flip)' if info['flip'] else 'unchanged (no flip)'})",
    ]

    # high == 2x low box, after the same flip
    s_ = info["scale"]
    assert info["high_box"] == (tl * s_, ll * s_, th * s_, tw * s_)
    assert out["rgb_high"].shape[1:] == (th * s_, tw * s_)
    assert out["rgb_low"].shape[1:] == (th, tw)

    if args.input_mode:
        # assembled-tensor path must equal transform-then-assemble (consistency)
        x_pre = transform_input(assemble_input(s, args.input_mode), args.input_mode, info)
        x_post = assemble_input(out, args.input_mode)
        match = bool(np.array_equal(x_pre, x_post))
        lines.append(f"  assembled({args.input_mode}) shape={tuple(x_post.shape)} "
                     f"transform_input==transform-then-assemble: {match}")
        if not match:
            raise ValueError("transform_input diverged from transform-then-assemble")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="output_buffers", type=Path,
                    help="renderer output root containing manifest.json")
    ap.add_argument("--split", default=None,
                    help="restrict the example/sample listing to this split")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the per-file existence check")
    ap.add_argument("--load", action="store_true",
                    help="load one sample's pixels and print a tensor summary")
    ap.add_argument("--index", type=int, default=0,
                    help="sample index to --load (default 0)")
    ap.add_argument("--input-mode", default=None, choices=INPUT_MODES,
                    help="also assemble + summarise the input tensor for this mode")
    ap.add_argument("--crop", nargs=2, type=int, metavar=("H", "W"), default=None,
                    help="paired random crop to low-res HxW (high-res cropped at 2x)")
    ap.add_argument("--hflip", action="store_true",
                    help="enable random paired horizontal flip (Nx negated)")
    ap.add_argument("--transform-seed", type=int, default=0,
                    help="seed for the deterministic transform RNG (default 0)")
    args = ap.parse_args(argv)

    try:
        man = RenderSRManifest(args.root, validate=not args.no_validate)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(_summary(man, args.split))
    print(f"\nOK: {len(man)} sample(s), files validated"
          f"{' (skipped)' if args.no_validate else ''}.")

    if args.load:
        if not (0 <= args.index < len(man)):
            print(f"error: --index {args.index} out of range [0, {len(man)})",
                  file=sys.stderr)
            return 1
        try:
            s = man.load(args.index)
        except (FileNotFoundError, ValueError) as e:
            print(f"error loading sample {args.index}: {e}", file=sys.stderr)
            return 1
        print(f"\nloaded sample[{args.index}] = {man[args.index].stem}:")
        print(_tensor_summary(s))

        if args.crop or args.hflip:
            try:
                print("\n" + _transform_summary(s, args))
            except ValueError as e:
                print(f"error in transform: {e}", file=sys.stderr)
                return 1

        if args.input_mode:
            x = assemble_input(s, args.input_mode)
            print(f"\ninput tensor (mode={args.input_mode}):")
            print(f"  shape={tuple(x.shape)} dtype={x.dtype} "
                  f"min={x.min():.4f} max={x.max():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
