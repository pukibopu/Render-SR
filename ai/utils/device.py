"""Device selection (Phase 2.5).

One place to pick the training/inference device so nothing downstream hardcodes
CUDA. Preference order is ``cuda > mps > cpu``. torch is treated as optional so
this (and the dataset smoke test) still runs on a machine without torch — the
dataset layer is numpy and device-agnostic; torch only matters at training time.
"""

from __future__ import annotations


def select_device():
    """Return ``(name, torch_device_or_None)``.

    ``name`` is one of ``"cuda" | "mps" | "cpu"``. ``torch_device`` is a
    ``torch.device`` when torch is importable, else ``None`` (name is "cpu").
    """
    try:
        import torch
    except ImportError:
        return "cpu", None

    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
