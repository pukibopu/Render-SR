"""Config loading + path resolution (Phase 2.5).

Keeps paths config-driven (no hardcoded machine-specific absolutes) so the same
code runs on a local Mac, Colab/Kaggle, or a remote GPU box. Paths in the config
are interpreted relative to the current working directory (run tools from the
repo root); ``~`` is expanded.
"""

from __future__ import annotations

from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a dict. Raises if missing or not a mapping."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {p} did not parse to a mapping")
    return cfg


def resolve_path(value: str | Path) -> Path:
    """Expand ``~`` and return a Path (kept relative to CWD if not absolute)."""
    return Path(value).expanduser()
