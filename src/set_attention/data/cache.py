from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_DATA_ROOT = Path.home() / ".cache" / "set-attention"


def resolve_data_root(user_path: Optional[str]) -> Path:
    """Return a writable root directory for datasets, creating it if missing."""
    root = Path(user_path).expanduser() if user_path else DEFAULT_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_hf_cache(data_root: Path, user_cache: Optional[str] = None) -> Path:
    """Set HuggingFace cache directories so datasets are reused across runs."""
    cache_dir = Path(user_cache).expanduser() if user_cache else data_root / "hf_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    return cache_dir


__all__ = ["resolve_data_root", "configure_hf_cache", "DEFAULT_DATA_ROOT"]
