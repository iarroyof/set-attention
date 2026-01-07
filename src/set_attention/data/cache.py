from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from set_attention.data.hf_cache import ensure_hf_cache

DEFAULT_DATA_ROOT = Path.home() / ".cache" / "set-attention"


def resolve_data_root(user_path: Optional[str]) -> Path:
    """Return a writable root directory for datasets, creating it if missing."""
    root = Path(user_path).expanduser() if user_path else DEFAULT_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_hf_cache(data_root: Path, user_cache: Optional[str] = None) -> Path:
    """Set HuggingFace cache directories so datasets are reused across runs."""
    if user_cache:
        return ensure_hf_cache(user_cache)
    # If env is already set, respect it; otherwise default under data_root/hf
    if os.environ.get("HF_HOME") or os.environ.get("HF_DATASETS_CACHE"):
        return ensure_hf_cache(None)
    return ensure_hf_cache(str(data_root / "hf"))


__all__ = ["resolve_data_root", "configure_hf_cache", "DEFAULT_DATA_ROOT"]
