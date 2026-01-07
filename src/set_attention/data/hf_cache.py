"""Helpers to ensure Hugging Face cache roots are consistent across tasks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def ensure_hf_cache(user_root: Optional[str] = None) -> Path:
    """
    Configure HF cache env vars in a consistent way.

    Precedence:
    1) user_root if provided (treated as HF_HOME)
    2) HF_HOME env
    3) HF_DATASETS_CACHE env (falls back to its parent as HF_HOME)
    4) ~/.cache/set-attention/hf

    Sets:
    - HF_HOME=<root>
    - HF_DATASETS_CACHE=<root>/datasets
    - TRANSFORMERS_CACHE=<root>/hub  (if not already set)

    Does NOT enable offline mode automatically.
    Returns the datasets cache path.
    """
    env_home = os.environ.get("HF_HOME")
    env_ds = os.environ.get("HF_DATASETS_CACHE")

    if user_root:
        root = Path(user_root).expanduser()
    elif env_home:
        root = Path(env_home).expanduser()
    elif env_ds:
        root = Path(env_ds).expanduser().parent
    else:
        root = Path("~/.cache/set-attention/hf").expanduser()

    datasets_dir = root / "datasets"
    hub_dir = Path(os.environ.get("TRANSFORMERS_CACHE", root / "hub")).expanduser()

    datasets_dir.mkdir(parents=True, exist_ok=True)
    hub_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_dir))

    return datasets_dir


__all__ = ["ensure_hf_cache"]
