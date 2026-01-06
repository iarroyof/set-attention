"""Deterministic worker seeding helper for DataLoader-like iterators."""
from __future__ import annotations

import random


def make_worker_init_fn(base_seed: int):
    """Return a worker_init_fn that seeds random/np/torch deterministically."""
    def _init(worker_id: int):
        seed = int(base_seed) + int(worker_id)
        random.seed(seed)
        try:
            import numpy as np  # type: ignore
            np.random.seed(seed % (2**32 - 1))
        except Exception:
            pass
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
        except Exception:
            pass
    return _init


__all__ = ["make_worker_init_fn"]
