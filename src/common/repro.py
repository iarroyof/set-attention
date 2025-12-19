import random

import numpy as np

try:  # pragma: no cover
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def set_seed(seed: int, deterministic: bool = False, benchmark_mode: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic or benchmark_mode:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = not deterministic and benchmark_mode
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    if not _HAS_TORCH:
        print("torch not installed; skipping test.")
        raise SystemExit(0)
    set_seed(1337, deterministic=True)
    t1 = torch.rand(3, 3)
    set_seed(1337, deterministic=True)
    t2 = torch.rand(3, 3)
    print(t1)
    print(t2)
