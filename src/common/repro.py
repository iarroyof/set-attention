import os
import random

import numpy as np

try:  # pragma: no cover
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def set_seed(
    seed: int,
    deterministic: bool = False,
    benchmark_mode: bool = False,
    strict_deterministic: bool = False,
) -> None:
    if strict_deterministic and not deterministic:
        raise ValueError("strict_deterministic requires deterministic=True")
    workspace_config = None
    if strict_deterministic:
        workspace_config = os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )
        if workspace_config not in {":4096:8", ":16:8"}:
            raise ValueError(
                "strict deterministic CUDA requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic or benchmark_mode:
            torch.use_deterministic_algorithms(
                True,
                warn_only=not strict_deterministic,
            )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = not deterministic and benchmark_mode
            if strict_deterministic and hasattr(torch.backends, "cuda"):
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
        else:
            torch.use_deterministic_algorithms(False)
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
