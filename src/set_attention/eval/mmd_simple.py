from __future__ import annotations
import torch


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    # x: (n,d), y: (m,d)
    x2 = (x**2).sum(dim=1, keepdim=True)
    y2 = (y**2).sum(dim=1, keepdim=True).t()
    d2 = x2 + y2 - 2.0 * (x @ y.t())
    return torch.exp(-gamma * d2.clamp_min(0.0))


def mmd2_unbiased_from_feats(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    Kxx = rbf_kernel(x, x, gamma)
    Kyy = rbf_kernel(y, y, gamma)
    Kxy = rbf_kernel(x, y, gamma)
    n = x.shape[0]
    m = y.shape[0]
    return ((Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1) + 1e-8)
            + (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1) + 1e-8)
            - 2.0 * Kxy.mean())

