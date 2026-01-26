from __future__ import annotations

import torch


def delta_indices(set_positions: torch.Tensor) -> torch.Tensor:
    idx = set_positions.to(torch.long)
    return (idx[:, None] - idx[None, :]).abs()


def geom_bias_from_delta(
    delta: torch.Tensor, gamma: float = 1.0, beta: float = 0.0
) -> torch.Tensor:
    return -gamma * delta + beta
