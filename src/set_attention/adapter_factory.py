from __future__ import annotations

import math
from typing import Literal

from torch import nn

from set_attention.adapters import HybridAdapter, LinearAdapter, NonlinearAdapter


AdapterType = Literal["linear", "nonlinear", "hybrid"]


def select_adapter_type(phi_dim: int, d_head: int) -> AdapterType:
    effective_rank = min(phi_dim, d_head)
    if effective_rank < 20:
        return "nonlinear"
    if effective_rank >= 32:
        return "linear"
    return "hybrid"


def create_adapter(
    adapter_type: AdapterType,
    num_heads: int,
    d_head: int,
    phi_dim: int,
    hidden_multiplier: int = 2,
    max_params: int | None = None,
) -> nn.Module:
    if adapter_type == "linear":
        return LinearAdapter(num_heads=num_heads, d_head=d_head, phi_dim=phi_dim)
    if adapter_type == "nonlinear":
        hidden_dim = max(phi_dim * hidden_multiplier, 64)
        if max_params is not None:
            est_params = num_heads * (phi_dim * hidden_dim + hidden_dim * d_head) * 2
            if est_params > max_params:
                hidden_dim = max(16, math.floor(max_params / (num_heads * phi_dim * 2)))
        return NonlinearAdapter(
            num_heads=num_heads, d_head=d_head, phi_dim=phi_dim, hidden_dim=hidden_dim
        )
    if adapter_type == "hybrid":
        adapters = []
        hidden_dim = max(phi_dim * hidden_multiplier, 64)
        for head in range(num_heads):
            if head % 2 == 0:
                adapters.append(LinearAdapter(num_heads=1, d_head=d_head, phi_dim=phi_dim))
            else:
                adapters.append(
                    NonlinearAdapter(
                        num_heads=1,
                        d_head=d_head,
                        phi_dim=phi_dim,
                        hidden_dim=hidden_dim,
                    )
                )
        return HybridAdapter(adapters)
    raise ValueError(f"Unknown adapter_type: {adapter_type}")
