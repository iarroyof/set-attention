from __future__ import annotations

import torch
from torch import nn


class LinearAdapter(nn.Module):
    def __init__(self, num_heads: int, d_head: int, phi_dim: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn(num_heads, d_head, phi_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(num_heads, d_head, phi_dim) * 0.01)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.dim() == 2:
            phi = phi.unsqueeze(0)
        q = torch.einsum("bmd,hkd->bhmk", phi, self.A)
        k = torch.einsum("bmd,hkd->bhmk", phi, self.B)
        return torch.einsum("bhmd,bhnd->bhmn", q, k)


class NonlinearAdapter(nn.Module):
    def __init__(self, num_heads: int, d_head: int, phi_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(phi_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, d_head),
                )
                for _ in range(num_heads)
            ]
        )
        self.k_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(phi_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, d_head),
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.dim() == 2:
            phi = phi.unsqueeze(0)
        q = torch.stack([mlp(phi) for mlp in self.q_mlps], dim=1)
        k = torch.stack([mlp(phi) for mlp in self.k_mlps], dim=1)
        return torch.einsum("bhmd,bhnd->bhmn", q, k)


class HybridAdapter(nn.Module):
    def __init__(self, adapters: list[nn.Module]) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(adapters)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.dim() == 2:
            phi = phi.unsqueeze(0)
        biases = [adapter(phi) for adapter in self.adapters]
        return torch.cat(biases, dim=1)
