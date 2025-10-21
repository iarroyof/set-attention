from __future__ import annotations
import torch
import torch.nn as nn


class AtomFeatureAdapter(nn.Module):
    """Low-rank adapter over atom feature bank phi_z.

    Computes adapted features: phi_adapt = phi_z + alpha * ((phi_z @ W1) @ W2^T)

    Args:
        dim: atom feature dimension D
        rank: adapter rank r
        alpha: scaling for the residual adapter
    """

    def __init__(self, dim: int, rank: int = 32, alpha: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(dim, rank, bias=False)
        self.W2 = nn.Linear(rank, dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, phi_z: torch.Tensor) -> torch.Tensor:
        # phi_z: (V, D)
        return phi_z + self.alpha * self.W2(self.W1(phi_z))

