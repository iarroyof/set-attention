from __future__ import annotations

import torch
from torch import nn

from set_attention.features.base import SetFeatures


class KernelFeatureBuilder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_phi: int,
        max_sets: int,
        gamma: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.max_sets = max_sets
        self.attn_proj = nn.Linear(max_sets, d_phi)
        self.router_proj = nn.Linear(max_sets, d_model)

    def forward(self, sig: torch.Tensor, sizes: torch.Tensor) -> SetFeatures:
        if sig.dim() != 2:
            raise ValueError("sig must be [m, k]")
        m, k = sig.shape
        if m > self.max_sets:
            raise ValueError(f"m={m} exceeds max_sets={self.max_sets}")

        matches = (sig.unsqueeze(1) == sig.unsqueeze(0)).float().mean(dim=-1)
        sizes_row = sizes.view(m, 1).float()
        sizes_col = sizes.view(1, m).float()
        j_hat = matches
        denom = (1.0 + j_hat).clamp_min(1e-6)
        delta_hat = sizes_row + sizes_col - 2.0 * (j_hat / denom) * (sizes_row + sizes_col)
        k_delta = torch.exp(-self.gamma * delta_hat + self.beta)

        if m < self.max_sets:
            pad = torch.zeros((m, self.max_sets - m), device=sig.device)
            k_delta = torch.cat([k_delta, pad], dim=-1)

        phi_attn = self.attn_proj(k_delta)
        desc_router = self.router_proj(k_delta)
        return SetFeatures(phi_attn=phi_attn, desc_router=desc_router, geom_bias=None)
