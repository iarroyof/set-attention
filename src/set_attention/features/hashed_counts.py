from __future__ import annotations

import torch
from torch import nn

from set_attention.features.base import SetFeatures
from set_attention.geometry import delta_indices, geom_bias_from_delta


class HashedCountFeatureBuilder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_bins: int = 128,
        gamma: float = 1.0,
        beta: float = 0.0,
        normalize: bool = True,
        hash_seed: int = 13,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.normalize = normalize
        self.hash_seed = hash_seed
        self.attn_proj = nn.Linear(num_bins, d_model)
        self.router_proj = nn.Linear(num_bins, d_model)
        self.gamma = gamma
        self.beta = beta

    def _hash_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids * 1315423911 + self.hash_seed) % self.num_bins

    def forward(self, token_ids: torch.Tensor, bank) -> SetFeatures:
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be [seq]")
        m, max_set_size = bank.set_indices.shape
        counts = torch.zeros((m, self.num_bins), device=token_ids.device)

        for j in range(m):
            indices = bank.set_indices[j]
            valid = indices >= 0
            if not valid.any():
                continue
            tokens = token_ids[indices[valid]]
            bins = self._hash_ids(tokens)
            counts[j].scatter_add_(0, bins, torch.ones_like(bins, dtype=counts.dtype))

        if self.normalize:
            denom = bank.set_sizes.clamp_min(1).unsqueeze(-1)
            counts = counts / denom

        phi_attn = self.attn_proj(counts)
        desc_router = self.router_proj(counts)

        delta = delta_indices(bank.set_positions)
        geom_bias = geom_bias_from_delta(delta, gamma=self.gamma, beta=self.beta)
        return SetFeatures(phi_attn=phi_attn, desc_router=desc_router, geom_bias=geom_bias)
