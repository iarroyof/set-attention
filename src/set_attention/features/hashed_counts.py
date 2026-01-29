from __future__ import annotations

import torch
from torch import nn

from set_attention.features.base import SetFeatures
from set_attention.geometry import delta_indices, geom_bias_from_delta


class HashedCountFeatureBuilder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_phi: int,
        max_sets: int,
        num_bins: int = 128,
        gamma: float = 1.0,
        beta: float = 0.0,
        normalize: bool = True,
        hash_seed: int = 13,
        fusion: str = "mlp",
        include_geom_in_attn: bool = True,
    ) -> None:
        super().__init__()
        if fusion not in {"mlp", "linear"}:
            raise ValueError("fusion must be 'mlp' or 'linear'")
        self.num_bins = num_bins
        self.max_sets = max_sets
        self.normalize = normalize
        self.hash_seed = hash_seed
        self.geom_proj = nn.Linear(max_sets, d_phi)
        self.count_proj = nn.Linear(num_bins, d_phi)
        self.router_count_proj = nn.Linear(num_bins, d_model)
        self.router_fuse = nn.Linear(d_model * 2, d_model)
        if fusion == "mlp":
            self.fuse_attn = nn.Sequential(
                nn.Linear(d_phi * 2, d_phi),
                nn.GELU(),
                nn.Linear(d_phi, d_phi),
            )
        else:
            self.fuse_attn = nn.Linear(d_phi * 2, d_phi)
        self.gamma = gamma
        self.beta = beta
        self.fusion = fusion
        self.include_geom_in_attn = include_geom_in_attn

    def _hash_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids * 1315423911 + self.hash_seed) % self.num_bins

    def forward(self, token_ids: torch.Tensor, bank, set_states: torch.Tensor | None = None) -> SetFeatures:
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

        delta = delta_indices(bank.set_positions)
        geom_bias = geom_bias_from_delta(delta, gamma=self.gamma, beta=self.beta)
        geom_row = torch.exp(geom_bias)
        if m < self.max_sets:
            pad = torch.zeros((m, self.max_sets - m), device=geom_row.device)
            geom_row = torch.cat([geom_row, pad], dim=-1)

        if self.include_geom_in_attn:
            proj_geom = self.geom_proj(geom_row)
        else:
            d_phi = self.count_proj.out_features
            proj_geom = torch.zeros((m, d_phi), device=geom_row.device)
        proj_counts = self.count_proj(counts)
        fused = torch.cat([proj_geom, proj_counts], dim=-1)
        phi_attn = self.fuse_attn(fused)

        router_counts = self.router_count_proj(counts)
        if set_states is None:
            desc_router = router_counts
        else:
            desc_router = self.router_fuse(torch.cat([set_states, router_counts], dim=-1))

        return SetFeatures(phi_attn=phi_attn, desc_router=desc_router, geom_bias=geom_bias)
