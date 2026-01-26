from __future__ import annotations

import torch
from torch import nn

from set_attention.features.base import SetFeatures
from set_attention.geometry import delta_indices, geom_bias_from_delta


class GeometryOnlyFeatureBuilder(nn.Module):
    def __init__(self, d_model: int, max_sets: int, gamma: float = 1.0, beta: float = 0.0) -> None:
        super().__init__()
        self.router_emb = nn.Embedding(max_sets, d_model)
        self.gamma = gamma
        self.beta = beta

    def forward(self, set_positions: torch.Tensor) -> SetFeatures:
        delta = delta_indices(set_positions)
        geom_bias = geom_bias_from_delta(delta, gamma=self.gamma, beta=self.beta)
        desc_router = self.router_emb(set_positions)
        return SetFeatures(phi_attn=None, desc_router=desc_router, geom_bias=geom_bias)
