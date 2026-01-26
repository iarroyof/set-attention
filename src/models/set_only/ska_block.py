from __future__ import annotations

import torch
from torch import nn

from set_attention.backends.base import SetAttentionBackend


class SetAttentionBlock(nn.Module):
    def __init__(self, d_model: int, backend: SetAttentionBackend, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.backend = backend
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(
        self,
        z: torch.Tensor,
        geom_bias: torch.Tensor | None,
        content_bias: torch.Tensor | None,
        sig_mask: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor:
        attn_out = self.backend(self.norm1(z), geom_bias, content_bias, sig_mask, seq_len)
        z = z + attn_out
        z = z + self.mlp(self.norm2(z))
        return z
