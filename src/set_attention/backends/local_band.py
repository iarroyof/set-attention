from __future__ import annotations

import math
import torch
from torch import nn

from set_attention.backends.base import SetAttentionBackend
from set_attention.core import apply_score_biases, assert_set_only_scores


class LocalBandBackend(SetAttentionBackend):
    def __init__(self, d_model: int, num_heads: int, radius: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if radius < 0:
            raise ValueError("radius must be non-negative")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.radius = radius
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        geom_bias: torch.Tensor | None,
        content_bias: torch.Tensor | None,
        sig_mask: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor:
        batch, m, _ = z.shape
        q = self.q_proj(z).view(batch, m, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(z).view(batch, m, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(z).view(batch, m, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        assert_set_only_scores(scores, seq_len=seq_len)

        idx = torch.arange(m, device=z.device)
        band_mask = (idx[:, None] - idx[None, :]).abs() <= self.radius
        if sig_mask is not None:
            band_mask = band_mask & sig_mask

        scores = apply_score_biases(
            scores, geom_bias=geom_bias, content_bias=content_bias, sig_mask=band_mask
        )
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, m, self.d_model)
        return self.out_proj(out)
