from __future__ import annotations

import math
import torch
from torch import nn

from set_attention.backends.base import SetAttentionBackend
from set_attention.core import apply_score_biases, assert_set_only_scores


class DenseExactBackend(SetAttentionBackend):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        allow_token_token: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.allow_token_token = allow_token_token

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
        assert_set_only_scores(scores, seq_len=seq_len, allow_token_token=self.allow_token_token)
        scores = apply_score_biases(
            scores, geom_bias=geom_bias, content_bias=content_bias, sig_mask=sig_mask
        )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, m, self.d_model)
        return self.out_proj(out)
