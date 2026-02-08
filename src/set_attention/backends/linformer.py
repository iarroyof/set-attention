from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from set_attention.core import apply_score_biases, assert_set_only_scores


class LinformerBackend(nn.Module):
    """Linformer-style set attention: projects keys/values along set dimension."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_sets: int,
        k: int,
        dropout: float = 0.0,
        allow_token_token: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if k <= 0:
            raise ValueError("k must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.max_sets = max_sets
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.allow_token_token = allow_token_token

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.E_k = nn.Linear(max_sets, k, bias=False)
        self.E_v = nn.Linear(max_sets, k, bias=False)

    def _pad_to_max(self, x: torch.Tensor, size: int) -> torch.Tensor:
        if x.shape[-1] == size:
            return x
        if x.shape[-1] > size:
            return x[..., :size]
        pad = size - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad))

    def _project_seq(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        # x: [B, H, D, M] -> [B, H, D, K]
        x = self._pad_to_max(x, self.max_sets)
        return proj(x)

    def _project_bias(self, bias: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if bias is None:
            return None
        # bias: [B, H, M, M] -> [B, H, M, K]
        bias = self._pad_to_max(bias, self.max_sets)
        return self.E_k(bias)

    def forward(
        self,
        z: torch.Tensor,
        geom_bias: Optional[torch.Tensor],
        content_bias: Optional[torch.Tensor],
        sig_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> torch.Tensor:
        bsz, m, _ = z.shape
        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)

        q = q.view(bsz, m, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, m, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, m, self.num_heads, self.d_head).transpose(1, 2)

        k_t = k.transpose(-2, -1)
        v_t = v.transpose(-2, -1)
        k_proj = self._project_seq(k_t, self.E_k).transpose(-2, -1)
        v_proj = self._project_seq(v_t, self.E_v).transpose(-2, -1)

        scores = torch.matmul(q, k_proj.transpose(-2, -1)) / math.sqrt(self.d_head)

        geom_proj = self._project_bias(geom_bias)
        content_proj = self._project_bias(content_bias)
        scores = apply_score_biases(scores, geom_proj, content_proj, None)

        if sig_mask is not None:
            # No exact projection for sig_mask; warn and ignore.
            pass

        if not self.allow_token_token:
            assert_set_only_scores(scores, seq_len=seq_len)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_proj)
        out = out.transpose(1, 2).contiguous().view(bsz, m, self.d_model)
        return self.out_proj(out)
