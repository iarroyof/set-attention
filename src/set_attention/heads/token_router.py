from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

from set_attention.sets.bank_utils import pad_segments_from_ptrs


class TokenSetRouter(nn.Module):
    """Route token states to concatenated set outputs via learned gates."""

    def __init__(self, d_model: int, num_heads: int, topk: int = 0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.Wg = nn.Linear(d_model, d_model)
        self.Wd = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.topk = int(topk)

    def forward(
        self,
        token_states: torch.Tensor,   # (B,L,D)
        Z_sets: torch.Tensor,         # (Nq,H,Dh)
        desc_q: torch.Tensor,         # (Nq,D)
        q_ptrs: torch.Tensor,         # (B+1,)
    ) -> torch.Tensor:
        B, L, D = token_states.shape
        desc_pad, mask = pad_segments_from_ptrs(desc_q, q_ptrs, fill_value=0.0)
        z_pad, _ = pad_segments_from_ptrs(Z_sets, q_ptrs, fill_value=0.0)
        Tproj = self.Wg(token_states)         # (B,L,D)
        Dproj = self.Wd(desc_pad)             # (B,S,D)
        logits = torch.matmul(Tproj, Dproj.transpose(1, 2))  # (B,L,S)
        mask_scores = ~mask.unsqueeze(1).expand_as(logits)
        logits = logits.masked_fill(mask_scores, float("-inf"))
        if self.topk > 0 and logits.size(2) > self.topk:
            topk_vals, topk_idx = torch.topk(logits, k=self.topk, dim=-1)
            tmp = torch.full_like(logits, float("-inf"))
            tmp.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            gates = torch.softmax(tmp, dim=-1)
        else:
            gates = torch.softmax(logits, dim=-1)
        mix = torch.einsum("bls,bshd->blhd", gates, z_pad)
        return self.out(mix.reshape(B, L, D))
