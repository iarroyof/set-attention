from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


class TokenSetRouter(nn.Module):
    """Route token states to a bank of set outputs per sequence via learned gates.

    Inputs:
      - token_states: (B, L, D)
      - Z_sets: (Nq, H, Dh) concatenated set outputs across batch
      - desc_q: (Nq, Dd) set descriptors (e.g., Phi_q @ W_d)
      - q_ptrs: (B+1,) delimiters for Z_sets/desc_q per sequence

    The router computes gates g_{b,l} over the sets of sequence b, then mixes
    Z_sets to token outputs and merges heads.
    """

    def __init__(self, d_model: int, num_heads: int, topk: int = 0):
        super().__init__()
        assert d_model % num_heads == 0
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
        Z_sets: torch.Tensor,         # (B,Sq,H,Dh)
        desc_q: torch.Tensor,         # (B,Sq,D)
        mask_q: torch.Tensor,         # (B,Sq) bool
    ) -> torch.Tensor:
        B, L, D = token_states.shape
        H = self.num_heads
        Dh = self.head_dim
        Tproj = self.Wg(token_states)         # (B,L,D)
        Dproj = self.Wd(desc_q)               # (B,Sq,D)
        logits = torch.matmul(Tproj, Dproj.transpose(1, 2))  # (B,L,Sq)
        # mask invalid sets
        mask = ~mask_q.unsqueeze(1).expand_as(logits)
        logits = logits.masked_fill(mask, float("-inf"))
        if self.topk > 0 and logits.size(2) > self.topk:
            topk_vals, topk_idx = torch.topk(logits, k=self.topk, dim=-1)
            tmp = torch.full_like(logits, float("-inf"))
            tmp.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            G = torch.softmax(tmp, dim=-1)
        else:
            G = torch.softmax(logits, dim=-1)
        mix = torch.einsum('bls,bshd->blhd', G, Z_sets)  # (B,L,H,Dh)
        return self.out(mix.reshape(B, L, D))
