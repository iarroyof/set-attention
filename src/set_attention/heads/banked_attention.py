from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn

from set_attention.kernels.sketches import symdiff_from_jaccard


class SetBankAttention(nn.Module):
    """Multihead attention over banks of sets per sequence (blockwise).

    Inputs are concatenated over sequences; `q_ptrs` and `k_ptrs` delimit blocks
    per sequence. Computes per-sequence W and returns concatenated set outputs.
    """

    def __init__(self, d_model: int, num_heads: int = 4, tau: float = 1.0, gamma: float = 0.3, beta: float = 1.0,
                 score_mode: str = "delta_plus_dot", eta: float = 1.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.eta = float(eta)
        self.score_mode = score_mode  # one of: delta_rbf, delta_plus_dot, intersect_norm, intersect_plus_dot, dot

        self.proj_A = nn.Linear(d_model, d_model, bias=False)
        self.proj_B = nn.Linear(d_model, d_model, bias=False)
        self.val_proj = nn.Linear(d_model, self.head_dim, bias=True)

    def forward(
        self,
        phi_q: torch.Tensor,  # (B, Sq, D)
        sig_q: torch.Tensor,  # (B, Sq, K)
        size_q: torch.Tensor,  # (B, Sq)
        mask_q: Optional[torch.Tensor],  # (B, Sq) bool, True=valid
        phi_k: torch.Tensor,  # (B, Sk, D)
        sig_k: torch.Tensor,  # (B, Sk, K)
        size_k: torch.Tensor,  # (B, Sk)
        mask_k: Optional[torch.Tensor],  # (B, Sk) bool, True=valid
    ) -> torch.Tensor:
        """Batched set×set attention.

        Returns Z_sets: (B, Sq, H, Dh)
        """
        B, Sq, D = phi_q.shape
        Sk = phi_k.shape[1]
        # Δ-RBF via MinHash Jaccard
        eq = (sig_q.unsqueeze(2) == sig_k.unsqueeze(1)).float()  # (B, Sq, Sk, K)
        jacc = eq.mean(dim=-1)  # (B, Sq, Sk)
        # Expected |S∩K| from Jaccard; then Δ = |S|+|K|-2|S∩K|
        SA = size_q.unsqueeze(2).float()  # (B, Sq, 1)
        SB = size_k.unsqueeze(1).float()  # (B, 1, Sk)
        inter = (jacc / (1.0 + jacc + 1e-8)) * (SA + SB)
        delta = SA + SB - 2.0 * inter  # (B, Sq, Sk)
        s_delta = torch.exp(-self.gamma * delta)

        # Content term via projections
        AQ = self.proj_A(phi_q)  # (B, Sq, D)
        BK = self.proj_B(phi_k)  # (B, Sk, D)
        dot = torch.matmul(AQ, BK.transpose(1, 2))  # (B, Sq, Sk)

        # Normalized intersection score
        inter_norm = inter / (torch.sqrt(SA * SB) + 1e-8)

        if self.score_mode == "delta_rbf":
            scores = s_delta
        elif self.score_mode == "delta_plus_dot":
            scores = s_delta + self.beta * dot
        elif self.score_mode == "intersect_norm":
            scores = self.eta * inter_norm
        elif self.score_mode == "intersect_plus_dot":
            scores = self.eta * inter_norm + self.beta * dot
        elif self.score_mode == "dot":
            scores = self.beta * dot
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode}")
        if mask_k is not None:
            mask = ~mask_k.unsqueeze(1).expand_as(scores)  # invalid -> True
            scores = scores.masked_fill(mask, float("-inf"))
        W = torch.softmax(scores / max(1e-6, self.tau), dim=-1)  # (B, Sq, Sk)

        V = self.val_proj(phi_k)  # (B, Sk, Dh)
        Z = torch.matmul(W, V)  # (B, Sq, Dh)
        Z = Z.unsqueeze(2).expand(B, Sq, self.num_heads, self.head_dim)
        return Z
