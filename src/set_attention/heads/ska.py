from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(x, k=k, dim=-1, largest=True).indices


def _binary_mask(idxs: torch.Tensor, depth: int, dtype: torch.dtype) -> torch.Tensor:
    # idxs: (..., k)
    shape = (*idxs.shape[:-1], depth)
    m = torch.zeros(shape, dtype=dtype, device=idxs.device)
    return m.scatter(-1, idxs, 1.0)


def _delta_rbf_sets(q_idx: torch.Tensor, k_idx: torch.Tensor, depth: int, gamma: float) -> torch.Tensor:
    # q_idx: (B,H,Lq,tk), k_idx: (B,H,Lk,tk)
    tk = q_idx.size(-1)
    m_q = _binary_mask(q_idx, depth, dtype=torch.float32)  # (B,H,Lq,D)
    m_k = _binary_mask(k_idx, depth, dtype=torch.float32)  # (B,H,Lk,D)
    inter = torch.matmul(m_q, m_k.transpose(-2, -1))  # (B,H,Lq,Lk)
    delta = (2 * tk) - 2.0 * inter
    return torch.exp(-gamma * delta.clamp_min(0.0))


class SetKernelMultiheadAttention(nn.Module):
    """
    True set-kernel attention using Δ-RBF on sets of top-k feature indices per head.
    Scores are computed via Nyström features built from per-batch landmarks.

    Args:
        embed_dim: embedding dimension
        num_heads: number of heads
        batch_first: input layout flag
        ska_topk: number of top features to define set atoms per token and head
        ska_gamma: Δ-RBF temperature
        ska_rank: Nyström rank (number of landmarks per head)
        ska_ridge: small ridge for Kzz eigendecomposition
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        batch_first: bool = False,
        ska_topk: int = 16,
        ska_gamma: float = 0.3,
        ska_rank: int = 32,
        ska_ridge: float = 1e-4,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.ska_topk = int(ska_topk)
        self.ska_gamma = float(ska_gamma)
        self.ska_rank = int(ska_rank)
        self.ska_ridge = float(ska_ridge)

        self.in_proj_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

        # Torch MHA compatibility shims for Transformer internals
        self.in_proj_weight = None  # type: ignore[attr-defined]
        self.in_proj_bias = None  # type: ignore[attr-defined]
        self.bias_k = None  # type: ignore[attr-defined]
        self.bias_v = None  # type: ignore[attr-defined]
        self.add_zero_attn = False  # type: ignore[attr-defined]

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1)
        bsz, seq, emb = x.shape
        return x.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,D)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, h, l, d = x.shape
        y = x.transpose(1, 2).contiguous().view(bsz, l, h * d)
        if not self.batch_first:
            y = y.transpose(0, 1)
        return y

    def _apply_masks(
        self,
        scores: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        if is_causal:
            lq, lk = scores.size(-2), scores.size(-1)
            causal = torch.triu(torch.ones(lq, lk, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal.view(1, 1, lq, lk), float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                if attn_mask.dim() == 2:
                    scores = scores.masked_fill(attn_mask.view(1, 1, *attn_mask.shape), float("-inf"))
                elif attn_mask.dim() == 3:
                    scores = scores.masked_fill(attn_mask.view(scores.shape[0], 1, *attn_mask.shape[1:]), float("-inf"))
            else:
                if attn_mask.dim() == 2:
                    scores = scores + attn_mask.view(1, 1, *attn_mask.shape)
                elif attn_mask.dim() == 3:
                    scores = scores + attn_mask.view(scores.shape[0], 1, *attn_mask.shape[1:])
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(scores.shape[0], 1, 1, scores.shape[-1]), float("-inf"))
        return scores

    def _nystrom_features(self, k_q: torch.Tensor, k_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # k_q: (B,H,Lq,tk) indices sets, k_k: (B,H,Lk,tk)
        bsz, heads, lq, tk = k_q.shape
        lk = k_k.size(-3)
        rank = max(1, min(self.ska_rank, lk))
        # choose evenly spaced landmark indices per head
        idx = torch.linspace(0, lk - 1, steps=rank, device=k_k.device).round().long()  # (rank,)
        # gather landmarks
        k_land = k_k[:, :, idx, :]  # (B,H,rank,tk)
        # compute kernels
        k_xz = _delta_rbf_sets(k_q, k_land, depth=self.head_dim, gamma=self.ska_gamma)  # (B,H,Lq,rank)
        k_yz = _delta_rbf_sets(k_k, k_land, depth=self.head_dim, gamma=self.ska_gamma)  # (B,H,Lk,rank)
        k_zz = _delta_rbf_sets(k_land, k_land, depth=self.head_dim, gamma=self.ska_gamma)  # (B,H,rank,rank)
        # eigendecomposition per (B,H)
        # add ridge for stability
        ridge = self.ska_ridge
        phi_q = torch.empty_like(k_xz)
        phi_k = torch.empty_like(k_yz)
        for b in range(bsz):
            for h in range(heads):
                kzz = k_zz[b, h] + ridge * torch.eye(rank, device=k_zz.device)
                evals, evecs = torch.linalg.eigh(kzz)
                # keep all rank dims; safe due to small rank
                evals = evals.clamp_min(ridge)
                e = evecs / torch.sqrt(evals)[None, :]
                phi_q[b, h] = k_xz[b, h] @ e
                phi_k[b, h] = k_yz[b, h] @ e
        return phi_q, phi_k

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self._shape_heads(self.in_proj_q(query))  # (B,H,Lq,D)
        k = self._shape_heads(self.in_proj_k(key))    # (B,H,Lk,D)
        v = self._shape_heads(self.in_proj_v(value))  # (B,H,Lk,D)

        # build sets via top-k indices per token
        tk = max(1, min(self.ska_topk, self.head_dim))
        q_idx = _topk_indices(q, tk)
        k_idx = _topk_indices(k, tk)

        # Nyström features and similarity
        phi_q, phi_k = self._nystrom_features(q_idx, k_idx)
        scores = torch.matmul(phi_q, phi_k.transpose(-2, -1))  # (B,H,Lq,Lk)
        scores = self._apply_masks(scores, attn_mask, key_padding_mask, is_causal)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,H,Lq,D)
        out = self._merge_heads(out)
        out = self.out_proj(out)

        attn_weights = None
        if need_weights:
            aw = attn.mean(dim=1) if average_attn_weights else attn
            if not self.batch_first:
                if aw.dim() == 3:
                    aw = aw.transpose(0, 1)
                else:
                    aw = aw.transpose(0, 2)
            attn_weights = aw
        return out, attn_weights

