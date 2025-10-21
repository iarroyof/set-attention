from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from .similarity import build_similarity


class KernelMultiheadAttention(nn.Module):
    """
    A drop-in replacement for nn.MultiheadAttention supporting kernel-based similarities.

    Args:
        embed_dim: input/output embedding dimension
        num_heads: number of attention heads
        batch_first: if True, expects inputs (B, L, E); else (L, B, E)
        sim: one of {"dot", "cosine", "rbf"}
        temperature: softmax temperature (dot, cosine)
        rbf_gamma: gamma for RBF similarity
        dropout: attention dropout
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        batch_first: bool = False,
        sim: str = "dot",
        temperature: float = 1.0,
        rbf_gamma: float = 0.5,
        dropout: float = 0.0,
        bias: bool = True,
        inter_topk: int = 16,
        inter_normalize: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.sim = sim
        self.temperature = float(temperature)
        self.rbf_gamma = float(rbf_gamma)
        self.inter_topk = int(inter_topk)
        self.inter_normalize = bool(inter_normalize)
        # similarity implementation (strategy)
        self._sim_impl = build_similarity(
            sim,
            head_dim=self.head_dim,
            temperature=self.temperature,
            rbf_gamma=self.rbf_gamma,
            inter_topk=self.inter_topk,
            inter_normalize=self.inter_normalize,
        )

        self.in_proj_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

        # scaling for dot-product heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Torch MHA compatibility shims (introspection in TransformerEncoderLayer)
        # MultiheadAttention exposes combined projection params; we don't use them,
        # but define placeholders so upstream checks (e.g., `in_proj_bias is None`) succeed.
        self.in_proj_weight = None  # type: ignore[attr-defined]
        self.in_proj_bias = None    # type: ignore[attr-defined]
        self.bias_k = None          # type: ignore[attr-defined]
        self.bias_v = None          # type: ignore[attr-defined]
        self.add_zero_attn = False  # type: ignore[attr-defined]

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,E) or (L,B,E) -> always convert to (B, heads, L, head_dim)
        if not self.batch_first:
            x = x.transpose(0, 1)  # (L,B,E) -> (B,L,E)
        B, L, E = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return x  # (B, H, L, D)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,L,D) -> (B,L,E)
        B, H, L, D = x.shape
        x = x.transpose(1, 2).contiguous().view(B, L, H * D)
        if not self.batch_first:
            x = x.transpose(0, 1)  # (B,L,E) -> (L,B,E)
        return x

    def _apply_mask(
        self,
        scores: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # scores: (B,H,Lq,Lk)
        if attn_mask is not None:
            # support (Lq, Lk) or (B, Lq, Lk)
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.view(1, 1, *attn_mask.shape)
            elif attn_mask.dim() == 3:
                scores = scores + attn_mask.view(scores.shape[0], 1, *attn_mask.shape[1:])
            else:
                raise ValueError("attn_mask must be 2D or 3D")
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = mask
            mask = key_padding_mask.view(scores.shape[0], 1, 1, scores.shape[-1])
            scores = scores.masked_fill(mask, float("-inf"))
        return scores

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
        # Project and shape into heads
        q = self._shape_heads(self.in_proj_q(query))  # (B,H,Lq,D)
        k = self._shape_heads(self.in_proj_k(key))    # (B,H,Lk,D)
        v = self._shape_heads(self.in_proj_v(value))  # (B,H,Lk,D)

        # Compute similarity per head
        scores = self._sim_impl.score(q, k)

        # Support PyTorch-style attn_mask: bool mask or additive mask
        # Also support causal masking if requested
        if is_causal:
            # build upper-triangular mask (Lq,Lk) with -inf where j>i
            Lq, Lk = q.size(-2), k.size(-2)
            causal_bool = torch.triu(torch.ones(Lq, Lk, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_bool.view(1, 1, Lq, Lk), float("-inf"))
        # apply provided masks
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # True means mask out
                if attn_mask.dim() == 2:
                    scores = scores.masked_fill(attn_mask.view(1, 1, *attn_mask.shape), float("-inf"))
                elif attn_mask.dim() == 3:
                    scores = scores.masked_fill(attn_mask.view(scores.shape[0], 1, *attn_mask.shape[1:]), float("-inf"))
                else:
                    raise ValueError("attn_mask must be 2D or 3D")
            else:
                # additive
                scores = self._apply_mask(scores, attn_mask, None)
        if key_padding_mask is not None:
            scores = self._apply_mask(scores, None, key_padding_mask)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,H,Lq,D)
        out = self._merge_heads(out)
        out = self.out_proj(out)

        # Aggregate attention weights across heads if requested
        attn_weights = None
        if need_weights:
            if average_attn_weights:
                aw = attn.mean(dim=1)  # (B,Lq,Lk)
            else:
                aw = attn  # (B,H,Lq,Lk)
            if not self.batch_first:
                if aw.dim() == 3:
                    aw = aw.transpose(0, 1)  # (Lq,B,Lk)
                else:
                    aw = aw.transpose(0, 2)  # (Lq,H,B,Lk) unusual but matches dims
            attn_weights = aw
        return out, attn_weights
