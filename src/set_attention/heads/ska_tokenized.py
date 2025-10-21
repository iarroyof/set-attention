from __future__ import annotations
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from set_attention.kernels.sketches import MinHasher, symdiff_from_jaccard


class SetKernelMultiheadAttentionTokenized(nn.Module):
    """
    Tokenizer-aware SKA: attention scores from Δ-RBF between per-token atom sets.

    Inputs:
      - token_sets_q: concatenated atom ids for all query tokens (LongTensor)
      - token_offs_q: (nQ+1,) CSR offsets over token_sets_q
      - token_sets_k, token_offs_k: same for keys

    If token sets are not provided, falls back to standard softmax over dot(QK^T).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        batch_first: bool = False,
        gamma: float = 0.3,
        minhash_k: int = 128,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.gamma = float(gamma)
        self.mh = MinHasher(k=minhash_k)

        self.in_proj_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

        # Torch MHA compatibility shims
        self.in_proj_weight = None  # type: ignore[attr-defined]
        self.in_proj_bias = None  # type: ignore[attr-defined]
        self.bias_k = None  # type: ignore[attr-defined]
        self.bias_v = None  # type: ignore[attr-defined]
        self.add_zero_attn = False  # type: ignore[attr-defined]

        # Optional learnable gating over tokenizer atoms (disabled by default)
        self.vocab_size: Optional[int] = None
        self.atom_dim: int = 64
        self.gate_topk: int = 0
        self.atom_emb: Optional[nn.Embedding] = None
        self.ctx_proj: Optional[nn.Linear] = None

    def enable_gating(self, vocab_size: int, atom_dim: int = 64, gate_topk: int = 8):
        """Enable learnable gating over tokenizer-provided atom sets.

        This does not discover new atoms. It selects a learned subset of the
        provided set per token using a dot-product between per-token context
        and learnable atom embeddings, then keeps top-k per token.
        """
        self.vocab_size = int(vocab_size)
        self.atom_dim = int(atom_dim)
        self.gate_topk = int(gate_topk)
        self.atom_emb = nn.Embedding(self.vocab_size + 1, self.atom_dim)
        self.ctx_proj = nn.Linear(self.head_dim, self.atom_dim)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1)
        b, l, e = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,D)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, l, d = x.shape
        y = x.transpose(1, 2).contiguous().view(b, l, h * d)
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

    def _scores_from_token_sets(
        self,
        q_ids: torch.LongTensor,
        q_offs: torch.LongTensor,
        k_ids: torch.LongTensor,
        k_offs: torch.LongTensor,
        ctx_q: Optional[torch.Tensor] = None,
        ctx_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute Jaccard via MinHash, then Δ and RBF; shared across heads
        sig_q = self.mh.sketch(q_ids, q_offs)  # (nQ,k)
        sig_k = self.mh.sketch(k_ids, k_offs)  # (nK,k)
        jacc = MinHasher.jaccard_from_signatures(sig_q, sig_k)  # (nQ,nK)
        size_q = (q_offs[1:] - q_offs[:-1]).to(torch.int64)
        size_k = (k_offs[1:] - k_offs[:-1]).to(torch.int64)
        delta = symdiff_from_jaccard(jacc, size_q, size_k)  # (nQ,nK)
        s = torch.exp(-self.gamma * delta)
        return s

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
        token_sets_q: Optional[torch.LongTensor] = None,
        token_offs_q: Optional[torch.LongTensor] = None,
        token_sets_k: Optional[torch.LongTensor] = None,
        token_offs_k: Optional[torch.LongTensor] = None,
        token_sigs_q: Optional[torch.LongTensor] = None,
        token_sigs_k: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self._shape_heads(self.in_proj_q(query))  # (B,H,Lq,D)
        k = self._shape_heads(self.in_proj_k(key))    # (B,H,Lk,D)
        v = self._shape_heads(self.in_proj_v(value))  # (B,H,Lk,D)

        if token_sets_q is not None and token_offs_q is not None and token_sets_k is not None and token_offs_k is not None:
            # Optional learned gating over tokenizer atoms (shared across heads)
            q_ids, q_offs = token_sets_q, token_offs_q
            k_ids, k_offs = token_sets_k, token_offs_k
            if self.gate_topk and self.atom_emb is not None and self.ctx_proj is not None:
                # Build per-token contexts from mean over heads
                ctx_q = q.mean(dim=1)  # (B,L,D)
                ctx_k = k.mean(dim=1)  # (B,L,D)
                if not self.batch_first:
                    ctx_q = ctx_q.transpose(0, 1)
                    ctx_k = ctx_k.transpose(0, 1)
                # Flatten tokens across batch
                CQ = self.ctx_proj(ctx_q).reshape(-1, self.atom_dim)  # (nQ,atom_dim)
                CK = self.ctx_proj(ctx_k).reshape(-1, self.atom_dim)  # (nK,atom_dim)
                # Re-gate ids per token
                def gate(ids: torch.LongTensor, offs: torch.LongTensor, C: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
                    out_vals: List[torch.Tensor] = []
                    out_offs = [0]
                    for i in range(offs.numel() - 1):
                        sl = ids[offs[i] : offs[i + 1]]
                        if sl.numel() == 0:
                            out_offs.append(out_offs[-1])
                            continue
                        emb = self.atom_emb(sl)
                        score = (emb * C[i].unsqueeze(0)).sum(dim=-1)
                        ktop = min(self.gate_topk, sl.numel())
                        keep = torch.topk(score, k=ktop, dim=0).indices
                        out_vals.append(sl[keep])
                        out_offs.append(out_offs[-1] + ktop)
                    if out_vals:
                        vals = torch.cat(out_vals, dim=0)
                    else:
                        vals = torch.empty(0, dtype=torch.long, device=ids.device)
                    return vals, torch.tensor(out_offs, dtype=torch.long, device=ids.device)

                q_ids, q_offs = gate(q_ids, q_offs, CQ)
                k_ids, k_offs = gate(k_ids, k_offs, CK)

            # Determine batch size and lengths
            if self.batch_first:
                bsz = query.size(0)
                Lq = query.size(1)
                Lk = key.size(1)
            else:
                bsz = query.size(1)
                Lq = query.size(0)
                Lk = key.size(0)

            # Compute scores per batch item. Support two layouts for offsets:
            # (A) per-token replication: len(offsets) == bsz*L + 1
            # (B) per-sequence sets:    len(offsets) == bsz + 1 (one set per sequence)
            scores_b = []
            for b in range(bsz):
                if q_offs.numel() == bsz * Lq + 1 and k_offs.numel() == bsz * Lk + 1:
                    # per-token replicated sets
                    q_start = b * Lq
                    q_end = (b + 1) * Lq
                    k_start = b * Lk
                    k_end = (b + 1) * Lk
                    q_off_slice = q_offs[q_start : q_end + 1]
                    k_off_slice = k_offs[k_start : k_end + 1]
                    q_base = q_off_slice[0]
                    k_base = k_off_slice[0]
                    q_vals = q_ids[q_base : q_off_slice[-1]]
                    k_vals = k_ids[k_base : k_off_slice[-1]]
                    q_off_re = (q_off_slice - q_base).contiguous()
                    k_off_re = (k_off_slice - k_base).contiguous()
                    if token_sigs_q is not None and token_sigs_k is not None:
                        # slice sigs per tokens
                        sig_q = token_sigs_q[q_start:q_end]
                        sig_k = token_sigs_k[k_start:k_end]
                        # compute jaccard via equality across signature columns
                        jacc = (sig_q[:, None, :] == sig_k[None, :, :]).float().mean(dim=-1)
                        delta = symdiff_from_jaccard(jacc, (q_off_re[1:] - q_off_re[:-1]), (k_off_re[1:] - k_off_re[:-1]))
                        s_b = torch.exp(-self.gamma * delta)
                    else:
                        s_b = self._scores_from_token_sets(q_vals, q_off_re, k_vals, k_off_re)
                else:
                    # per-sequence sets (one set per example)
                    q_off_slice = q_offs[b : b + 2]
                    k_off_slice = k_offs[b : b + 2]
                    q_base = q_off_slice[0]
                    k_base = k_off_slice[0]
                    q_vals = q_ids[q_base : q_off_slice[-1]]
                    k_vals = k_ids[k_base : k_off_slice[-1]]
                    q_off_re = (q_off_slice - q_base)
                    k_off_re = (k_off_slice - k_base)
                    if token_sigs_q is not None and token_sigs_k is not None:
                        sig_q = token_sigs_q[b : b + 1]
                        sig_k = token_sigs_k[b : b + 1]
                        jacc = (sig_q[:, None, :] == sig_k[None, :, :]).float().mean(dim=-1)  # (1,1)
                        delta = symdiff_from_jaccard(jacc, torch.tensor([q_off_re[1]-q_off_re[0]], device=jacc.device), torch.tensor([k_off_re[1]-k_off_re[0]], device=jacc.device))
                        s_val = torch.exp(-self.gamma * delta)[0, 0]
                        s_b = s_val.expand(Lq, Lk)
                    else:
                        s_scalar = self._scores_from_token_sets(q_vals, q_off_re, k_vals, k_off_re)  # (1,1)
                        s_b = s_scalar.expand(Lq, Lk)
                scores_b.append(s_b)
            scores = torch.stack(scores_b, dim=0).unsqueeze(1).expand(bsz, self.num_heads, Lq, Lk)
        else:
            # Fallback: dot-product scores
            scores = (q @ k.transpose(-2, -1)) / max(1e-6, float(self.head_dim) ** 0.5)

        scores = self._apply_masks(scores, attn_mask, key_padding_mask, is_causal)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
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
