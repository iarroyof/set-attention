from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from set_attention.backends.dense_exact import DenseExactBackend
from set_attention.backends.landmark import LandmarkAttentionBackend
from set_attention.backends.local_band import LocalBandBackend
from set_attention.backends.nystrom import NystromBackend
from set_attention.backends.sparse_topk import SparseTopKBackend


class BaselineAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        attention_family: str,
        backend: str,
        backend_params: Optional[dict] = None,
        max_seq_len: int | None = None,
        causal: bool = False,
        is_cross: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.is_cross = is_cross
        self.attention_family = attention_family
        self.backend = backend
        self.backend_params = backend_params or {}
        self.max_seq_len = max_seq_len

        self._use_manual = backend in {"exact", "local_band", "sparse_topk", "linformer"}
        if self._use_manual:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            if backend == "linformer":
                if self.max_seq_len is None:
                    raise ValueError("linformer baseline requires max_seq_len")
                k = int(self.backend_params.get("k", 32))
                self.E_k = nn.Linear(self.max_seq_len, k, bias=False)
                self.E_v = nn.Linear(self.max_seq_len, k, bias=False)
        else:
            if is_cross:
                # Cross-attn with linear backends is not implemented; fall back to exact.
                import warnings

                warnings.warn(
                    "linear baseline cross-attn is not implemented; falling back to exact attention.",
                    RuntimeWarning,
                )
                self._use_manual = True
                self.backend = "exact"
                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                return
            if backend == "landmark":
                self.backend_impl = LandmarkAttentionBackend(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_landmarks=self.backend_params.get("num_landmarks", 32),
                    dropout=dropout,
                    allow_token_token=True,
                )
            elif backend == "nystrom":
                self.backend_impl = NystromBackend(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_landmarks=self.backend_params.get("num_landmarks", 32),
                    dropout=dropout,
                    allow_token_token=True,
                    bias_scale=self.backend_params.get("bias_scale", 0.1),
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")

    def _apply_masks(
        self,
        scores: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))
        if self.causal:
            nq = scores.shape[-2]
            nk = scores.shape[-1]
            causal = torch.triu(torch.ones(nq, nk, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(causal, float("-inf"))
        return scores

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._use_manual:
            kv = x if memory is None else memory
            q = self.q_proj(x)
            k = self.k_proj(kv)
            v = self.v_proj(kv)

            b, nq, _ = q.shape
            nk = k.shape[1]
            q = q.view(b, nq, self.num_heads, self.d_head).transpose(1, 2)
            k = k.view(b, nk, self.num_heads, self.d_head).transpose(1, 2)
            v = v.view(b, nk, self.num_heads, self.d_head).transpose(1, 2)

            if self.backend == "linformer":
                k_t = k.transpose(-2, -1)
                v_t = v.transpose(-2, -1)
                if nk > self.max_seq_len:
                    k_t = k_t[..., : self.max_seq_len]
                    v_t = v_t[..., : self.max_seq_len]
                elif nk < self.max_seq_len:
                    pad = self.max_seq_len - nk
                    k_t = torch.nn.functional.pad(k_t, (0, pad))
                    v_t = torch.nn.functional.pad(v_t, (0, pad))
                k_proj = self.E_k(k_t).transpose(-2, -1)
                v_proj = self.E_v(v_t).transpose(-2, -1)
                scores = torch.matmul(q, k_proj.transpose(-2, -1)) / math.sqrt(self.d_head)
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

            if self.backend == "local_band":
                radius = int(self.backend_params.get("radius", 4))
                idx_q = torch.arange(nq, device=x.device)
                idx_k = torch.arange(nk, device=x.device)
                band = (idx_q[:, None] - idx_k[None, :]).abs() <= radius
                global_indices = self.backend_params.get("global_indices", [])
                if global_indices:
                    global_mask = torch.zeros_like(band)
                    for g_idx in global_indices:
                        if g_idx < 0:
                            g_idx = nk + g_idx
                        if 0 <= g_idx < nk:
                            global_mask[g_idx, :] = True
                            global_mask[:, g_idx] = True
                    band = band | global_mask
                scores = scores.masked_fill(~band, float("-inf"))
            scores = self._apply_masks(scores, key_padding_mask)

            if self.backend == "sparse_topk":
                k_s = int(self.backend_params.get("k_s", 16))
                if k_s and k_s < nk:
                    topk_scores, topk_idx = torch.topk(scores, k_s, dim=-1)
                    masked = torch.full_like(scores, float("-inf"))
                    masked.scatter_(-1, topk_idx, topk_scores)
                    scores = masked

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            if self.backend == "linformer":
                out = torch.matmul(attn, v_proj)
            else:
                out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(b, nq, self.d_model)
            return self.out_proj(out), attn.mean(dim=1)

        # Linear backends (self-attention only).
        z = x
        b, m, _ = z.shape
        sig_mask = None
        if key_padding_mask is not None:
            allowed = ~key_padding_mask
            sig_mask = allowed.unsqueeze(1) & allowed.unsqueeze(2)
        if self.causal:
            causal = torch.tril(torch.ones(m, m, device=z.device)).bool()
            sig_mask = causal if sig_mask is None else (sig_mask & causal)
        out = self.backend_impl(z, geom_bias=None, content_bias=None, sig_mask=sig_mask, seq_len=m)
        return out, None
