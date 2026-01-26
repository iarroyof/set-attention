from __future__ import annotations

import math
import torch
from torch import nn

from set_attention.backends.base import SetAttentionBackend
from set_attention.core import apply_score_biases, assert_set_only_scores


class LandmarkAttentionBackend(SetAttentionBackend):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_landmarks: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.num_landmarks = num_landmarks
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _select_landmarks(self, m: int, device: torch.device) -> torch.Tensor:
        if self.num_landmarks >= m:
            return torch.arange(m, device=device)
        stride = max(m // self.num_landmarks, 1)
        idx = torch.arange(0, m, stride, device=device)[: self.num_landmarks]
        return idx

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

        landmark_idx = self._select_landmarks(m, z.device)
        q_l = q[:, :, landmark_idx, :]
        k_l = k[:, :, landmark_idx, :]

        scores_mL = torch.matmul(q, k_l.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores_Lm = torch.matmul(q_l, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        assert_set_only_scores(scores_mL, seq_len=seq_len)
        assert_set_only_scores(scores_Lm, seq_len=seq_len)

        if geom_bias is not None:
            geom_mL = geom_bias[:, landmark_idx] if geom_bias.dim() == 2 else geom_bias[
                :, :, landmark_idx
            ]
            geom_Lm = geom_bias[landmark_idx] if geom_bias.dim() == 2 else geom_bias[
                :, landmark_idx
            ]
        else:
            geom_mL = None
            geom_Lm = None

        if content_bias is not None:
            content_mL = (
                content_bias[:, landmark_idx]
                if content_bias.dim() == 2
                else content_bias[:, :, landmark_idx]
            )
            content_Lm = (
                content_bias[landmark_idx]
                if content_bias.dim() == 2
                else content_bias[:, landmark_idx]
            )
        else:
            content_mL = None
            content_Lm = None

        scores_mL = apply_score_biases(
            scores_mL, geom_bias=geom_mL, content_bias=content_mL, sig_mask=None
        )
        scores_Lm = apply_score_biases(
            scores_Lm, geom_bias=geom_Lm, content_bias=content_Lm, sig_mask=None
        )

        attn_mL = torch.softmax(scores_mL, dim=-1)
        attn_Lm = torch.softmax(scores_Lm, dim=-1)
        attn_mL = self.dropout(attn_mL)
        attn_Lm = self.dropout(attn_Lm)

        v_l = torch.matmul(attn_Lm, v)
        out = torch.matmul(attn_mL, v_l)

        out = out.transpose(1, 2).contiguous().view(batch, m, self.d_model)
        return self.out_proj(out)
