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
        landmark_coverage: float | None = None,
        coverage: float | None = None,
        num_landmarks: int | None = None,
        dropout: float = 0.0,
        allow_token_token: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if landmark_coverage is not None and coverage is not None:
            raise ValueError("Use only one of landmark_coverage or coverage")
        resolved_coverage = landmark_coverage if landmark_coverage is not None else coverage
        if resolved_coverage is None:
            resolved_coverage = 0.25
        resolved_coverage = float(resolved_coverage)
        if resolved_coverage <= 0.0:
            raise ValueError("landmark_coverage must be > 0")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.landmark_coverage = resolved_coverage
        self.num_landmarks = num_landmarks
        self.last_landmark_count: int | None = None
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.allow_token_token = allow_token_token

    def _landmark_count(self, m: int) -> int:
        if m <= 0:
            return 0
        if self.num_landmarks is not None:
            floor = 2 if m > 1 else 1
            return min(max(int(self.num_landmarks), floor), m)
        return min(max(round(self.landmark_coverage * m), 2), m)

    def _select_landmarks(self, m: int, device: torch.device) -> torch.Tensor:
        k = self._landmark_count(m)
        self.last_landmark_count = k
        if k >= m:
            return torch.arange(m, device=device)
        return torch.tensor(
            [round(i * (m - 1) / (k - 1)) for i in range(k)],
            device=device,
            dtype=torch.long,
        )

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
        assert_set_only_scores(
            scores_mL, seq_len=seq_len, allow_token_token=self.allow_token_token
        )
        assert_set_only_scores(
            scores_Lm, seq_len=seq_len, allow_token_token=self.allow_token_token
        )

        if geom_bias is not None:
            if geom_bias.dim() == 2:
                geom_mL = geom_bias[:, landmark_idx]
                geom_Lm = geom_bias[landmark_idx]
            elif geom_bias.dim() == 3:
                geom_mL = geom_bias[:, :, landmark_idx]
                geom_Lm = geom_bias[:, landmark_idx]
            else:
                geom_mL = geom_bias[:, :, :, landmark_idx]
                geom_Lm = geom_bias[:, :, landmark_idx, :]
        else:
            geom_mL = None
            geom_Lm = None

        if content_bias is not None:
            if content_bias.dim() == 2:
                content_mL = content_bias[:, landmark_idx]
                content_Lm = content_bias[landmark_idx]
            elif content_bias.dim() == 3:
                content_mL = content_bias[:, :, landmark_idx]
                content_Lm = content_bias[:, landmark_idx]
            else:
                content_mL = content_bias[:, :, :, landmark_idx]
                content_Lm = content_bias[:, :, landmark_idx, :]
        else:
            content_mL = None
            content_Lm = None

        if sig_mask is not None:
            sig_mL = sig_mask[:, landmark_idx] if sig_mask.dim() == 2 else sig_mask[:, :, landmark_idx]
            sig_Lm = sig_mask[landmark_idx] if sig_mask.dim() == 2 else sig_mask[:, landmark_idx]
        else:
            sig_mL = None
            sig_Lm = None

        scores_mL = apply_score_biases(
            scores_mL, geom_bias=geom_mL, content_bias=content_mL, sig_mask=sig_mL
        )
        scores_Lm = apply_score_biases(
            scores_Lm, geom_bias=geom_Lm, content_bias=content_Lm, sig_mask=sig_Lm
        )

        attn_mL = torch.softmax(scores_mL, dim=-1)
        attn_Lm = torch.softmax(scores_Lm, dim=-1)
        attn_mL = self.dropout(attn_mL)
        attn_Lm = self.dropout(attn_Lm)

        v_l = torch.matmul(attn_Lm, v)
        out = torch.matmul(attn_mL, v_l)

        out = out.transpose(1, 2).contiguous().view(batch, m, self.d_model)
        return self.out_proj(out)
