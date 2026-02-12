from __future__ import annotations

import math
import torch
from torch import nn

from set_attention.backends.base import SetAttentionBackend
from set_attention.core import apply_score_biases, assert_set_only_scores


class NystromBackend(SetAttentionBackend):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_landmarks: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
        normalize: bool = True,
        allow_token_token: bool = False,
        bias_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.num_landmarks = num_landmarks
        self.eps = eps
        self.normalize = normalize
        self.bias_scale = bias_scale
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.allow_token_token = allow_token_token

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

        # Ensure no token-token attention is constructed.

        landmark_idx = self._select_landmarks(m, z.device)
        q_l = q[:, :, landmark_idx, :]
        k_l = k[:, :, landmark_idx, :]

        scores_mL = torch.matmul(q, k_l.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores_LL = torch.matmul(q_l, k_l.transpose(-2, -1)) / math.sqrt(self.d_head)
        assert_set_only_scores(
            scores_mL, seq_len=seq_len, allow_token_token=self.allow_token_token
        )
        assert_set_only_scores(
            scores_LL, seq_len=seq_len, allow_token_token=self.allow_token_token
        )

        if geom_bias is not None:
            if geom_bias.dim() == 2:
                geom_mL = geom_bias[:, landmark_idx]
                geom_LL = geom_bias[landmark_idx][:, landmark_idx]
            elif geom_bias.dim() == 3:
                geom_mL = geom_bias[:, :, landmark_idx]
                geom_LL = geom_bias[:, landmark_idx][:, :, landmark_idx]
            else:
                geom_mL = geom_bias[:, :, :, landmark_idx]
                geom_LL = geom_bias[:, :, landmark_idx, :][:, :, :, landmark_idx]
        else:
            geom_mL = None
            geom_LL = None

        if content_bias is not None:
            if content_bias.dim() == 2:
                content_mL = content_bias[:, landmark_idx]
                content_LL = content_bias[landmark_idx][:, landmark_idx]
            elif content_bias.dim() == 3:
                content_mL = content_bias[:, :, landmark_idx]
                content_LL = content_bias[:, landmark_idx][:, :, landmark_idx]
            else:
                content_mL = content_bias[:, :, :, landmark_idx]
                content_LL = content_bias[:, :, landmark_idx, :][:, :, :, landmark_idx]
        else:
            content_mL = None
            content_LL = None

        if self.bias_scale != 1.0:
            if geom_mL is not None:
                geom_mL = geom_mL * self.bias_scale
            if geom_LL is not None:
                geom_LL = geom_LL * self.bias_scale
            if content_mL is not None:
                content_mL = content_mL * self.bias_scale
            if content_LL is not None:
                content_LL = content_LL * self.bias_scale

        if sig_mask is not None:
            sig_mL = sig_mask[:, landmark_idx] if sig_mask.dim() == 2 else sig_mask[:, :, landmark_idx]
            sig_LL = sig_mask[landmark_idx][:, landmark_idx] if sig_mask.dim() == 2 else sig_mask[:, landmark_idx][:, :, landmark_idx]
        else:
            sig_mL = None
            sig_LL = None

        scores_mL = apply_score_biases(
            scores_mL, geom_bias=geom_mL, content_bias=content_mL, sig_mask=sig_mL
        )
        scores_LL = apply_score_biases(
            scores_LL, geom_bias=geom_LL, content_bias=content_LL, sig_mask=sig_LL
        )

        k_mL = torch.exp(scores_mL)
        k_LL = torch.exp(scores_LL)

        # Proper regularization with correct broadcasting
        eye = torch.eye(k_LL.shape[-1], device=z.device).unsqueeze(0).unsqueeze(0)
        k_LL = k_LL + 0.1 * eye

        k_Lm = k_mL.transpose(-2, -1)
        kv = torch.matmul(k_Lm, v)
        k_LL_reg = k_LL
        x = torch.linalg.solve(k_LL_reg, kv)
        out = torch.matmul(k_mL, x)

        if self.normalize:
            ones = torch.ones((batch, self.num_heads, m, 1), device=z.device)
            k_Lm_ones = torch.matmul(k_Lm, ones)
            tilde = torch.linalg.solve(k_LL_reg, k_Lm_ones)
            denom = torch.matmul(k_mL, tilde).clamp_min(self.eps)
            out = out / denom

        # Apply dropout after solve/normalization to avoid destabilizing k_LL inversion.
        out = self.dropout(out)

        out = out.transpose(1, 2).contiguous().view(batch, m, self.d_model)
        return self.out_proj(out)
