from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RouterOutput:
    token_repr: torch.Tensor
    bank_indices: torch.Tensor
    num_sets: int
    probs: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None


class UniformRouter(nn.Module):
    def forward(self, set_states: torch.Tensor, token_to_sets: torch.Tensor) -> RouterOutput:
        if set_states.dim() != 3:
            raise ValueError("set_states must be [batch, m, d]")
        batch, _, d_model = set_states.shape
        seq_len, _ = token_to_sets.shape

        num_sets = set_states.shape[1]
        indices = token_to_sets.clamp_min(0).clamp_max(num_sets - 1)
        gathered = set_states[:, indices]  # [batch, seq, max_sets, d]
        mask = (token_to_sets >= 0).unsqueeze(0).unsqueeze(-1)
        summed = (gathered * mask).sum(dim=2)
        counts = mask.sum(dim=2).clamp_min(1)
        token_repr = summed / counts
        bank_indices = token_to_sets[:, 0].clamp_min(0).unsqueeze(0).expand(batch, -1)
        # Keep max_sets dimension even when max_sets_per_token == 1.
        weights = (token_to_sets >= 0).unsqueeze(0).squeeze(0).float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs = torch.zeros((seq_len, num_sets), device=weights.device)
        valid = token_to_sets >= 0
        if valid.any():
            probs.scatter_(1, token_to_sets.clamp_min(0).clamp_max(num_sets - 1), weights)
        return RouterOutput(
            token_repr=token_repr,
            bank_indices=bank_indices,
            num_sets=num_sets,
            probs=probs.unsqueeze(0).expand(batch, -1, -1),
        )


class LearnedRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        d_phi: int | None = None,
        topk: int = 0,
        restrict_to_sets: bool = True,
        multihead: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.multihead = bool(multihead)
        if self.multihead:
            if self.num_heads <= 0:
                raise ValueError("num_heads must be > 0 for multihead routing")
            if self.d_model % self.num_heads != 0:
                raise ValueError(
                    f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}"
                )
            self.d_head = self.d_model // self.num_heads
            self.d_phi = int(d_phi) if d_phi is not None else self.d_head
            if self.d_phi <= 0:
                raise ValueError("d_phi must be > 0 for multihead routing")
            self.query = nn.Linear(self.d_model, self.num_heads * self.d_phi)
            self.desc_key = nn.Linear(self.d_model, self.num_heads * self.d_phi)
            self.scale = 1.0 / math.sqrt(self.d_phi)
        else:
            self.query = nn.Linear(self.d_model, self.d_model)
            self.desc_key = None
            self.scale = 1.0 / math.sqrt(self.d_model)
        self.register_buffer("temperature", torch.ones(1))
        self.min_temp = 0.5
        self.topk = int(topk)
        self.restrict_to_sets = bool(restrict_to_sets)

    def forward(
        self,
        token_states: torch.Tensor,
        set_states: torch.Tensor,
        desc_router: torch.Tensor,
        token_to_sets: torch.Tensor,
    ) -> RouterOutput:
        if token_states.dim() != 3:
            raise ValueError("token_states must be [batch, seq, d]")
        if set_states.dim() != 3:
            raise ValueError("set_states must be [batch, m, d]")
        if desc_router.dim() != 3:
            raise ValueError("desc_router must be [batch, m, d_desc]")
        if token_to_sets.dim() != 2:
            raise ValueError("token_to_sets must be [seq, candidates]")

        batch, seq_len, d_model = token_states.shape
        batch_s, num_sets, d_set = set_states.shape
        batch_d, num_sets_desc, d_desc = desc_router.shape
        if batch_s != batch or batch_d != batch:
            raise ValueError("token_states, set_states, and desc_router must share batch size")
        if num_sets_desc != num_sets:
            raise ValueError("desc_router and set_states must share num_sets")
        if d_model != self.d_model or d_set != self.d_model:
            raise ValueError(
                f"token/set states must have d_model={self.d_model} (got {d_model} and {d_set})"
            )
        if token_to_sets.shape[0] != seq_len:
            raise ValueError("token_to_sets first dimension must equal sequence length")

        if self.multihead:
            if self.desc_key is None:
                raise RuntimeError("multihead router missing descriptor projection")
            if d_desc != self.d_model:
                raise ValueError(
                    f"multihead router expects desc_router dim {self.d_model}, got {d_desc}"
                )
            q = self.query(token_states).view(batch, seq_len, self.num_heads, self.d_phi)
            q = q.transpose(1, 2)  # [B,H,T,d_phi]
            k = self.desc_key(desc_router).view(batch, num_sets, self.num_heads, self.d_phi)
            k = k.transpose(1, 2)  # [B,H,M,d_phi]
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,M]

            if self.restrict_to_sets:
                mask = torch.zeros((seq_len, num_sets), dtype=torch.bool, device=scores.device)
                rows = torch.arange(seq_len, device=scores.device).unsqueeze(1).expand_as(token_to_sets)
                valid = token_to_sets >= 0
                if valid.any():
                    mask[rows[valid], token_to_sets[valid].clamp_max(num_sets - 1)] = True
                scores = scores.masked_fill(~mask.view(1, 1, seq_len, num_sets), float("-inf"))

            topk_indices = None
            if self.topk and self.topk < num_sets:
                topk_scores, topk_idx = torch.topk(scores, self.topk, dim=-1)
                keep = torch.full_like(scores, float("-inf"))
                keep.scatter_(-1, topk_idx, topk_scores)
                scores = keep
                topk_indices = topk_idx

            temp = self.temperature.clamp(min=self.min_temp)
            finite_rows = torch.isfinite(scores).any(dim=-1, keepdim=True)
            safe_scores = torch.where(finite_rows, scores, torch.zeros_like(scores))
            weights = torch.softmax(safe_scores / temp, dim=-1)
            weights = torch.where(finite_rows, weights, torch.zeros_like(weights))

            s = set_states.view(batch, num_sets, self.num_heads, self.d_head).transpose(1, 2)
            token_h = torch.matmul(weights, s)  # [B,H,T,d_head]
            token_repr = token_h.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
            bank_indices = weights.mean(dim=1).argmax(dim=-1)  # [B,T], legacy-compatible
            return RouterOutput(
                token_repr=token_repr,
                bank_indices=bank_indices,
                num_sets=num_sets,
                probs=weights,
                topk_indices=topk_indices,
            )

        q = self.query(token_states)
        scores = torch.matmul(q, desc_router.transpose(-2, -1)) * self.scale

        if self.restrict_to_sets:
            mask = torch.zeros((seq_len, num_sets), dtype=torch.bool, device=scores.device)
            rows = torch.arange(seq_len, device=scores.device).unsqueeze(1).expand_as(token_to_sets)
            valid = token_to_sets >= 0
            if valid.any():
                mask[rows[valid], token_to_sets[valid].clamp_max(num_sets - 1)] = True
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

        if self.topk and self.topk < num_sets:
            topk_scores, topk_idx = torch.topk(scores, self.topk, dim=-1)
            keep = torch.full_like(scores, float("-inf"))
            keep.scatter_(-1, topk_idx, topk_scores)
            scores = keep

        temp = self.temperature.clamp(min=self.min_temp)
        finite_rows = torch.isfinite(scores).any(dim=-1, keepdim=True)
        safe_scores = torch.where(finite_rows, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores / temp, dim=-1)
        weights = torch.where(finite_rows, weights, torch.zeros_like(weights))
        token_repr = torch.matmul(weights, set_states)
        bank_indices = weights.argmax(dim=-1)
        topk_indices = None

        if self.topk and self.topk < num_sets:
            topk_indices = torch.topk(scores, self.topk, dim=-1).indices

        return RouterOutput(
            token_repr=token_repr,
            bank_indices=bank_indices,
            num_sets=num_sets,
            probs=weights,
            topk_indices=topk_indices,
        )
