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
    def __init__(self, d_model: int, topk: int = 0, restrict_to_sets: bool = True) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)
        self.temperature = nn.Parameter(torch.ones(1))
        self.min_temp = 0.5
        self.topk = topk
        self.restrict_to_sets = restrict_to_sets

    def forward(
        self,
        token_states: torch.Tensor,
        set_states: torch.Tensor,
        desc_router: torch.Tensor,
        token_to_sets: torch.Tensor,
    ) -> RouterOutput:
        if token_states.dim() != 3:
            raise ValueError("token_states must be [batch, seq, d]")
        batch, seq_len, _ = token_states.shape
        _, num_sets, _ = set_states.shape

        q = self.query(token_states)
        scores = torch.matmul(q, desc_router.transpose(-2, -1)) * self.scale

        if self.restrict_to_sets:
            mask = torch.zeros((seq_len, num_sets), dtype=torch.bool, device=scores.device)
            rows = torch.arange(seq_len, device=scores.device).unsqueeze(1).expand_as(token_to_sets)
            valid = token_to_sets >= 0
            mask[rows[valid], token_to_sets[valid].clamp_max(num_sets - 1)] = True
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

        if self.topk and self.topk < num_sets:
            topk_scores, topk_idx = torch.topk(scores, self.topk, dim=-1)
            keep = torch.full_like(scores, float("-inf"))
            keep.scatter_(-1, topk_idx, topk_scores)
            scores = keep

        if (
            scores.isnan().any()
            or (scores == float("inf")).any()
            or (scores == float("-inf")).sum() == scores.numel()
        ):
            print(
                f"[DEBUG] Bad scores! shape={scores.shape}, has_nan={scores.isnan().any()}, "
                f"all_-inf={(scores == float('-inf')).sum() == scores.numel()}"
            )
            print(
                f"  desc_router stats: min={desc_router.min()}, "
                f"max={desc_router.max()}, mean={desc_router.mean()}"
            )
            if hasattr(self, "temperature"):
                print(f"  temperature: {self.temperature.item()}")

        temp = self.temperature.clamp(min=self.min_temp)
        weights = torch.softmax(scores / temp, dim=-1)
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
