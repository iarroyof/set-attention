from __future__ import annotations

from typing import Dict, Optional

import torch


class BaselineAttentionDiagnostics:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._prev_top_indices: Optional[torch.Tensor] = None
        self._prev_params: Optional[dict[str, torch.Tensor]] = None
        self._prev_epoch_stats: dict[str, float] | None = None

    def _add(self, key: str, value: float) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + float(value)
        self._counts[key] = self._counts.get(key, 0) + 1

    def update(
        self,
        attention_weights: torch.Tensor,
    ) -> None:
        if attention_weights.numel() == 0:
            return
        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=1)
        else:
            attn = attention_weights

        B, N, N_kv = attn.shape
        entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)
        self._add("baseline/attention_entropy_mean", float(entropy.mean().item()))
        max_entropy = torch.log(torch.tensor(float(N_kv), device=attn.device))
        entropy_norm = entropy.mean() / max_entropy
        self._add("baseline/attention_entropy_norm", float(entropy_norm.item()))

        top1 = attn.max(dim=-1).values
        self._add("baseline/attention_top1_mean", float(top1.mean().item()))
        self._add("baseline/attention_top1_std", float(top1.std().item()))

        top_indices = attn.topk(k=min(5, N_kv), dim=-1).indices
        if self._prev_top_indices is not None:
            curr = set(top_indices.flatten().tolist())
            prev = set(self._prev_top_indices.flatten().tolist())
            inter = len(curr & prev)
            union = len(curr | prev)
            jaccard = inter / union if union else 0.0
            self._add("baseline/attention_pattern_jaccard", float(jaccard))
        self._prev_top_indices = top_indices.detach()

    def update_params(self, params: Dict[str, torch.Tensor]) -> None:
        grad_norm = 0.0
        param_norm = 0.0
        for _, param in params.items():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
            param_norm += param.norm().item() ** 2
        self._add("baseline/attention_gradient_norm", float(grad_norm ** 0.5))
        self._add("baseline/attention_param_norm", float(param_norm ** 0.5))

        if self._prev_params is not None:
            change = 0.0
            for name, param in params.items():
                prev = self._prev_params.get(name)
                if prev is not None:
                    change += (param.detach() - prev).norm().item() ** 2
            self._add("baseline/attention_weight_change", float(change ** 0.5))
        self._prev_params = {k: v.detach().clone() for k, v in params.items()}

    def get_epoch_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for key, total in self._sums.items():
            count = self._counts.get(key, 0)
            stats[key] = total / count if count else float("nan")

        if self._prev_epoch_stats is not None:
            for base_key, delta_key in (
                ("baseline/attention_entropy_mean", "baseline/delta_attention_entropy"),
                ("baseline/attention_top1_mean", "baseline/delta_attention_confidence"),
            ):
                prev = self._prev_epoch_stats.get(base_key)
                cur = stats.get(base_key)
                if prev is not None and cur is not None:
                    stats[delta_key] = cur - prev
        else:
            stats["baseline/delta_attention_entropy"] = float("nan")
            stats["baseline/delta_attention_confidence"] = float("nan")

        self._prev_epoch_stats = stats.copy()
        return stats
