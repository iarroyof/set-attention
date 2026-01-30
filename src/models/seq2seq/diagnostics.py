from __future__ import annotations

from typing import Dict, Optional

import torch


class BaselineSeq2SeqDiagnostics:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._prev_top_indices: dict[str, torch.Tensor] = {}
        self._prev_params: dict[str, dict[str, torch.Tensor]] = {}
        self._prev_epoch_stats: dict[str, float] | None = None

    def _add(self, key: str, value: float) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + float(value)
        self._counts[key] = self._counts.get(key, 0) + 1

    def _update_attn(self, attention_weights: Optional[torch.Tensor], prefix: str) -> None:
        if attention_weights is None or attention_weights.numel() == 0:
            return
        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=1)
        else:
            attn = attention_weights

        B, N, N_kv = attn.shape
        entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)
        self._add(f"{prefix}_entropy_mean", float(entropy.mean().item()))
        max_entropy = torch.log(torch.tensor(float(N_kv), device=attn.device))
        entropy_norm = entropy.mean() / max_entropy
        self._add(f"{prefix}_entropy_norm", float(entropy_norm.item()))

        top1 = attn.max(dim=-1).values
        self._add(f"{prefix}_top1_mean", float(top1.mean().item()))
        self._add(f"{prefix}_top1_std", float(top1.std().item()))

        top_indices = attn.topk(k=min(5, N_kv), dim=-1).indices
        prev = self._prev_top_indices.get(prefix)
        if prev is not None:
            curr = set(top_indices.flatten().tolist())
            prev_set = set(prev.flatten().tolist())
            inter = len(curr & prev_set)
            union = len(curr | prev_set)
            jaccard = inter / union if union else 0.0
            self._add(f"{prefix}_pattern_jaccard", float(jaccard))
        self._prev_top_indices[prefix] = top_indices.detach()

    def update(
        self,
        encoder_attn: Optional[torch.Tensor],
        decoder_self_attn: Optional[torch.Tensor],
        decoder_cross_attn: Optional[torch.Tensor],
    ) -> None:
        self._update_attn(encoder_attn, "baseline/encoder_attention")
        self._update_attn(decoder_self_attn, "baseline/decoder_self_attention")
        self._update_attn(decoder_cross_attn, "baseline/decoder_cross_attention")

    def update_params(self, params: Dict[str, torch.Tensor]) -> None:
        buckets: dict[str, dict[str, torch.Tensor]] = {
            "baseline/encoder_attention": {},
            "baseline/decoder_self_attention": {},
            "baseline/decoder_cross_attention": {},
        }
        for name, param in params.items():
            if name.startswith("encoder."):
                buckets["baseline/encoder_attention"][name] = param
            elif name.startswith("decoder_self."):
                buckets["baseline/decoder_self_attention"][name] = param
            elif name.startswith("decoder_cross."):
                buckets["baseline/decoder_cross_attention"][name] = param

        for prefix, params_dict in buckets.items():
            if not params_dict:
                continue
            grad_norm = 0.0
            param_norm = 0.0
            for _, param in params_dict.items():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
                param_norm += param.norm().item() ** 2
            self._add(f"{prefix}_gradient_norm", float(grad_norm ** 0.5))
            self._add(f"{prefix}_param_norm", float(param_norm ** 0.5))

            prev = self._prev_params.get(prefix)
            if prev is not None:
                change = 0.0
                for name, param in params_dict.items():
                    prev_param = prev.get(name)
                    if prev_param is not None:
                        change += (param.detach() - prev_param).norm().item() ** 2
                self._add(f"{prefix}_weight_change", float(change ** 0.5))
            self._prev_params[prefix] = {k: v.detach().clone() for k, v in params_dict.items()}

    def get_epoch_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for key, total in self._sums.items():
            count = self._counts.get(key, 0)
            stats[key] = total / count if count else float("nan")

        if self._prev_epoch_stats is not None:
            for base_key, delta_key in (
                ("baseline/encoder_attention_entropy_mean", "baseline/encoder_delta_attention_entropy"),
                ("baseline/encoder_attention_top1_mean", "baseline/encoder_delta_attention_confidence"),
                ("baseline/decoder_self_attention_entropy_mean", "baseline/decoder_self_delta_attention_entropy"),
                ("baseline/decoder_self_attention_top1_mean", "baseline/decoder_self_delta_attention_confidence"),
                ("baseline/decoder_cross_attention_entropy_mean", "baseline/decoder_cross_delta_attention_entropy"),
                ("baseline/decoder_cross_attention_top1_mean", "baseline/decoder_cross_delta_attention_confidence"),
            ):
                prev = self._prev_epoch_stats.get(base_key)
                cur = stats.get(base_key)
                if prev is not None and cur is not None:
                    stats[delta_key] = cur - prev
        else:
            stats["baseline/encoder_delta_attention_entropy"] = float("nan")
            stats["baseline/encoder_delta_attention_confidence"] = float("nan")
            stats["baseline/decoder_self_delta_attention_entropy"] = float("nan")
            stats["baseline/decoder_self_delta_attention_confidence"] = float("nan")
            stats["baseline/decoder_cross_delta_attention_entropy"] = float("nan")
            stats["baseline/decoder_cross_delta_attention_confidence"] = float("nan")

        self._prev_epoch_stats = stats.copy()
        return stats
