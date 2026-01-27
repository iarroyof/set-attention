from __future__ import annotations

from typing import Dict

import torch


class SetDiagnostics:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._jaccard_count = 0
        self._prev_active = None
        self._sums = {
            "ausa/active_set_ratio": 0.0,
            "ausa/active_set_size_mean": 0.0,
            "ausa/active_set_size_std": 0.0,
            "ausa/routing_entropy": 0.0,
            "ausa/routing_entropy_norm": 0.0,
            "ausa/routing_gini": 0.0,
            "ausa/routing_top1_prob_mean": 0.0,
            "ausa/set_reuse_jaccard": 0.0,
        }

    def update(self, bank_indices: torch.Tensor, num_sets: int) -> None:
        if bank_indices.numel() == 0 or num_sets <= 0:
            return
        flat = bank_indices.reshape(-1).clamp(min=0, max=num_sets - 1)
        counts = torch.bincount(flat, minlength=num_sets).float()
        total = counts.sum().clamp_min(1.0)
        p = counts / total
        active_mask = counts > 0
        active_sets = active_mask.sum().float()

        active_sizes = counts[active_mask]
        if active_sizes.numel() == 0:
            active_mean = torch.tensor(0.0, device=counts.device)
            active_std = torch.tensor(0.0, device=counts.device)
        else:
            active_mean = active_sizes.mean()
            active_std = active_sizes.std(unbiased=False)

        p_nonzero = p[p > 0]
        entropy = -(p_nonzero * torch.log(p_nonzero)).sum() if p_nonzero.numel() else p.sum() * 0.0
        entropy_norm = entropy / torch.log(torch.tensor(float(num_sets), device=counts.device)) if num_sets > 1 else entropy * 0.0

        p_sorted, _ = torch.sort(p)
        n = num_sets
        idx = torch.arange(1, n + 1, device=counts.device, dtype=p_sorted.dtype)
        gini = 1.0 - 2.0 * torch.sum(p_sorted * (n - idx + 0.5)) / n

        top1 = p.max()
        ratio = active_sets / float(num_sets)

        if self._prev_active is not None:
            inter = (active_mask & self._prev_active).sum().float()
            union = (active_mask | self._prev_active).sum().float().clamp_min(1.0)
            jaccard = inter / union
            self._sums["ausa/set_reuse_jaccard"] += float(jaccard.item())
            self._jaccard_count += 1
        self._prev_active = active_mask

        self._sums["ausa/active_set_ratio"] += float(ratio.item())
        self._sums["ausa/active_set_size_mean"] += float(active_mean.item())
        self._sums["ausa/active_set_size_std"] += float(active_std.item())
        self._sums["ausa/routing_entropy"] += float(entropy.item())
        self._sums["ausa/routing_entropy_norm"] += float(entropy_norm.item())
        self._sums["ausa/routing_gini"] += float(gini.item())
        self._sums["ausa/routing_top1_prob_mean"] += float(top1.item())
        self._count += 1

    def get_epoch_stats(self) -> Dict[str, float]:
        stats = {}
        for key, total in self._sums.items():
            if key == "ausa/set_reuse_jaccard":
                if self._jaccard_count == 0:
                    stats[key] = float("nan")
                else:
                    stats[key] = total / self._jaccard_count
            else:
                stats[key] = total / self._count if self._count else float("nan")
        return stats
