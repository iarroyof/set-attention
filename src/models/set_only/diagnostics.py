from __future__ import annotations

from typing import Dict, Optional
import warnings

import torch


class SetDiagnostics:
    def __init__(
        self,
        spectral_top_eig_warn: float = 0.7,
        spectral_entropy_warn: float = 0.3,
    ) -> None:
        self.spectral_top_eig_warn = spectral_top_eig_warn
        self.spectral_entropy_warn = spectral_entropy_warn
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._prev_active = None
        self._prev_bank_indices = None
        self._prev_router_params = None
        self._prev_epoch_stats: dict[str, float] | None = None

    def _add(self, key: str, value: float) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + float(value)
        self._counts[key] = self._counts.get(key, 0) + 1

    def update_with_router_state(
        self,
        bank_indices: torch.Tensor,
        num_sets: int,
        router_probs: Optional[torch.Tensor] = None,
        set_embeddings: Optional[torch.Tensor] = None,
        set_attention_weights: Optional[torch.Tensor] = None,
    ) -> None:
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
            self._add("ausa/set_reuse_jaccard", float(jaccard.item()))
        self._prev_active = active_mask

        self._add("ausa/active_set_ratio", float(ratio.item()))
        self._add("ausa/active_set_size_mean", float(active_mean.item()))
        self._add("ausa/active_set_size_std", float(active_std.item()))
        self._add("ausa/routing_entropy", float(entropy.item()))
        self._add("ausa/routing_entropy_norm", float(entropy_norm.item()))
        self._add("ausa/routing_gini", float(gini.item()))
        self._add("ausa/routing_top1_prob_mean", float(top1.item()))
        self._add("ausa/tokens_per_set_variance", float(counts.var(unbiased=False).item()))

        if self._prev_bank_indices is not None and bank_indices.shape == self._prev_bank_indices.shape:
            consistency = (bank_indices == self._prev_bank_indices).float().mean()
            self._add("ausa/routing_consistency", float(consistency.item()))
        self._prev_bank_indices = bank_indices.detach()

        if router_probs is not None:
            confidence = router_probs.max(dim=-1).values
            self._add("ausa/router_confidence_mean", float(confidence.mean().item()))
            self._add("ausa/router_confidence_std", float(confidence.std().item()))
            probs = router_probs.clamp_min(1e-12)
            router_entropy = -(probs * torch.log(probs)).sum(dim=-1)
            self._add("ausa/router_entropy", float(router_entropy.mean().item()))
            self._add("ausa/router_top1_weight", float(confidence.mean().item()))
            if probs.dim() >= 3 and probs.shape[-1] > 1:
                util = probs.mean(dim=tuple(range(probs.dim() - 1)))
                util = util / util.sum().clamp_min(1e-12)
                util_sorted, _ = torch.sort(util)
                n_util = util_sorted.shape[0]
                idx_util = torch.arange(
                    1, n_util + 1, device=util.device, dtype=util.dtype
                )
                util_gini = 1.0 - 2.0 * torch.sum(
                    util_sorted * (n_util - idx_util + 0.5)
                ) / n_util
                self._add("ausa/router_set_utilization_gini", float(util_gini.item()))
            if num_sets > 1:
                kl = (router_probs * torch.log(router_probs * float(num_sets) + 1e-8)).sum(dim=-1)
                self._add("ausa/top1_vs_random_kl", float(kl.mean().item()))

        if set_embeddings is not None:
            B, S, D = set_embeddings.shape
            variance = set_embeddings.var(dim=(0, 1), unbiased=False).mean().item()
            self._add("ausa/set_embedding_variance", float(variance))
            norms = set_embeddings.norm(dim=-1)
            self._add("ausa/set_embedding_norm_mean", float(norms.mean().item()))

            # Spectrum diagnostic for set attention conditioning:
            # K = ZZ^T / M, where Z in R^{M x d}. Fast decay indicates an overconstrained set space.
            if S > 1:
                z = set_embeddings.float()
                gram = torch.matmul(z, z.transpose(-2, -1)) / float(S)
                try:
                    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
                    eig_sum = eigvals.sum(dim=-1).clamp_min(1e-8)
                    top_ratio = eigvals[:, -1] / eig_sum
                    p = eigvals / eig_sum.unsqueeze(-1)
                    spectral_entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
                    spectral_entropy_norm = spectral_entropy / torch.log(
                        torch.tensor(float(S), device=spectral_entropy.device)
                    )
                    self._add("ausa/set_gram_top_eig_ratio", float(top_ratio.mean().item()))
                    self._add(
                        "ausa/set_gram_spectral_entropy_norm",
                        float(spectral_entropy_norm.mean().item()),
                    )
                except RuntimeError:
                    # Keep training robust if an eigendecomposition fails on a rare batch.
                    pass

            flat = set_embeddings.reshape(-1, D)
            if flat.shape[0] > 1:
                n_samples = min(100, flat.shape[0])
                idx = torch.randperm(flat.shape[0], device=flat.device)[:n_samples]
                samples = flat[idx]
                sims = torch.mm(samples, samples.t())
                denom = (samples.norm(dim=1, keepdim=True) @ samples.norm(dim=1, keepdim=True).t()).clamp_min(1e-8)
                cos_sim = sims / denom
                mask = ~torch.eye(n_samples, dtype=torch.bool, device=cos_sim.device)
                self._add("ausa/set_cosine_similarity_mean", float(cos_sim[mask].mean().item()))

                centered = samples - samples.mean(dim=0, keepdim=True)
                cov = (centered.transpose(0, 1) @ centered) / centered.shape[0]
                trace = torch.trace(cov)
                fro = torch.norm(cov, p="fro")
                eff_rank = (trace ** 2) / (fro ** 2 + 1e-8)
                self._add("ausa/set_rank_effective", float(eff_rank.item()))

        if set_attention_weights is not None:
            attn_entropy = -(set_attention_weights * torch.log(set_attention_weights + 1e-10)).sum(dim=-1)
            self._add("ausa/set_attention_entropy_mean", float(attn_entropy.mean().item()))
            top1_prob = set_attention_weights.max(dim=-1).values
            self._add("ausa/set_attention_top1_mean", float(top1_prob.mean().item()))

    def update_router_params(self, router_params: Dict[str, torch.Tensor]) -> None:
        grad_norm = 0.0
        param_norm = 0.0
        for _, param in router_params.items():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
            param_norm += param.norm().item() ** 2
        self._add("ausa/router_gradient_norm", float(grad_norm ** 0.5))
        self._add("ausa/router_param_norm", float(param_norm ** 0.5))

        if self._prev_router_params is not None:
            change = 0.0
            for name, param in router_params.items():
                prev = self._prev_router_params.get(name)
                if prev is not None:
                    change += (param.detach() - prev).norm().item() ** 2
            self._add("ausa/router_weight_change", float(change ** 0.5))
        self._prev_router_params = {k: v.detach().clone() for k, v in router_params.items()}

    def update(self, bank_indices: torch.Tensor, num_sets: int) -> None:
        self.update_with_router_state(bank_indices, num_sets)

    def update_with_pooling_stats(self, stats: Dict[str, float]) -> None:
        for key, value in stats.items():
            self._add(key, float(value))

    def get_epoch_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for key, total in self._sums.items():
            count = self._counts.get(key, 0)
            stats[key] = total / count if count else float("nan")

        # Epoch-to-epoch deltas
        if self._prev_epoch_stats is not None:
            for base_key, delta_key in (
                ("ausa/routing_entropy", "ausa/delta_routing_entropy"),
                ("ausa/set_embedding_variance", "ausa/delta_set_variance"),
                ("ausa/router_confidence_mean", "ausa/delta_router_confidence"),
            ):
                prev = self._prev_epoch_stats.get(base_key)
                cur = stats.get(base_key)
                if prev is not None and cur is not None:
                    stats[delta_key] = cur - prev
        else:
            stats["ausa/delta_routing_entropy"] = float("nan")
            stats["ausa/delta_set_variance"] = float("nan")
            stats["ausa/delta_router_confidence"] = float("nan")

        top_eig_ratio = stats.get("ausa/set_gram_top_eig_ratio")
        spectral_entropy = stats.get("ausa/set_gram_spectral_entropy_norm")
        if (
            top_eig_ratio is not None
            and top_eig_ratio == top_eig_ratio
            and top_eig_ratio > self.spectral_top_eig_warn
        ) or (
            spectral_entropy is not None
            and spectral_entropy == spectral_entropy
            and spectral_entropy < self.spectral_entropy_warn
        ):
            warnings.warn(
                "Set Gram spectrum indicates possible overconstraint "
                f"(top_eig_ratio={top_eig_ratio:.4f}, "
                f"spectral_entropy_norm={spectral_entropy:.4f}).",
                RuntimeWarning,
            )

        self._prev_epoch_stats = stats.copy()
        return stats
