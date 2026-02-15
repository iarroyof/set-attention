from __future__ import annotations

from typing import Dict, Optional
import warnings
import math

import torch


class SetDiagnostics:
    def __init__(
        self,
        spectral_top_eig_warn: float = 0.3,
        spectral_entropy_warn: float = 0.3,
        pooling_neff_ratio_warn: float = 0.3,
        router_top1_weight_warn: float = 0.8,
        spectrum_interval: int = 200,
        embedding_structure_interval: int = 200,
        max_spectrum_sets: int = 48,
        max_embedding_samples: int = 64,
        powerlaw_collapse_gate: float = 0.5,
        condition_number_cap: float = 1e6,
    ) -> None:
        self.spectral_top_eig_warn = spectral_top_eig_warn
        self.spectral_entropy_warn = spectral_entropy_warn
        self.pooling_neff_ratio_warn = pooling_neff_ratio_warn
        self.router_top1_weight_warn = router_top1_weight_warn
        self.spectrum_interval = max(1, int(spectrum_interval))
        self.embedding_structure_interval = max(1, int(embedding_structure_interval))
        self.max_spectrum_sets = max(8, int(max_spectrum_sets))
        self.max_embedding_samples = max(8, int(max_embedding_samples))
        self.powerlaw_collapse_gate = float(powerlaw_collapse_gate)
        self.condition_number_cap = float(condition_number_cap)
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._prev_active = None
        self._prev_bank_indices = None
        self._prev_router_params = None
        self._prev_epoch_stats: dict[str, float] | None = None
        self._step = 0

    def _add(self, key: str, value: float) -> None:
        v = float(value)
        if not math.isfinite(v):
            self._sums["ausa/diagnostics_nonfinite_count"] = (
                self._sums.get("ausa/diagnostics_nonfinite_count", 0.0) + 1.0
            )
            self._counts["ausa/diagnostics_nonfinite_count"] = (
                self._counts.get("ausa/diagnostics_nonfinite_count", 0) + 1
            )
            return
        self._sums[key] = self._sums.get(key, 0.0) + v
        self._counts[key] = self._counts.get(key, 0) + 1

    def update_with_router_state(
        self,
        bank_indices: torch.Tensor,
        num_sets: int,
        router_probs: Optional[torch.Tensor] = None,
        set_embeddings: Optional[torch.Tensor] = None,
        set_attention_weights: Optional[torch.Tensor] = None,
        token_to_sets: Optional[torch.Tensor] = None,
    ) -> None:
        self._step += 1
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
            router_probs = router_probs.detach().to(torch.float32)
            confidence = router_probs.max(dim=-1).values
            self._add("ausa/router_confidence_mean", float(confidence.mean().item()))
            self._add("ausa/router_confidence_std", float(confidence.std().item()))
            probs = router_probs.clamp_min(1e-12)
            router_entropy = -(probs * torch.log(probs)).sum(dim=-1)
            self._add("ausa/router_entropy", float(router_entropy.mean().item()))
            self._add("ausa/router_top1_weight", float(confidence.mean().item()))
            if token_to_sets is not None and token_to_sets.numel() > 0:
                token_to_sets = token_to_sets.to(router_probs.device)
                if token_to_sets.dim() == 2:
                    # [T, C] -> [B, T, C]
                    cand_idx = token_to_sets.unsqueeze(0).expand(router_probs.shape[0], -1, -1)
                elif token_to_sets.dim() == 3:
                    cand_idx = token_to_sets
                    if cand_idx.shape[0] == 1 and router_probs.shape[0] > 1:
                        cand_idx = cand_idx.expand(router_probs.shape[0], -1, -1)
                else:
                    cand_idx = None

                if cand_idx is not None and cand_idx.shape[1] == router_probs.shape[1]:
                    valid = cand_idx >= 0
                    cand_raw = valid.sum(dim=-1)
                    cand = cand_raw.clamp_min(1)
                    idx_safe = cand_idx.clamp(min=0, max=num_sets - 1)
                    probs_c = probs.gather(dim=-1, index=idx_safe)
                    probs_c = probs_c * valid.to(probs.dtype)
                    den = probs_c.sum(dim=-1, keepdim=True)
                    probs_c = torch.where(
                        den > 0,
                        probs_c / den.clamp_min(1e-12),
                        torch.zeros_like(probs_c),
                    )
                    router_entropy_c = -(probs_c * torch.log(probs_c.clamp_min(1e-12))).sum(dim=-1)
                    top1_c = probs_c.max(dim=-1).values
                else:
                    # Fallback when candidate topology does not align with current batch shape.
                    cand_raw = torch.full(
                        router_entropy.shape,
                        num_sets,
                        dtype=torch.long,
                        device=router_probs.device,
                    )
                    cand = cand_raw
                    router_entropy_c = router_entropy
                    top1_c = confidence

                cand_f_raw = cand_raw.to(router_probs.dtype)
                self._add("ausa/router_candidate_count_mean", float(cand_f_raw.mean().item()))
                self._add("ausa/router_candidate_count_min", float(cand_f_raw.min().item()))
                self._add("ausa/router_candidate_count_max", float(cand_f_raw.max().item()))
                cand_f = cand.to(router_probs.dtype)
                log_c = torch.log(cand_f.clamp_min(1.0)).clamp_min(1e-12)
                entropy_norm_c = torch.where(
                    cand_f > 1.0,
                    router_entropy_c / log_c,
                    torch.zeros_like(router_entropy_c),
                )
                self._add(
                    "ausa/router_entropy_norm_by_candidates",
                    float(entropy_norm_c.mean().item()),
                )
                u = 1.0 / cand_f.clamp_min(1.0)
                top1_gap_norm = torch.where(
                    cand_f > 1.0,
                    (top1_c - u) / (1.0 - u + 1e-12),
                    torch.zeros_like(top1_c),
                )
                self._add(
                    "ausa/router_top1_gap_norm",
                    float(top1_gap_norm.mean().item()),
                )
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
            set_embeddings = set_embeddings.detach().to(torch.float32)
            B, S, D = set_embeddings.shape
            variance = set_embeddings.var(dim=(0, 1), unbiased=False).mean().item()
            self._add("ausa/set_embedding_variance", float(variance))
            norms = set_embeddings.norm(dim=-1)
            self._add("ausa/set_embedding_norm_mean", float(norms.mean().item()))

            # Spectrum diagnostic for set attention conditioning:
            # K = ZZ^T / M, where Z in R^{M x d}. Fast decay indicates an overconstrained set space.
            do_spectrum = (self._step % self.spectrum_interval == 0)
            if S > 1 and do_spectrum:
                z = set_embeddings
                if S > self.max_spectrum_sets:
                    idx_sets = torch.randperm(S, device=z.device)[: self.max_spectrum_sets]
                    z = z[:, idx_sets]
                s_eff = z.shape[1]
                try:
                    if D < s_eff:
                        cov = torch.matmul(z.transpose(-2, -1), z) / float(s_eff)
                        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
                    else:
                        gram = torch.matmul(z, z.transpose(-2, -1)) / float(s_eff)
                        eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
                    eig_sum = eigvals.sum(dim=-1).clamp_min(1e-8)
                    top_ratio = eigvals[:, -1] / eig_sum
                    p = eigvals / eig_sum.unsqueeze(-1)
                    spectral_entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
                    n_eval = float(eigvals.shape[-1])
                    spectral_entropy_norm = spectral_entropy / torch.log(
                        torch.tensor(max(n_eval, 2.0), device=spectral_entropy.device)
                    )
                    self._add("ausa/set_gram_top_eig_ratio", float(top_ratio.mean().item()))
                    self._add(
                        "ausa/set_gram_spectral_entropy_norm",
                        float(spectral_entropy_norm.mean().item()),
                    )
                    cond_eps = 1e-12
                    cond_num = eigvals[:, -1] / eigvals[:, 0].clamp_min(cond_eps)
                    cond_num = cond_num.clamp_max(self.condition_number_cap)
                    self._add("ausa/set_gram_condition_number", float(cond_num.mean().item()))

                    logdet_eps = 1e-8
                    logdet = torch.log(eigvals + logdet_eps).sum(dim=-1)
                    self._add("ausa/set_gram_logdet", float(logdet.mean().item()))

                    # Power-law slope in the eigenspectrum tail:
                    # lambda_i ~ i^{-alpha} => log(lambda_i) ~ -alpha * log(i) + c.
                    n_eigs = eigvals.shape[-1]
                    i_min = max(3, int(math.ceil(0.1 * n_eigs)))
                    i_max = min(n_eigs, int(math.floor(0.6 * n_eigs)))
                    if i_max - i_min >= 1:
                        alpha_vals: list[torch.Tensor] = []
                        idx_1based = torch.arange(
                            1, n_eigs + 1, device=eigvals.device, dtype=eigvals.dtype
                        )
                        for b_idx in range(eigvals.shape[0]):
                            if float(top_ratio[b_idx].item()) > self.powerlaw_collapse_gate:
                                continue
                            vals_desc = torch.flip(eigvals[b_idx], dims=[0])
                            floor = vals_desc[0].clamp_min(1e-12) * 1e-12
                            fit_mask = (
                                (idx_1based >= i_min)
                                & (idx_1based <= i_max)
                                & (vals_desc >= floor)
                            )
                            if int(fit_mask.sum().item()) < 2:
                                continue
                            x = torch.log(idx_1based[fit_mask].to(vals_desc.dtype))
                            y = torch.log(vals_desc[fit_mask].clamp_min(1e-12))
                            x_center = x - x.mean()
                            denom = (x_center * x_center).sum()
                            if float(denom.item()) <= 0.0:
                                continue
                            slope = (x_center * (y - y.mean())).sum() / denom
                            alpha_vals.append((-slope).detach())
                        if alpha_vals:
                            alpha = torch.stack(alpha_vals).mean()
                            self._add("ausa/set_gram_powerlaw_alpha", float(alpha.item()))
                except RuntimeError:
                    # Keep training robust if an eigendecomposition fails on a rare batch.
                    pass

            flat = set_embeddings.reshape(-1, D)
            do_structure = (self._step % self.embedding_structure_interval == 0)
            if flat.shape[0] > 1 and do_structure:
                n_samples = min(self.max_embedding_samples, flat.shape[0])
                idx = torch.randperm(flat.shape[0], device=flat.device)[:n_samples]
                samples = flat[idx]
                sims = torch.mm(samples, samples.t())
                denom = (samples.norm(dim=1, keepdim=True) @ samples.norm(dim=1, keepdim=True).t()).clamp_min(1e-8)
                cos_sim = sims / denom
                mask = ~torch.eye(n_samples, dtype=torch.bool, device=cos_sim.device)
                self._add("ausa/set_cosine_similarity_mean", float(cos_sim[mask].mean().item()))

                centered = samples - samples.mean(dim=0, keepdim=True)
                svals = torch.linalg.svdvals(centered)
                s2 = svals.pow(2)
                eff_rank = (s2.sum() ** 2) / (s2.pow(2).sum() + 1e-8)
                self._add("ausa/set_rank_effective", float(eff_rank.item()))

        if set_attention_weights is not None:
            set_attention_weights = set_attention_weights.detach().to(torch.float32)
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

    def update_with_gradient_probe(
        self,
        grad_h: float,
        grad_z0: float,
        grad_zl: float,
    ) -> None:
        vals = torch.tensor([grad_h, grad_z0, grad_zl], dtype=torch.float32)
        nonfinite = (~torch.isfinite(vals)).sum().item()
        if nonfinite > 0:
            self._add("ausa/grad_probe_nonfinite_count", float(nonfinite))
        vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        grad_h, grad_z0, grad_zl = [float(v.item()) for v in vals]
        eps = 1e-12
        self._add("ausa/grad_norm_token_pre_pool", float(grad_h))
        self._add("ausa/grad_norm_set_post_pool", float(grad_z0))
        self._add("ausa/grad_norm_set_post_blocks", float(grad_zl))

        rho_p = grad_z0 / (grad_h + eps)
        rho_a = grad_zl / (grad_z0 + eps)
        rho_total = grad_zl / (grad_h + eps)
        self._add("ausa/grad_ratio_pool_rho_p", float(rho_p))
        self._add("ausa/grad_ratio_set_stack_rho_a", float(rho_a))
        self._add("ausa/grad_ratio_total_rho_pa", float(rho_total))

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
        spectrum_collapse = (
            top_eig_ratio is not None
            and top_eig_ratio == top_eig_ratio
            and top_eig_ratio > self.spectral_top_eig_warn
        ) or (
            spectral_entropy is not None
            and spectral_entropy == spectral_entropy
            and spectral_entropy < self.spectral_entropy_warn
        )
        if spectrum_collapse:
            warnings.warn(
                "Set Gram spectrum indicates possible overconstraint "
                f"(top_eig_ratio={top_eig_ratio:.4f}, "
                f"spectral_entropy_norm={spectral_entropy:.4f}).",
                RuntimeWarning,
            )
        pooling_neff_ratio = stats.get("ausa/pooling_neff_ratio")
        if (
            pooling_neff_ratio is not None
            and pooling_neff_ratio == pooling_neff_ratio
            and pooling_neff_ratio < self.pooling_neff_ratio_warn
        ):
            warnings.warn(
                "Pooling effective support is low; potential pooling collapse "
                f"(pooling_neff_ratio={pooling_neff_ratio:.4f}).",
                RuntimeWarning,
            )
        router_top1 = stats.get("ausa/router_top1_weight")
        if (
            router_top1 is not None
            and router_top1 == router_top1
            and router_top1 > self.router_top1_weight_warn
        ):
            warnings.warn(
                "Routing appears near-hard top-1; sparse gate may be overconfident "
                f"(router_top1_weight={router_top1:.4f}).",
                RuntimeWarning,
            )

        self._prev_epoch_stats = stats.copy()
        return stats
