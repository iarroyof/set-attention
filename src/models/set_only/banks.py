# src/models/set_only/banks.py
from __future__ import annotations

from dataclasses import dataclass
import warnings
import math
import torch
from torch import nn


@dataclass
class Bank:
    set_indices: torch.Tensor  # [m, max_set_size] with -1 padding
    set_sizes: torch.Tensor  # [m]
    token_to_sets: torch.Tensor  # [seq, max_sets_per_token] with -1 padding
    set_positions: torch.Tensor  # [m]
    neighbor_mask: torch.Tensor | None = None

    def compute_neighbor_mask(
        self,
        method: str = "pos_topk",
        k: int = 16,
        delta_threshold: float = 0.25,
        include_self: bool = True,
        symmetric: bool = True,
        sig: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from set_attention.geometry import delta_indices

        if method.startswith("minhash"):
            if sig is None:
                raise ValueError("sig_gating requires signatures for minhash methods")
            if sig.dim() != 2:
                raise ValueError("sig must be [m, k]")
            matches = (sig.unsqueeze(1) == sig.unsqueeze(0)).float().mean(dim=-1)
            delta = 1.0 - matches
        else:
            delta = delta_indices(self.set_positions).to(torch.float32)
            if method == "pos_threshold":
                denom = max(1, delta.shape[0] - 1)
                delta = delta / float(denom)

        m = delta.shape[0]
        if m == 0:
            self.neighbor_mask = torch.zeros((0, 0), dtype=torch.bool, device=delta.device)
            return self.neighbor_mask
        if method in {"pos_threshold", "minhash_threshold"}:
            mask = delta <= delta_threshold
        elif method in {"pos_topk", "minhash_topk"}:
            k = max(1, min(k, m))
            idx = torch.topk(delta, k, dim=1, largest=False).indices
            mask = torch.zeros((m, m), dtype=torch.bool, device=delta.device)
            mask.scatter_(1, idx, True)
        else:
            raise ValueError(f"Unknown sig_gating method: {method}")

        if include_self:
            mask.fill_diagonal_(True)
        if symmetric:
            mask = mask | mask.t()
        self.neighbor_mask = mask
        return mask

    def pool(
        self,
        token_embeddings: torch.Tensor,
        mode: str = "mean",
        params: dict | None = None,
        pooling_module: "InformativeBoltzmannPooling | None" = None,
    ) -> torch.Tensor:
        if token_embeddings.dim() != 3:
            raise ValueError("token_embeddings must be [batch, seq, d]")
        batch, _, d_model = token_embeddings.shape

        indices = self.set_indices.clamp_min(0)
        gathered = token_embeddings[:, indices]  # [batch, m, max_set_size, d]
        mask = (self.set_indices >= 0).unsqueeze(0).unsqueeze(-1)
        if mode == "mean":
            summed = (gathered * mask).sum(dim=2)
            denom = self.set_sizes.clamp_min(1).unsqueeze(0).unsqueeze(-1)
            return summed / denom
        if mode == "soft_trimmed_boltzmann":
            params = params or {}
            if pooling_module is None:
                pooling_module = InformativeBoltzmannPooling(
                    tau=float(params.get("tau", 0.1)),
                    q=float(params.get("q", 0.8)),
                    alpha=float(params.get("alpha", 10.0)),
                    learnable_alpha=bool(params.get("learnable_alpha", False)),
                    tiny_set_n=int(params.get("tiny_set_n", 3)),
                    isotropy_eps=float(params.get("isotropy_eps", 1e-4)),
                    pooling_multihead=bool(params.get("pooling_multihead", False)),
                    num_heads=params.get("num_heads"),
                )
            # Flatten [batch, m, W, d] -> [batch*m, W, d]
            b, m, w, d = gathered.shape
            flat = gathered.reshape(b * m, w, d)
            flat_mask = (self.set_indices >= 0).unsqueeze(0).expand(b, -1, -1)
            flat_mask = flat_mask.reshape(b * m, w)
            pooled = pooling_module(flat, mask=flat_mask)
            return pooled.view(b, m, d)
        raise ValueError(f"Unknown pooling mode: {mode}")


def num_sets_for_length(seq_len: int, window_size: int, stride: int) -> int:
    """Calculate number of sets created by build_window_bank."""
    if seq_len <= 0:
        return 0
    # Number of window starting positions: range(0, seq_len, stride)
    return math.ceil(seq_len / stride)


def build_window_bank(
    seq_len: int,
    window_size: int,
    stride: int,
    device: torch.device,
) -> Bank:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    starts = list(range(0, seq_len, stride))
    set_indices_list: list[list[int]] = []
    for start in starts:
        end = min(start + window_size, seq_len)
        set_indices_list.append(list(range(start, end)))

    m = len(set_indices_list)
    max_set_size = max(len(s) for s in set_indices_list) if m else 0
    set_indices = torch.full((m, max_set_size), -1, dtype=torch.long, device=device)
    set_sizes = torch.zeros((m,), dtype=torch.long, device=device)

    for j, indices in enumerate(set_indices_list):
        if not indices:
            continue
        set_sizes[j] = len(indices)
        set_indices[j, : len(indices)] = torch.tensor(indices, device=device)

    token_sets: list[list[int]] = [[] for _ in range(seq_len)]
    for j, indices in enumerate(set_indices_list):
        for idx in indices:
            token_sets[idx].append(j)

    max_sets_per_token = max((len(s) for s in token_sets), default=0)
    token_to_sets = torch.full(
        (seq_len, max_sets_per_token), -1, dtype=torch.long, device=device
    )
    for t, sets in enumerate(token_sets):
        if sets:
            token_to_sets[t, : len(sets)] = torch.tensor(sets, device=device)

    set_positions = torch.arange(m, device=device, dtype=torch.long)
    return Bank(
        set_indices=set_indices,
        set_sizes=set_sizes,
        token_to_sets=token_to_sets,
        set_positions=set_positions,
    )


class InformativeBoltzmannPooling(nn.Module):
    def __init__(
        self,
        tau: float = 0.1,
        q: float = 0.8,
        alpha: float = 10.0,
        learnable_alpha: bool = False,
        tiny_set_n: int = 3,
        isotropy_eps: float = 1e-4,
        pooling_multihead: bool = False,
        num_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.q = q
        self.tiny_set_n = tiny_set_n
        self.isotropy_eps = isotropy_eps
        self.pooling_multihead = bool(pooling_multihead)
        self.num_heads = int(num_heads) if num_heads is not None else None
        if self.pooling_multihead and (self.num_heads is None or self.num_heads <= 0):
            raise ValueError("pooling_multihead=True requires num_heads > 0")
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha_buf", torch.tensor(float(alpha)))
        self._last_stats: dict[str, float] = {}

    def _alpha_value(self) -> float:
        if isinstance(getattr(self, "alpha", None), torch.Tensor):
            return float(self.alpha.detach().item())
        return float(self.alpha_buf.detach().item())

    def _alpha_tensor(self) -> torch.Tensor:
        if isinstance(getattr(self, "alpha", None), torch.Tensor):
            return self.alpha.clamp(min=1.0, max=50.0)
        return self.alpha_buf

    def _pool_single(
        self,
        x: torch.Tensor,  # [N, W, D]
        mask: torch.Tensor | None = None,  # [N, W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            mu = x.sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        else:
            mu = x.mean(dim=1, keepdim=True)

        d2 = ((x - mu) ** 2).sum(dim=-1) / x.shape[-1]
        if mask is not None:
            d2 = d2.masked_fill(~mask, float("inf"))

        mean_pooled = mu.squeeze(1)
        if mask is not None:
            tiny_set = denom.squeeze(1) <= self.tiny_set_n
            masked_d2 = d2.masked_fill(~mask, 0.0)
            mean_d2 = masked_d2.sum(dim=1) / denom.squeeze(1).clamp_min(1)
            var_d2 = (
                ((masked_d2 - mean_d2.unsqueeze(1)) ** 2) * mask
            ).sum(dim=1) / denom.squeeze(1).clamp_min(1)
            spread = var_d2 / (mean_d2 + 1e-8)
            isotropic = spread <= self.isotropy_eps
            uniform = torch.full_like(d2, 1.0 / d2.shape[1])
            weights_mean = torch.where(mask.any(dim=1, keepdim=True), mask.float() / denom, uniform)
        else:
            tiny_set = torch.zeros((x.shape[0],), dtype=torch.bool, device=x.device)
            var_d2 = d2.var(dim=1, unbiased=False)
            mean_d2 = d2.mean(dim=1)
            spread = var_d2 / (mean_d2 + 1e-8)
            isotropic = spread <= self.isotropy_eps
            weights_mean = torch.full_like(d2, 1.0 / d2.shape[1])

        k = max(1, int(self.q * d2.shape[1]))
        thresh = torch.topk(d2, k, dim=1, largest=False).values[:, -1:].detach()
        alpha = self._alpha_tensor()
        mask_soft = torch.sigmoid(alpha * (thresh - d2))
        logits = (-d2 / self.tau) + torch.log(mask_soft + 1e-8)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        finite_rows = torch.isfinite(logits).any(dim=1, keepdim=True)
        safe_logits = torch.where(finite_rows, logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=1)
        weights = torch.where(finite_rows, weights, torch.zeros_like(weights))

        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        use_mean = tiny_set | isotropic | (~finite_rows.squeeze(1))
        if use_mean.any():
            pooled = torch.where(use_mean.unsqueeze(-1), mean_pooled, pooled)
            weights = torch.where(use_mean.unsqueeze(1), weights_mean, weights)

        return pooled, weights

    def _compute_base_stats(
        self,
        weights: torch.Tensor,  # [B,W]
        x: torch.Tensor,  # [B,W,D]
        pooled: torch.Tensor,  # [B,D]
        mask: torch.Tensor | None,
    ) -> dict[str, float]:
        ent = -(weights * torch.log(weights + 1e-12)).sum(dim=1)
        top1 = weights.max(dim=1).values
        support = (weights > 1e-3).sum(dim=1).float()
        neff = 1.0 / (weights.pow(2).sum(dim=1) + 1e-12)
        p_sorted, _ = torch.sort(weights, dim=1)
        n = p_sorted.shape[1]
        idx = torch.arange(1, n + 1, device=p_sorted.device, dtype=p_sorted.dtype)
        gini = 1.0 - 2.0 * torch.sum(p_sorted * (n - idx + 0.5), dim=1) / n

        if mask is not None:
            valid = mask.float()
            valid_tokens = valid.sum(dim=1).clamp_min(1.0)
            neff_ratio = neff / valid_tokens
            x_norm = x.norm(dim=-1)
            x_norm_mean = (x_norm * valid).sum(dim=1) / valid_tokens
        else:
            valid_tokens = torch.full_like(neff, float(weights.shape[1]))
            neff_ratio = neff / valid_tokens
            x_norm_mean = x.norm(dim=-1).mean(dim=1)
        pooled_norm = pooled.norm(dim=-1)
        norm_ratio = pooled_norm / (x_norm_mean + 1e-12)

        if support.min().item() <= 1:
            warnings.warn(
                "Pooling collapsed to single-set dominance; this run may behave like hard routing.",
                RuntimeWarning,
            )

        return {
            "ausa/pooling_weight_entropy": float(ent.mean().item()),
            "ausa/pooling_top1_weight": float(top1.mean().item()),
            "ausa/pooling_effective_support": float(support.mean().item()),
            "ausa/pooling_neff_l2": float(neff.mean().item()),
            "ausa/pooling_neff_ratio": float(neff_ratio.mean().item()),
            "ausa/pooling_norm_ratio": float(norm_ratio.mean().item()),
            "ausa/pooling_weight_gini": float(gini.mean().item()),
            "ausa/pooling_alpha_value": self._alpha_value(),
        }

    def _compute_head_stats(self, weights_h: torch.Tensor) -> dict[str, float]:
        # weights_h: [B,H,W]
        eps = 1e-12
        w = weights_h.clamp_min(eps)
        ent_h = -(w * torch.log(w)).sum(dim=-1)  # [B,H]
        top1_idx = weights_h.argmax(dim=-1)  # [B,H]
        modal_counts = torch.nn.functional.one_hot(
            top1_idx, num_classes=weights_h.shape[-1]
        ).sum(dim=1)  # [B,W]
        top1_agreement = modal_counts.amax(dim=-1).to(weights_h.dtype) / float(
            max(weights_h.shape[1], 1)
        )
        pbar = weights_h.mean(dim=1, keepdim=True).clamp_min(eps)
        m = (0.5 * (weights_h + pbar)).clamp_min(eps)
        kl_h_m = (w * (torch.log(w) - torch.log(m))).sum(dim=-1)
        kl_p_m = (pbar * (torch.log(pbar) - torch.log(m))).sum(dim=-1)
        js = 0.5 * (kl_h_m + kl_p_m)
        return {
            "ausa/pooling_head_entropy_mean": float(ent_h.mean().item()),
            "ausa/pooling_head_entropy_std": float(
                ent_h.std(dim=1, unbiased=False).mean().item()
            ),
            "ausa/pooling_head_js_to_mean": float(js.mean().item()),
            "ausa/pooling_head_top1_agreement": float(top1_agreement.mean().item()),
            "ausa/pooling_head_prob_var": float(
                weights_h.var(dim=1, unbiased=False).mean().item()
            ),
        }

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, W, D], mask: [B, W]
        if x.dim() != 3:
            raise ValueError("x must be [batch, window, dim]")
        if mask is not None and mask.dim() != 2:
            raise ValueError("mask must be [batch, window]")
        if mask is not None and mask.shape[1] != x.shape[1] and mask.shape[1] == x.shape[2]:
            # If x arrives as [B, D, W], transpose to [B, W, D] to match mask.
            x = x.transpose(1, 2)
        elif mask is None and x.shape[1] > x.shape[2]:
            # Heuristic fallback when no mask is provided.
            x = x.transpose(1, 2)

        if self.pooling_multihead:
            if self.num_heads is None or self.num_heads <= 0:
                raise ValueError("pooling_multihead=True requires num_heads > 0")
            if x.shape[-1] % self.num_heads != 0:
                raise ValueError(
                    f"Pooling head split requires dim {x.shape[-1]} divisible by num_heads {self.num_heads}"
                )
            b, w, d = x.shape
            d_h = d // self.num_heads
            x_h = x.view(b, w, self.num_heads, d_h).permute(0, 2, 1, 3).contiguous()  # [B,H,W,dh]
            x_flat = x_h.view(b * self.num_heads, w, d_h)
            mask_flat = None
            if mask is not None:
                mask_flat = mask.unsqueeze(1).expand(b, self.num_heads, w).reshape(
                    b * self.num_heads, w
                )
            pooled_flat, weights_flat = self._pool_single(x_flat, mask_flat)
            pooled_h = pooled_flat.view(b, self.num_heads, d_h)
            pooled = pooled_h.reshape(b, d)
            weights_h = weights_flat.view(b, self.num_heads, w)

            # Backward-compatible pooling metrics on mean-over-head weights.
            weights_bar = weights_h.mean(dim=1)  # [B,W]
            pooled_bar = torch.sum(x * weights_bar.unsqueeze(-1), dim=1)
            self._last_stats = self._compute_base_stats(weights_bar, x, pooled_bar, mask)
            self._last_stats.update(self._compute_head_stats(weights_h))
            return pooled

        pooled, weights = self._pool_single(x, mask)
        self._last_stats = self._compute_base_stats(weights, x, pooled, mask)
        return pooled

    def get_last_stats(self) -> dict[str, float]:
        return dict(self._last_stats)
