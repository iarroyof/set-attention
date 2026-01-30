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
    ) -> None:
        super().__init__()
        self.tau = tau
        self.q = q
        self.tiny_set_n = tiny_set_n
        self.isotropy_eps = isotropy_eps
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha_buf", torch.tensor(float(alpha)))
        self._last_stats: dict[str, float] = {}

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, W, D], mask: [B, W]
        if x.dim() != 3:
            raise ValueError("x must be [batch, window, dim]")
        if mask is not None and mask.dim() != 2:
            raise ValueError("mask must be [batch, window]")

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
        else:
            tiny_set = torch.zeros((x.shape[0],), dtype=torch.bool, device=x.device)
            var_d2 = d2.var(dim=1, unbiased=False)
            mean_d2 = d2.mean(dim=1)
            spread = var_d2 / (mean_d2 + 1e-8)
            isotropic = spread <= self.isotropy_eps

        with torch.no_grad():
            k = max(1, int(self.q * d2.shape[1]))
            # Prefer topk for small/fixed windows; quantile can be used for larger windows.
            thresh = torch.topk(d2, k, dim=1, largest=False).values[:, -1:]

        if isinstance(getattr(self, "alpha", None), torch.Tensor):
            alpha = self.alpha
        else:
            alpha = self.alpha_buf
        mask_soft = torch.sigmoid(alpha * (thresh - d2))
        logits = (-d2 / self.tau) + torch.log(mask_soft + 1e-8)

        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(x * weights, dim=1)
        use_mean = tiny_set | isotropic
        if use_mean.any():
            pooled = torch.where(use_mean.unsqueeze(-1), mean_pooled, pooled)
            if mask is not None:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
                weights_mean = mask.float() / denom
            else:
                weights_mean = torch.full_like(d2, 1.0 / d2.shape[1])
            weights = torch.where(use_mean.unsqueeze(-1), weights_mean.unsqueeze(-1), weights)

        weights_flat = weights.squeeze(-1)
        ent = -(weights_flat * torch.log(weights_flat + 1e-12)).sum(dim=1)
        top1 = weights_flat.max(dim=1).values
        support = (weights_flat > 1e-3).sum(dim=1).float()
        p_sorted, _ = torch.sort(weights_flat, dim=1)
        n = p_sorted.shape[1]
        idx = torch.arange(1, n + 1, device=p_sorted.device, dtype=p_sorted.dtype)
        gini = 1.0 - 2.0 * torch.sum(p_sorted * (n - idx + 0.5), dim=1) / n

        if support.min().item() <= 1:
            warnings.warn(
                "Pooling collapsed to single-set dominance; this run may behave like hard routing.",
                RuntimeWarning,
            )

        if isinstance(getattr(self, "alpha", None), torch.Tensor):
            alpha_value = float(self.alpha.detach().item())
        else:
            alpha_value = float(self.alpha_buf.detach().item())

        self._last_stats = {
            "ausa/pooling_weight_entropy": float(ent.mean().item()),
            "ausa/pooling_top1_weight": float(top1.mean().item()),
            "ausa/pooling_effective_support": float(support.mean().item()),
            "ausa/pooling_weight_gini": float(gini.mean().item()),
            "ausa/pooling_alpha_value": alpha_value,
        }
        return pooled

    def get_last_stats(self) -> dict[str, float]:
        return dict(self._last_stats)
