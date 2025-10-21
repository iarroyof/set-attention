from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F


class BaseSim:
    def score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class DotSim(BaseSim):
    head_dim: int
    temperature: float = 1.0

    def score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_dim)
        s = (q @ k.transpose(-2, -1)) * scale
        return s / max(self.temperature, 1e-6)


@dataclass
class CosineSim(BaseSim):
    temperature: float = 1.0

    def score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(k, dim=-1)
        return (qn @ kn.transpose(-2, -1)) / max(self.temperature, 1e-6)


@dataclass
class RbfSim(BaseSim):
    gamma: float = 0.5

    def score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q2 = (q**2).sum(dim=-1, keepdim=True)
        k2 = (k**2).sum(dim=-1, keepdim=True).transpose(-2, -1)
        d2 = q2 + k2 - 2.0 * (q @ k.transpose(-2, -1))
        return torch.exp(-self.gamma * d2.clamp_min(0.0))


@dataclass
class IntersectSim(BaseSim):
    head_dim: int
    topk: int = 16
    normalize: bool = True

    def score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        tk = max(1, min(self.topk, self.head_dim))
        q_idx = torch.topk(q, k=tk, dim=-1, largest=True).indices
        k_idx = torch.topk(k, k=tk, dim=-1, largest=True).indices
        Mq = torch.zeros_like(q).to(q.dtype)
        Mk = torch.zeros_like(k).to(k.dtype)
        Mq.scatter_(-1, q_idx, 1.0)
        Mk.scatter_(-1, k_idx, 1.0)
        s = Mq @ Mk.transpose(-2, -1)
        if self.normalize:
            s = s / float(tk)
        return s


def build_similarity(kind: str, head_dim: int, **kwargs) -> BaseSim:
    kind = kind.lower()
    if kind == "dot":
        return DotSim(head_dim=head_dim, temperature=float(kwargs.get("temperature", 1.0)))
    if kind == "cosine":
        return CosineSim(temperature=float(kwargs.get("temperature", 1.0)))
    if kind == "rbf":
        return RbfSim(gamma=float(kwargs.get("rbf_gamma", 0.5)))
    if kind in ("intersect", "intersection"):
        return IntersectSim(
            head_dim=head_dim,
            topk=int(kwargs.get("inter_topk", 16)),
            normalize=bool(kwargs.get("inter_normalize", True)),
        )
    raise ValueError(f"Unknown similarity kind: {kind}")

