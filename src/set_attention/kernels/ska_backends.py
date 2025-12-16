from __future__ import annotations

from typing import Optional
import warnings

import torch

from set_attention.kernels.sketches import symdiff_from_jaccard


class PythonSKABackend:
    """Reference backend that operates block-by-block using torch tensors."""

    def __init__(self, *, score_mode: str):
        self.score_mode = score_mode

    def __call__(
        self,
        sig_q: torch.Tensor,
        size_q: torch.Tensor,
        sig_k: torch.Tensor,
        size_k: torch.Tensor,
        q_ptrs: torch.Tensor,
        k_ptrs: torch.Tensor,
        values: torch.Tensor,
        *,
        gamma: float,
        beta: float,
        tau: float,
        eta: float,
        proj_q: Optional[torch.Tensor],
        proj_k: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = values.device
        dtype = values.dtype
        nq = sig_q.shape[0]
        H, Dh = values.shape[1], values.shape[2]
        context = torch.zeros(nq, H, Dh, device=device, dtype=dtype)
        B = int(q_ptrs.numel() - 1)
        for b in range(B):
            qa = int(q_ptrs[b].item())
            qb = int(q_ptrs[b + 1].item())
            ka = int(k_ptrs[b].item())
            kb = int(k_ptrs[b + 1].item())
            if qb <= qa or kb <= ka:
                continue
            ctx_block = self._block_attention(
                sig_q[qa:qb],
                size_q[qa:qb],
                sig_k[ka:kb],
                size_k[ka:kb],
                values[ka:kb],
                gamma=gamma,
                beta=beta,
                tau=tau,
                eta=eta,
                proj_q=None if proj_q is None else proj_q[qa:qb],
                proj_k=None if proj_k is None else proj_k[ka:kb],
            )
            context[qa:qb] = ctx_block
        return context

    def _block_attention(
        self,
        sig_q: torch.Tensor,
        size_q: torch.Tensor,
        sig_k: torch.Tensor,
        size_k: torch.Tensor,
        values: torch.Tensor,
        *,
        gamma: float,
        beta: float,
        tau: float,
        eta: float,
        proj_q: Optional[torch.Tensor],
        proj_k: Optional[torch.Tensor],
    ) -> torch.Tensor:
        dtype = values.dtype
        # Δ-RBF score
        eq = (sig_q[:, None, :] == sig_k[None, :, :]).to(dtype)
        jacc = eq.mean(dim=-1)
        if jacc.numel() == 0:
            return torch.zeros(sig_q.size(0), values.size(1), values.size(2), device=values.device, dtype=dtype)
        delta = symdiff_from_jaccard(jacc, size_q.to(dtype), size_k.to(dtype))
        score_delta = torch.exp(-gamma * delta)
        inter = (jacc / (1.0 + jacc + 1e-8)) * (size_q[:, None].to(dtype) + size_k[None, :].to(dtype))
        inter_norm = inter / (torch.sqrt(size_q[:, None].to(dtype) * size_k[None, :].to(dtype)) + 1e-8)

        if self.score_mode == "delta_rbf":
            scores = score_delta
        elif self.score_mode == "delta_plus_dot":
            content = self._content_term(proj_q, proj_k)
            scores = score_delta + beta * content if content is not None else score_delta
        elif self.score_mode == "intersect_norm":
            scores = eta * inter_norm
        elif self.score_mode == "intersect_plus_dot":
            content = self._content_term(proj_q, proj_k)
            scores = eta * inter_norm
            if content is not None:
                scores = scores + beta * content
        elif self.score_mode == "dot":
            content = self._content_term(proj_q, proj_k)
            scores = beta * content if content is not None else torch.zeros_like(score_delta)
        else:
            raise ValueError(f"Unknown score_mode '{self.score_mode}'")

        weights = torch.softmax(scores / max(tau, 1e-6), dim=-1)
        return torch.einsum("qs,shd->qhd", weights, values)

    @staticmethod
    def _content_term(q_proj: Optional[torch.Tensor], k_proj: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if q_proj is None or k_proj is None:
            return None
        return torch.matmul(q_proj, k_proj.transpose(0, 1))


class _UnavailableBackend:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"The '{self.name}' SKA backend is not available. "
            f"Please install the required dependencies or select --ska-backend python."
        )


def get_backend(name: str, *, score_mode: str):
    name = name.lower()
    if name == "python":
        return PythonSKABackend(score_mode=score_mode)
    if name in {"triton", "keops"}:
        warnings.warn(
            f"SKA backend '{name}' is not yet available; falling back to the reference python backend.",
            stacklevel=2,
        )
        return PythonSKABackend(score_mode=score_mode)
    raise ValueError(f"Unknown ska backend '{name}'")
