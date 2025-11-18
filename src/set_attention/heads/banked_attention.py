from __future__ import annotations

from typing import Tuple

import contextlib

import torch
import torch.nn as nn

from set_attention.kernels.ska_backends import get_backend
from set_attention.sets.bank_utils import pad_segments_from_ptrs


class SetBankAttention(nn.Module):
    """Multi-head attention operating on concatenated set banks."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        tau: float = 1.0,
        gamma: float = 0.3,
        beta: float = 1.0,
        score_mode: str = "delta_plus_dot",
        eta: float = 1.0,
        backend: str = "python",
        precision: str = "fp32",
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of {'fp32','fp16','bf16'}.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.eta = float(eta)
        self.score_mode = score_mode
        self.precision = precision
        self.backend_name = backend

        self.proj_A = nn.Linear(d_model, d_model, bias=False)
        self.proj_B = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.backend = get_backend(backend, score_mode=score_mode)

    def forward(
        self,
        phi_q: torch.Tensor,
        sig_q: torch.Tensor,
        size_q: torch.Tensor,
        q_ptrs: torch.Tensor,
        phi_k: torch.Tensor,
        sig_k: torch.Tensor,
        size_k: torch.Tensor,
        k_ptrs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute set outputs given concatenated query & key banks."""
        phi_q, sig_q, size_q, q_ptrs = self._ensure_flat(phi_q, sig_q, size_q, q_ptrs)
        phi_k, sig_k, size_k, k_ptrs = self._ensure_flat(phi_k, sig_k, size_k, k_ptrs)
        nq = phi_q.shape[0]
        if nq == 0:
            empty = torch.zeros(0, self.num_heads, self.head_dim, device=phi_q.device, dtype=phi_q.dtype)
            return empty, q_ptrs

        orig_dtype = phi_q.dtype
        with self._autocast(phi_q.device):
            phi_q_cast = phi_q
            phi_k_cast = phi_k
            proj_q = self.proj_A(phi_q_cast) if self.beta != 0 else None
            proj_k = self.proj_B(phi_k_cast) if self.beta != 0 else None
            value_heads = self.value_proj(phi_k_cast).view(-1, self.num_heads, self.head_dim)
            context = self.backend(
                sig_q=sig_q,
                size_q=size_q,
                sig_k=sig_k,
                size_k=size_k,
                q_ptrs=q_ptrs,
                k_ptrs=k_ptrs,
                values=value_heads,
                gamma=self.gamma,
                beta=self.beta,
                tau=self.tau,
                eta=self.eta,
                proj_q=proj_q,
                proj_k=proj_k,
            )
        return context.to(orig_dtype), q_ptrs

    def padded_context(
        self,
        context_flat: torch.Tensor,
        q_ptrs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Utility to obtain padded context/mask for router-side diagnostics."""
        padded, mask = pad_segments_from_ptrs(context_flat, q_ptrs, fill_value=0.0)
        return padded, mask

    def _autocast(self, device: torch.device):
        if device.type != "cuda" or self.precision == "fp32":
            return contextlib.nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _ensure_flat(
        self,
        phi: torch.Tensor,
        sig: torch.Tensor,
        size: torch.Tensor,
        ptrs_or_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if ptrs_or_mask.dtype == torch.bool:
            mask = ptrs_or_mask
            B = mask.size(0)
            counts = mask.sum(dim=1, dtype=torch.long)
            ptrs = torch.zeros(B + 1, dtype=torch.long, device=mask.device)
            if B > 0:
                ptrs[1:] = torch.cumsum(counts, dim=0)
            flat_phi = phi[mask]
            flat_sig = sig[mask]
            flat_size = size[mask]
            return flat_phi, flat_sig, flat_size, ptrs
        if ptrs_or_mask.dim() != 1:
            raise ValueError("Expected pointer tensor of shape (B+1,).")
        return phi, sig, size, ptrs_or_mask
