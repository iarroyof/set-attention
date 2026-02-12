from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from models.set_only.banks import InformativeBoltzmannPooling, build_window_bank, num_sets_for_length
from models.set_only.router import LearnedRouter
from set_attention.features import (
    GeometryOnlyFeatureBuilder,
    HashedCountFeatureBuilder,
    KernelFeatureBuilder,
)
from set_attention.features.base import SetFeatures
from set_attention.minhash import minhash_signatures


class SetOnlyCrossAttention(nn.Module):
    """Set-only cross-attention: decoder tokens attend to encoder sets."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        stride: int,
        max_seq_len: int,
        pooling: str | dict,
        feature_mode: str,
        feature_params: Optional[dict],
        router_type: str,
        router_topk: int,
        sig_gating: Optional[dict],
        d_phi: Optional[int],
        gamma: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.feature_mode = feature_mode
        self.feature_params = feature_params or {}
        self.sig_gating = sig_gating or {}
        self.router_type = router_type
        self.router_topk = router_topk

        if isinstance(pooling, dict):
            self.pooling_mode = pooling.get("mode", "mean")
            self.pooling_params = {
                "tau": pooling.get("tau", 0.1),
                "q": pooling.get("q", 0.8),
                "alpha": pooling.get("alpha", 10.0),
                "learnable_alpha": pooling.get("learnable_alpha", False),
                "tiny_set_n": pooling.get("tiny_set_n", 3),
                "isotropy_eps": pooling.get("isotropy_eps", 1e-4),
            }
        else:
            self.pooling_mode = pooling
            self.pooling_params = {}

        self.pooling_module: InformativeBoltzmannPooling | None = None
        if self.pooling_mode == "soft_trimmed_boltzmann":
            self.pooling_module = InformativeBoltzmannPooling(
                tau=float(self.pooling_params.get("tau", 0.1)),
                q=float(self.pooling_params.get("q", 0.8)),
                alpha=float(self.pooling_params.get("alpha", 10.0)),
                learnable_alpha=bool(self.pooling_params.get("learnable_alpha", False)),
                tiny_set_n=int(self.pooling_params.get("tiny_set_n", 3)),
                isotropy_eps=float(self.pooling_params.get("isotropy_eps", 1e-4)),
            )

        max_sets = num_sets_for_length(max_seq_len, window_size, stride)
        d_phi = d_phi or d_model
        if self.feature_mode == "geometry_only":
            self.feature_builder = GeometryOnlyFeatureBuilder(
                d_model=d_model,
                max_sets=max_sets,
                gamma=gamma,
                beta=beta,
            )
        elif self.feature_mode == "hashed_counts":
            num_bins = int(self.feature_params.get("num_bins", 128))
            fusion = self.feature_params.get("fusion", "mlp")
            include_geom = self.feature_params.get("include_geom_in_attn", True)
            self.feature_builder = HashedCountFeatureBuilder(
                d_model=d_model,
                d_phi=d_phi,
                max_sets=max_sets,
                num_bins=num_bins,
                gamma=gamma,
                beta=beta,
                fusion=fusion,
                include_geom_in_attn=include_geom,
            )
        else:
            self.feature_builder = KernelFeatureBuilder(
                d_model=d_model,
                d_phi=d_phi,
                max_sets=max_sets,
                gamma=gamma,
                beta=beta,
            )

        if self.router_type == "uniform":
            self.router = None
        elif self.router_type == "learned":
            self.router = LearnedRouter(
                d_model=d_model,
                topk=router_topk,
                restrict_to_sets=False,
            )
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

    def _build_features(
        self,
        src_ids: torch.Tensor,
        bank,
        set_states: torch.Tensor,
    ) -> SetFeatures:
        if self.feature_mode == "geometry_only":
            return self.feature_builder(bank.set_positions)
        if self.feature_mode == "hashed_counts":
            per_batch = [
                self.feature_builder(src_ids[i], bank, set_states[i])
                for i in range(src_ids.size(0))
            ]
            desc_router = torch.stack([f.desc_router for f in per_batch], dim=0)
            phi_attn = torch.stack([f.phi_attn for f in per_batch], dim=0)
            return SetFeatures(
                phi_attn=phi_attn,
                desc_router=desc_router,
                geom_bias=per_batch[0].geom_bias,
            )

        k = int(self.feature_params.get("minhash_k", 64))
        per_batch = []
        for i in range(src_ids.size(0)):
            token_ids = src_ids[i]
            set_tokens = token_ids[bank.set_indices.clamp_min(0)]
            set_tokens = set_tokens.masked_fill(bank.set_indices < 0, -1)
            sig = minhash_signatures(set_tokens, k, max_id=self.vocab_size)
            per_batch.append(self.feature_builder(sig, bank.set_sizes))
        phi_attn = torch.stack([f.phi_attn for f in per_batch], dim=0)
        desc_router = torch.stack([f.desc_router for f in per_batch], dim=0)
        return SetFeatures(
            phi_attn=phi_attn,
            desc_router=desc_router,
            geom_bias=per_batch[0].geom_bias,
        )

    def forward(
        self,
        token_states: torch.Tensor,
        memory_tokens: torch.Tensor,
        src_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if memory_tokens.dim() != 3:
            raise ValueError("memory_tokens must be [batch, seq, d]")
        batch, seq_len, _ = memory_tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        bank = build_window_bank(
            seq_len=seq_len,
            window_size=self.window_size,
            stride=self.stride,
            device=memory_tokens.device,
        )
        set_states = bank.pool(
            token_embeddings=memory_tokens,
            mode=self.pooling_mode,
            params=self.pooling_params,
            pooling_module=self.pooling_module,
        )
        features = self._build_features(src_ids, bank, set_states)
        desc_router = features.desc_router

        num_sets = set_states.shape[1]
        if self.router is None:
            token_repr = set_states.mean(dim=1, keepdim=True).expand(
                batch, token_states.shape[1], -1
            )
            weights = torch.full(
                (batch, token_states.shape[1], num_sets),
                1.0 / max(1, num_sets),
                device=token_states.device,
            )
            return token_repr, weights

        token_to_sets = torch.zeros(
            (token_states.shape[1], 1),
            dtype=torch.long,
            device=token_states.device,
        )
        router_out = self.router(token_states, set_states, desc_router, token_to_sets)
        return router_out.token_repr, router_out.probs


class SetOnlyCrossLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        cross_attn: SetOnlyCrossAttention,
        attn_dropout: float | None = None,
        resid_dropout: float | None = None,
        ffn_dropout: float | None = None,
    ) -> None:
        super().__init__()
        attn_drop = attn_dropout if attn_dropout is not None else dropout
        resid_drop = resid_dropout if resid_dropout is not None else dropout
        ffn_drop = ffn_dropout if ffn_dropout is not None else dropout
        self.cross_attn = cross_attn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.dropout = nn.Dropout(ffn_drop)
        self.dropout1 = nn.Dropout(resid_drop)
        self.dropout2 = nn.Dropout(resid_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cross_input = self.norm1(x)
        cross_out, cross_weights = self.cross_attn(cross_input, memory, src_ids)
        cross_out = self.attn_dropout(cross_out)
        x = x + self.dropout1(cross_out)
        ff_input = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ff_input))))
        x = x + self.dropout2(ff_output)
        return x, cross_weights
