from __future__ import annotations

import torch
from torch import nn

from .banks import build_window_bank, num_sets_for_length
from .router import LearnedRouter, UniformRouter
from .ska_block import SetAttentionBlock
from set_attention.adapter_factory import create_adapter, select_adapter_type
from set_attention.backends.dense_exact import DenseExactBackend
from set_attention.backends.landmark import LandmarkAttentionBackend
from set_attention.backends.local_band import LocalBandBackend
from set_attention.backends.nystrom import NystromBackend
from set_attention.backends.sparse_topk import SparseTopKBackend
from set_attention.features.base import SetFeatures
from set_attention.features.geometry_only import GeometryOnlyFeatureBuilder
from set_attention.features.hashed_counts import HashedCountFeatureBuilder
from set_attention.features.kernel_features import KernelFeatureBuilder
from set_attention.minhash import minhash_signatures


class SetOnlyLM(nn.Module):
    """Set-only LM: token-to-set pooling, set attention, set-to-token routing."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        window_size: int = 32,
        stride: int = 16,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        router_type: str = "uniform",
        router_topk: int = 0,
        backend: str = "dense_exact",
        backend_params: dict | None = None,
        feature_mode: str = "geometry_only",
        feature_params: dict | None = None,
        adapter_type: str = "auto",
        adapter_hidden_multiplier: int = 2,
        adapter_budget_fraction: float = 0.15,
        gamma: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.window_size = window_size
        self.stride = stride
        self.max_seq_len = max_seq_len

        max_sets = num_sets_for_length(max_seq_len, window_size, stride)
        feature_params = feature_params or {}
        if feature_mode == "geometry_only":
            self.feature_builder = GeometryOnlyFeatureBuilder(
                d_model=d_model, max_sets=max_sets, gamma=gamma, beta=beta
            )
        elif feature_mode == "hashed_counts":
            self.feature_builder = HashedCountFeatureBuilder(
                d_model=d_model,
                num_bins=feature_params.get("num_bins", 128),
                gamma=gamma,
                beta=beta,
            )
        elif feature_mode == "kernel":
            self.feature_builder = KernelFeatureBuilder(
                d_model=d_model, max_sets=max_sets, gamma=gamma, beta=beta
            )
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
        self.feature_mode = feature_mode
        self.feature_params = feature_params

        backend_params = backend_params or {}
        def make_backend() -> nn.Module:
            if backend == "dense_exact":
                return DenseExactBackend(
                    d_model=d_model, num_heads=num_heads, dropout=dropout
                )
            if backend == "local_band":
                return LocalBandBackend(
                    d_model=d_model,
                    num_heads=num_heads,
                    radius=backend_params.get("radius", 4),
                    dropout=dropout,
                )
            if backend == "nystrom":
                return NystromBackend(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_landmarks=backend_params.get("num_landmarks", 32),
                    dropout=dropout,
                )
            if backend == "landmark":
                return LandmarkAttentionBackend(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_landmarks=backend_params.get("num_landmarks", 32),
                    dropout=dropout,
                )
            if backend == "sparse_topk":
                return SparseTopKBackend(
                    d_model=d_model,
                    num_heads=num_heads,
                    k_s=backend_params.get("k_s", 16),
                    dropout=dropout,
                )
            raise ValueError(f"Unknown backend: {backend}")

        self.blocks = nn.ModuleList(
            [
                SetAttentionBlock(d_model=d_model, backend=make_backend())
                for _ in range(num_layers)
            ]
        )

        if router_type == "uniform":
            self.router = UniformRouter()
        elif router_type == "learned":
            self.router = LearnedRouter(d_model=d_model, topk=router_topk)
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.adapter = None
        if feature_mode != "geometry_only":
            phi_dim = d_model
            d_head = d_model // num_heads
            if adapter_type == "auto":
                adapter_type = select_adapter_type(phi_dim, d_head)
            self.adapter = create_adapter(
                adapter_type=adapter_type,
                num_heads=num_heads,
                d_head=d_head,
                phi_dim=phi_dim,
                hidden_multiplier=adapter_hidden_multiplier,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [batch, seq]")
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(batch, seq_len)

        token_states = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        token_states = self.token_mlp(token_states)

        bank = build_window_bank(
            seq_len=seq_len,
            window_size=self.window_size,
            stride=self.stride,
            device=input_ids.device,
        )
        set_states = bank.pool(token_states)

        if self.feature_mode == "geometry_only":
            features = self.feature_builder(bank.set_positions)
        elif self.feature_mode == "hashed_counts":
            per_batch = [self.feature_builder(input_ids[i], bank) for i in range(batch)]
            phi_attn = torch.stack([f.phi_attn for f in per_batch], dim=0)
            desc_router = torch.stack([f.desc_router for f in per_batch], dim=0)
            features = SetFeatures(
                phi_attn=phi_attn,
                desc_router=desc_router,
                geom_bias=per_batch[0].geom_bias,
            )
        else:
            k = self.feature_params.get("minhash_k", 64)
            per_batch = []
            for i in range(batch):
                token_ids = input_ids[i]
                set_tokens = token_ids[bank.set_indices.clamp_min(0)]
                set_tokens = set_tokens.masked_fill(bank.set_indices < 0, -1)
                sig = minhash_signatures(
                    set_tokens, k, max_id=self.token_emb.num_embeddings
                )
                per_batch.append(self.feature_builder(sig, bank.set_sizes))
            phi_attn = torch.stack([f.phi_attn for f in per_batch], dim=0)
            desc_router = torch.stack([f.desc_router for f in per_batch], dim=0)
            features = SetFeatures(
                phi_attn=phi_attn,
                desc_router=desc_router,
                geom_bias=per_batch[0].geom_bias,
            )
        geom_bias = features.geom_bias
        content_bias = None
        if self.adapter is not None and features.phi_attn is not None:
            content_bias = self.adapter(features.phi_attn)

        for block in self.blocks:
            set_states = block(set_states, geom_bias, content_bias, None, seq_len)

        if isinstance(self.router, UniformRouter):
            token_out = self.router(set_states, bank.token_to_sets)
        else:
            desc_router = features.desc_router
            token_out = self.router(token_states, set_states, desc_router, bank.token_to_sets)

        return self.lm_head(token_out)
