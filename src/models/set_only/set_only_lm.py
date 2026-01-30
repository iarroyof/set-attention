from __future__ import annotations

import torch
from torch import nn
import warnings

from .banks import build_window_bank, num_sets_for_length
from .diagnostics import SetDiagnostics
from .router import LearnedRouter, UniformRouter, RouterOutput
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
        pooling: str = "mean",
        multiscale: bool = False,
        sig_gating: dict | None = None,
        d_phi: int | None = None,
        geometry: dict | None = None,
        features: dict | None = None,
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
        token_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.token_emb = token_embedding or nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.window_size = window_size
        self.stride = stride
        self.max_seq_len = max_seq_len
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
        self.pooling_module = None
        if self.pooling_mode == "soft_trimmed_boltzmann":
            from .banks import InformativeBoltzmannPooling

            self.pooling_module = InformativeBoltzmannPooling(
                tau=float(self.pooling_params.get("tau", 0.1)),
                q=float(self.pooling_params.get("q", 0.8)),
                alpha=float(self.pooling_params.get("alpha", 10.0)),
                learnable_alpha=bool(self.pooling_params.get("learnable_alpha", False)),
                tiny_set_n=int(self.pooling_params.get("tiny_set_n", 3)),
                isotropy_eps=float(self.pooling_params.get("isotropy_eps", 1e-4)),
            )
        self.multiscale = multiscale
        self.sig_gating = sig_gating or {}
        if not self.multiscale:
            warnings.warn(
                "multiscale disabled; using single-scale bank",
                RuntimeWarning,
            )
        if (
            self.pooling_mode != "mean"
            and router_type == "uniform"
            and feature_mode == "geometry_only"
        ):
            warnings.warn(
                "Pooling is configured but has no effect with "
                "geometry_only + uniform router. "
                "This run will not test pooling behavior.",
                RuntimeWarning,
            )

        max_sets = num_sets_for_length(max_seq_len, window_size, stride)
        if d_phi is None:
            d_phi = d_model
        self.d_phi = d_phi
        feature_params = feature_params or {}
        features_cfg = features or {}
        if isinstance(features_cfg, dict) and feature_mode in features_cfg:
            mode_cfg = features_cfg.get(feature_mode, {})
            if isinstance(mode_cfg, dict):
                feature_params = {**feature_params, **mode_cfg}
        geometry_cfg = geometry or {}
        geom_enabled = bool(geometry_cfg.get("enabled", True))
        geom_apply_bias = bool(geometry_cfg.get("apply_as_bias", True))
        geom_apply_in_phi = bool(geometry_cfg.get("apply_in_phi_attn", True))
        if not geom_enabled:
            geom_apply_bias = False
            geom_apply_in_phi = False
        self.geom_enabled = geom_enabled
        self.geom_apply_bias = geom_apply_bias
        self.geom_apply_in_phi = geom_apply_in_phi

        if feature_mode == "geometry_only":
            self.feature_builder = GeometryOnlyFeatureBuilder(
                d_model=d_model, max_sets=max_sets, gamma=gamma, beta=beta
            )
        elif feature_mode == "hashed_counts":
            self.feature_builder = HashedCountFeatureBuilder(
                d_model=d_model,
                d_phi=d_phi,
                max_sets=max_sets,
                num_bins=feature_params.get("num_bins", 128),
                gamma=gamma,
                beta=beta,
                fusion=feature_params.get("fusion", "mlp"),
                include_geom_in_attn=geom_apply_in_phi,
            )
        elif feature_mode == "kernel":
            self.feature_builder = KernelFeatureBuilder(
                d_model=d_model, d_phi=d_phi, max_sets=max_sets, gamma=gamma, beta=beta
            )
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
        self.feature_mode = feature_mode
        self.feature_params = feature_params

        if self.multiscale:
            raise ValueError("multiscale is not implemented in SetOnlyLM")

        print(
            {
                "pooling": {"mode": self.pooling_mode, **self.pooling_params},
                "sig_gating": self.sig_gating,
                "d_phi": self.d_phi,
                "geometry": {
                    "enabled": self.geom_enabled,
                    "apply_as_bias": self.geom_apply_bias,
                    "apply_in_phi_attn": self.geom_apply_in_phi,
                },
            }
        )

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
        self.diagnostics = SetDiagnostics()
        if feature_mode != "geometry_only":
            phi_dim = d_phi
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

    def _encode_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RouterOutput]:
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
        set_states = bank.pool(
            token_embeddings=token_states,
            mode=self.pooling_mode,
            params=self.pooling_params,
            pooling_module=self.pooling_module,
        )
        if self.training and self.pooling_module is not None:
            pooling_stats = self.pooling_module.get_last_stats()
            if pooling_stats:
                self.diagnostics.update_with_pooling_stats(pooling_stats)

        sig_for_gating = None
        if self.feature_mode == "geometry_only":
            features = self.feature_builder(bank.set_positions)
        elif self.feature_mode == "hashed_counts":
            per_batch = [
                self.feature_builder(input_ids[i], bank, set_states[i])
                for i in range(batch)
            ]
            phi_attn = torch.stack([f.phi_attn for f in per_batch], dim=0)
            desc_router = torch.stack([f.desc_router for f in per_batch], dim=0)
            features = SetFeatures(
                phi_attn=phi_attn,
                desc_router=desc_router,
                geom_bias=per_batch[0].geom_bias,
            )
            if self.sig_gating.get("enabled") and self.sig_gating.get("method", "").startswith("minhash"):
                k = int(self.sig_gating["sig_k"])
                token_ids = input_ids[0]
                set_tokens = token_ids[bank.set_indices.clamp_min(0)]
                set_tokens = set_tokens.masked_fill(bank.set_indices < 0, -1)
                sig_for_gating = minhash_signatures(
                    set_tokens, k, max_id=self.token_emb.num_embeddings
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
            if self.sig_gating.get("enabled") and self.sig_gating.get("method", "").startswith("minhash"):
                sig_k = int(self.sig_gating["sig_k"])
                sig_for_gating = minhash_signatures(
                    input_ids[0][bank.set_indices.clamp_min(0)].masked_fill(bank.set_indices < 0, -1),
                    sig_k,
                    max_id=self.token_emb.num_embeddings,
                )
        geom_bias = features.geom_bias
        if not self.geom_enabled or not self.geom_apply_bias:
            geom_bias = None
        content_bias = None
        if self.adapter is not None and features.phi_attn is not None:
            content_bias = self.adapter(features.phi_attn)

        sig_mask = None
        if self.sig_gating and self.sig_gating.get("enabled"):
            method = self.sig_gating.get("method", "pos_topk")
            k = int(self.sig_gating.get("k", 16))
            delta_threshold = float(self.sig_gating.get("delta_threshold", 0.25))
            include_self = bool(self.sig_gating.get("include_self", True))
            symmetric = bool(self.sig_gating.get("symmetric", True))
            sig_mask = bank.compute_neighbor_mask(
                method=method,
                k=k,
                delta_threshold=delta_threshold,
                include_self=include_self,
                symmetric=symmetric,
                sig=sig_for_gating,
            )

        for block in self.blocks:
            set_states = block(set_states, geom_bias, content_bias, sig_mask, seq_len)

        if isinstance(self.router, UniformRouter):
            router_out: RouterOutput = self.router(set_states, bank.token_to_sets)
        else:
            desc_router = features.desc_router
            router_out = self.router(token_states, set_states, desc_router, bank.token_to_sets)

        if self.training:
            self.diagnostics.update_with_router_state(
                bank_indices=router_out.bank_indices,
                num_sets=router_out.num_sets,
                router_probs=router_out.probs,
                set_embeddings=set_states,
                set_attention_weights=None,
            )

        return router_out.token_repr, router_out

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_repr, _ = self._encode_tokens(input_ids)
        return token_repr

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_repr, _ = self._encode_tokens(input_ids, attention_mask=attention_mask)
        return self.lm_head(token_repr)

    def get_diagnostics(self) -> dict[str, float]:
        stats = self.diagnostics.get_epoch_stats()
        self.diagnostics.reset()
        return stats
