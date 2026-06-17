from __future__ import annotations

import warnings
import math

import torch
from torch import nn

from models.baseline_token.transformer_lm import BaselineEncoderLayer
from models.baseline_token.diagnostics import BaselineAttentionDiagnostics
from models.set_only.banks import (
    InformativeBoltzmannPooling,
    build_window_bank,
    num_sets_for_length,
)
from models.set_only.diagnostics import SetDiagnostics
from models.set_only.router import LearnedRouter, RouterOutput, UniformRouter
from models.set_only.ska_block import SetAttentionBlock
from set_attention.adapter_factory import create_adapter, select_adapter_type
from set_attention.backends.dense_exact import DenseExactBackend
from set_attention.backends.landmark import LandmarkAttentionBackend
from set_attention.backends.local_band import LocalBandBackend
from set_attention.backends.sparse_topk import SparseTopKBackend
from set_attention.features.base import SetFeatures
from set_attention.features.geometry_only import GeometryOnlyFeatureBuilder
from set_attention.features.hashed_counts import HashedCountFeatureBuilder


class HybridSetLayer(nn.Module):
    """One set layer that maps a shared token stream through pooled set states."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        attn_dropout: float | None,
        resid_dropout: float | None,
        ffn_dropout: float | None,
        max_seq_len: int,
        window_size: int,
        stride: int,
        pooling: str | dict,
        pooling_multihead: bool,
        d_phi: int | None,
        set_state_dim: int | None,
        backend: str,
        backend_params: dict | None,
        feature_mode: str,
        feature_params: dict | None,
        router_type: str,
        router_topk: int,
        router_multihead: bool,
        router_temperature: float,
        router_min_temp: float,
        router_score_mode: str,
        adapter_type: str,
        adapter_hidden_multiplier: int,
        gamma: float,
        beta: float,
        causal: bool | None,
        set_causality_mode: str,
        output_residual_mode: str,
        token_embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.max_seq_len = int(max_seq_len)
        if causal is not None:
            warnings.warn(
                "HybridSetLayer(causal=...) is deprecated; set_causality_mode is the "
                "single source of truth and wins when both are supplied.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.set_causality_mode = set_causality_mode
        if self.set_causality_mode not in {"strict_past", "noncausal"}:
            raise ValueError("set_causality_mode must be strict_past or noncausal")
        self.causal = self.set_causality_mode == "strict_past"
        self.output_residual_mode = output_residual_mode
        self.token_embedding = token_embedding
        self.feature_mode = feature_mode
        self.router_multihead = bool(router_multihead)
        self.router_temperature = float(router_temperature)
        self.router_min_temp = float(router_min_temp)
        if router_score_mode not in {"candidate_gather", "dense"}:
            raise ValueError("router_score_mode must be 'candidate_gather' or 'dense'")
        self.router_score_mode = router_score_mode
        self.backend = backend
        self.backend_params = dict(backend_params or {})

        if d_phi is None:
            d_phi = d_model
        self.d_phi = int(d_phi)
        if set_state_dim is None:
            set_state_dim = d_model
        self.set_state_dim = int(set_state_dim)
        if self.set_state_dim % num_heads != 0:
            raise ValueError("set_state_dim must be divisible by num_heads")

        self.set_input_proj = (
            nn.Identity()
            if self.set_state_dim == d_model
            else nn.Linear(d_model, self.set_state_dim)
        )
        self.set_output_proj = (
            nn.Identity()
            if self.set_state_dim == d_model
            else nn.Linear(self.set_state_dim, d_model)
        )

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
            self.pooling_mode = str(pooling)
            self.pooling_params = {"alpha": 10.0, "learnable_alpha": False}
        self.resolved_pooling_alpha = float(self.pooling_params.get("alpha", 10.0))
        self.resolved_pooling_learnable_alpha = bool(
            self.pooling_params.get("learnable_alpha", False)
        )
        self.pooling_module = None
        if self.pooling_mode == "soft_trimmed_boltzmann":
            self.pooling_module = InformativeBoltzmannPooling(
                tau=float(self.pooling_params.get("tau", 0.1)),
                q=float(self.pooling_params.get("q", 0.8)),
                alpha=float(self.pooling_params.get("alpha", 10.0)),
                learnable_alpha=bool(self.pooling_params.get("learnable_alpha", False)),
                tiny_set_n=int(self.pooling_params.get("tiny_set_n", 3)),
                isotropy_eps=float(self.pooling_params.get("isotropy_eps", 1e-4)),
                pooling_multihead=bool(pooling_multihead),
                num_heads=num_heads,
            )

        max_sets = num_sets_for_length(
            max_seq_len,
            self.window_size,
            self.stride,
            causality_mode=self.set_causality_mode,
        )
        feature_params = {
            "num_bins": 128,
            "hash_seed": 13,
            "normalize": True,
            **(feature_params or {}),
        }
        self.resolved_hash_num_bins = int(feature_params.get("num_bins", 128))
        self.resolved_hash_seed = int(feature_params.get("hash_seed", 13))
        self.resolved_hash_normalize = bool(feature_params.get("normalize", True))

        if feature_mode == "geometry_only":
            self.feature_builder = GeometryOnlyFeatureBuilder(
                d_model=self.set_state_dim,
                max_sets=max_sets,
                gamma=gamma,
                beta=beta,
            )
        elif feature_mode == "hashed_counts":
            self.feature_builder = HashedCountFeatureBuilder(
                d_model=self.set_state_dim,
                d_phi=self.d_phi,
                max_sets=max_sets,
                num_bins=self.resolved_hash_num_bins,
                gamma=gamma,
                beta=beta,
                normalize=self.resolved_hash_normalize,
                hash_seed=self.resolved_hash_seed,
                fusion=feature_params.get("fusion", "mlp"),
                include_geom_in_attn=True,
            )
        else:
            raise ValueError("hybrid set layers support geometry_only or hashed_counts")
        self.feature_params = feature_params

        attn_drop = attn_dropout if attn_dropout is not None else dropout
        resid_drop = resid_dropout if resid_dropout is not None else dropout
        ffn_drop = ffn_dropout if ffn_dropout is not None else dropout

        def make_backend() -> nn.Module:
            if backend in {"exact", "dense_exact"}:
                return DenseExactBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    dropout=attn_drop,
                    allow_token_token=False,
                )
            if backend == "local_band":
                return LocalBandBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    radius=self.backend_params.get("radius", 4),
                    dropout=attn_drop,
                    allow_token_token=False,
                    global_set_indices=self.backend_params.get("global_set_indices"),
                )
            if backend == "landmark":
                return LandmarkAttentionBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    landmark_coverage=self.backend_params.get("landmark_coverage", 0.25),
                    dropout=attn_drop,
                    allow_token_token=False,
                )
            if backend == "sparse_topk":
                return SparseTopKBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    k_s=self.backend_params.get("k_s", 16),
                    dropout=attn_drop,
                    allow_token_token=False,
                )
            raise ValueError(f"Unknown hybrid set backend: {backend}")

        self.block = SetAttentionBlock(
            d_model=self.set_state_dim,
            backend=make_backend(),
            dim_feedforward=dim_feedforward,
            resid_dropout=resid_drop,
            ffn_dropout=ffn_drop,
        )

        if router_type == "uniform":
            self.router = UniformRouter()
        elif router_type == "learned":
            self.router = LearnedRouter(
                d_model=d_model,
                set_dim=self.set_state_dim,
                desc_dim=self.set_state_dim,
                num_heads=num_heads,
                d_phi=self.d_phi,
                topk=router_topk,
                multihead=router_multihead,
                min_temp=router_min_temp,
                score_mode=router_score_mode,
            )
            self.router.temperature.fill_(router_temperature)
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

        self.adapter = None
        self.resolved_adapter_type = "none"
        if feature_mode != "geometry_only":
            d_head = self.set_state_dim // num_heads
            requested = adapter_type
            if requested == "auto":
                requested = select_adapter_type(self.d_phi, d_head)
            self.resolved_adapter_type = requested
            self.adapter = create_adapter(
                adapter_type=requested,
                num_heads=num_heads,
                d_head=d_head,
                phi_dim=self.d_phi,
                hidden_multiplier=adapter_hidden_multiplier,
            )

        self.resolved_landmark_coverage = None
        self.resolved_landmark_count = None
        if backend == "landmark":
            self.resolved_landmark_coverage = float(
                self.backend_params.get("landmark_coverage", 0.25)
            )
            self.resolved_landmark_count = min(
                max(round(self.resolved_landmark_coverage * max_sets), 2),
                max_sets,
            )
        self.diagnostics = SetDiagnostics()

    def forward(
        self,
        token_states: torch.Tensor,
        input_ids: torch.Tensor,
        diagnostics: SetDiagnostics | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = token_states.shape
        bank = build_window_bank(
            seq_len=seq_len,
            window_size=self.window_size,
            stride=self.stride,
            device=token_states.device,
            causality_mode=self.set_causality_mode,
        )
        set_states = bank.pool(
            token_embeddings=token_states,
            mode=self.pooling_mode,
            params=self.pooling_params,
            pooling_module=self.pooling_module,
        )
        set_states = self.set_input_proj(set_states)

        if self.feature_mode == "geometry_only":
            features = self.feature_builder(bank.set_positions)
        else:
            per_batch = [
                self.feature_builder(input_ids[i], bank, set_states[i])
                for i in range(batch)
            ]
            features = SetFeatures(
                phi_attn=torch.stack([f.phi_attn for f in per_batch], dim=0),
                desc_router=torch.stack([f.desc_router for f in per_batch], dim=0),
                geom_bias=per_batch[0].geom_bias,
            )

        content_bias = None
        if self.adapter is not None and features.phi_attn is not None:
            content_bias = self.adapter(features.phi_attn)

        sig_mask = None
        if self.causal:
            sig_mask = bank.set_positions[:, None] >= bank.set_positions[None, :]

        set_states = self.block(
            set_states,
            features.geom_bias,
            content_bias,
            sig_mask,
            seq_len=-1,
        )

        if isinstance(self.router, UniformRouter):
            router_out: RouterOutput = self.router(set_states, bank.token_to_sets)
        else:
            desc_router = features.desc_router
            if desc_router is not None and desc_router.dim() == 2:
                desc_router = desc_router.unsqueeze(0).expand(batch, -1, -1)
            router_out = self.router(
                token_states,
                set_states,
                desc_router,
                bank.token_to_sets,
            )
        routed_repr = self.set_output_proj(router_out.token_repr)

        if self.set_causality_mode == "strict_past":
            if self.output_residual_mode == "direct":
                token_repr = token_states + routed_repr
            elif self.output_residual_mode == "empty_only":
                has_candidates = (bank.token_to_sets >= 0).any(dim=-1)
                token_repr = torch.where(
                    has_candidates.view(1, seq_len, 1),
                    routed_repr,
                    token_states,
                )
            elif self.output_residual_mode == "none":
                token_repr = routed_repr
            else:
                raise ValueError(f"Unknown output_residual_mode: {self.output_residual_mode}")
        else:
            token_repr = routed_repr

        if diagnostics is not None:
            diagnostics.update_with_router_state(
                bank_indices=router_out.bank_indices,
                num_sets=router_out.num_sets,
                router_probs=router_out.probs,
                router_prob_indices=router_out.prob_indices,
                set_embeddings=set_states,
                set_attention_weights=None,
                token_to_sets=bank.token_to_sets,
            )
            if self.pooling_module is not None:
                pooling_stats = self.pooling_module.get_last_stats()
                if pooling_stats:
                    diagnostics.update_with_pooling_stats(pooling_stats)

        return token_repr


class HybridTokenSetLM(nn.Module):
    """Layer-level token/set hybrid LM with one shared residual token stream."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        attn_dropout: float | None = None,
        resid_dropout: float | None = None,
        ffn_dropout: float | None = None,
        max_seq_len: int = 512,
        attention_family: str = "sparse",
        backend: str = "local_band",
        backend_params: dict | None = None,
        hybrid: dict | None = None,
        pooling: str | dict = "mean",
        pooling_multihead: bool = False,
        d_phi: int | None = None,
        set_state_dim: int | None = None,
        feature_mode: str = "hashed_counts",
        feature_params: dict | None = None,
        router_type: str = "learned",
        router_topk: int = 16,
        router_multihead: bool = True,
        router_temperature: float = 1.0,
        router_min_temp: float = 0.5,
        router_score_mode: str = "candidate_gather",
        adapter_type: str = "auto",
        adapter_hidden_multiplier: int = 2,
        gamma: float = 1.0,
        beta: float = 0.0,
        causal: bool | None = None,
        set_causality_mode: str = "strict_past",
        output_residual_mode: str = "empty_only",
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = int(max_seq_len)
        if causal is not None:
            warnings.warn(
                "HybridTokenSetLM(causal=...) is deprecated; set_causality_mode is the "
                "single source of truth and wins when both are supplied.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.attention_family = attention_family
        self.backend = backend
        self.backend_params = dict(backend_params or {})
        self.hybrid_cfg = hybrid or {}
        self.pattern = str(self.hybrid_cfg.get("pattern", "TTSSSS")).upper()
        if len(self.pattern) != num_layers:
            raise ValueError("hybrid.pattern length must equal num_layers")
        if any(ch not in {"T", "S"} for ch in self.pattern):
            raise ValueError("hybrid.pattern may contain only T and S")

        set_topologies = self.hybrid_cfg.get("set_topologies")
        if set_topologies is None:
            set_topologies = [{"window_size": 4, "stride": 2} for _ in self.pattern if _ == "S"]
        if len(set_topologies) != self.pattern.count("S"):
            raise ValueError("hybrid.set_topologies must match number of S layers")

        mode_aliases = {
            "option1": "strict_past",
            "end_aligned": "strict_past",
            "causal": "strict_past",
            "bidirectional": "noncausal",
        }
        self.set_causality_mode = mode_aliases.get(set_causality_mode, set_causality_mode)
        if self.set_causality_mode not in {"strict_past", "noncausal"}:
            raise ValueError("set_causality_mode must be strict_past or noncausal")
        self.causal = self.set_causality_mode == "strict_past"
        self.output_residual_mode = output_residual_mode
        if self.output_residual_mode not in {"direct", "empty_only", "none"}:
            raise ValueError("output_residual_mode must be direct, empty_only, or none")

        self.layers = nn.ModuleList()
        self.layer_kinds: list[str] = []
        self.set_layer_topologies: list[dict[str, int]] = []
        set_idx = 0
        for kind in self.pattern:
            if kind == "T":
                self.layers.append(
                    BaselineEncoderLayer(
                        d_model=d_model,
                        nhead=num_heads,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        resid_dropout=resid_dropout,
                        ffn_dropout=ffn_dropout,
                        attention_family=attention_family,
                        backend=backend,
                        backend_params=backend_params,
                        max_seq_len=max_seq_len,
                        causal=self.causal,
                    )
                )
                self.layer_kinds.append("T")
                continue
            topo = set_topologies[set_idx]
            window_size = int(topo.get("window_size", topo.get("w")))
            stride = int(topo.get("stride", topo.get("s")))
            self.layers.append(
                HybridSetLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    ffn_dropout=ffn_dropout,
                    max_seq_len=max_seq_len,
                    window_size=window_size,
                    stride=stride,
                    pooling=pooling,
                    pooling_multihead=pooling_multihead,
                    d_phi=d_phi,
                    set_state_dim=set_state_dim,
                    backend=backend,
                    backend_params=backend_params,
                    feature_mode=feature_mode,
                    feature_params=feature_params,
                    router_type=router_type,
                    router_topk=router_topk,
                    router_multihead=router_multihead,
                    router_temperature=router_temperature,
                    router_min_temp=router_min_temp,
                    router_score_mode=router_score_mode,
                    adapter_type=adapter_type,
                    adapter_hidden_multiplier=adapter_hidden_multiplier,
                    gamma=gamma,
                    beta=beta,
                    causal=None,
                    set_causality_mode=self.set_causality_mode,
                    output_residual_mode=self.output_residual_mode,
                    token_embedding=self.token_emb,
                )
            )
            self.layer_kinds.append("S")
            self.set_layer_topologies.append({"window_size": window_size, "stride": stride})
            set_idx += 1

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.baseline_diagnostics = BaselineAttentionDiagnostics()
        self.set_diagnostics = SetDiagnostics()
        if self.pattern.startswith("S"):
            warnings.warn(
                "Hybrid pattern starts with a set layer; early compression may discard fine-grained token information.",
                RuntimeWarning,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [batch, seq]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(batch_size, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError("attention_mask must be [batch, seq]")
            key_padding_mask = attention_mask == 0

        attn_sum = None
        token_layer_count = 0
        for kind, layer in zip(self.layer_kinds, self.layers):
            if kind == "T":
                x, attn = layer(x, key_padding_mask=key_padding_mask)
                if attn is not None:
                    attn_sum = attn if attn_sum is None else attn_sum + attn
                    token_layer_count += 1
            else:
                x = layer(
                    x,
                    input_ids,
                    diagnostics=layer.diagnostics if self.training else None,
                )

        if self.training and attn_sum is not None and token_layer_count:
            self.baseline_diagnostics.update((attn_sum / token_layer_count).detach())
        return self.lm_head(x)

    def get_diagnostics(self) -> dict[str, float]:
        stats = {}
        set_layer_stats: list[dict[str, float]] = []
        for kind, layer in zip(self.layer_kinds, self.layers):
            if kind == "S":
                layer_stats = layer.diagnostics.get_epoch_stats()
                set_layer_stats.append(layer_stats)
                layer.diagnostics.reset()
        for key in sorted({k for layer_stats in set_layer_stats for k in layer_stats}):
            values = [
                float(layer_stats[key])
                for layer_stats in set_layer_stats
                if key in layer_stats and math.isfinite(float(layer_stats[key]))
            ]
            if values:
                stats[key] = sum(values) / len(values)
        stats.update(self.baseline_diagnostics.get_epoch_stats())
        self.baseline_diagnostics.reset()
        return stats

    def get_resolved_metadata(self) -> dict[str, object]:
        set_layers = [layer for kind, layer in zip(self.layer_kinds, self.layers) if kind == "S"]
        landmark_counts = [
            layer.resolved_landmark_count
            for layer in set_layers
            if getattr(layer, "resolved_landmark_count", None) is not None
        ]
        first_set = set_layers[0] if set_layers else None
        return {
            "d_phi": getattr(first_set, "d_phi", "NA") if first_set is not None else "NA",
            "set_state_dim": getattr(first_set, "set_state_dim", "NA") if first_set is not None else "NA",
            "adapter_type": (
                getattr(first_set, "resolved_adapter_type", "NA")
                if first_set is not None
                else "NA"
            ),
            "router_min_temp": (
                getattr(first_set, "router_min_temp", "NA")
                if first_set is not None
                else "NA"
            ),
            "router_score_mode": (
                getattr(first_set, "router_score_mode", "NA")
                if first_set is not None
                else "NA"
            ),
            "pooling_alpha": (
                getattr(first_set, "resolved_pooling_alpha", "NA")
                if first_set is not None
                else "NA"
            ),
            "hash_seed": (
                getattr(first_set, "resolved_hash_seed", "NA")
                if first_set is not None
                else "NA"
            ),
            "hash_normalize": (
                getattr(first_set, "resolved_hash_normalize", "NA")
                if first_set is not None
                else "NA"
            ),
            "hash_num_bins": (
                getattr(first_set, "resolved_hash_num_bins", "NA")
                if first_set is not None
                else "NA"
            ),
            "landmark_coverage": (
                getattr(first_set, "resolved_landmark_coverage", "NA")
                if first_set is not None and getattr(first_set, "resolved_landmark_coverage", None) is not None
                else "NA"
            ),
            "landmark_count": ",".join(str(v) for v in landmark_counts)
            if landmark_counts
            else "NA",
            "output_residual_mode": self.output_residual_mode,
            "hybrid_pattern": self.pattern,
            "hybrid_set_topologies": ";".join(
                f"{t['window_size']}:{t['stride']}" for t in self.set_layer_topologies
            ),
        }
