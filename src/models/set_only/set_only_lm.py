from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn
import warnings

from .banks import build_window_bank, num_sets_for_length
from .diagnostics import SetDiagnostics
from .router import LearnedRouter, UniformRouter, RouterOutput
from .ska_block import SetAttentionBlock
from set_attention.adapter_factory import create_adapter, select_adapter_type
from set_attention.backends.dense_exact import DenseExactBackend
from set_attention.backends.linformer import LinformerBackend
from set_attention.backends.landmark import LandmarkAttentionBackend
from set_attention.backends.local_band import LocalBandBackend
from set_attention.backends.nystrom import NystromBackend
from set_attention.backends.sparse_topk import SparseTopKBackend
from set_attention.features.base import SetFeatures
from set_attention.features.geometry_only import GeometryOnlyFeatureBuilder
from set_attention.features.hashed_counts import HashedCountFeatureBuilder
from set_attention.features.kernel_features import KernelFeatureBuilder
from set_attention.minhash import minhash_signatures


class CausalPreEncoder(nn.Module):
    """Shallow causal token pre-encoder used only for the anchor target."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        pre_encoder_head: bool,
    ) -> None:
        super().__init__()
        self.pre_encoder_head = bool(pre_encoder_head)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = (
            nn.Linear(d_model, vocab_size, bias=False)
            if self.pre_encoder_head
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        return x

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("anchor.pre_encoder_head=true is required for L_CE_pre")
        return self.lm_head(hidden)


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
        attn_dropout: float | None = None,
        resid_dropout: float | None = None,
        ffn_dropout: float | None = None,
        max_seq_len: int = 512,
        dim_feedforward: int | None = None,
        pooling: str = "mean",
        pooling_multihead: bool = False,
        multiscale: bool = False,
        sig_gating: dict | None = None,
        d_phi: int | None = None,
        set_state_dim: int | None = None,
        geometry: dict | None = None,
        features: dict | None = None,
        router_type: str = "uniform",
        router_topk: int = 0,
        router_multihead: bool = False,
        router_temperature: float = 1.0,
        router_min_temp: float = 0.5,
        router_score_mode: str = "candidate_gather",
        backend: str = "exact",
        backend_params: dict | None = None,
        feature_mode: str = "geometry_only",
        feature_params: dict | None = None,
        token_mlp: bool | dict | None = None,
        adapter_type: str = "auto",
        adapter_hidden_multiplier: int = 2,
        adapter_budget_fraction: float = 0.15,
        gamma: float = 1.0,
        beta: float = 0.0,
        token_embedding: nn.Embedding | None = None,
        allow_token_token: bool = False,
        causal: bool | None = None,
        set_causality_mode: str | None = None,
        output_residual_mode: str = "direct",
        anchor: dict | None = None,
        set_diversity: dict | None = None,
        multivector_basis: dict | None = None,
        candidate_fiber: str = "endpoint_window",
        multiresolution: dict | None = None,
    ) -> None:
        super().__init__()
        self.token_emb = token_embedding or nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.d_model = int(d_model)
        if isinstance(token_mlp, dict):
            token_mlp_enabled = bool(token_mlp.get("enabled", True))
        elif token_mlp is None:
            token_mlp_enabled = True
        else:
            token_mlp_enabled = bool(token_mlp)
        self.token_mlp_enabled = token_mlp_enabled
        self.token_mlp = (
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.token_mlp_enabled
            else nn.Identity()
        )
        self.grad_probe_interval = 200
        self._forward_step = 0
        self._grad_probe_active = False
        self._grad_probe_tensors: dict[str, object] = {}
        self._last_set_embeddings: torch.Tensor | None = None
        self.window_size = window_size
        self.stride = stride
        self.max_seq_len = max_seq_len
        self.attn_dropout = attn_dropout if attn_dropout is not None else dropout
        self.resid_dropout = resid_dropout if resid_dropout is not None else dropout
        self.ffn_dropout = ffn_dropout if ffn_dropout is not None else dropout
        self.router_multihead = bool(router_multihead)
        self.router_temperature = float(router_temperature)
        self.router_min_temp = float(router_min_temp)
        if router_score_mode not in {"candidate_gather", "dense"}:
            raise ValueError("router_score_mode must be 'candidate_gather' or 'dense'")
        self.router_score_mode = router_score_mode
        self.pooling_multihead = bool(pooling_multihead)
        self.allow_token_token = bool(allow_token_token)
        if causal is not None:
            warnings.warn(
                "SetOnlyLM(causal=...) is deprecated; set_causality_mode is the "
                "single source of truth and wins when both are supplied.",
                DeprecationWarning,
                stacklevel=2,
            )
        mode = set_causality_mode
        if mode is None:
            mode = "strict_past" if (causal is None or bool(causal)) else "noncausal"
        mode_aliases = {
            "option1": "strict_past",
            "end_aligned": "strict_past",
            "causal": "strict_past",
            "bidirectional": "noncausal",
        }
        self.set_causality_mode = mode_aliases.get(mode, mode)
        if self.set_causality_mode not in {"strict_past", "noncausal"}:
            raise ValueError(
                "set_causality_mode must be 'strict_past' or 'noncausal'"
            )
        self.causal = self.set_causality_mode == "strict_past"
        output_residual_mode = str(output_residual_mode)
        if output_residual_mode not in {"direct", "empty_only", "none", "anchor_span"}:
            raise ValueError(
                "output_residual_mode must be 'direct', 'empty_only', 'none', or 'anchor_span'"
            )
        self.output_residual_mode = output_residual_mode
        self.candidate_fiber = str(candidate_fiber)
        if self.candidate_fiber not in {"endpoint_window", "all_past"}:
            raise ValueError("candidate_fiber must be 'endpoint_window' or 'all_past'")
        self.set_diversity_cfg = {"lambda_div": 0.0, **(set_diversity or {})}
        self.multivector_basis_cfg = {
            "enabled": False,
            "r": 1,
            **(multivector_basis or {}),
        }
        if bool(self.multivector_basis_cfg.get("enabled", False)):
            raise ValueError("multivector_basis is deferred; keep enabled=false")
        if int(self.multivector_basis_cfg.get("r", 1)) != 1:
            raise ValueError("multivector_basis.r must stay 1 unless enabled in a later gate")
        self.multiresolution_cfg = {
            "enabled": False,
            "groups": [],
            **(multiresolution or {}),
        }
        self.multiresolution_enabled = bool(
            self.multiresolution_cfg.get("enabled", False)
        )
        anchor_cfg = anchor or {}
        teacher_cfg = dict(anchor_cfg.get("teacher", {}) or {})
        self.anchor_cfg = {
            "enabled": False,
            "target": "pre_encoder",
            "pre_encoder_layers": 2,
            "lambda_h": 0.1,
            "lambda_pre": 1.0,
            "pre_encoder_head": True,
            "detach_target": True,
            "norm": "layernorm",
            "teacher": {"enabled": False, **teacher_cfg},
            **{k: v for k, v in anchor_cfg.items() if k != "teacher"},
        }
        if self.anchor_cfg["target"] != "pre_encoder":
            raise ValueError("anchor.target must be 'pre_encoder'")
        if bool(self.anchor_cfg["teacher"].get("enabled", False)):
            raise ValueError("anchor.teacher.enabled is deferred and must stay false")
        self.anchor_enabled = bool(self.anchor_cfg.get("enabled", False))
        self.anchor_lambda_h = float(self.anchor_cfg.get("lambda_h", 0.1))
        self.anchor_lambda_pre = float(self.anchor_cfg.get("lambda_pre", 1.0))
        self.anchor_pre_encoder_head_enabled = bool(
            self.anchor_cfg.get("pre_encoder_head", True)
        )
        if self.anchor_enabled and self.anchor_lambda_pre <= 0.0:
            raise ValueError("anchor.lambda_pre must be > 0 when anchor.enabled=true")
        if self.anchor_enabled and not self.anchor_pre_encoder_head_enabled:
            raise ValueError("anchor.pre_encoder_head must be true when anchor.enabled=true")
        self.anchor_detach_target = bool(self.anchor_cfg.get("detach_target", True))
        self.anchor_norm_mode = str(self.anchor_cfg.get("norm", "layernorm"))
        if self.anchor_norm_mode != "layernorm":
            raise ValueError("anchor.norm must be 'layernorm'")
        self.anchor_pre_encoder_layers = int(self.anchor_cfg.get("pre_encoder_layers", 2))
        if self.anchor_pre_encoder_layers not in {1, 2}:
            raise ValueError("anchor.pre_encoder_layers must be 1 or 2")
        self.anchor_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.anchor_pre_encoder = (
            CausalPreEncoder(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward or d_model * 4,
                num_layers=self.anchor_pre_encoder_layers,
                dropout=dropout,
                pre_encoder_head=self.anchor_pre_encoder_head_enabled,
            )
            if self.anchor_enabled
            else None
        )
        self._last_aux_losses: dict[str, torch.Tensor] = {}
        self._last_aux_metrics: dict[str, float] = {}
        self.span_ablation_enabled = False
        self.span_ablation_mode = "none"
        self._probe_metric_sums: dict[str, float] = {}
        self._probe_metric_counts: dict[str, float] = {}
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
            self.pooling_params = {
                "alpha": 10.0,
                "learnable_alpha": False,
            }
        self.resolved_pooling_alpha = float(self.pooling_params.get("alpha", 10.0))
        self.resolved_pooling_learnable_alpha = bool(
            self.pooling_params.get("learnable_alpha", False)
        )
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
                pooling_multihead=self.pooling_multihead,
                num_heads=num_heads,
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
        if self.allow_token_token:
            warnings.warn(
                "Set-only guard disabled; token-token attention is allowed. "
                "Use with care for singleton-set experiments.",
                RuntimeWarning,
            )

        max_sets = num_sets_for_length(
            max_seq_len,
            window_size,
            stride,
            causality_mode=self.set_causality_mode,
        )
        if d_phi is None:
            d_phi = d_model
        self.d_phi = d_phi
        if set_state_dim is None:
            set_state_dim = d_model
        self.set_state_dim = int(set_state_dim)
        if self.set_state_dim <= 0:
            raise ValueError("set_state_dim must be > 0")
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
        self.multiresolution_streams = nn.ModuleList()
        self.multiresolution_group_metadata: list[dict[str, object]] = []
        feature_params = feature_params or {}
        features_cfg = features or {}
        if isinstance(features_cfg, dict) and feature_mode in features_cfg:
            mode_cfg = features_cfg.get(feature_mode, {})
            if isinstance(mode_cfg, dict):
                feature_params = {**feature_params, **mode_cfg}
        feature_params = {
            "num_bins": 128,
            "hash_seed": 13,
            "normalize": True,
            **feature_params,
        }
        self.resolved_hash_num_bins = int(feature_params.get("num_bins", 128))
        self.resolved_hash_seed = int(feature_params.get("hash_seed", 13))
        self.resolved_hash_normalize = bool(feature_params.get("normalize", True))
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

        def make_feature_builder(
            *,
            stream_dim: int,
            stream_d_phi: int,
            stream_max_sets: int,
        ) -> nn.Module:
            if feature_mode == "geometry_only":
                return GeometryOnlyFeatureBuilder(
                    d_model=stream_dim,
                    max_sets=stream_max_sets,
                    gamma=gamma,
                    beta=beta,
                )
            if feature_mode == "hashed_counts":
                return HashedCountFeatureBuilder(
                    d_model=stream_dim,
                    d_phi=stream_d_phi,
                    max_sets=stream_max_sets,
                    num_bins=self.resolved_hash_num_bins,
                    gamma=gamma,
                    beta=beta,
                    normalize=self.resolved_hash_normalize,
                    hash_seed=self.resolved_hash_seed,
                    fusion=feature_params.get("fusion", "mlp"),
                    include_geom_in_attn=geom_apply_in_phi,
                )
            if feature_mode == "kernel":
                return KernelFeatureBuilder(
                    d_model=stream_dim,
                    d_phi=stream_d_phi,
                    max_sets=stream_max_sets,
                    gamma=gamma,
                    beta=beta,
                )
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        self.feature_builder = None
        if not self.multiresolution_enabled:
            self.feature_builder = make_feature_builder(
                stream_dim=self.set_state_dim,
                stream_d_phi=self.d_phi,
                stream_max_sets=max_sets,
            )
        self.feature_mode = feature_mode
        self.feature_params = feature_params

        if self.multiscale:
            raise ValueError("multiscale is not implemented in SetOnlyLM")

        print(
            {
                "pooling": {"mode": self.pooling_mode, **self.pooling_params},
                "pooling_multihead": self.pooling_multihead,
                "sig_gating": self.sig_gating,
                "d_phi": self.d_phi,
                "set_state_dim": self.set_state_dim,
                "geometry": {
                    "enabled": self.geom_enabled,
                    "apply_as_bias": self.geom_apply_bias,
                    "apply_in_phi_attn": self.geom_apply_in_phi,
                },
                "router_multihead": self.router_multihead,
                "router_temperature": self.router_temperature,
                "router_min_temp": self.router_min_temp,
                "router_score_mode": self.router_score_mode,
                "token_mlp": {"enabled": self.token_mlp_enabled},
                "set_causality_mode": self.set_causality_mode,
                "output_residual_mode": self.output_residual_mode,
                "anchor": {
                    "enabled": self.anchor_enabled,
                    "target": self.anchor_cfg["target"],
                    "pre_encoder_layers": self.anchor_pre_encoder_layers,
                    "lambda_h": self.anchor_lambda_h,
                    "lambda_pre": self.anchor_lambda_pre,
                    "pre_encoder_head": self.anchor_pre_encoder_head_enabled,
                },
                "candidate_fiber": self.candidate_fiber,
                "multiresolution": {
                    "enabled": self.multiresolution_enabled,
                    "groups": self.multiresolution_cfg.get("groups", []),
                },
            }
        )

        backend_params = backend_params or {}
        self.backend = backend
        self.backend_params = dict(backend_params)
        self.resolved_landmark_coverage = None
        self.resolved_landmark_count = None
        if backend == "landmark":
            self.resolved_landmark_coverage = float(
                backend_params.get("landmark_coverage", 0.25)
            )
            self.resolved_landmark_count = min(
                max(round(self.resolved_landmark_coverage * max_sets), 2),
                max_sets,
            )

        def make_backend_for(stream_dim: int, stream_heads: int, stream_max_sets: int) -> nn.Module:
            if backend in {"exact", "dense_exact"}:
                return DenseExactBackend(
                    d_model=stream_dim,
                    num_heads=stream_heads,
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                )
            if backend == "local_band":
                return LocalBandBackend(
                    d_model=stream_dim,
                    num_heads=stream_heads,
                    radius=backend_params.get("radius", 4),
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                    global_set_indices=backend_params.get("global_set_indices"),
                )
            if backend == "nystrom":
                return NystromBackend(
                    d_model=stream_dim,
                    num_heads=stream_heads,
                    num_landmarks=backend_params.get("num_landmarks", 32),
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                    bias_scale=backend_params.get("bias_scale", 0.1),
                )
            if backend == "landmark":
                return LandmarkAttentionBackend(
                    d_model=stream_dim,
                    num_heads=stream_heads,
                    landmark_coverage=backend_params.get("landmark_coverage", 0.25),
                    num_landmarks=backend_params.get("num_landmarks"),  # fixed k -> genuinely O(M*k); takes precedence over coverage
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                )
            if backend == "sparse_topk":
                return SparseTopKBackend(
                    d_model=stream_dim,
                    num_heads=stream_heads,
                    k_s=backend_params.get("k_s", 16),
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                )
            if backend == "linformer":
                return LinformerBackend(
                    d_model=stream_dim,
                    num_heads=stream_heads,
                    max_sets=stream_max_sets,
                    k=backend_params.get("k", 32),
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                )
            raise ValueError(f"Unknown backend: {backend}")

        self.blocks = nn.ModuleList()
        self.router: nn.Module | None = None
        self.adapter = None
        self.adapter_type_requested = adapter_type
        self.resolved_adapter_type = "none"
        if self.multiresolution_enabled:
            self._build_multiresolution_streams(
                groups=self.multiresolution_cfg.get("groups", []),
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                router_type=router_type,
                router_topk=router_topk,
                adapter_type=adapter_type,
                adapter_hidden_multiplier=adapter_hidden_multiplier,
                make_backend_for=make_backend_for,
                make_feature_builder=make_feature_builder,
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    SetAttentionBlock(
                        d_model=self.set_state_dim,
                        backend=make_backend_for(self.set_state_dim, num_heads, max_sets),
                        dim_feedforward=dim_feedforward,
                        resid_dropout=self.resid_dropout,
                        ffn_dropout=self.ffn_dropout,
                    )
                    for _ in range(num_layers)
                ]
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
                    multihead=self.router_multihead,
                    min_temp=self.router_min_temp,
                    score_mode=self.router_score_mode,
                )
                self.router.temperature.fill_(self.router_temperature)
            else:
                raise ValueError(f"Unknown router_type: {router_type}")

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.diagnostics = SetDiagnostics()
        self.multiresolution_diagnostics: dict[str, SetDiagnostics] = {
            str(meta["name"]): SetDiagnostics()
            for meta in self.multiresolution_group_metadata
        }
        if feature_mode != "geometry_only" and not self.multiresolution_enabled:
            phi_dim = d_phi
            d_head = self.set_state_dim // num_heads
            if adapter_type == "auto":
                adapter_type = select_adapter_type(phi_dim, d_head)
            self.resolved_adapter_type = adapter_type
            self.adapter = create_adapter(
                adapter_type=adapter_type,
                num_heads=num_heads,
                d_head=d_head,
                phi_dim=phi_dim,
                hidden_multiplier=adapter_hidden_multiplier,
            )

    def _build_multiresolution_streams(
        self,
        *,
        groups: list[dict],
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | None,
        router_type: str,
        router_topk: int,
        adapter_type: str,
        adapter_hidden_multiplier: int,
        make_backend_for,
        make_feature_builder,
    ) -> None:
        if not groups:
            raise ValueError("multiresolution.groups must be non-empty when enabled")
        if sum(int(group["num_heads"]) for group in groups) != num_heads:
            raise ValueError("multiresolution group head counts must sum to num_heads")

        resolved_adapter_types: list[str] = []
        for idx, group in enumerate(groups):
            group_heads = int(group["num_heads"])
            window_size = int(group.get("window_size", group.get("w")))
            stride = int(group.get("stride", group.get("s")))
            name = str(group.get("name", f"group{idx}"))
            stream_dim = (self.set_state_dim * group_heads) // num_heads
            stream_d_phi = (int(self.d_phi) * group_heads) // num_heads
            if stream_dim <= 0 or stream_dim % group_heads != 0:
                raise ValueError(
                    f"multiresolution group {name!r} has invalid stream_dim={stream_dim}"
                )
            if stream_d_phi <= 0:
                raise ValueError(
                    f"multiresolution group {name!r} has invalid d_phi={stream_d_phi}"
                )
            max_sets = num_sets_for_length(
                self.max_seq_len,
                window_size,
                stride,
                causality_mode=self.set_causality_mode,
            )
            if max_sets <= 0:
                raise ValueError(
                    f"multiresolution group {name!r} creates no sets at max_seq_len"
                )
            set_input_proj: nn.Module = (
                nn.Identity()
                if stream_dim == self.d_model
                else nn.Linear(self.d_model, stream_dim)
            )
            pooling_module: nn.Module = nn.Identity()
            if self.pooling_mode == "soft_trimmed_boltzmann":
                from .banks import InformativeBoltzmannPooling

                pooling_module = InformativeBoltzmannPooling(
                    tau=float(self.pooling_params.get("tau", 0.1)),
                    q=float(self.pooling_params.get("q", 0.8)),
                    alpha=float(self.pooling_params.get("alpha", 10.0)),
                    learnable_alpha=bool(
                        self.pooling_params.get("learnable_alpha", False)
                    ),
                    tiny_set_n=int(self.pooling_params.get("tiny_set_n", 3)),
                    isotropy_eps=float(self.pooling_params.get("isotropy_eps", 1e-4)),
                    pooling_multihead=self.pooling_multihead,
                    num_heads=group_heads,
                )
            blocks = nn.ModuleList(
                [
                    SetAttentionBlock(
                        d_model=stream_dim,
                        backend=make_backend_for(stream_dim, group_heads, max_sets),
                        dim_feedforward=dim_feedforward,
                        resid_dropout=self.resid_dropout,
                        ffn_dropout=self.ffn_dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            if router_type == "uniform":
                router: nn.Module = UniformRouter()
            elif router_type == "learned":
                router = LearnedRouter(
                    d_model=self.d_model,
                    set_dim=stream_dim,
                    desc_dim=stream_dim,
                    num_heads=group_heads,
                    d_phi=stream_d_phi,
                    topk=router_topk,
                    multihead=self.router_multihead,
                    min_temp=self.router_min_temp,
                    score_mode=self.router_score_mode,
                )
                router.temperature.fill_(self.router_temperature)
            else:
                raise ValueError(f"Unknown router_type: {router_type}")

            adapter: nn.Module = nn.Identity()
            stream_adapter_type = "none"
            if self.feature_mode != "geometry_only":
                stream_adapter_type = adapter_type
                if stream_adapter_type == "auto":
                    stream_adapter_type = select_adapter_type(
                        stream_d_phi,
                        stream_dim // group_heads,
                    )
                adapter = create_adapter(
                    adapter_type=stream_adapter_type,
                    num_heads=group_heads,
                    d_head=stream_dim // group_heads,
                    phi_dim=stream_d_phi,
                    hidden_multiplier=adapter_hidden_multiplier,
                )
                resolved_adapter_types.append(f"{name}:{stream_adapter_type}")

            self.multiresolution_streams.append(
                nn.ModuleDict(
                    {
                        "set_input_proj": set_input_proj,
                        "pooling_module": pooling_module,
                        "feature_builder": make_feature_builder(
                            stream_dim=stream_dim,
                            stream_d_phi=stream_d_phi,
                            stream_max_sets=max_sets,
                        ),
                        "blocks": blocks,
                        "router": router,
                        "adapter": adapter,
                    }
                )
            )
            landmark_count = "NA"
            if self.backend == "landmark":
                coverage = float(self.backend_params.get("landmark_coverage", 0.25))
                landmark_count = min(max(round(coverage * max_sets), 2), max_sets)
            self.multiresolution_group_metadata.append(
                {
                    "name": name,
                    "num_heads": group_heads,
                    "window_size": window_size,
                    "stride": stride,
                    "set_state_dim": stream_dim,
                    "d_phi": stream_d_phi,
                    "M": max_sets,
                    "landmark_count": landmark_count,
                }
            )
        if resolved_adapter_types:
            self.resolved_adapter_type = ",".join(resolved_adapter_types)

    def get_resolved_metadata(self) -> dict[str, object]:
        return {
            "d_phi": self.d_phi,
            "set_state_dim": self.set_state_dim,
            "adapter_type": self.resolved_adapter_type,
            "router_min_temp": self.router_min_temp,
            "router_score_mode": self.router_score_mode,
            "pooling_alpha": self.resolved_pooling_alpha,
            "hash_seed": self.resolved_hash_seed,
            "hash_normalize": self.resolved_hash_normalize,
            "hash_num_bins": self.resolved_hash_num_bins,
            "landmark_coverage": (
                self.resolved_landmark_coverage
                if self.resolved_landmark_coverage is not None
                else "NA"
            ),
            "landmark_count": (
                self.resolved_landmark_count
                if self.resolved_landmark_count is not None
                else "NA"
            ),
            "output_residual_mode": self.output_residual_mode,
            "anchor_enabled": self.anchor_enabled,
            "anchor_target": self.anchor_cfg["target"],
            "anchor_pre_encoder_layers": (
                self.anchor_pre_encoder_layers if self.anchor_enabled else 0
            ),
            "anchor_lambda_h": self.anchor_lambda_h,
            "anchor_lambda_pre": self.anchor_lambda_pre,
            "anchor_pre_encoder_head": self.anchor_pre_encoder_head_enabled,
            "anchor_detach_target": self.anchor_detach_target,
            "anchor_norm": self.anchor_norm_mode,
            "anchor_teacher_enabled": bool(
                self.anchor_cfg["teacher"].get("enabled", False)
            ),
            "set_diversity_lambda_div": float(
                self.set_diversity_cfg.get("lambda_div", 0.0)
            ),
            "multivector_basis_enabled": bool(
                self.multivector_basis_cfg.get("enabled", False)
            ),
            "multivector_basis_r": int(self.multivector_basis_cfg.get("r", 1)),
            "candidate_fiber": self.candidate_fiber,
            "multiresolution_enabled": self.multiresolution_enabled,
            "multiresolution_groups": self.multiresolution_group_metadata,
            "multiresolution_num_groups": len(self.multiresolution_group_metadata),
        }

    def _thin_anchor(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(batch, seq_len)
        return self.token_emb(input_ids) + self.pos_emb(pos_ids)

    def compute_anchor_target(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.anchor_pre_encoder is None:
            raise RuntimeError("anchor pre-encoder is only constructed when anchor.enabled=true")
        return self.anchor_pre_encoder(self._thin_anchor(input_ids))

    def compute_anchor_pre_encoder_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.anchor_pre_encoder is None:
            raise RuntimeError("anchor pre-encoder is only constructed when anchor.enabled=true")
        return self.anchor_pre_encoder.logits(self.compute_anchor_target(input_ids))

    def _update_anchor_loss(
        self,
        span_repr: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> None:
        self._last_aux_losses = {}
        self._last_aux_metrics = {}
        if not self.training or not self.anchor_enabled or self.anchor_pre_encoder is None:
            return
        target = self.compute_anchor_target(input_ids)
        if self.anchor_lambda_pre > 0.0:
            if labels is None:
                raise RuntimeError(
                    "anchor.lambda_pre>0 requires labels during training so "
                    "CausalPreEncoder receives L_CE_pre gradients"
                )
            pre_logits = self.anchor_pre_encoder.logits(target)
            pre_ce = F.cross_entropy(
                pre_logits.reshape(-1, pre_logits.size(-1)),
                labels.reshape(-1),
            )
        else:
            pre_ce = None
        if self.anchor_detach_target:
            target = target.detach()
        span_norm = self.anchor_norm(span_repr)
        target_norm = self.anchor_norm(target)
        anchor_mse = F.mse_loss(span_norm, target_norm)
        diff_norm = (span_norm - target_norm).norm()
        target_norm_value = target_norm.norm().clamp_min(1e-12)
        recon_error = diff_norm / target_norm_value
        self._last_aux_losses = {"anchor_loss": self.anchor_lambda_h * anchor_mse}
        if pre_ce is not None:
            self._last_aux_losses["anchor_pre_ce_loss"] = self.anchor_lambda_pre * pre_ce
            self._last_aux_losses["anchor_pre_ce"] = pre_ce.detach()
        self._last_aux_losses["anchor_mse"] = anchor_mse.detach()
        self._last_aux_metrics = {
            "anchor/lambda_h": self.anchor_lambda_h,
            "anchor/lambda_pre": self.anchor_lambda_pre,
            "anchor/mse": float(anchor_mse.detach().item()),
            "anchor/recon_error_norm": float(recon_error.detach().item()),
        }
        if pre_ce is not None:
            self._last_aux_metrics["anchor/pre_ce"] = float(pre_ce.detach().item())

    def get_auxiliary_losses(self) -> dict[str, torch.Tensor]:
        return dict(self._last_aux_losses)

    def get_auxiliary_metrics(self) -> dict[str, float]:
        return dict(self._last_aux_metrics)

    def set_span_ablation(self, enabled: bool = True) -> None:
        self.span_ablation_enabled = bool(enabled)
        self.span_ablation_mode = "all" if enabled else "none"

    def set_span_ablation_mode(self, mode: str = "none") -> None:
        valid_modes = {"none", "all"}
        valid_modes.update(str(m["name"]) for m in self.multiresolution_group_metadata)
        if mode not in valid_modes:
            raise ValueError(
                f"span ablation mode must be one of {sorted(valid_modes)}, got {mode!r}"
            )
        self.span_ablation_mode = mode
        self.span_ablation_enabled = mode == "all"

    def reset_probe_metrics(self) -> None:
        self._probe_metric_sums = {}
        self._probe_metric_counts = {}

    def get_probe_metrics(self, reset: bool = True) -> dict[str, float]:
        metrics = {
            key: self._probe_metric_sums[key] / max(self._probe_metric_counts.get(key, 0.0), 1.0)
            for key in self._probe_metric_sums
        }
        if reset:
            self.reset_probe_metrics()
        return metrics

    def _accumulate_probe_metric(self, key: str, value: torch.Tensor, count: torch.Tensor) -> None:
        count_f = float(count.detach().item())
        if count_f <= 0.0:
            return
        self._probe_metric_sums[key] = self._probe_metric_sums.get(key, 0.0) + (
            float(value.detach().item()) * count_f
        )
        self._probe_metric_counts[key] = self._probe_metric_counts.get(key, 0.0) + count_f

    def _update_effective_range_probe(
        self,
        *,
        group_name: str,
        bank,
        router_out: RouterOutput,
    ) -> None:
        probs = router_out.probs
        if probs is None or probs.numel() == 0:
            return
        with torch.no_grad():
            if probs.dim() == 3:
                probs4 = probs.unsqueeze(1)  # [B,1,T,C]
            elif probs.dim() == 4:
                probs4 = probs
            else:
                return
            _, _, seq_len, cand = probs4.shape
            prob_indices = router_out.prob_indices
            if prob_indices is None:
                idx = torch.arange(cand, device=probs4.device).view(1, cand).expand(seq_len, cand)
            else:
                idx = prob_indices.to(probs4.device)
                if idx.shape != (seq_len, cand):
                    return
            valid = idx >= 0
            if not valid.any():
                return
            centers = (
                bank.set_starts.to(probs4.device, dtype=torch.float32)
                + bank.set_endpoints.to(probs4.device, dtype=torch.float32)
            ) * 0.5
            idx_safe = idx.clamp(min=0, max=max(int(centers.numel()) - 1, 0))
            center_tc = centers.index_select(0, idx_safe.reshape(-1)).view(seq_len, cand)
            token_pos = torch.arange(seq_len, device=probs4.device, dtype=torch.float32).view(seq_len, 1)
            distance = (token_pos - center_tc).abs()
            valid4 = valid.view(1, 1, seq_len, cand)
            p = probs4.detach().masked_fill(~valid4, 0.0)
            active = p.sum(dim=-1) > 0
            active_count = active.sum().to(dtype=torch.float32)
            if float(active_count.item()) <= 0.0:
                return
            denom = p.sum().clamp_min(1e-12)
            range_mean = (p * distance.view(1, 1, seq_len, cand)).sum() / denom
            entropy = -(p.clamp_min(1e-12).log() * p).sum(dim=-1)
            top1 = p.max(dim=-1).values
            self._accumulate_probe_metric(
                f"effective_range_{group_name}",
                range_mean,
                active_count,
            )
            self._accumulate_probe_metric(
                f"routing_entropy_{group_name}",
                entropy.masked_select(active).mean(),
                active_count,
            )
            self._accumulate_probe_metric(
                f"routing_top1_{group_name}",
                top1.masked_select(active).mean(),
                active_count,
            )

    def _build_features_for_bank(
        self,
        *,
        input_ids: torch.Tensor,
        bank,
        set_states: torch.Tensor,
        feature_builder: nn.Module,
    ) -> tuple[SetFeatures, torch.Tensor | None]:
        batch = input_ids.shape[0]
        sig_for_gating = None
        if self.feature_mode == "geometry_only":
            features = feature_builder(bank.set_positions)
        elif self.feature_mode == "hashed_counts":
            per_batch = [
                feature_builder(input_ids[i], bank, set_states[i])
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
                per_batch.append(feature_builder(sig, bank.set_sizes))
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
        return features, sig_for_gating

    def _encode_multiresolution_tokens(
        self,
        *,
        thin_anchor: torch.Tensor,
        token_states: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RouterOutput]:
        batch, seq_len = input_ids.shape
        routed_parts: list[torch.Tensor] = []
        primary_router_out: RouterOutput | None = None
        self._grad_probe_active = bool(
            self.training
            and self.grad_probe_interval > 0
            and (self._forward_step % self.grad_probe_interval == 0)
        )
        self._forward_step += 1
        self._grad_probe_tensors = {}
        if self._grad_probe_active:
            token_states.retain_grad()
            self._grad_probe_tensors["token_pre_pool"] = token_states
            self._grad_probe_tensors["set_post_pool"] = {}
            self._grad_probe_tensors["set_post_blocks"] = {}

        for stream, meta in zip(
            self.multiresolution_streams,
            self.multiresolution_group_metadata,
        ):
            bank = build_window_bank(
                seq_len=seq_len,
                window_size=int(meta["window_size"]),
                stride=int(meta["stride"]),
                device=input_ids.device,
                causality_mode=self.set_causality_mode,
                candidate_fiber=self.candidate_fiber,
            )
            pooling_module = stream["pooling_module"]
            set_states = bank.pool(
                token_embeddings=token_states,
                mode=self.pooling_mode,
                params=self.pooling_params,
                pooling_module=(
                    None
                    if isinstance(pooling_module, nn.Identity)
                    else pooling_module
                ),
            )
            set_states = stream["set_input_proj"](set_states)
            group_name = str(meta["name"])
            group_diagnostics = self.multiresolution_diagnostics[group_name]
            if self._grad_probe_active:
                set_states.retain_grad()
                post_pool = self._grad_probe_tensors["set_post_pool"]
                if isinstance(post_pool, dict):
                    post_pool[group_name] = set_states
            if self.training and self.pooling_mode == "soft_trimmed_boltzmann":
                stats_getter = getattr(pooling_module, "get_last_stats", None)
                if stats_getter is not None:
                    pooling_stats = stats_getter()
                    if pooling_stats:
                        group_diagnostics.update_with_pooling_stats(pooling_stats)

            features, sig_for_gating = self._build_features_for_bank(
                input_ids=input_ids,
                bank=bank,
                set_states=set_states,
                feature_builder=stream["feature_builder"],
            )
            geom_bias = features.geom_bias
            if not self.geom_enabled or not self.geom_apply_bias:
                geom_bias = None
            content_bias = None
            if self.feature_mode != "geometry_only" and features.phi_attn is not None:
                content_bias = stream["adapter"](features.phi_attn)

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
            if self.causal:
                causal_mask = bank.set_positions[:, None] >= bank.set_positions[None, :]
                sig_mask = causal_mask if sig_mask is None else sig_mask & causal_mask

            guard_seq_len = (
                seq_len
                if (
                    int(meta["window_size"]) == 1
                    and int(meta["stride"]) == 1
                    and not self.allow_token_token
                )
                else -1
            )
            for block in stream["blocks"]:
                set_states = block(set_states, geom_bias, content_bias, sig_mask, guard_seq_len)
            if self._grad_probe_active:
                set_states.retain_grad()
                post_blocks = self._grad_probe_tensors["set_post_blocks"]
                if isinstance(post_blocks, dict):
                    post_blocks[group_name] = set_states

            router = stream["router"]
            if isinstance(router, UniformRouter):
                router_out: RouterOutput = router(set_states, bank.token_to_sets)
            else:
                desc_router = features.desc_router
                if desc_router is not None and desc_router.dim() == 2:
                    desc_router = desc_router.unsqueeze(0).expand(batch, -1, -1)
                router_out = router(token_states, set_states, desc_router, bank.token_to_sets)
            routed_part = router_out.token_repr
            if self.span_ablation_mode == group_name:
                routed_part = torch.zeros_like(routed_part)
            routed_parts.append(routed_part)
            if not self.training:
                self._update_effective_range_probe(
                    group_name=group_name,
                    bank=bank,
                    router_out=router_out,
                )
            if primary_router_out is None:
                primary_router_out = router_out
            if self.training:
                group_diagnostics.update_with_router_state(
                    bank_indices=router_out.bank_indices,
                    num_sets=router_out.num_sets,
                    router_probs=router_out.probs,
                    router_prob_indices=router_out.prob_indices,
                    set_embeddings=set_states,
                    set_attention_weights=None,
                    token_to_sets=bank.token_to_sets,
                )

        if not routed_parts or primary_router_out is None:
            raise RuntimeError("multiresolution produced no routed streams")
        routed_repr = self.set_output_proj(torch.cat(routed_parts, dim=-1))
        self._last_set_embeddings = None
        self._update_anchor_loss(routed_repr, input_ids, labels=labels)
        if self.span_ablation_enabled:
            routed_repr = torch.zeros_like(routed_repr)
        token_repr = routed_repr
        if self.set_causality_mode == "strict_past":
            if self.output_residual_mode == "direct":
                token_repr = token_states + routed_repr
            elif self.output_residual_mode == "empty_only":
                has_candidates = torch.ones(
                    (seq_len,), dtype=torch.bool, device=input_ids.device
                )
                token_repr = torch.where(
                    has_candidates.view(1, seq_len, 1),
                    routed_repr,
                    token_states,
                )
            elif self.output_residual_mode == "none":
                token_repr = routed_repr
            elif self.output_residual_mode == "anchor_span":
                token_repr = thin_anchor + routed_repr
        return token_repr, primary_router_out

    def anchor_pre_encoder_parameter_count(self) -> int:
        if self.anchor_pre_encoder is None:
            return 0
        return sum(p.numel() for p in self.anchor_pre_encoder.parameters())

    def inference_parameter_count(self) -> int:
        excluded_prefix = "anchor_pre_encoder."
        return sum(
            p.numel()
            for name, p in self.named_parameters()
            if not name.startswith(excluded_prefix)
        )

    def _encode_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RouterOutput]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [batch, seq]")
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        thin_anchor = self._thin_anchor(input_ids)
        token_states = self.token_mlp(thin_anchor)
        if self.multiresolution_enabled:
            return self._encode_multiresolution_tokens(
                thin_anchor=thin_anchor,
                token_states=token_states,
                input_ids=input_ids,
                labels=labels,
            )

        bank = build_window_bank(
            seq_len=seq_len,
            window_size=self.window_size,
            stride=self.stride,
            device=input_ids.device,
            causality_mode=self.set_causality_mode,
            candidate_fiber=self.candidate_fiber,
        )
        set_states = bank.pool(
            token_embeddings=token_states,
            mode=self.pooling_mode,
            params=self.pooling_params,
            pooling_module=self.pooling_module,
        )
        set_states = self.set_input_proj(set_states)
        self._grad_probe_active = bool(
            self.training
            and self.grad_probe_interval > 0
            and (self._forward_step % self.grad_probe_interval == 0)
        )
        self._forward_step += 1
        self._grad_probe_tensors = {}
        if self._grad_probe_active:
            token_states.retain_grad()
            set_states.retain_grad()
            self._grad_probe_tensors["token_pre_pool"] = token_states
            self._grad_probe_tensors["set_post_pool"] = set_states
        if self.training:
            self._last_set_embeddings = set_states
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
        if self.causal:
            causal_mask = bank.set_positions[:, None] >= bank.set_positions[None, :]
            if sig_mask is None:
                sig_mask = causal_mask
            else:
                sig_mask = sig_mask & causal_mask

        guard_seq_len = (
            seq_len
            if (self.window_size == 1 and self.stride == 1 and not self.allow_token_token)
            else -1
        )
        for block in self.blocks:
            set_states = block(set_states, geom_bias, content_bias, sig_mask, guard_seq_len)
        if self._grad_probe_active:
            set_states.retain_grad()
            self._grad_probe_tensors["set_post_blocks"] = set_states

        if isinstance(self.router, UniformRouter):
            router_out: RouterOutput = self.router(set_states, bank.token_to_sets)
        else:
            desc_router = features.desc_router
            if desc_router is not None and desc_router.dim() == 2:
                desc_router = desc_router.unsqueeze(0).expand(batch, -1, -1)
            router_out = self.router(token_states, set_states, desc_router, bank.token_to_sets)
        routed_repr = self.set_output_proj(router_out.token_repr)
        self._update_anchor_loss(routed_repr, input_ids, labels=labels)
        if self.span_ablation_enabled:
            routed_repr = torch.zeros_like(routed_repr)
        token_repr = routed_repr
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
            elif self.output_residual_mode == "anchor_span":
                token_repr = thin_anchor + routed_repr

        if self.training:
            self.diagnostics.update_with_router_state(
                bank_indices=router_out.bank_indices,
                num_sets=router_out.num_sets,
                router_probs=router_out.probs,
                router_prob_indices=router_out.prob_indices,
                set_embeddings=set_states,
                set_attention_weights=None,
                token_to_sets=bank.token_to_sets,
            )

        return token_repr, router_out

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_repr, _ = self._encode_tokens(input_ids)
        return token_repr

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_repr, _ = self._encode_tokens(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return self.lm_head(token_repr)

    def get_diagnostics(self) -> dict[str, float]:
        if self.multiresolution_enabled:
            self.diagnostics.reset()
            by_group: dict[str, dict[str, float]] = {}
            group_heads = {
                str(meta["name"]): int(meta["num_heads"])
                for meta in self.multiresolution_group_metadata
            }
            stats: dict[str, float] = {}
            for name, diagnostics in self.multiresolution_diagnostics.items():
                group_stats = diagnostics.get_epoch_stats()
                diagnostics.reset()
                by_group[name] = group_stats
                for key, value in group_stats.items():
                    suffix = key.removeprefix("ausa/")
                    stats[f"ausa/{name}/{suffix}"] = value

            all_keys = sorted({key for group in by_group.values() for key in group})
            for key in all_keys:
                values = [
                    (float(group_heads[name]), float(group[key]))
                    for name, group in by_group.items()
                    if key in group
                    and isinstance(group[key], (int, float))
                    and math.isfinite(float(group[key]))
                ]
                if not values:
                    continue
                if key.endswith("_min"):
                    stats[key] = min(value for _, value in values)
                elif key.endswith("_max"):
                    stats[key] = max(value for _, value in values)
                elif key.endswith("_count"):
                    stats[key] = sum(value for _, value in values)
                else:
                    total_weight = sum(weight for weight, _ in values)
                    stats[key] = sum(weight * value for weight, value in values) / total_weight
            self._reset_gradient_probe_schedule()
            return stats
        stats = self.diagnostics.get_epoch_stats()
        self.diagnostics.reset()
        self._reset_gradient_probe_schedule()
        return stats

    def _reset_gradient_probe_schedule(self) -> None:
        # Diagnostics are emitted once per epoch. Re-arm the probe so validation
        # forwards cannot shift every probe opportunity outside the next epoch.
        self._forward_step = 0
        self._grad_probe_active = False
        self._grad_probe_tensors = {}

    def update_parameter_diagnostics(self) -> None:
        if not self.multiresolution_enabled:
            return
        for stream, meta in zip(
            self.multiresolution_streams,
            self.multiresolution_group_metadata,
        ):
            name = str(meta["name"])
            router_params = dict(stream["router"].named_parameters())
            self.multiresolution_diagnostics[name].update_router_params(router_params)

    def get_last_set_embeddings(self) -> torch.Tensor | None:
        return self._last_set_embeddings

    def collect_grad_diagnostics(self) -> None:
        if not self._grad_probe_active:
            return
        t_h = self._grad_probe_tensors.get("token_pre_pool")
        t_z0 = self._grad_probe_tensors.get("set_post_pool")
        t_zl = self._grad_probe_tensors.get("set_post_blocks")
        if self.multiresolution_enabled:
            if (
                not torch.is_tensor(t_h)
                or not isinstance(t_z0, dict)
                or not isinstance(t_zl, dict)
            ):
                self._grad_probe_active = False
                self._grad_probe_tensors = {}
                return

            def _gnorm(t: torch.Tensor) -> float:
                if t.grad is None:
                    return float("nan")
                return float(t.grad.detach().norm().item())

            grad_h = _gnorm(t_h)
            for name, diagnostics in self.multiresolution_diagnostics.items():
                post_pool = t_z0.get(name)
                post_blocks = t_zl.get(name)
                if not torch.is_tensor(post_pool) or not torch.is_tensor(post_blocks):
                    continue
                diagnostics.update_with_gradient_probe(
                    grad_h=grad_h,
                    grad_z0=_gnorm(post_pool),
                    grad_zl=_gnorm(post_blocks),
                )
            self._grad_probe_active = False
            self._grad_probe_tensors = {}
            return
        if t_h is None or t_z0 is None or t_zl is None:
            self._grad_probe_active = False
            self._grad_probe_tensors = {}
            return

        def _gnorm(t: torch.Tensor) -> float:
            if t.grad is None:
                return float("nan")
            return float(t.grad.detach().norm().item())

        self.diagnostics.update_with_gradient_probe(
            grad_h=_gnorm(t_h),
            grad_z0=_gnorm(t_z0),
            grad_zl=_gnorm(t_zl),
        )
        self._grad_probe_active = False
        self._grad_probe_tensors = {}
