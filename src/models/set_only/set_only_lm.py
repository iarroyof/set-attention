from __future__ import annotations

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
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        return x


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
        self._grad_probe_tensors: dict[str, torch.Tensor] = {}
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
        if self.candidate_fiber != "endpoint_window":
            raise ValueError("Only candidate_fiber='endpoint_window' is implemented")
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
        anchor_cfg = anchor or {}
        teacher_cfg = dict(anchor_cfg.get("teacher", {}) or {})
        self.anchor_cfg = {
            "enabled": False,
            "target": "pre_encoder",
            "pre_encoder_layers": 2,
            "lambda_h": 0.1,
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
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward or d_model * 4,
                num_layers=self.anchor_pre_encoder_layers,
                dropout=dropout,
            )
            if self.anchor_enabled
            else None
        )
        self._last_aux_losses: dict[str, torch.Tensor] = {}
        self._last_aux_metrics: dict[str, float] = {}
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

        if feature_mode == "geometry_only":
            self.feature_builder = GeometryOnlyFeatureBuilder(
                d_model=self.set_state_dim, max_sets=max_sets, gamma=gamma, beta=beta
            )
        elif feature_mode == "hashed_counts":
            self.feature_builder = HashedCountFeatureBuilder(
                d_model=self.set_state_dim,
                d_phi=d_phi,
                max_sets=max_sets,
                num_bins=self.resolved_hash_num_bins,
                gamma=gamma,
                beta=beta,
                normalize=self.resolved_hash_normalize,
                hash_seed=self.resolved_hash_seed,
                fusion=feature_params.get("fusion", "mlp"),
                include_geom_in_attn=geom_apply_in_phi,
            )
        elif feature_mode == "kernel":
            self.feature_builder = KernelFeatureBuilder(
                d_model=self.set_state_dim, d_phi=d_phi, max_sets=max_sets, gamma=gamma, beta=beta
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
                },
                "candidate_fiber": self.candidate_fiber,
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

        def make_backend() -> nn.Module:
            if backend in {"exact", "dense_exact"}:
                return DenseExactBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                )
            if backend == "local_band":
                return LocalBandBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    radius=backend_params.get("radius", 4),
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                    global_set_indices=backend_params.get("global_set_indices"),
                )
            if backend == "nystrom":
                return NystromBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    num_landmarks=backend_params.get("num_landmarks", 32),
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                    bias_scale=backend_params.get("bias_scale", 0.1),
                )
            if backend == "landmark":
                return LandmarkAttentionBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    landmark_coverage=backend_params.get("landmark_coverage", 0.25),
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                )
            if backend == "sparse_topk":
                return SparseTopKBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    k_s=backend_params.get("k_s", 16),
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                )
            if backend == "linformer":
                return LinformerBackend(
                    d_model=self.set_state_dim,
                    num_heads=num_heads,
                    max_sets=max_sets,
                    k=backend_params.get("k", 32),
                    dropout=self.attn_dropout,
                    allow_token_token=self.allow_token_token,
                )
            raise ValueError(f"Unknown backend: {backend}")

        self.blocks = nn.ModuleList(
            [
                SetAttentionBlock(
                    d_model=self.set_state_dim,
                    backend=make_backend(),
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
        self.adapter = None
        self.adapter_type_requested = adapter_type
        self.resolved_adapter_type = "none"
        self.diagnostics = SetDiagnostics()
        if feature_mode != "geometry_only":
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

    def _update_anchor_loss(
        self,
        span_repr: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> None:
        self._last_aux_losses = {}
        self._last_aux_metrics = {}
        if not self.training or not self.anchor_enabled or self.anchor_pre_encoder is None:
            return
        target = self.compute_anchor_target(input_ids)
        if self.anchor_detach_target:
            target = target.detach()
        span_norm = self.anchor_norm(span_repr)
        target_norm = self.anchor_norm(target)
        anchor_mse = F.mse_loss(span_norm, target_norm)
        diff_norm = (span_norm - target_norm).norm()
        target_norm_value = target_norm.norm().clamp_min(1e-12)
        recon_error = diff_norm / target_norm_value
        self._last_aux_losses = {
            "anchor_loss": self.anchor_lambda_h * anchor_mse,
            "anchor_mse": anchor_mse.detach(),
        }
        self._last_aux_metrics = {
            "anchor/lambda_h": self.anchor_lambda_h,
            "anchor/mse": float(anchor_mse.detach().item()),
            "anchor/recon_error_norm": float(recon_error.detach().item()),
        }

    def get_auxiliary_losses(self) -> dict[str, torch.Tensor]:
        return dict(self._last_aux_losses)

    def get_auxiliary_metrics(self) -> dict[str, float]:
        return dict(self._last_aux_metrics)

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

        thin_anchor = self._thin_anchor(input_ids)
        token_states = self.token_mlp(thin_anchor)

        bank = build_window_bank(
            seq_len=seq_len,
            window_size=self.window_size,
            stride=self.stride,
            device=input_ids.device,
            causality_mode=self.set_causality_mode,
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
        self._update_anchor_loss(routed_repr, input_ids)
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
    ) -> torch.Tensor:
        token_repr, _ = self._encode_tokens(input_ids, attention_mask=attention_mask)
        return self.lm_head(token_repr)

    def get_diagnostics(self) -> dict[str, float]:
        stats = self.diagnostics.get_epoch_stats()
        self.diagnostics.reset()
        return stats

    def get_last_set_embeddings(self) -> torch.Tensor | None:
        return self._last_set_embeddings

    def collect_grad_diagnostics(self) -> None:
        if not self._grad_probe_active:
            return
        t_h = self._grad_probe_tensors.get("token_pre_pool")
        t_z0 = self._grad_probe_tensors.get("set_post_pool")
        t_zl = self._grad_probe_tensors.get("set_post_blocks")
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
