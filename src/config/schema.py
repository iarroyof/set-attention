from __future__ import annotations


class ConfigError(ValueError):
    pass


COMMON_KEYS = {
    "implementation",
    "attention_family",
    "backend",
    "encoder_attention_family",
    "encoder_backend",
    "decoder_attention_family",
    "decoder_backend",
    "cross_attention_family",
    "cross_backend",
    "cross_attention",
}

BASELINE_KEYS = {
    "architecture",
    "vocab_size",
    "d_model",
    "nhead",
    "num_heads",
    "num_layers",
    "dim_feedforward",
    "dropout",
    "attn_dropout",
    "resid_dropout",
    "ffn_dropout",
    "max_seq_len",
    "causal",
    "seq2seq",
}

SET_ONLY_KEYS = {
    "vocab_size",
    "d_model",
    "num_layers",
    "num_heads",
    "window_size",
    "stride",
    "dropout",
    "attn_dropout",
    "resid_dropout",
    "ffn_dropout",
    "max_seq_len",
    "pooling",
    "pooling_multihead",
    "multiscale",
    "sig_gating",
    "d_phi",
    "set_state_dim",
    "geometry",
    "features",
    "router_type",
    "router_topk",
    "router",
    "router_multihead",
    "router_temperature",
    "backend_params",
    "feature_mode",
    "feature_params",
    "gamma",
    "beta",
    "adapter_type",
    "adapter_hidden_multiplier",
    "adapter_budget_fraction",
    "allow_token_token",
    "token_mlp",
    "set_causality_mode",
    "output_residual_mode",
    "anchor",
    "set_diversity",
    "multivector_basis",
    "candidate_fiber",
    "seq2seq",
    "hybrid",
}

LOGGING_KEYS = {"wandb", "csv"}
WANDB_KEYS = {"enable", "project", "tags", "run_name"}
CSV_KEYS = {"path"}


def validate_config(cfg: dict) -> None:
    if "model" not in cfg:
        raise ConfigError("Missing 'model' section")
    if "data" not in cfg:
        raise ConfigError("Missing 'data' section")
    if "training" not in cfg:
        raise ConfigError("Missing 'training' section")

    model_cfg = cfg["model"]
    impl = model_cfg.get("implementation")
    if impl not in {
        "baseline_token",
        "hybrid_token_set",
        "set_only",
        "encoder_set_only",
        "decoder_set_only",
        "cross_attention_set_only",
        "encoder_set_decoder_baseline",
        "encoder_baseline_decoder_set",
    }:
        raise ConfigError("model.implementation must be a supported value")
    cross_attention = model_cfg.get("cross_attention")
    if cross_attention is not None and cross_attention not in {"baseline", "set_only"}:
        raise ConfigError("cross_attention must be 'baseline' or 'set_only'")

    allowed_keys = COMMON_KEYS | BASELINE_KEYS | SET_ONLY_KEYS
    unexpected = set(model_cfg.keys()) - allowed_keys
    if unexpected:
        raise ConfigError(f"Unexpected model keys: {sorted(unexpected)}")
    if impl in {"set_only", "hybrid_token_set"} and "causal" in model_cfg:
        raise ConfigError(
            "model.causal is deprecated for set-only models; use model.set_causality_mode"
        )

    if model_cfg.get("architecture") is not None:
        if model_cfg.get("architecture") not in {"transformer_lm", "transformer_seq2seq"}:
            raise ConfigError("baseline_token architecture must be 'transformer_lm' or 'transformer_seq2seq'")

    if model_cfg.get("backend") not in {
        None,
        "exact",
        "local_band",
        "linformer",
        "nystrom",
        "landmark",
        "sparse_topk",
    }:
        raise ConfigError("backend must be a supported backend")
    if model_cfg.get("router_type") is not None and model_cfg.get("router_type") not in {"uniform", "learned"}:
        raise ConfigError("router_type must be 'uniform' or 'learned'")
    router_cfg = model_cfg.get("router")
    if router_cfg is not None:
        if not isinstance(router_cfg, dict):
            raise ConfigError("model.router must be a mapping")
        unexpected_router = set(router_cfg.keys()) - {"min_temp", "score_mode"}
        if unexpected_router:
            raise ConfigError(f"Unexpected model.router keys: {sorted(unexpected_router)}")
        if router_cfg.get("score_mode") is not None and router_cfg.get("score_mode") not in {
            "candidate_gather",
            "dense",
        }:
            raise ConfigError("model.router.score_mode must be 'candidate_gather' or 'dense'")
    if model_cfg.get("router_temperature") is not None:
        try:
            router_temperature = float(model_cfg.get("router_temperature"))
        except (TypeError, ValueError):
            raise ConfigError("router_temperature must be a float > 0")
        if router_temperature <= 0:
            raise ConfigError("router_temperature must be > 0")
    if model_cfg.get("feature_mode") is not None and model_cfg.get("feature_mode", "geometry_only") not in {
        "geometry_only",
        "hashed_counts",
        "kernel",
    }:
        raise ConfigError("feature_mode must be geometry_only, hashed_counts, or kernel")
    if model_cfg.get("adapter_type") is not None and model_cfg.get("adapter_type") not in {
        "auto",
        "linear",
        "nonlinear",
        "hybrid",
    }:
        raise ConfigError("adapter_type must be auto, linear, nonlinear, or hybrid")
    if model_cfg.get("set_causality_mode") is not None and model_cfg.get("set_causality_mode") not in {
        "strict_past",
        "noncausal",
    }:
        raise ConfigError("set_causality_mode must be 'strict_past' or 'noncausal'")
    if model_cfg.get("output_residual_mode") is not None and model_cfg.get("output_residual_mode") not in {
        "direct",
        "empty_only",
        "none",
        "anchor_span",
    }:
        raise ConfigError("output_residual_mode must be 'direct', 'empty_only', 'none', or 'anchor_span'")
    if model_cfg.get("anchor") is not None:
        anchor_cfg = model_cfg["anchor"]
        if not isinstance(anchor_cfg, dict):
            raise ConfigError("model.anchor must be a mapping")
        unexpected_anchor = set(anchor_cfg.keys()) - {
            "enabled",
            "target",
            "pre_encoder_layers",
            "lambda_h",
            "detach_target",
            "norm",
            "teacher",
        }
        if unexpected_anchor:
            raise ConfigError(f"Unexpected model.anchor keys: {sorted(unexpected_anchor)}")
        teacher_cfg = anchor_cfg.get("teacher", {})
        if teacher_cfg is not None:
            if not isinstance(teacher_cfg, dict):
                raise ConfigError("model.anchor.teacher must be a mapping")
            unexpected_teacher = set(teacher_cfg.keys()) - {"enabled"}
            if unexpected_teacher:
                raise ConfigError(
                    f"Unexpected model.anchor.teacher keys: {sorted(unexpected_teacher)}"
                )
    if model_cfg.get("set_diversity") is not None:
        set_diversity_cfg = model_cfg["set_diversity"]
        if not isinstance(set_diversity_cfg, dict):
            raise ConfigError("model.set_diversity must be a mapping")
        unexpected_div = set(set_diversity_cfg.keys()) - {"lambda_div"}
        if unexpected_div:
            raise ConfigError(f"Unexpected model.set_diversity keys: {sorted(unexpected_div)}")
    if model_cfg.get("multivector_basis") is not None:
        multivector_cfg = model_cfg["multivector_basis"]
        if not isinstance(multivector_cfg, dict):
            raise ConfigError("model.multivector_basis must be a mapping")
        unexpected_multivec = set(multivector_cfg.keys()) - {"enabled", "r"}
        if unexpected_multivec:
            raise ConfigError(
                f"Unexpected model.multivector_basis keys: {sorted(unexpected_multivec)}"
            )
    if model_cfg.get("candidate_fiber") is not None and model_cfg.get("candidate_fiber") not in {
        "endpoint_window",
        "all_past",
        "window_plus_landmarks",
    }:
        raise ConfigError(
            "candidate_fiber must be endpoint_window, all_past, or window_plus_landmarks"
        )

    if "family" in cfg.get("data", {}):
        raise ConfigError("data.family is not allowed; use model.implementation only")

    if "logging" in cfg:
        if not isinstance(cfg["logging"], dict):
            raise ConfigError("logging must be a mapping")
        unexpected = set(cfg["logging"].keys()) - LOGGING_KEYS
        if unexpected:
            raise ConfigError(f"Unexpected logging keys: {sorted(unexpected)}")
        wandb_cfg = cfg["logging"].get("wandb", {})
        if wandb_cfg and not isinstance(wandb_cfg, dict):
            raise ConfigError("logging.wandb must be a mapping")
        if isinstance(wandb_cfg, dict):
            extra = set(wandb_cfg.keys()) - WANDB_KEYS
            if extra:
                raise ConfigError(f"Unexpected logging.wandb keys: {sorted(extra)}")
        csv_cfg = cfg["logging"].get("csv", {})
        if csv_cfg and not isinstance(csv_cfg, dict):
            raise ConfigError("logging.csv must be a mapping")
        if isinstance(csv_cfg, dict):
            extra = set(csv_cfg.keys()) - CSV_KEYS
            if extra:
                raise ConfigError(f"Unexpected logging.csv keys: {sorted(extra)}")
