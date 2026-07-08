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
    "multiresolution",
    "seq2seq",
    "hybrid",
}

LOGGING_KEYS = {"wandb", "csv", "metric_columns"}
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
            "lambda_pre",
            "pre_encoder_head",
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
    if model_cfg.get("multiresolution") is not None:
        multi_cfg = model_cfg["multiresolution"]
        if not isinstance(multi_cfg, dict):
            raise ConfigError("model.multiresolution must be a mapping")
        unexpected_multi = set(multi_cfg.keys()) - {"enabled", "groups"}
        if unexpected_multi:
            raise ConfigError(
                f"Unexpected model.multiresolution keys: {sorted(unexpected_multi)}"
            )
        groups = multi_cfg.get("groups", [])
        if groups is not None:
            if not isinstance(groups, list):
                raise ConfigError("model.multiresolution.groups must be a list")
            for idx, group in enumerate(groups):
                if not isinstance(group, dict):
                    raise ConfigError(
                        f"model.multiresolution.groups[{idx}] must be a mapping"
                    )
                unexpected_group = set(group.keys()) - {
                    "name",
                    "num_heads",
                    "window_size",
                    "stride",
                    "w",
                    "s",
                }
                if unexpected_group:
                    raise ConfigError(
                        "Unexpected model.multiresolution.groups"
                        f"[{idx}] keys: {sorted(unexpected_group)}"
                    )

    if "family" in cfg.get("data", {}):
        raise ConfigError("data.family is not allowed; use model.implementation only")

    training_cfg = cfg["training"]
    if not isinstance(training_cfg, dict):
        raise ConfigError("training must be a mapping")
    for key in ("deterministic", "strict_deterministic", "benchmark_mode"):
        if key in training_cfg and not isinstance(training_cfg[key], bool):
            raise ConfigError(f"training.{key} must be a boolean")
    if training_cfg.get("strict_deterministic") and not training_cfg.get(
        "deterministic"
    ):
        raise ConfigError(
            "training.strict_deterministic=true requires "
            "training.deterministic=true"
        )
    checkpoint_cfg = training_cfg.get("checkpoint", {})
    if not isinstance(checkpoint_cfg, dict):
        raise ConfigError("training.checkpoint must be a mapping")
    unexpected_checkpoint = set(checkpoint_cfg) - {
        "save_final",
        "save_every_epochs",
        "directory",
        "resume_from",
        "eval_only_from",
    }
    if unexpected_checkpoint:
        raise ConfigError(
            "Unexpected training.checkpoint keys: "
            f"{sorted(unexpected_checkpoint)}"
        )
    if not isinstance(checkpoint_cfg.get("save_final", False), bool):
        raise ConfigError("training.checkpoint.save_final must be a boolean")
    save_every = checkpoint_cfg.get("save_every_epochs", 0)
    if (
        not isinstance(save_every, int)
        or isinstance(save_every, bool)
        or save_every < 0
    ):
        raise ConfigError(
            "training.checkpoint.save_every_epochs must be an integer >= 0"
        )
    for key in ("directory", "resume_from", "eval_only_from"):
        value = checkpoint_cfg.get(key)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"training.checkpoint.{key} must be a string or null")

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
        metric_columns = cfg["logging"].get("metric_columns", [])
        if not isinstance(metric_columns, list) or not all(
            isinstance(column, str) and column
            for column in metric_columns
        ):
            raise ConfigError(
                "logging.metric_columns must be a list of non-empty strings"
            )
