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
    "max_seq_len",
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
    "max_seq_len",
    "pooling",
    "multiscale",
    "sig_gating",
    "d_phi",
    "geometry",
    "features",
    "router_type",
    "router_topk",
    "backend_params",
    "feature_mode",
    "feature_params",
    "gamma",
    "beta",
    "adapter_type",
    "adapter_hidden_multiplier",
    "adapter_budget_fraction",
    "allow_token_token",
    "seq2seq",
    "causal",
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
    if model_cfg.get("feature_mode") is not None and model_cfg.get("feature_mode", "geometry_only") not in {
        "geometry_only",
        "hashed_counts",
        "kernel",
    }:
        raise ConfigError("feature_mode must be geometry_only, hashed_counts, or kernel")

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
