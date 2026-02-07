from __future__ import annotations


class ConfigError(ValueError):
    pass


BASELINE_KEYS = {
    "family",
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
    "decoder_family",
    "decoder_set_only",
    "cross_attention",
    "cross_set_only",
}

SET_ONLY_KEYS = {
    "family",
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
    "backend",
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
    "decoder_family",
    "decoder_set_only",
    "cross_attention",
    "cross_set_only",
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
    family = model_cfg.get("family")
    if family not in {"baseline_token", "set_only", "encoder_set_only"}:
        raise ConfigError("model.family must be 'baseline_token', 'set_only', or 'encoder_set_only'")
    decoder_family = model_cfg.get("decoder_family")
    if decoder_family is not None and decoder_family not in {"baseline_token", "set_only"}:
        raise ConfigError("decoder_family must be 'baseline_token' or 'set_only'")
    cross_attention = model_cfg.get("cross_attention")
    if cross_attention is not None and cross_attention not in {"baseline", "set_only"}:
        raise ConfigError("cross_attention must be 'baseline' or 'set_only'")

    if family == "baseline_token":
        unexpected = set(model_cfg.keys()) - BASELINE_KEYS
        if unexpected:
            raise ConfigError(
                f"Unexpected baseline_token keys: {sorted(unexpected)}"
            )
        if model_cfg.get("architecture") not in {"transformer_lm", "transformer_seq2seq"}:
            raise ConfigError("baseline_token architecture must be 'transformer_lm' or 'transformer_seq2seq'")
    else:
        unexpected = set(model_cfg.keys()) - SET_ONLY_KEYS
        if unexpected:
            raise ConfigError(f"Unexpected set_only keys: {sorted(unexpected)}")
        if model_cfg.get("backend") not in {
            "dense_exact",
            "local_band",
            "nystrom",
            "landmark",
            "sparse_topk",
        }:
            raise ConfigError("set_only backend must be a supported backend")
        if model_cfg.get("router_type") not in {"uniform", "learned"}:
            raise ConfigError("router_type must be 'uniform' or 'learned'")
        if model_cfg.get("feature_mode", "geometry_only") not in {
            "geometry_only",
            "hashed_counts",
            "kernel",
        }:
            raise ConfigError("feature_mode must be geometry_only, hashed_counts, or kernel")

        max_seq_len = model_cfg.get("max_seq_len", 0)
        window_size = model_cfg.get("window_size", 1)
        stride = model_cfg.get("stride", 1)
        if stride <= 0 or window_size <= 0:
            raise ConfigError("window_size and stride must be positive")
        max_sets = 1 if max_seq_len <= window_size else (
            (max_seq_len - window_size + stride - 1) // stride + 1
        )
        if model_cfg.get("feature_mode") == "kernel" and max_sets > 500:
            raise ConfigError("Kernel features forbidden when max_sets > 500")

        if model_cfg.get("feature_mode") == "kernel" and model_cfg.get("backend") == "local_band":
            import warnings

            warnings.warn(
                "Kernel features with local_band backend may be redundant.",
                RuntimeWarning,
            )

    if "family" in cfg.get("data", {}):
        raise ConfigError("data.family is not allowed; use model.family only")

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
