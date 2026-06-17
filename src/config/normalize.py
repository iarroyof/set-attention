from __future__ import annotations

import warnings
from typing import Any, Dict


_IMPL_ALIASES = {
    "encoder_set_decoder_baseline": "encoder_set_only",
    "encoder_baseline_decoder_set": "decoder_set_only",
}


def _normalize_backend(name: str | None) -> str | None:
    if name is None:
        return None
    if name == "dense_exact":
        return "exact"
    return name


def _infer_attention_family(backend: str | None) -> str | None:
    if backend is None:
        return None
    if backend == "exact":
        return "dense"
    if backend in {"local_band", "sparse_topk"}:
        return "sparse"
    if backend in {"landmark", "nystrom", "linformer"}:
        return "linear"
    return None


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model = cfg.get("model", {})

    legacy_family = model.pop("family", None)
    decoder_family = model.pop("decoder_family", None)
    cross_attention = model.get("cross_attention")

    impl = model.get("implementation") or legacy_family
    if impl in _IMPL_ALIASES:
        impl = _IMPL_ALIASES[impl]

    if impl is None and legacy_family is None and decoder_family is not None:
        if decoder_family == "set_only":
            impl = "decoder_set_only"
        else:
            impl = "baseline_token"

    # Legacy combo inference.
    if impl in {"baseline_token", "encoder_set_only"} and decoder_family == "set_only":
        impl = "set_only" if impl == "encoder_set_only" else "decoder_set_only"

    if impl is not None:
        model["implementation"] = impl

    # Normalize backend names and attention family.
    backend = _normalize_backend(model.get("backend")) or "exact"
    model["backend"] = backend

    attention_family = model.get("attention_family") or _infer_attention_family(backend) or "dense"
    model["attention_family"] = attention_family

    if backend == "landmark":
        backend_params = model.get("backend_params")
        if backend_params is None:
            backend_params = {}
        if isinstance(backend_params, dict):
            backend_params.setdefault("landmark_coverage", 0.25)
        model["backend_params"] = backend_params

    # Per-component defaults (seq2seq only; harmless for LM).
    for prefix in ("encoder", "decoder", "cross"):
        comp_backend = _normalize_backend(model.get(f"{prefix}_backend")) or backend
        if comp_backend is not None:
            model[f"{prefix}_backend"] = comp_backend
        comp_family = model.get(f"{prefix}_attention_family") or _infer_attention_family(comp_backend) or attention_family
        if comp_family is not None:
            model[f"{prefix}_attention_family"] = comp_family

    if cross_attention is not None:
        model["cross_attention"] = cross_attention

    # Keep split dropout fields explicit for auditability/comparability in logs.
    dropout = model.get("dropout", 0.1)
    if model.get("attn_dropout") is None:
        model["attn_dropout"] = dropout
    if model.get("resid_dropout") is None:
        model["resid_dropout"] = dropout
    if model.get("ffn_dropout") is None:
        model["ffn_dropout"] = dropout

    impl = model.get("implementation")
    uses_set_only = impl in {
        "set_only",
        "hybrid_token_set",
        "encoder_set_only",
        "decoder_set_only",
        "cross_attention_set_only",
        "encoder_set_decoder_baseline",
        "encoder_baseline_decoder_set",
    }
    if uses_set_only or model.get("cross_attention") == "set_only":
        # Set-only parity/ablation defaults.
        model.setdefault("router_multihead", False)
        model.setdefault("router_temperature", 1.0)
        router_cfg = model.get("router")
        if router_cfg is None:
            router_cfg = {}
        elif not isinstance(router_cfg, dict):
            router_cfg = {"min_temp": router_cfg}
        router_cfg.setdefault("min_temp", 0.5)
        router_cfg.setdefault("score_mode", "candidate_gather")
        model["router"] = router_cfg
        model.setdefault("pooling_multihead", False)
        pooling_cfg = model.get("pooling", "mean")
        if isinstance(pooling_cfg, dict):
            pooling_cfg.setdefault("mode", "mean")
        else:
            pooling_cfg = {"mode": pooling_cfg}
        pooling_cfg.setdefault("alpha", 10.0)
        pooling_cfg.setdefault("learnable_alpha", False)
        model["pooling"] = pooling_cfg
        feature_params = model.get("feature_params")
        if feature_params is None:
            feature_params = {}
        if isinstance(feature_params, dict):
            feature_params.setdefault("num_bins", 128)
            feature_params.setdefault("hash_seed", 13)
            feature_params.setdefault("normalize", True)
        model["feature_params"] = feature_params
        model.setdefault("d_phi", None)
        model.setdefault("set_state_dim", None)
        model.setdefault("adapter_type", "auto")
        model.setdefault("output_residual_mode", "direct")
        data_cfg = cfg.get("data", {})
        task = cfg.get("task")
        if task is None:
            if data_cfg.get("seq_dataset"):
                task = "seq2seq"
            elif data_cfg.get("dataset"):
                task = "lm"
        legacy_causal = model.pop("causal", None)
        if legacy_causal is not None:
            message = (
                "model.causal is deprecated for set-only models; "
                "use model.set_causality_mode. set_causality_mode wins when both are set."
            )
            cfg.setdefault("_warnings", []).append(message)
            warnings.warn(message, DeprecationWarning, stacklevel=2)
        if "set_causality_mode" not in model:
            default_causality_mode = "strict_past" if task == "lm" else "noncausal"
            if legacy_causal is not None:
                default_causality_mode = "strict_past" if bool(legacy_causal) else "noncausal"
            model["set_causality_mode"] = default_causality_mode
        token_mlp = model.get("token_mlp")
        if token_mlp is None:
            model["token_mlp"] = {"enabled": True}
        elif isinstance(token_mlp, bool):
            model["token_mlp"] = {"enabled": token_mlp}
        elif isinstance(token_mlp, dict):
            token_mlp.setdefault("enabled", True)

    cfg["model"] = model
    return cfg
