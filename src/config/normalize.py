from __future__ import annotations

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

    cfg["model"] = model
    return cfg
