from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from config.schema import ConfigError
from config.validators import forbid, require, warn


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _max_sets(seq_len: int, stride: int) -> int:
    if seq_len <= 0 or stride <= 0:
        return 0
    return (seq_len + stride - 1) // stride


def _fingerprint(cfg: Dict[str, Any]) -> str:
    def _strip(obj: Any, path: str = "") -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                key_path = f"{path}.{k}" if path else k
                if key_path in {
                    "logging",
                    "training.seed",
                    "logging.wandb.run_name",
                }:
                    continue
                out[k] = _strip(v, key_path)
            return out
        if isinstance(obj, list):
            return [_strip(v, path) for v in obj]
        return obj

    payload = _strip(cfg)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _record_fingerprint(cfg: Dict[str, Any], fingerprint: str) -> None:
    output_dir = cfg.get("training", {}).get("output_dir", "out")
    path = Path(os.environ.get("SET_ATTENTION_FINGERPRINT_PATH", Path(output_dir) / "metrics" / "config_fingerprints.jsonl"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("fingerprint") == fingerprint:
                    warn(f"Config fingerprint {fingerprint} already seen; run may be redundant.")
                    return
        payload = {"fingerprint": fingerprint}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as exc:
        warn(f"Failed to record config fingerprint: {exc}")


def validate_compatibility(cfg: Dict[str, Any]) -> Dict[str, Any]:
    warnings_list = cfg.setdefault("_warnings", [])

    def _warn(message: str) -> None:
        warnings_list.append(message)
        warn(message)

    model = cfg.get("model", {})
    impl = model.get("implementation")
    task = cfg.get("task") or cfg.get("data", {}).get("task")
    if task is None:
        data_cfg = cfg.get("data", {})
        if data_cfg.get("seq_dataset"):
            task = "seq2seq"
        elif data_cfg.get("dataset"):
            task = "lm"

    seq_len = cfg.get("data", {}).get("seq_len") or model.get("max_seq_len") or 0
    window_size = model.get("window_size", 32) or 32
    stride = model.get("stride", 16) or 16
    max_sets = _max_sets(int(seq_len), int(stride))

    min_head_dim = _env_int("SET_ATTENTION_MIN_HEAD_DIM", 8)
    kernel_max_sets = _env_int("SET_ATTENTION_KERNEL_MAX_SETS", 500)
    adapter_min_rank = _env_int("SET_ATTENTION_ADAPTER_MIN_RANK", 20)
    min_landmarks = _env_int("SET_ATTENTION_MIN_LANDMARKS", 2)
    warn_landmark_min = _env_int("SET_ATTENTION_WARN_MIN_LANDMARKS", 10)
    warn_landmark_ratio = _env_float("SET_ATTENTION_WARN_LANDMARK_RATIO", 0.5)

    if task == "lm" and impl not in {"baseline_token", "set_only"}:
        raise ConfigError("LM only supports implementation=baseline_token or set_only")

    if task == "seq2seq" and impl not in {
        "baseline_token",
        "set_only",
        "encoder_set_only",
        "decoder_set_only",
        "cross_attention_set_only",
        "encoder_set_decoder_baseline",
        "encoder_baseline_decoder_set",
    }:
        raise ConfigError("Seq2Seq implementation is not supported")

    d_model = model.get("d_model", 0)
    nhead = model.get("nhead", model.get("num_heads", 1))
    require(d_model % nhead == 0, "baseline_token: d_model must be divisible by nhead")
    require(d_model // nhead >= min_head_dim, "baseline_token: head dimension too small")

    def _resolve_impls() -> tuple[str, str, str]:
        encoder = "baseline"
        decoder = "baseline"
        cross = "baseline"
        if impl == "set_only":
            encoder = decoder = cross = "set_only"
        elif impl in {"encoder_set_only", "encoder_set_decoder_baseline"}:
            encoder = "set_only"
        elif impl in {"decoder_set_only", "encoder_baseline_decoder_set"}:
            decoder = "set_only"
        elif impl == "cross_attention_set_only":
            cross = "set_only"
        elif impl == "baseline_token":
            pass
        if model.get("cross_attention") == "set_only":
            cross = "set_only"
        if model.get("cross_attention") == "baseline":
            cross = "baseline"
        return encoder, decoder, cross

    encoder_impl, decoder_impl, cross_impl = _resolve_impls()

    def _validate_family_backend(family: str | None, backend: str | None, scope: str) -> None:
        if family is None or backend is None:
            return
        if family == "dense":
            require(backend == "exact", f"{scope}: dense requires backend=exact")
        elif family == "sparse":
            require(backend in {"local_band", "sparse_topk"}, f"{scope}: sparse backend mismatch")
        elif family == "linear":
            require(backend in {"landmark", "nystrom", "linformer"}, f"{scope}: linear backend mismatch")
        else:
            raise ConfigError(f"{scope}: attention_family must be dense, sparse, or linear")

    _validate_family_backend(model.get("encoder_attention_family"), model.get("encoder_backend"), "encoder")
    _validate_family_backend(model.get("decoder_attention_family"), model.get("decoder_backend"), "decoder")
    _validate_family_backend(model.get("cross_attention_family"), model.get("cross_backend"), "cross")

    uses_set_only = encoder_impl == "set_only" or decoder_impl == "set_only" or cross_impl == "set_only"
    if not uses_set_only:
        return cfg

    pooling_cfg = model.get("pooling", "mean")
    if isinstance(pooling_cfg, dict):
        pooling_mode = pooling_cfg.get("mode", "mean")
    else:
        pooling_mode = pooling_cfg
    require(
        pooling_mode in {"mean", "soft_trimmed_boltzmann"},
        "set_only: pooling.mode must be 'mean' or 'soft_trimmed_boltzmann'",
    )
    if model.get("multiscale"):
        raise ConfigError("set_only: multiscale is not implemented in this runner")

    d_phi = model.get("d_phi")
    if d_phi is not None:
        require(int(d_phi) > 0, "set_only: d_phi must be positive")

    require(window_size <= seq_len, "set_only: window_size must be <= max_seq_len")
    require(stride <= window_size, "set_only: stride must be <= window_size")
    require(max_sets >= 1, "set_only: max_sets must be >= 1")

    d_model = model.get("d_model", 0)
    num_heads = model.get("num_heads", 1)
    require(d_model % num_heads == 0, "set_only: d_model must be divisible by num_heads")
    d_head = d_model // num_heads
    require(d_head >= min_head_dim, "set_only: head dimension too small")

    backend = model.get("backend")
    backend_params = model.get("backend_params") or {}
    if backend == "local_band":
        require("radius" in backend_params, "local_band backend requires backend_params.radius")
        require(backend_params["radius"] >= 1, "local_band radius must be >= 1")
        global_indices = backend_params.get("global_indices", [])
        global_set_indices = backend_params.get("global_set_indices", [])
        if global_indices and not isinstance(global_indices, list):
            raise ConfigError("local_band backend_params.global_indices must be a list")
        if global_set_indices and not isinstance(global_set_indices, list):
            raise ConfigError("local_band backend_params.global_set_indices must be a list")
    elif backend == "sparse_topk":
        _warn("backend sparse_topk is deprecated; use local_band (Longformer-style) instead.")
    elif backend == "nystrom":
        require("num_landmarks" in backend_params, "nystrom backend requires backend_params.num_landmarks")
        require(backend_params["num_landmarks"] >= min_landmarks, "nystrom num_landmarks too small")
        require(backend_params["num_landmarks"] < max_sets, "nystrom num_landmarks must be < max_sets")
    elif backend == "landmark":
        require("num_landmarks" in backend_params, "landmark backend requires backend_params.num_landmarks")
        require(backend_params["num_landmarks"] >= min_landmarks, "landmark num_landmarks too small")
        require(backend_params["num_landmarks"] < max_sets, "landmark num_landmarks must be < max_sets")
    elif backend == "linformer":
        require("k" in backend_params, "linformer backend requires backend_params.k")
        require(backend_params["k"] >= min_landmarks, "linformer k too small")
        require(backend_params["k"] <= max_sets, "linformer k must be <= max_sets")
    elif backend == "exact":
        forbid(bool(backend_params), "exact backend forbids backend_params")

    if backend in {"nystrom", "landmark"}:
        num_landmarks = backend_params.get("num_landmarks", 0)
        if num_landmarks and num_landmarks < warn_landmark_min:
            _warn("num_landmarks is very small; approximation may be ineffective.")
        if num_landmarks and num_landmarks > int(max_sets * warn_landmark_ratio):
            _warn("num_landmarks is large relative to max_sets; approximation may be wasteful.")

    router_type = model.get("router_type", "uniform")
    router_topk = model.get("router_topk", None)
    if router_type == "learned":
        require(router_topk is not None, "learned router requires router_topk")
        require(router_topk >= 1, "learned router_topk must be >= 1")
        require(router_topk <= max_sets, "learned router_topk must be <= max_sets")
    else:
        if router_topk is not None:
            _warn("router_topk is ignored for uniform router")
    if router_topk == max_sets:
        _warn("router_topk == max_sets is equivalent to full softmax")

    sig_gating = model.get("sig_gating", {})
    if sig_gating and sig_gating.get("enabled"):
        method = sig_gating.get("method", "pos_topk")
        require(
            method in {"pos_topk", "pos_threshold", "minhash_topk", "minhash_threshold"},
            "sig_gating.method must be pos_topk, pos_threshold, minhash_topk, or minhash_threshold",
        )
        if method.endswith("topk"):
            k = int(sig_gating.get("k", 16))
            require(k >= 1, "sig_gating.k must be >= 1")
            require(k <= max_sets, "sig_gating.k must be <= max_sets")
        else:
            delta_threshold = float(sig_gating.get("delta_threshold", 0.25))
            require(0 <= delta_threshold <= 1.0, "sig_gating.delta_threshold must be in [0,1]")
        if method.startswith("minhash"):
            sig_k = sig_gating.get("sig_k", None)
            require(sig_k is not None, "sig_gating.sig_k is required for minhash gating")
            require(int(sig_k) >= 1, "sig_gating.sig_k must be >= 1")

    features_cfg = model.get("features", {})
    if isinstance(features_cfg, dict):
        hashed_cfg = features_cfg.get("hashed_counts", {})
        if isinstance(hashed_cfg, dict) and "fusion" in hashed_cfg:
            require(
                hashed_cfg["fusion"] in {"mlp", "linear"},
                "features.hashed_counts.fusion must be 'mlp' or 'linear'",
            )
        geometry_cfg = model.get("geometry", {})
        if isinstance(geometry_cfg, dict):
            for key in ("enabled", "apply_as_bias", "apply_in_phi_attn"):
                if key in geometry_cfg:
                    require(
                        isinstance(geometry_cfg[key], bool),
                        f"geometry.{key} must be a boolean",
                    )

    feature_mode = model.get("feature_mode", "geometry_only")
    feature_params = model.get("feature_params") or {}
    allow_unsafe = bool(feature_params.get("allow_unsafe") or os.environ.get("SET_ATTENTION_KERNEL_ALLOW_UNSAFE") == "1")
    if feature_mode == "kernel":
        if max_sets > kernel_max_sets and not allow_unsafe:
            raise ConfigError(
                "Kernel features forbidden for max_sets above limit; set allow_unsafe to override."
            )
        if backend == "local_band":
            _warn("Kernel features with local_band backend may be redundant.")

    adapter_type = model.get("adapter_type", "auto")
    adapter_hidden_multiplier = model.get("adapter_hidden_multiplier", 2)
    require(adapter_hidden_multiplier > 0, "adapter_hidden_multiplier must be > 0")
    effective_rank = min(max_sets, d_head)
    if adapter_type == "linear" and max_sets < 2:
        raise ConfigError("Linear adapter requires at least 2 sets")
    if adapter_type == "auto" and effective_rank < adapter_min_rank:
        _warn("Auto adapter switched to nonlinear due to low effective rank")
        model["adapter_type"] = "nonlinear"
    elif adapter_type == "linear" and effective_rank < adapter_min_rank:
        _warn("Linear adapter rank-limited; consider nonlinear")

    fingerprint = _fingerprint(cfg)
    cfg["_fingerprint"] = fingerprint
    _record_fingerprint(cfg, fingerprint)
    return cfg
