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


def _require_bool(value: Any, label: str) -> None:
    require(isinstance(value, bool), f"{label} must be a boolean")


def _require_positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{label} must be numeric")
    require(parsed > 0.0, f"{label} must be > 0")
    return parsed


def _require_nonnegative_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{label} must be numeric")
    require(parsed >= 0.0, f"{label} must be >= 0")
    return parsed


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{label} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ConfigError(f"{label} must be an integer")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise ConfigError(f"{label} must be an integer")
    require(parsed > 0, f"{label} must be > 0")
    return parsed


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{label} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ConfigError(f"{label} must be an integer")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise ConfigError(f"{label} must be an integer")
    return parsed


NYSTROM_DEPRECATION_MESSAGE = (
    "backend=nystrom is deprecated for this revision cycle; use backend=landmark "
    "with model.backend_params.landmark_coverage."
)


def _max_sets(seq_len: int, window_size: int, stride: int, set_causality_mode: str) -> int:
    if seq_len <= 0 or window_size <= 0 or stride <= 0:
        return 0
    if set_causality_mode == "strict_past":
        if seq_len < window_size:
            return 0
        return ((seq_len - window_size) // stride) + 1
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
        elif data_cfg.get("dataset") == "mqar":
            task = "mqar"
        elif data_cfg.get("dataset"):
            task = "lm"

    training_cfg = cfg.get("training", {})
    checkpoint_cfg = training_cfg.get("checkpoint", {}) or {}
    resume_from = checkpoint_cfg.get("resume_from")
    eval_only_from = checkpoint_cfg.get("eval_only_from")
    require(
        not (resume_from and eval_only_from),
        "training.checkpoint.resume_from and eval_only_from are mutually exclusive",
    )
    if training_cfg.get("strict_deterministic"):
        require(
            training_cfg.get("deterministic") is True,
            "strict_deterministic requires training.deterministic=true",
        )
        require(
            training_cfg.get("benchmark_mode") is False,
            "strict_deterministic requires training.benchmark_mode=false",
        )

    seq_len = cfg.get("data", {}).get("seq_len") or model.get("max_seq_len") or 0
    window_size = model.get("window_size", 32) or 32
    stride = model.get("stride", 16) or 16
    if impl in {"set_only", "hybrid_token_set"} and "causal" in model:
        _warn(
            "model.causal is deprecated for set-only models; "
            "model.set_causality_mode is the single source of truth."
        )
    set_causality_mode = model.get(
        "set_causality_mode",
        (
            "strict_past"
            if task in {"lm", "mqar"}
            and impl in {"set_only", "hybrid_token_set"}
            else "noncausal"
        ),
    )
    max_sets = _max_sets(
        int(seq_len),
        int(window_size),
        int(stride),
        str(set_causality_mode),
    )

    min_head_dim = _env_int("SET_ATTENTION_MIN_HEAD_DIM", 8)
    kernel_max_sets = _env_int("SET_ATTENTION_KERNEL_MAX_SETS", 500)
    adapter_min_rank = _env_int("SET_ATTENTION_ADAPTER_MIN_RANK", 20)
    min_landmarks = _env_int("SET_ATTENTION_MIN_LANDMARKS", 2)
    warn_landmark_min = _env_int("SET_ATTENTION_WARN_MIN_LANDMARKS", 10)
    warn_landmark_ratio = _env_float("SET_ATTENTION_WARN_LANDMARK_RATIO", 0.5)

    if task in {"lm", "mqar"} and impl not in {
        "baseline_token",
        "set_only",
        "hybrid_token_set",
    }:
        raise ConfigError(
            "causal LM tasks only support implementation=baseline_token, "
            "set_only, or hybrid_token_set"
        )

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
        if impl in {"set_only", "hybrid_token_set"}:
            encoder = decoder = cross = "set_only"
        elif impl in {"encoder_set_only", "encoder_set_decoder_baseline"}:
            encoder = "set_only"
        elif impl in {"decoder_set_only", "encoder_baseline_decoder_set"}:
            decoder = "set_only"
        elif impl == "cross_attention_set_only":
            cross = "set_only"
        elif impl in {"baseline_token", "hybrid_token_set"}:
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
            if backend == "nystrom":
                raise ConfigError(f"{scope}: {NYSTROM_DEPRECATION_MESSAGE}")
            require(backend in {"landmark", "nystrom", "linformer"}, f"{scope}: linear backend mismatch")
        else:
            raise ConfigError(f"{scope}: attention_family must be dense, sparse, or linear")

    _validate_family_backend(model.get("attention_family"), model.get("backend"), "model")
    _validate_family_backend(model.get("encoder_attention_family"), model.get("encoder_backend"), "encoder")
    _validate_family_backend(model.get("decoder_attention_family"), model.get("decoder_backend"), "decoder")
    _validate_family_backend(model.get("cross_attention_family"), model.get("cross_backend"), "cross")

    uses_set_only = (
        encoder_impl == "set_only"
        or decoder_impl == "set_only"
        or cross_impl == "set_only"
        or impl == "hybrid_token_set"
    )
    if not uses_set_only:
        backend = model.get("backend")
        raw_backend_params = model.get("backend_params")
        backend_params = raw_backend_params or {}
        if raw_backend_params is not None and not isinstance(raw_backend_params, dict):
            raise ConfigError("backend_params must be a mapping")
        if backend == "local_band":
            require("radius" in backend_params, "local_band backend requires backend_params.radius")
            require(backend_params["radius"] >= 1, "local_band radius must be >= 1")
            global_indices = backend_params.get("global_indices", [])
            if global_indices and not isinstance(global_indices, list):
                raise ConfigError("local_band backend_params.global_indices must be a list")
        elif backend == "sparse_topk":
            _warn("backend sparse_topk is deprecated; use local_band (Longformer-style) instead.")
        elif backend == "nystrom":
            raise ConfigError(f"baseline_token: {NYSTROM_DEPRECATION_MESSAGE}")
        elif backend == "landmark":
            if "num_landmarks" in backend_params:
                raise ConfigError(
                    "landmark backend uses backend_params.landmark_coverage; "
                    "num_landmarks is reserved for deprecated nystrom paths"
                )
            landmark_coverage = _require_positive_float(
                backend_params.get("landmark_coverage", 0.25),
                "landmark backend_params.landmark_coverage",
            )
            backend_params["landmark_coverage"] = landmark_coverage
            landmark_count = min(
                max(round(landmark_coverage * int(seq_len)), min_landmarks),
                int(seq_len),
            )
            if landmark_count < warn_landmark_min:
                _warn("landmark_count is very small; approximation may be ineffective.")
            if landmark_count > int(int(seq_len) * warn_landmark_ratio):
                _warn("landmark_count is large relative to sequence length; approximation may be wasteful.")
        elif backend == "linformer":
            require("k" in backend_params, "linformer backend requires backend_params.k")
            require(backend_params["k"] >= min_landmarks, "linformer k too small")
            require(backend_params["k"] <= int(seq_len), "linformer k must be <= seq_len")
        elif backend == "exact":
            forbid(bool(backend_params), "exact backend forbids backend_params")
        if backend_params:
            model["backend_params"] = backend_params
        fingerprint = _fingerprint(cfg)
        cfg["_fingerprint"] = fingerprint
        _record_fingerprint(cfg, fingerprint)
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
    multiresolution_cfg = model.get("multiresolution", {})
    if multiresolution_cfg and not isinstance(multiresolution_cfg, dict):
        raise ConfigError("set_only: multiresolution must be a mapping")
    multiresolution_enabled = bool((multiresolution_cfg or {}).get("enabled", False))
    multiresolution_groups = (multiresolution_cfg or {}).get("groups", [])
    if multiresolution_enabled:
        require(
            isinstance(multiresolution_groups, list) and len(multiresolution_groups) >= 1,
            "set_only: multiresolution.groups must be a non-empty list when enabled",
        )
        total_group_heads = 0
        group_names: set[str] = set()
        for idx, group in enumerate(multiresolution_groups):
            require(isinstance(group, dict), f"set_only: multiresolution.groups[{idx}] must be a mapping")
            group_name = str(group.get("name", f"group{idx}")).strip()
            require(
                bool(group_name),
                f"set_only: multiresolution.groups[{idx}].name must be non-empty",
            )
            require(
                group_name not in group_names,
                f"set_only: duplicate multiresolution group name {group_name!r}",
            )
            group_names.add(group_name)
            group_heads = _require_positive_int(
                group.get("num_heads"),
                f"set_only: multiresolution.groups[{idx}].num_heads",
            )
            group_w = _require_positive_int(
                group.get("window_size", group.get("w")),
                f"set_only: multiresolution.groups[{idx}].window_size",
            )
            group_s = _require_positive_int(
                group.get("stride", group.get("s")),
                f"set_only: multiresolution.groups[{idx}].stride",
            )
            require(
                group_w <= seq_len,
                f"set_only: multiresolution.groups[{idx}].window_size must be <= max_seq_len",
            )
            require(
                group_s <= group_w,
                f"set_only: multiresolution.groups[{idx}].stride must be <= window_size",
            )
            require(
                _max_sets(seq_len, group_w, group_s, set_causality_mode) >= 1,
                f"set_only: multiresolution.groups[{idx}] must create at least one set",
            )
            total_group_heads += group_heads
        require(
            total_group_heads == int(model.get("num_heads", 1)),
            "set_only: multiresolution group head counts must sum to num_heads",
        )
    if isinstance(pooling_cfg, dict):
        if "alpha" in pooling_cfg:
            _require_positive_float(pooling_cfg["alpha"], "set_only: pooling.alpha")
        if "learnable_alpha" in pooling_cfg:
            _require_bool(
                pooling_cfg["learnable_alpha"],
                "set_only: pooling.learnable_alpha",
            )
        if "tau" in pooling_cfg:
            _require_positive_float(pooling_cfg["tau"], "set_only: pooling.tau")
        if "q" in pooling_cfg:
            q = float(pooling_cfg["q"])
            require(0.0 < q <= 1.0, "set_only: pooling.q must be in (0, 1]")

    d_phi = model.get("d_phi")
    if d_phi is not None:
        _require_positive_int(d_phi, "set_only: d_phi")
    set_state_dim = model.get("set_state_dim")
    if set_state_dim is not None:
        _require_positive_int(set_state_dim, "set_only: set_state_dim")
    router_multihead = model.get("router_multihead", False)
    require(isinstance(router_multihead, bool), "set_only: router_multihead must be a boolean")
    pooling_multihead = model.get("pooling_multihead", False)
    require(isinstance(pooling_multihead, bool), "set_only: pooling_multihead must be a boolean")
    token_mlp = model.get("token_mlp", {"enabled": True})
    if isinstance(token_mlp, bool):
        model["token_mlp"] = {"enabled": token_mlp}
    elif isinstance(token_mlp, dict):
        if "enabled" in token_mlp:
            require(
                isinstance(token_mlp.get("enabled"), bool),
                "set_only: token_mlp.enabled must be a boolean",
            )
        else:
            token_mlp["enabled"] = True
    else:
        raise ConfigError("set_only: token_mlp must be a bool or mapping")

    require(window_size <= seq_len, "set_only: window_size must be <= max_seq_len")
    require(stride <= window_size, "set_only: stride must be <= window_size")
    require(max_sets >= 1, "set_only: max_sets must be >= 1")
    require(
        set_causality_mode in {"strict_past", "noncausal"},
        "set_only: set_causality_mode must be 'strict_past' or 'noncausal'",
    )
    output_residual_mode = model.get("output_residual_mode", "direct")
    require(
        output_residual_mode in {"direct", "empty_only", "none", "anchor_span"},
        "set_only: output_residual_mode must be 'direct', 'empty_only', 'none', or 'anchor_span'",
    )
    anchor_cfg = model.get("anchor", {})
    if anchor_cfg and not isinstance(anchor_cfg, dict):
        raise ConfigError("set_only: anchor must be a mapping")
    if anchor_cfg:
        _require_bool(anchor_cfg.get("enabled", False), "set_only: anchor.enabled")
        require(
            anchor_cfg.get("target", "pre_encoder") == "pre_encoder",
            "set_only: anchor.target must be 'pre_encoder'",
        )
        pre_encoder_layers = _require_positive_int(
            anchor_cfg.get("pre_encoder_layers", 2),
            "set_only: anchor.pre_encoder_layers",
        )
        require(
            pre_encoder_layers in {1, 2},
            "set_only: anchor.pre_encoder_layers must be 1 or 2",
        )
        _require_nonnegative_float(
            anchor_cfg.get("lambda_h", 0.1),
            "set_only: anchor.lambda_h",
        )
        lambda_pre = _require_nonnegative_float(
            anchor_cfg.get("lambda_pre", 1.0),
            "set_only: anchor.lambda_pre",
        )
        _require_bool(
            anchor_cfg.get("pre_encoder_head", True),
            "set_only: anchor.pre_encoder_head",
        )
        if bool(anchor_cfg.get("enabled", False)):
            require(
                lambda_pre > 0.0,
                "set_only: anchor.lambda_pre must be > 0 when anchor.enabled=true",
            )
            require(
                bool(anchor_cfg.get("pre_encoder_head", True)),
                "set_only: anchor.pre_encoder_head must be true when anchor.enabled=true",
            )
        _require_bool(
            anchor_cfg.get("detach_target", True),
            "set_only: anchor.detach_target",
        )
        require(
            anchor_cfg.get("norm", "layernorm") == "layernorm",
            "set_only: anchor.norm must be 'layernorm'",
        )
        teacher_cfg = anchor_cfg.get("teacher", {})
        if teacher_cfg and not isinstance(teacher_cfg, dict):
            raise ConfigError("set_only: anchor.teacher must be a mapping")
        require(
            not bool((teacher_cfg or {}).get("enabled", False)),
            "set_only: anchor.teacher.enabled is deferred and must stay false",
        )
    if output_residual_mode == "anchor_span":
        require(
            model.get("token_mlp", {}).get("enabled") is False,
            "set_only: output_residual_mode=anchor_span requires token_mlp.enabled=false",
        )
    set_diversity_cfg = model.get("set_diversity", {})
    if set_diversity_cfg and not isinstance(set_diversity_cfg, dict):
        raise ConfigError("set_only: set_diversity must be a mapping")
    _require_nonnegative_float(
        (set_diversity_cfg or {}).get("lambda_div", 0.0),
        "set_only: set_diversity.lambda_div",
    )
    multivector_cfg = model.get("multivector_basis", {})
    if multivector_cfg and not isinstance(multivector_cfg, dict):
        raise ConfigError("set_only: multivector_basis must be a mapping")
    _require_bool(
        (multivector_cfg or {}).get("enabled", False),
        "set_only: multivector_basis.enabled",
    )
    multivec_r = _require_positive_int(
        (multivector_cfg or {}).get("r", 1),
        "set_only: multivector_basis.r",
    )
    require(1 <= multivec_r <= 4, "set_only: multivector_basis.r must be in [1, 4]")
    require(
        not bool((multivector_cfg or {}).get("enabled", False)),
        "set_only: multivector_basis.enabled is deferred and must stay false",
    )
    require(
        multivec_r == 1,
        "set_only: multivector_basis.r must stay 1 while multivector_basis is disabled",
    )
    candidate_fiber = model.get("candidate_fiber", "endpoint_window")
    require(
        candidate_fiber in {"endpoint_window", "all_past", "window_plus_landmarks"},
        "set_only: candidate_fiber must be endpoint_window, all_past, or window_plus_landmarks",
    )
    require(
        candidate_fiber in {"endpoint_window", "all_past"},
        "set_only: candidate_fiber=window_plus_landmarks is deferred",
    )
    if task in {"lm", "mqar"} and impl in {"set_only", "hybrid_token_set"}:
        require(
            set_causality_mode == "strict_past",
            "set/hybrid causal tasks require set_causality_mode=strict_past",
        )

    if impl == "hybrid_token_set":
        hybrid = model.get("hybrid")
        require(isinstance(hybrid, dict), "hybrid_token_set requires model.hybrid mapping")
        pattern = str(hybrid.get("pattern", ""))
        require(pattern, "hybrid_token_set requires model.hybrid.pattern")
        require(
            len(pattern) == int(model.get("num_layers", 0)),
            "hybrid.pattern length must equal model.num_layers",
        )
        require(
            all(ch in {"T", "S", "t", "s"} for ch in pattern),
            "hybrid.pattern may contain only T and S",
        )
        set_topologies = hybrid.get("set_topologies")
        require(isinstance(set_topologies, list), "hybrid.set_topologies must be a list")
        require(
            len(set_topologies) == pattern.upper().count("S"),
            "hybrid.set_topologies length must equal number of S layers",
        )
        for idx, topo in enumerate(set_topologies):
            require(isinstance(topo, dict), f"hybrid.set_topologies[{idx}] must be a mapping")
            w = _require_positive_int(
                topo.get("window_size", topo.get("w")),
                f"hybrid.set_topologies[{idx}].window_size",
            )
            s = _require_positive_int(
                topo.get("stride", topo.get("s")),
                f"hybrid.set_topologies[{idx}].stride",
            )
            require(
                w <= int(seq_len),
                f"hybrid.set_topologies[{idx}].window_size must be <= seq_len",
            )
            require(
                s <= w,
                f"hybrid.set_topologies[{idx}].stride must be <= window_size",
            )

    d_model = model.get("d_model", 0)
    num_heads = model.get("num_heads", 1)
    require(d_model % num_heads == 0, "set_only: d_model must be divisible by num_heads")
    d_head = d_model // num_heads
    require(d_head >= min_head_dim, "set_only: head dimension too small")
    set_state_dim_for_heads = int(set_state_dim) if set_state_dim is not None else int(d_model)
    require(
        set_state_dim_for_heads % num_heads == 0,
        "set_only: set_state_dim must be divisible by num_heads",
    )
    set_d_head = set_state_dim_for_heads // num_heads
    require(set_d_head >= min_head_dim, "set_only: set_state_dim head dimension too small")
    if pooling_multihead:
        require(d_model % num_heads == 0, "set_only: pooling_multihead requires d_model divisible by num_heads")
    if multiresolution_enabled:
        d_phi_for_heads = int(d_phi) if d_phi is not None else int(d_model)
        require(
            d_phi_for_heads % num_heads == 0,
            "set_only: d_phi must be divisible by num_heads for multiresolution",
        )
        for idx, group in enumerate(multiresolution_groups):
            group_heads = int(group.get("num_heads"))
            require(
                (int(d_model) * group_heads) % num_heads == 0,
                f"set_only: multiresolution.groups[{idx}] d_model share must be integral",
            )
            require(
                (set_state_dim_for_heads * group_heads) % num_heads == 0,
                f"set_only: multiresolution.groups[{idx}] set_state_dim share must be integral",
            )
            require(
                (d_phi_for_heads * group_heads) % num_heads == 0,
                f"set_only: multiresolution.groups[{idx}] d_phi share must be integral",
            )
            group_dim = (set_state_dim_for_heads * group_heads) // num_heads
            require(
                group_dim % group_heads == 0,
                f"set_only: multiresolution.groups[{idx}] set_state_dim share must be divisible by group heads",
            )
            require(
                group_dim // group_heads >= min_head_dim,
                f"set_only: multiresolution.groups[{idx}] set head dimension too small",
            )

    backend = model.get("backend")
    raw_backend_params = model.get("backend_params")
    backend_params = raw_backend_params or {}
    if raw_backend_params is not None and not isinstance(raw_backend_params, dict):
        raise ConfigError("backend_params must be a mapping")
    if backend_params:
        model["backend_params"] = backend_params
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
        raise ConfigError(f"set_only: {NYSTROM_DEPRECATION_MESSAGE}")
    elif backend == "landmark":
        if "num_landmarks" in backend_params:
            raise ConfigError(
                "landmark backend uses backend_params.landmark_coverage; "
                "num_landmarks is reserved for deprecated nystrom paths"
            )
        landmark_coverage = _require_positive_float(
            backend_params.get("landmark_coverage", 0.25),
            "landmark backend_params.landmark_coverage",
        )
        backend_params["landmark_coverage"] = landmark_coverage
    elif backend == "linformer":
        require("k" in backend_params, "linformer backend requires backend_params.k")
        require(backend_params["k"] >= min_landmarks, "linformer k too small")
        require(backend_params["k"] <= max_sets, "linformer k must be <= max_sets")
    elif backend == "exact":
        forbid(bool(backend_params), "exact backend forbids backend_params")

    if backend_params:
        model["backend_params"] = backend_params

    if backend == "nystrom":
        num_landmarks = backend_params.get("num_landmarks", 0)
        if num_landmarks and num_landmarks < warn_landmark_min:
            _warn("num_landmarks is very small; approximation may be ineffective.")
        if num_landmarks and num_landmarks > int(max_sets * warn_landmark_ratio):
            _warn("num_landmarks is large relative to max_sets; approximation may be wasteful.")
    if backend == "landmark":
        landmark_count = min(
            max(round(float(backend_params.get("landmark_coverage", 0.25)) * max_sets), min_landmarks),
            max_sets,
        )
        if landmark_count < warn_landmark_min:
            _warn("landmark_count is very small; approximation may be ineffective.")
        if landmark_count > int(max_sets * warn_landmark_ratio):
            _warn("landmark_count is large relative to max_sets; approximation may be wasteful.")

    router_type = model.get("router_type", "uniform")
    router_topk = model.get("router_topk", None)
    router_temperature = model.get("router_temperature", 1.0)
    require(
        isinstance(router_temperature, (int, float)),
        "set_only: router_temperature must be numeric",
    )
    require(float(router_temperature) > 0.0, "set_only: router_temperature must be > 0")
    router_cfg = model.get("router", {})
    if router_cfg is None:
        router_cfg = {}
    if not isinstance(router_cfg, dict):
        raise ConfigError("set_only: router must be a mapping")
    router_min_temp = router_cfg.get("min_temp", 0.5)
    _require_positive_float(router_min_temp, "set_only: router.min_temp")
    router_score_mode = router_cfg.get("score_mode", "candidate_gather")
    require(
        router_score_mode in {"candidate_gather", "dense"},
        "set_only: router.score_mode must be 'candidate_gather' or 'dense'",
    )
    if router_type == "learned":
        require(router_topk is not None, "learned router requires router_topk")
        require(router_topk >= 1, "learned router_topk must be >= 1")
        require(router_topk <= max_sets, "learned router_topk must be <= max_sets")
    else:
        if router_topk is not None:
            _warn("router_topk is ignored for uniform router")
        if router_multihead:
            _warn("router_multihead is ignored when router_type=uniform")
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
    if feature_params and not isinstance(feature_params, dict):
        raise ConfigError("set_only: feature_params must be a mapping")
    if "num_bins" in feature_params:
        _require_positive_int(
            feature_params["num_bins"],
            "set_only: feature_params.num_bins",
        )
    if "hash_seed" in feature_params:
        _require_int(
            feature_params["hash_seed"],
            "set_only: feature_params.hash_seed",
        )
    if "normalize" in feature_params:
        _require_bool(
            feature_params["normalize"],
            "set_only: feature_params.normalize",
        )
    allow_unsafe = bool(feature_params.get("allow_unsafe") or os.environ.get("SET_ATTENTION_KERNEL_ALLOW_UNSAFE") == "1")
    if feature_mode == "kernel":
        if max_sets > kernel_max_sets and not allow_unsafe:
            raise ConfigError(
                "Kernel features forbidden for max_sets above limit; set allow_unsafe to override."
            )
        if backend == "local_band":
            _warn("Kernel features with local_band backend may be redundant.")

    adapter_type = model.get("adapter_type", "auto")
    require(
        adapter_type in {"auto", "linear", "nonlinear", "hybrid"},
        "adapter_type must be auto, linear, nonlinear, or hybrid",
    )
    adapter_hidden_multiplier = model.get("adapter_hidden_multiplier", 2)
    require(adapter_hidden_multiplier > 0, "adapter_hidden_multiplier must be > 0")
    d_phi_for_adapter = model.get("d_phi")
    if d_phi_for_adapter is None:
        d_phi_for_adapter = d_model
    effective_rank = min(int(d_phi_for_adapter), set_d_head)
    if adapter_type == "linear" and max_sets < 2:
        raise ConfigError("Linear adapter requires at least 2 sets")
    if adapter_type == "auto" and effective_rank < adapter_min_rank:
        _warn("Auto adapter switched to nonlinear due to low effective rank")
    elif adapter_type == "linear" and effective_rank < adapter_min_rank:
        _warn("Linear adapter rank-limited; consider nonlinear")

    fingerprint = _fingerprint(cfg)
    cfg["_fingerprint"] = fingerprint
    _record_fingerprint(cfg, fingerprint)
    return cfg
