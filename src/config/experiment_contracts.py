from __future__ import annotations

from typing import Any

from config.schema import ConfigError


SD_GRID_CONTRACT = "sd_grid_seeded_v1"
SD_GRID_DIAGNOSTICS_CONTRACT = "current_matrix_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(f"{SD_GRID_CONTRACT}: {message}")


def _value(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _require_equal(cfg: dict[str, Any], path: str, expected: Any) -> None:
    actual = _value(cfg, path)
    if isinstance(expected, float):
        try:
            actual_float = float(actual)
        except (TypeError, ValueError):
            _require(False, f"{path} must be numeric {expected!r}, got {actual!r}")
            return
        _require(
            abs(actual_float - expected) <= 1e-12,
            f"{path} must be {expected!r}, got {actual!r}",
        )
        return
    _require(actual == expected, f"{path} must be {expected!r}, got {actual!r}")


def _validate_sd_grid_seeded_v1(cfg: dict[str, Any]) -> None:
    seed = _value(cfg, "training.seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed in range(5),
        f"training.seed must be one of 0..4, got {seed!r}",
    )
    for path, expected in {
        "training.epochs": 10,
        "training.lr": 0.0001,
        "training.warmup_steps": 1000,
        "training.deterministic": True,
        "training.benchmark_mode": False,
        "training.diagnostics_contract": SD_GRID_DIAGNOSTICS_CONTRACT,
        "data.dataset": "wikitext2",
        "model.attention_family": "dense",
        "model.backend": "exact",
        "model.d_model": 384,
        "model.dim_feedforward": 1536,
        "model.num_layers": 6,
        "model.num_heads": 8,
        "model.dropout": 0.1,
        "model.attn_dropout": 0.1,
        "model.resid_dropout": 0.1,
        "model.ffn_dropout": 0.1,
    }.items():
        _require_equal(cfg, path, expected)

    _require(
        _value(cfg, "data.limit") is None and _value(cfg, "data.val_limit") is None,
        "data.limit and data.val_limit must be absent/null (full dataset only)",
    )
    seq_len = _value(cfg, "data.seq_len")
    batch_size = _value(cfg, "data.batch_size")
    _require(
        (seq_len, batch_size)
        in {
            (512, 16),
            (512, 4),
            (512, 3),
            (1024, 3),
            (1024, 4),
            (2048, 3),
            (2048, 4),
            (3584, 3),
            (3584, 4),
            (4096, 3),
            (4096, 4),
        },
        f"unsupported (data.seq_len,data.batch_size)=({seq_len!r},{batch_size!r})",
    )
    _require_equal(cfg, "model.max_seq_len", seq_len)

    implementation = _value(cfg, "model.implementation")
    _require(
        implementation in {"baseline_token", "set_only"},
        f"model.implementation must be baseline_token or set_only, got {implementation!r}",
    )
    _require(
        not bool(_value(cfg, "model.backend_params", {})),
        "exact backend requires empty/absent model.backend_params",
    )

    if implementation == "baseline_token":
        _require_equal(cfg, "model.architecture", "transformer_lm")
        _require_equal(cfg, "model.causal", True)
        return

    for path, expected in {
        "model.d_phi": 384,
        "model.set_state_dim": 384,
        "model.set_causality_mode": "strict_past",
        "model.output_residual_mode": "anchor_span",
        "model.allow_token_token": False,
        "model.candidate_fiber": "endpoint_window",
        "model.multiresolution.enabled": True,
        "model.token_mlp.enabled": False,
        "model.anchor.enabled": False,
        "model.anchor.teacher.enabled": False,
        "model.set_diversity.lambda_div": 0.0,
        "model.multivector_basis.enabled": False,
        "model.multivector_basis.r": 1,
        "model.pooling.mode": "soft_trimmed_boltzmann",
        "model.pooling.tau": 0.1,
        "model.pooling.q": 0.85,
        "model.router_type": "learned",
        "model.router_topk": 16,
        "model.router_multihead": True,
        "model.router.score_mode": "candidate_gather",
    }.items():
        _require_equal(cfg, path, expected)

    groups = _value(cfg, "model.multiresolution.groups")
    _require(isinstance(groups, list) and groups, "multiresolution.groups must be non-empty")
    expected_topology = {
        "fine": (2, 1),
        "coarse": (4, 2),
    }
    names: list[str] = []
    total_heads = 0
    for group in groups:
        _require(isinstance(group, dict), "every multiresolution group must be a mapping")
        name = str(group.get("name", ""))
        _require(name in expected_topology, f"unexpected multiresolution group {name!r}")
        _require(name not in names, f"duplicate multiresolution group name {name!r}")
        names.append(name)
        expected_w, expected_s = expected_topology[name]
        _require(
            (group.get("window_size"), group.get("stride")) == (expected_w, expected_s),
            f"group {name!r} must use (window,stride)=({expected_w},{expected_s})",
        )
        heads = group.get("num_heads")
        _require(
            isinstance(heads, int) and not isinstance(heads, bool) and heads > 0,
            f"group {name!r} num_heads must be a positive integer",
        )
        total_heads += heads
    _require(total_heads == 8, f"group head counts must total 8, got {total_heads}")


def validate_experiment_contract(cfg: dict[str, Any]) -> None:
    contract = _value(cfg, "training.experiment_contract")
    if contract is None:
        return
    if contract != SD_GRID_CONTRACT:
        raise ConfigError(f"Unknown training.experiment_contract {contract!r}")
    _validate_sd_grid_seeded_v1(cfg)
