from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.compatibility import validate_compatibility  # noqa: E402
from config.normalize import normalize_config  # noqa: E402
from config.schema import ConfigError, validate_config  # noqa: E402


def _base_config() -> dict:
    return {
        "model": {
            "implementation": "baseline_token",
            "attention_family": "dense",
            "backend": "exact",
            "architecture": "transformer_lm",
            "vocab_size": 32,
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "max_seq_len": 8,
            "causal": True,
        },
        "data": {
            "dataset": "wikitext2",
            "batch_size": 2,
            "seq_len": 8,
        },
        "training": {
            "seed": 0,
            "epochs": 1,
            "lr": 1e-4,
            "deterministic": True,
            "strict_deterministic": True,
            "benchmark_mode": False,
            "checkpoint": {
                "save_final": True,
            },
        },
    }


def test_checkpoint_and_metric_config_is_normalized_and_valid() -> None:
    cfg = normalize_config(_base_config())
    cfg["logging"]["metric_columns"] = ["val/query_accuracy"]
    validate_config(cfg)
    validate_compatibility(cfg)
    assert cfg["training"]["checkpoint"]["save_final"] is True
    assert cfg["training"]["checkpoint"]["save_every_epochs"] == 0
    assert cfg["logging"]["metric_columns"] == ["val/query_accuracy"]


def test_resume_and_eval_only_are_mutually_exclusive() -> None:
    cfg = normalize_config(_base_config())
    cfg["training"]["checkpoint"]["resume_from"] = "resume.pt"
    cfg["training"]["checkpoint"]["eval_only_from"] = "eval.pt"
    validate_config(cfg)
    with pytest.raises(ConfigError, match="mutually exclusive"):
        validate_compatibility(cfg)


def test_strict_determinism_requires_deterministic_mode() -> None:
    cfg = _base_config()
    cfg["training"]["deterministic"] = False
    normalized = normalize_config(cfg)
    with pytest.raises(ConfigError, match="requires"):
        validate_config(normalized)


def test_mqar_is_normalized_as_strict_past_causal_task() -> None:
    cfg = _base_config()
    cfg["data"]["dataset"] = "mqar"
    cfg["model"]["implementation"] = "set_only"
    cfg["model"].pop("causal")
    cfg["model"]["token_mlp"] = {"enabled": False}
    cfg["model"]["output_residual_mode"] = "anchor_span"
    normalized = normalize_config(cfg)
    assert normalized["task"] == "mqar"
    assert normalized["data"]["task"] == "mqar"
    assert normalized["model"]["set_causality_mode"] == "strict_past"
