import pytest

from config.schema import ConfigError, validate_config


def test_rejects_mixed_keys():
    cfg = {
        "model": {
            "family": "baseline_token",
            "architecture": "transformer_lm",
            "vocab_size": 100,
            "d_model": 64,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "max_seq_len": 32,
            "backend": "dense_exact",
        },
        "data": {"dataset": "wikitext2", "batch_size": 2, "seq_len": 32},
        "training": {"epochs": 1, "lr": 1e-3},
    }
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_accepts_set_only():
    cfg = {
        "model": {
            "family": "set_only",
            "vocab_size": 100,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 2,
            "window_size": 8,
            "stride": 4,
            "dropout": 0.1,
            "max_seq_len": 32,
            "router_type": "uniform",
            "router_topk": 0,
            "backend": "dense_exact",
            "feature_mode": "geometry_only",
        },
        "data": {"dataset": "wikitext2", "batch_size": 2, "seq_len": 32},
        "training": {"epochs": 1, "lr": 1e-3},
    }
    validate_config(cfg)


def test_accepts_logging_block():
    cfg = {
        "model": {
            "family": "baseline_token",
            "architecture": "transformer_lm",
            "vocab_size": 100,
            "d_model": 64,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "max_seq_len": 32,
        },
        "data": {"dataset": "wikitext2", "batch_size": 2, "seq_len": 32},
        "training": {"epochs": 1, "lr": 1e-3},
        "logging": {
            "wandb": {"enable": False, "project": "test", "tags": ["a"], "run_name": "run"},
            "csv": {"path": "out/metrics.csv"},
        },
    }
    validate_config(cfg)


def test_rejects_logging_keys():
    cfg = {
        "model": {
            "family": "baseline_token",
            "architecture": "transformer_lm",
            "vocab_size": 100,
            "d_model": 64,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "max_seq_len": 32,
        },
        "data": {"dataset": "wikitext2", "batch_size": 2, "seq_len": 32},
        "training": {"epochs": 1, "lr": 1e-3},
        "logging": {"wandb": {"invalid": True}},
    }
    with pytest.raises(ConfigError):
        validate_config(cfg)
