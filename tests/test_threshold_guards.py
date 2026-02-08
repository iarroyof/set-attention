import pytest

from config.schema import ConfigError, validate_config


def test_kernel_features_reject_large_sets():
    cfg = {
        "model": {
            "implementation": "set_only",
            "attention_family": "dense",
            "backend": "exact",
            "vocab_size": 100,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 2,
            "window_size": 1,
            "stride": 1,
            "dropout": 0.1,
            "max_seq_len": 1024,
            "router_type": "uniform",
            "router_topk": 0,
            "feature_mode": "kernel",
        },
        "data": {"dataset": "wikitext2", "batch_size": 2, "seq_len": 32},
        "training": {"epochs": 1, "lr": 1e-3},
    }
    with pytest.raises(ConfigError):
        validate_config(cfg)
