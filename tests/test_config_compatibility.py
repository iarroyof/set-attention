import pytest

from config.compatibility import validate_compatibility
from config.schema import ConfigError


def _base_set_only_cfg():
    return {
        "model": {
            "implementation": "set_only",
            "attention_family": "dense",
            "backend": "exact",
            "vocab_size": 100,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "window_size": 8,
            "stride": 4,
            "dropout": 0.1,
            "max_seq_len": 32,
            "router_type": "uniform",
            "router_topk": 0,
            "feature_mode": "geometry_only",
        },
        "data": {"dataset": "wikitext2", "batch_size": 2, "seq_len": 32},
        "training": {"epochs": 1, "lr": 1e-3},
    }


def test_backend_params_required():
    cfg = _base_set_only_cfg()
    cfg["model"]["backend"] = "local_band"
    with pytest.raises(ConfigError):
        validate_compatibility(cfg)


def test_router_learned_requires_topk():
    cfg = _base_set_only_cfg()
    cfg["model"]["router_type"] = "learned"
    cfg["model"]["router_topk"] = 0
    with pytest.raises(ConfigError):
        validate_compatibility(cfg)


def test_kernel_max_sets_guard():
    cfg = _base_set_only_cfg()
    cfg["model"]["feature_mode"] = "kernel"
    cfg["data"]["seq_len"] = 2048
    cfg["model"]["max_seq_len"] = 2048
    cfg["model"]["stride"] = 1
    with pytest.raises(ConfigError):
        validate_compatibility(cfg)


def test_uniform_router_warns_on_topk():
    cfg = _base_set_only_cfg()
    with pytest.warns(RuntimeWarning):
        validate_compatibility(cfg)
