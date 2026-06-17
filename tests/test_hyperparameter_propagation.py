from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config.compatibility import validate_compatibility  # noqa: E402
from config.normalize import normalize_config  # noqa: E402
from config.schema import validate_config  # noqa: E402
from scripts.run_experiment import attach_resolved_metadata, build_model  # noqa: E402
from src.models.set_only import SetOnlyLM  # noqa: E402
from train.experiment_logger import ExperimentLogger  # noqa: E402


def _base_cfg() -> dict:
    return {
        "model": {
            "implementation": "set_only",
            "attention_family": "dense",
            "backend": "exact",
            "vocab_size": 101,
            "d_model": 128,
            "num_layers": 1,
            "num_heads": 4,
            "dim_feedforward": 256,
            "window_size": 8,
            "stride": 4,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "resid_dropout": 0.0,
            "ffn_dropout": 0.0,
            "max_seq_len": 32,
            "router_type": "learned",
            "router_topk": 3,
            "router_multihead": True,
            "feature_mode": "hashed_counts",
            "token_mlp": False,
            "causal": True,
        },
        "data": {"dataset": "wikitext2", "batch_size": 2, "seq_len": 32},
        "training": {"epochs": 1, "lr": 1e-3, "seed": 0},
    }


def _validated(cfg: dict) -> dict:
    cfg = normalize_config(cfg)
    validate_config(cfg)
    return validate_compatibility(cfg)


def test_a1_4_normalize_preserves_canonical_defaults():
    cfg = _validated(_base_cfg())
    model = cfg["model"]

    assert model["pooling"]["alpha"] == 10.0
    assert model["pooling"]["learnable_alpha"] is False
    assert model["feature_params"]["hash_seed"] == 13
    assert model["feature_params"]["normalize"] is True
    assert model["feature_params"]["num_bins"] == 128
    assert model["router"]["min_temp"] == 0.5
    assert model["router"]["score_mode"] == "candidate_gather"
    assert model["d_phi"] is None
    assert model["set_state_dim"] is None
    assert model["adapter_type"] == "auto"
    assert model["output_residual_mode"] == "direct"


def test_a1_4_config_values_reach_set_only_submodules():
    raw = _base_cfg()
    raw["model"]["pooling"] = {
        "mode": "soft_trimmed_boltzmann",
        "tau": 0.2,
        "q": 0.75,
        "alpha": 7.5,
        "learnable_alpha": True,
    }
    raw["model"]["feature_params"] = {
        "num_bins": 17,
        "hash_seed": 99,
        "normalize": False,
    }
    raw["model"]["router"] = {"min_temp": 0.25}
    raw["model"]["set_state_dim"] = 192
    cfg = _validated(raw)

    torch.manual_seed(0)
    model = build_model(cfg["model"])

    assert model.d_phi == 128
    assert model.set_state_dim == 192
    assert model.resolved_adapter_type == "linear"
    assert model.pooling_module is not None
    assert model.pooling_module._alpha_value() == 7.5
    assert isinstance(model.pooling_module.alpha, torch.nn.Parameter)
    assert model.router.min_temp == 0.25
    assert model.router.score_mode == "candidate_gather"
    assert model.feature_builder.num_bins == 17
    assert model.feature_builder.hash_seed == 99
    assert model.feature_builder.normalize is False


def test_a1_4_resolved_metadata_reaches_json_and_csv():
    raw = _base_cfg()
    raw["model"]["feature_params"] = {"num_bins": 19, "hash_seed": 23}
    raw["model"]["router"] = {"min_temp": 0.75}
    raw["model"]["set_state_dim"] = 192
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw["training"]["output_dir"] = str(tmp_path)
        raw["logging"] = {
            "wandb": {"enable": False},
            "csv": {"path": str(tmp_path / "metrics.csv")},
        }
        cfg = _validated(raw)
        torch.manual_seed(0)
        model = build_model(cfg["model"])
        attach_resolved_metadata(cfg, model)

        logger = ExperimentLogger(config=cfg)
        logger.log_model_complexity(model)
        logger.start_epoch(num_train_samples=2)
        logger.log_epoch(
            1,
            train_metrics={"loss": 1.0},
            val_metrics={"loss": 1.1},
            set_diagnostics=None,
        )
        logger.finish()

        json_payload = json.loads((tmp_path / "metrics.json").read_text())
        assert json_payload["resolved.d_phi"] == 128
        assert json_payload["resolved.set_state_dim"] == 192
        assert json_payload["resolved.adapter_type"] == "linear"
        assert json_payload["resolved.router_min_temp"] == 0.75
        assert json_payload["resolved.router_score_mode"] == "candidate_gather"
        assert json_payload["resolved.pooling_alpha"] == 10.0
        assert json_payload["resolved.hash_seed"] == 23
        assert json_payload["resolved.hash_normalize"] is True
        assert json_payload["resolved.hash_num_bins"] == 19
        assert json_payload["resolved.output_residual_mode"] == "direct"

        rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
        assert rows[0]["resolved.d_phi"] == "128"
        assert rows[0]["resolved.set_state_dim"] == "192"
        assert rows[0]["resolved.adapter_type"] == "linear"
        assert rows[0]["resolved.router_min_temp"] == "0.75"
        assert rows[0]["resolved.router_score_mode"] == "candidate_gather"
        assert rows[0]["resolved.pooling_alpha"] == "10.0"
        assert rows[0]["resolved.hash_seed"] == "23"
        assert rows[0]["resolved.hash_normalize"] == "True"
        assert rows[0]["resolved.hash_num_bins"] == "19"
        assert rows[0]["resolved.output_residual_mode"] == "direct"


def test_a1_4_explicit_defaults_preserve_eval_forward_fingerprint():
    base_kwargs = {
        "vocab_size": 97,
        "d_model": 32,
        "num_layers": 1,
        "num_heads": 4,
        "window_size": 8,
        "stride": 4,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "resid_dropout": 0.0,
        "ffn_dropout": 0.0,
        "max_seq_len": 16,
        "dim_feedforward": 64,
        "router_type": "learned",
        "router_topk": 2,
        "router_multihead": True,
        "router_temperature": 1.0,
        "backend": "exact",
        "feature_mode": "hashed_counts",
        "token_mlp": False,
        "causal": True,
        "set_causality_mode": "strict_past",
    }
    explicit_defaults = {
        "pooling": {"mode": "mean", "alpha": 10.0, "learnable_alpha": False},
        "feature_params": {"num_bins": 128, "hash_seed": 13, "normalize": True},
        "router_min_temp": 0.5,
        "router_score_mode": "candidate_gather",
        "d_phi": None,
        "set_state_dim": None,
        "adapter_type": "auto",
    }
    input_ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.long)

    torch.manual_seed(2026)
    implicit_model = SetOnlyLM(**base_kwargs)
    implicit_model.eval()
    torch.manual_seed(2026)
    explicit_model = SetOnlyLM(**base_kwargs, **explicit_defaults)
    explicit_model.eval()

    with torch.no_grad():
        implicit_logits = implicit_model(input_ids)
        explicit_logits = explicit_model(input_ids)

    assert torch.equal(implicit_logits, explicit_logits)


def test_set_state_dim_changes_set_stack_width_without_changing_lm_head_width():
    cfg = _validated(_base_cfg())
    cfg["model"]["set_state_dim"] = 192
    cfg = _validated(cfg)
    torch.manual_seed(0)
    model = build_model(cfg["model"])
    model.eval()
    input_ids = torch.randint(0, cfg["model"]["vocab_size"], (2, 16))

    with torch.no_grad():
        logits = model(input_ids)

    assert model.set_state_dim == 192
    assert model.set_input_proj.out_features == 192
    assert model.set_output_proj.in_features == 192
    assert model.blocks[0].norm1.normalized_shape == (192,)
    assert model.router.set_dim == 192
    assert model.lm_head.in_features == cfg["model"]["d_model"]
    assert logits.shape == (2, 16, cfg["model"]["vocab_size"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {Path(__file__).name}:{name}")
