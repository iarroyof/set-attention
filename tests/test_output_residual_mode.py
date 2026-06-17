from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config.compatibility import validate_compatibility  # noqa: E402
from config.normalize import normalize_config  # noqa: E402
from config.schema import ConfigError, validate_config  # noqa: E402
from models.set_only import SetOnlyLM  # noqa: E402


def _make_model(mode: str, window_size: int = 4, stride: int = 2) -> SetOnlyLM:
    torch.manual_seed(2026)
    model = SetOnlyLM(
        vocab_size=31,
        d_model=8,
        num_layers=0,
        num_heads=2,
        window_size=window_size,
        stride=stride,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        max_seq_len=8,
        dim_feedforward=16,
        pooling="mean",
        router_type="uniform",
        router_topk=0,
        backend="exact",
        feature_mode="geometry_only",
        token_mlp=False,
        causal=True,
        set_causality_mode="strict_past",
        output_residual_mode=mode,
        allow_token_token=(window_size == 1 and stride == 1),
    )
    model.eval()
    return model


def _token_states(model: SetOnlyLM, input_ids: torch.Tensor) -> torch.Tensor:
    pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    pos_ids = pos_ids.expand_as(input_ids)
    return model.token_mlp(model.token_emb(input_ids) + model.pos_emb(pos_ids))


def test_output_residual_modes_match_strict_past_candidate_fiber():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    direct = _make_model("direct")
    empty_only = _make_model("empty_only")
    none = _make_model("none")

    with torch.no_grad():
        direct_repr = direct.encode(input_ids)
        empty_repr = empty_only.encode(input_ids)
        none_repr = none.encode(input_ids)
        tokens = _token_states(direct, input_ids)

    empty_positions = slice(0, 3)
    nonempty_positions = slice(3, None)

    assert torch.allclose(direct_repr[:, empty_positions], tokens[:, empty_positions])
    assert torch.allclose(empty_repr[:, empty_positions], tokens[:, empty_positions])
    assert torch.allclose(none_repr[:, empty_positions], torch.zeros_like(none_repr[:, empty_positions]))

    assert torch.allclose(empty_repr[:, nonempty_positions], none_repr[:, nonempty_positions])
    assert torch.allclose(
        direct_repr[:, nonempty_positions],
        empty_repr[:, nonempty_positions] + tokens[:, nonempty_positions],
    )


def test_empty_only_matches_none_when_singleton_bank_has_no_empty_tokens():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    direct = _make_model("direct", window_size=1, stride=1)
    empty_only = _make_model("empty_only", window_size=1, stride=1)
    none = _make_model("none", window_size=1, stride=1)

    with torch.no_grad():
        direct_repr = direct.encode(input_ids)
        empty_repr = empty_only.encode(input_ids)
        none_repr = none.encode(input_ids)
        tokens = _token_states(direct, input_ids)

    assert torch.allclose(empty_repr, none_repr)
    assert torch.allclose(direct_repr, none_repr + tokens)


def test_output_residual_mode_config_validation():
    cfg = {
        "model": {
            "implementation": "set_only",
            "attention_family": "dense",
            "backend": "exact",
            "vocab_size": 31,
            "d_model": 32,
            "num_layers": 1,
            "num_heads": 4,
            "dim_feedforward": 64,
            "window_size": 4,
            "stride": 2,
            "dropout": 0.0,
            "max_seq_len": 8,
            "router_type": "uniform",
            "router_topk": 0,
            "feature_mode": "geometry_only",
            "token_mlp": False,
            "causal": True,
            "output_residual_mode": "invalid",
        },
        "data": {"dataset": "wikitext2", "batch_size": 1, "seq_len": 8},
        "training": {"epochs": 1, "lr": 1e-3, "seed": 0},
    }
    normalized = normalize_config(cfg)
    try:
        validate_config(normalized)
    except ConfigError as exc:
        assert "output_residual_mode" in str(exc)
    else:
        raise AssertionError("invalid output_residual_mode should fail schema validation")

    cfg["model"]["output_residual_mode"] = "empty_only"
    normalized = normalize_config(cfg)
    validate_config(normalized)
    validate_compatibility(normalized)


def test_geometry_only_learned_router_batches_descriptors():
    torch.manual_seed(2026)
    model = SetOnlyLM(
        vocab_size=31,
        d_model=16,
        num_layers=1,
        num_heads=4,
        window_size=1,
        stride=1,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        max_seq_len=8,
        dim_feedforward=32,
        pooling="mean",
        router_type="learned",
        router_topk=4,
        router_multihead=True,
        backend="exact",
        feature_mode="geometry_only",
        geometry={"enabled": False},
        token_mlp=False,
        causal=True,
        set_causality_mode="strict_past",
        output_residual_mode="empty_only",
        allow_token_token=True,
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)

    assert torch.isfinite(logits).all()
    assert logits.shape == (1, 8, 31)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {Path(__file__).name}:{name}")
