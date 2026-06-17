from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.load import load_config  # noqa: E402
from config.schema import ConfigError  # noqa: E402
from models.baseline_token import TransformerLM  # noqa: E402


def _make_model(backend: str, backend_params: dict | None = None) -> TransformerLM:
    torch.manual_seed(123)
    family = "sparse" if backend == "local_band" else "linear"
    return TransformerLM(
        vocab_size=97,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        max_seq_len=24,
        attention_family=family,
        backend=backend,
        backend_params=backend_params or {},
        causal=True,
    ).eval()


def _assert_future_perturbation_causal(backend: str, backend_params: dict | None = None) -> None:
    model = _make_model(backend, backend_params)
    base = torch.arange(24, dtype=torch.long).unsqueeze(0) % 97
    for t in [0, 1, 4, 8, 12, 20]:
        for seed in range(5):
            perturbed = base.clone()
            gen = torch.Generator().manual_seed(1000 + 97 * t + seed)
            if t + 1 < base.shape[1]:
                perturbed[:, t + 1 :] = torch.randint(
                    0,
                    97,
                    perturbed[:, t + 1 :].shape,
                    generator=gen,
                    dtype=torch.long,
                )
            with torch.no_grad():
                logits_base = model(base)
                logits_pert = model(perturbed)
            torch.testing.assert_close(
                logits_base[:, : t + 1],
                logits_pert[:, : t + 1],
                atol=1e-5,
                rtol=1e-5,
            )


def test_baseline_local_band_future_perturbation_causal():
    _assert_future_perturbation_causal("local_band", {"radius": 4})


def test_baseline_landmark_future_perturbation_causal():
    _assert_future_perturbation_causal("landmark", {"landmark_coverage": 0.25})


def test_baseline_control_configs_parse_and_resolve():
    sparse = load_config("configs/paper_lr_norm/baseline_sparse_local_band.yaml")
    linear = load_config("configs/paper_lr_norm/baseline_linear_landmark.yaml")
    assert sparse["model"]["implementation"] == "baseline_token"
    assert sparse["model"]["backend"] == "local_band"
    assert sparse["model"]["backend_params"]["radius"] == 4
    assert linear["model"]["implementation"] == "baseline_token"
    assert linear["model"]["backend"] == "landmark"
    assert linear["model"]["backend_params"]["landmark_coverage"] == 0.25
    model = TransformerLM(
        vocab_size=101,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=512,
        attention_family="linear",
        backend="landmark",
        backend_params=linear["model"]["backend_params"],
        causal=True,
    )
    assert model.get_resolved_metadata()["landmark_coverage"] == 0.25
    assert model.get_resolved_metadata()["landmark_count"] == 128


def test_baseline_control_validation_rejects_bad_backend_params():
    cfg = load_config("configs/paper_lr_norm/baseline_sparse_local_band.yaml")
    bad = copy.deepcopy(cfg)
    bad["model"]["backend_params"] = {}
    try:
        from config.compatibility import validate_compatibility

        validate_compatibility(bad)
    except ConfigError as exc:
        assert "local_band backend requires backend_params.radius" in str(exc)
    else:
        raise AssertionError("missing local_band radius should fail validation")


if __name__ == "__main__":
    test_baseline_local_band_future_perturbation_causal()
    test_baseline_landmark_future_perturbation_causal()
    test_baseline_control_configs_parse_and_resolve()
    test_baseline_control_validation_rejects_bad_backend_params()
    print("baseline backend control tests passed")
