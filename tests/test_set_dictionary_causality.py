from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from models.set_only import SetOnlyLM  # noqa: E402


SEQ_LEN = 10
VOCAB_SIZE = 64


def _make_model(anchor_enabled: bool) -> SetOnlyLM:
    torch.manual_seed(1234)
    model = SetOnlyLM(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        num_layers=1,
        num_heads=4,
        window_size=4,
        stride=2,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        max_seq_len=SEQ_LEN,
        dim_feedforward=32,
        pooling="mean",
        d_phi=16,
        geometry={"enabled": True, "apply_as_bias": True, "apply_in_phi_attn": True},
        router_type="learned",
        router_topk=3,
        router_multihead=True,
        router_temperature=1.0,
        backend="exact",
        feature_mode="hashed_counts",
        feature_params={"num_bins": 32},
        token_mlp=False,
        set_causality_mode="strict_past",
        output_residual_mode="anchor_span",
        anchor={
            "enabled": anchor_enabled,
            "pre_encoder_layers": 2,
            "lambda_h": 0.1,
        },
    )
    model.eval()
    return model


def _base_input() -> torch.Tensor:
    return torch.tensor(
        [
            [3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        ],
        dtype=torch.long,
    )


def _perturb_future(input_ids: torch.Tensor, t: int) -> torch.Tensor:
    perturbed = input_ids.clone()
    if t + 1 >= input_ids.size(1):
        return perturbed
    offsets = torch.arange(1, perturbed[:, t + 1 :].numel() + 1).view_as(
        perturbed[:, t + 1 :]
    )
    perturbed[:, t + 1 :] = (perturbed[:, t + 1 :] + offsets) % VOCAB_SIZE
    return perturbed


def test_anchor_span_logits_do_not_change_under_future_perturbations():
    model = _make_model(anchor_enabled=False)
    input_ids = _base_input()
    with torch.no_grad():
        base_logits = model(input_ids)

    for t in range(input_ids.size(1)):
        perturbed = _perturb_future(input_ids, t)
        with torch.no_grad():
            logits = model(perturbed)
        assert torch.allclose(logits[:, : t + 1], base_logits[:, : t + 1], atol=1e-5)


def test_anchor_pre_encoder_target_do_not_change_under_future_perturbations():
    model = _make_model(anchor_enabled=True)
    input_ids = _base_input()
    with torch.no_grad():
        base_target = model.compute_anchor_target(input_ids)

    for t in range(input_ids.size(1)):
        perturbed = _perturb_future(input_ids, t)
        with torch.no_grad():
            target = model.compute_anchor_target(perturbed)
        assert torch.allclose(target[:, : t + 1], base_target[:, : t + 1], atol=1e-5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {Path(__file__).name}:{name}")
