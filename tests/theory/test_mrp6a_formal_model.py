from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from models.set_only import SetOnlyLM  # noqa: E402
from models.set_only.banks import build_window_bank, num_sets_for_length  # noqa: E402


def _candidate_lists(bank):
    return [row[row >= 0].detach().cpu().tolist() for row in bank.token_to_sets]


def _formula_candidates(seq_len: int, window: int, stride: int, t: int) -> list[int]:
    if seq_len < window:
        return []
    m_count = ((seq_len - window) // stride) + 1
    endpoints = [m * stride + window - 1 for m in range(m_count)]
    return [m for m, endpoint in enumerate(endpoints) if t - window < endpoint <= t]


def test_endpoint_window_formula_matches_bank_exhaustively_small_cases():
    for seq_len in range(1, 13):
        for window in range(1, 7):
            for stride in range(1, 5):
                bank = build_window_bank(
                    seq_len=seq_len,
                    window_size=window,
                    stride=stride,
                    device=torch.device("cpu"),
                    causality_mode="strict_past",
                    candidate_fiber="endpoint_window",
                )
                assert bank.set_endpoints.numel() == num_sets_for_length(
                    seq_len, window, stride, causality_mode="strict_past"
                )
                observed = _candidate_lists(bank)
                expected = [
                    _formula_candidates(seq_len, window, stride, t)
                    for t in range(seq_len)
                ]
                assert observed == expected


def test_active_group_candidate_counts_and_empty_prefixes():
    fine = build_window_bank(16, 2, 1, torch.device("cpu"), "strict_past")
    coarse = build_window_bank(16, 4, 2, torch.device("cpu"), "strict_past")

    fine_counts = [len(c) for c in _candidate_lists(fine)]
    coarse_counts = [len(c) for c in _candidate_lists(coarse)]

    assert fine_counts[:3] == [0, 1, 2]
    assert all(count == 2 for count in fine_counts[2:])
    assert coarse_counts[:6] == [0, 0, 0, 1, 1, 2]
    assert all(count == 2 for count in coarse_counts[5:-1])


def _make_multires_model() -> SetOnlyLM:
    torch.manual_seed(7)
    model = SetOnlyLM(
        vocab_size=37,
        d_model=16,
        num_layers=1,
        num_heads=4,
        window_size=2,
        stride=1,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        max_seq_len=12,
        dim_feedforward=32,
        pooling="mean",
        d_phi=16,
        set_state_dim=16,
        geometry={"enabled": True, "apply_as_bias": True, "apply_in_phi_attn": True},
        router_type="learned",
        router_topk=4,
        router_multihead=True,
        router_temperature=1.0,
        router_min_temp=0.5,
        router_score_mode="candidate_gather",
        backend="exact",
        feature_mode="hashed_counts",
        feature_params={"num_bins": 16, "hash_seed": 3, "normalize": True},
        token_mlp=False,
        set_causality_mode="strict_past",
        output_residual_mode="anchor_span",
        candidate_fiber="endpoint_window",
        multiresolution={
            "enabled": True,
            "groups": [
                {"name": "fine", "num_heads": 3, "window_size": 2, "stride": 1},
                {"name": "coarse", "num_heads": 1, "window_size": 4, "stride": 2},
            ],
        },
    )
    model.eval()
    return model


def test_multiresolution_direct_sum_metadata_matches_width_and_bank_counts():
    model = _make_multires_model()
    metadata = {group["name"]: group for group in model.multiresolution_group_metadata}

    assert model.token_mlp_enabled is False
    assert model.output_residual_mode == "anchor_span"
    assert model.candidate_fiber == "endpoint_window"
    assert metadata["fine"]["set_state_dim"] == 12
    assert metadata["coarse"]["set_state_dim"] == 4
    assert sum(group["set_state_dim"] for group in metadata.values()) == model.set_state_dim
    assert metadata["fine"]["M"] == num_sets_for_length(12, 2, 1, "strict_past")
    assert metadata["coarse"]["M"] == num_sets_for_length(12, 4, 2, "strict_past")


def test_context_path_logit_identity_for_anchor_span():
    model = _make_multires_model()
    input_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    input_b = torch.tensor([[9, 8, 7, 6, 5, 4, 7, 8]])
    t = 6
    assert input_a[0, t].item() == input_b[0, t].item()

    with torch.no_grad():
        logits_a = model(input_a)
        logits_b = model(input_b)
        anchor_a = model._thin_anchor(input_a)
        anchor_b = model._thin_anchor(input_b)
        encoded_a = model.encode(input_a)
        encoded_b = model.encode(input_b)

    routed_span_a = encoded_a - anchor_a
    routed_span_b = encoded_b - anchor_b
    predicted_delta = model.lm_head(routed_span_a[:, t] - routed_span_b[:, t])
    observed_delta = logits_a[:, t] - logits_b[:, t]

    assert torch.allclose(anchor_a[:, t], anchor_b[:, t], atol=1e-6)
    assert torch.allclose(observed_delta, predicted_delta, atol=1e-5)


def test_block_supported_multigroup_rank_bound():
    fine = torch.tensor(
        [
            [0.7, 0.3],
            [0.1, 0.9],
            [0.4, 0.6],
        ],
        dtype=torch.float32,
    )
    coarse = torch.tensor([[1.0]], dtype=torch.float32)
    block = torch.zeros((4, 3), dtype=torch.float32)
    block[:3, :2] = fine
    block[3:, 2:] = coarse

    rank_block = torch.linalg.matrix_rank(block).item()
    rank_sum = torch.linalg.matrix_rank(fine).item() + torch.linalg.matrix_rank(coarse).item()

    assert rank_block == rank_sum
    assert rank_block <= min(3, 2) + min(1, 1)
    assert rank_block == 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {Path(__file__).name}:{name}")
