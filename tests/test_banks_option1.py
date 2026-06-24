import torch

from src.models.set_only import SetOnlyLM
from src.models.set_only.banks import build_window_bank, num_sets_for_length


def _candidate_lists(bank):
    return [
        row[row >= 0].detach().cpu().tolist()
        for row in bank.token_to_sets
    ]


def test_strict_past_bank_drops_partial_trailing_windows():
    bank = build_window_bank(
        seq_len=20,
        window_size=6,
        stride=4,
        device=torch.device("cpu"),
        causality_mode="strict_past",
    )

    assert bank.set_indices.tolist() == [
        [0, 1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8, 9],
        [8, 9, 10, 11, 12, 13],
        [12, 13, 14, 15, 16, 17],
    ]
    assert bank.set_sizes.tolist() == [6, 6, 6, 6]
    assert bank.set_endpoints.tolist() == [5, 9, 13, 17]
    assert num_sets_for_length(20, 6, 4, causality_mode="strict_past") == 4


def test_noncausal_bank_preserves_clipped_membership_mode():
    bank = build_window_bank(
        seq_len=20,
        window_size=6,
        stride=4,
        device=torch.device("cpu"),
        causality_mode="noncausal",
    )

    assert bank.set_sizes.tolist() == [6, 6, 6, 6, 4]
    assert bank.set_endpoints.tolist() == [5, 9, 13, 17, 19]
    assert _candidate_lists(bank)[0] == [0]
    assert _candidate_lists(bank)[5] == [0, 1]
    assert _candidate_lists(bank)[19] == [4]
    assert num_sets_for_length(20, 6, 4, causality_mode="noncausal") == 5


def test_option1_candidate_fiber_uses_strict_endpoint_window():
    bank = build_window_bank(
        seq_len=24,
        window_size=8,
        stride=4,
        device=torch.device("cpu"),
        causality_mode="strict_past",
    )
    candidates = _candidate_lists(bank)

    assert bank.set_endpoints.tolist() == [7, 11, 15, 19, 23]
    assert candidates[0] == []
    assert candidates[6] == []
    assert candidates[7] == [0]
    assert candidates[11] == [0, 1]
    assert candidates[15] == [1, 2]
    assert candidates[23] == [3, 4]

    for t, sets in enumerate(candidates):
        endpoints = bank.set_endpoints[sets].tolist() if sets else []
        assert all(t - 8 < endpoint <= t for endpoint in endpoints)


def test_all_past_candidate_fiber_uses_all_sealed_sets():
    bank = build_window_bank(
        seq_len=24,
        window_size=8,
        stride=4,
        device=torch.device("cpu"),
        causality_mode="strict_past",
        candidate_fiber="all_past",
    )
    candidates = _candidate_lists(bank)

    assert bank.set_endpoints.tolist() == [7, 11, 15, 19, 23]
    assert candidates[0] == []
    assert candidates[6] == []
    assert candidates[7] == [0]
    assert candidates[11] == [0, 1]
    assert candidates[15] == [0, 1, 2]
    assert candidates[23] == [0, 1, 2, 3, 4]

    for t, sets in enumerate(candidates):
        endpoints = bank.set_endpoints[sets].tolist() if sets else []
        assert all(endpoint <= t for endpoint in endpoints)


def test_reference_strict_past_set_counts():
    assert num_sets_for_length(512, 16, 8, causality_mode="strict_past") == 63
    assert num_sets_for_length(2048, 16, 8, causality_mode="strict_past") == 255

    bank = build_window_bank(
        seq_len=512,
        window_size=16,
        stride=8,
        device=torch.device("cpu"),
        causality_mode="strict_past",
    )
    assert bank.set_endpoints[0].item() == 15
    assert bank.set_endpoints[-1].item() == 511
    assert bank.set_endpoints.numel() == 63


def test_strict_past_residual_supplies_current_token_when_no_candidates():
    torch.manual_seed(0)
    model = SetOnlyLM(
        vocab_size=32,
        d_model=12,
        num_layers=0,
        num_heads=3,
        window_size=4,
        stride=2,
        max_seq_len=8,
        pooling="mean",
        router_type="uniform",
        token_mlp=False,
        causal=True,
        set_causality_mode="strict_past",
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    with torch.no_grad():
        final_states = model.encode(input_ids)
        pos_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
        direct_states = model.token_emb(input_ids) + model.pos_emb(pos_ids)

    assert torch.allclose(final_states[:, 0], direct_states[:, 0], atol=1e-6)
    assert torch.allclose(final_states[:, 1], direct_states[:, 1], atol=1e-6)
    assert torch.allclose(final_states[:, 2], direct_states[:, 2], atol=1e-6)
