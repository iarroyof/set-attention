from __future__ import annotations

import math

import torch

from src.models.set_only.diagnostics import SetDiagnostics
from src.models.set_only.router import LearnedRouter


def _make_router(score_mode: str, multihead: bool, topk: int) -> LearnedRouter:
    router = LearnedRouter(
        d_model=12,
        set_dim=12,
        desc_dim=12,
        num_heads=3,
        d_phi=5,
        topk=topk,
        restrict_to_sets=True,
        multihead=multihead,
        min_temp=0.5,
        score_mode=score_mode,
    )
    router.temperature.fill_(1.0)
    return router


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(2026)
    token_states = torch.randn(2, 7, 12)
    set_states = torch.randn(2, 5, 12)
    desc_router = torch.randn(2, 5, 12)
    token_to_sets = torch.tensor(
        [
            [-1, -1, -1],
            [0, -1, -1],
            [0, 1, -1],
            [1, 2, -1],
            [2, 3, 4],
            [3, 4, -1],
            [4, -1, -1],
        ],
        dtype=torch.long,
    )
    return token_states, set_states, desc_router, token_to_sets


def _assert_outputs_match(multihead: bool, topk: int) -> None:
    dense = _make_router("dense", multihead=multihead, topk=topk)
    gather = _make_router("candidate_gather", multihead=multihead, topk=topk)
    gather.load_state_dict(dense.state_dict())
    token_states, set_states, desc_router, token_to_sets = _inputs()

    out_dense = dense(token_states, set_states, desc_router, token_to_sets)
    out_gather = gather(token_states, set_states, desc_router, token_to_sets)

    assert out_gather.prob_indices is token_to_sets
    assert out_gather.probs.shape[-1] == token_to_sets.shape[-1]
    assert torch.allclose(out_gather.token_repr, out_dense.token_repr, atol=1e-6, rtol=1e-6)
    assert torch.equal(out_gather.bank_indices, out_dense.bank_indices)

    idx_safe = token_to_sets.clamp(min=0, max=out_dense.num_sets - 1)
    valid = token_to_sets >= 0
    if multihead:
        idx = idx_safe.view(1, 1, idx_safe.shape[0], idx_safe.shape[1]).expand(
            out_dense.probs.shape[0],
            out_dense.probs.shape[1],
            -1,
            -1,
        )
        dense_local = out_dense.probs.gather(dim=-1, index=idx)
        dense_local = dense_local * valid.view(1, 1, valid.shape[0], valid.shape[1])
    else:
        idx = idx_safe.view(1, idx_safe.shape[0], idx_safe.shape[1]).expand(
            out_dense.probs.shape[0],
            -1,
            -1,
        )
        dense_local = out_dense.probs.gather(dim=-1, index=idx)
        dense_local = dense_local * valid.view(1, valid.shape[0], valid.shape[1])
    assert torch.allclose(out_gather.probs, dense_local, atol=1e-6, rtol=1e-6)


def test_candidate_gather_matches_dense_multihead_full_candidate_softmax():
    _assert_outputs_match(multihead=True, topk=16)


def test_candidate_gather_matches_dense_multihead_top1():
    _assert_outputs_match(multihead=True, topk=1)


def test_candidate_gather_matches_dense_singlehead_top1():
    _assert_outputs_match(multihead=False, topk=1)


def test_compact_router_diagnostics_match_dense_candidate_metrics():
    dense = _make_router("dense", multihead=True, topk=16)
    gather = _make_router("candidate_gather", multihead=True, topk=16)
    gather.load_state_dict(dense.state_dict())
    token_states, set_states, desc_router, token_to_sets = _inputs()

    dense_out = dense(token_states, set_states, desc_router, token_to_sets)
    gather_out = gather(token_states, set_states, desc_router, token_to_sets)

    dense_diag = SetDiagnostics()
    dense_diag.update_with_router_state(
        bank_indices=dense_out.bank_indices,
        num_sets=dense_out.num_sets,
        router_probs=dense_out.probs,
        token_to_sets=token_to_sets,
    )
    dense_stats = dense_diag.get_epoch_stats()

    gather_diag = SetDiagnostics()
    gather_diag.update_with_router_state(
        bank_indices=gather_out.bank_indices,
        num_sets=gather_out.num_sets,
        router_probs=gather_out.probs,
        router_prob_indices=gather_out.prob_indices,
        token_to_sets=token_to_sets,
    )
    gather_stats = gather_diag.get_epoch_stats()

    for key in (
        "ausa/router_entropy_norm",
        "ausa/router_top1_weight",
        "ausa/router_top1_gap_norm",
        "ausa/candidate_count_mean",
        "ausa/candidate_count_max",
        "ausa/router_candidate_count_eff_mean",
    ):
        assert math.isclose(dense_stats[key], gather_stats[key], rel_tol=1e-6, abs_tol=1e-6)


if __name__ == "__main__":
    test_candidate_gather_matches_dense_multihead_full_candidate_softmax()
    test_candidate_gather_matches_dense_multihead_top1()
    test_candidate_gather_matches_dense_singlehead_top1()
    test_compact_router_diagnostics_match_dense_candidate_metrics()
