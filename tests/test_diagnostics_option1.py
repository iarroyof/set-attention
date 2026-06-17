from __future__ import annotations

import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.models.set_only.banks import build_window_bank  # noqa: E402
from src.models.set_only.diagnostics import SetDiagnostics  # noqa: E402


def _probs_from_candidates(token_to_sets: torch.Tensor, num_sets: int, sharp: bool = False) -> torch.Tensor:
    probs = torch.zeros((1, token_to_sets.shape[0], num_sets), dtype=torch.float32)
    for t, row in enumerate(token_to_sets):
        valid = row[row >= 0]
        if valid.numel() == 0:
            continue
        if sharp:
            probs[0, t, int(valid[0].item())] = 1.0
        else:
            probs[0, t, valid] = 1.0 / float(valid.numel())
    return probs


def _stats(router_probs: torch.Tensor, token_to_sets: torch.Tensor) -> dict[str, float]:
    diag = SetDiagnostics()
    if router_probs.dim() == 4:
        bank_indices = router_probs.mean(dim=1).argmax(dim=-1)
        num_sets = router_probs.shape[-1]
    else:
        bank_indices = router_probs.argmax(dim=-1)
        num_sets = router_probs.shape[-1]
    diag.update_with_router_state(
        bank_indices=bank_indices,
        num_sets=num_sets,
        router_probs=router_probs,
        token_to_sets=token_to_sets,
    )
    return diag.get_epoch_stats()


def test_uniform_candidate_probs_have_normalized_entropy_one():
    token_to_sets = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    router_probs = _probs_from_candidates(token_to_sets, num_sets=4, sharp=False)
    stats = _stats(router_probs, token_to_sets)

    assert math.isclose(stats["ausa/router_entropy_norm"], 1.0, abs_tol=1e-6)
    assert math.isclose(
        stats["ausa/router_entropy_norm_by_candidates"],
        1.0,
        abs_tol=1e-6,
    )
    assert math.isclose(stats["ausa/routing_entropy_norm"], 1.0, abs_tol=1e-6)


def test_sharp_candidate_probs_have_top1_and_gap_one():
    token_to_sets = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    router_probs = _probs_from_candidates(token_to_sets, num_sets=4, sharp=True)
    stats = _stats(router_probs, token_to_sets)

    assert math.isclose(stats["ausa/router_top1_weight"], 1.0, abs_tol=1e-6)
    assert math.isclose(stats["ausa/router_top1_gap_norm"], 1.0, abs_tol=1e-6)
    assert math.isclose(stats["ausa/router_entropy_norm"], 0.0, abs_tol=1e-6)


def test_candidate_counts_match_strict_past_bank_fiber():
    bank = build_window_bank(
        seq_len=12,
        window_size=4,
        stride=2,
        device=torch.device("cpu"),
        causality_mode="strict_past",
    )
    num_sets = int(bank.set_indices.shape[0])
    router_probs = _probs_from_candidates(bank.token_to_sets, num_sets=num_sets)
    stats = _stats(router_probs, bank.token_to_sets)
    expected_counts = (bank.token_to_sets >= 0).sum(dim=-1).to(torch.float32)

    assert math.isclose(
        stats["ausa/candidate_count_mean"],
        float(expected_counts.mean().item()),
        abs_tol=1e-6,
    )
    assert math.isclose(
        stats["ausa/candidate_count_max"],
        float(expected_counts.max().item()),
        abs_tol=1e-6,
    )
    assert math.isclose(
        stats["ausa/router_candidate_count_mean"],
        float(expected_counts.mean().item()),
        abs_tol=1e-6,
    )
    assert stats["ausa/candidate_count_max"] == 2.0


def test_diagnostics_follow_supplied_fiber_not_legacy_membership():
    strict_fiber = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    legacy_membership = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    router_probs = torch.tensor(
        [[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]],
        dtype=torch.float32,
    )

    strict_stats = _stats(router_probs, strict_fiber)
    legacy_stats = _stats(router_probs, legacy_membership)

    assert strict_stats["ausa/candidate_count_mean"] == 2.0
    assert legacy_stats["ausa/candidate_count_mean"] == 3.0
    assert math.isclose(strict_stats["ausa/router_entropy_norm"], 1.0, abs_tol=1e-6)
    assert legacy_stats["ausa/router_entropy_norm"] < strict_stats["ausa/router_entropy_norm"]
    assert legacy_stats["ausa/router_top1_gap_norm"] > strict_stats["ausa/router_top1_gap_norm"]


def test_zero_and_single_candidate_rows_are_finite():
    token_to_sets = torch.tensor([[-1, -1], [0, -1], [0, 1]], dtype=torch.long)
    router_probs = _probs_from_candidates(token_to_sets, num_sets=2)
    stats = _stats(router_probs, token_to_sets)

    for key in (
        "ausa/router_entropy",
        "ausa/router_entropy_norm",
        "ausa/router_top1_gap_norm",
        "ausa/candidate_count_mean",
        "ausa/candidate_count_max",
        "ausa/delta_routing_entropy",
        "ausa/delta_set_variance",
        "ausa/delta_router_confidence",
    ):
        assert math.isfinite(stats[key]), (key, stats[key])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {Path(__file__).name}:{name}")
