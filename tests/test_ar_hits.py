from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.ar_hits import (  # noqa: E402
    ARHitAccumulator,
    ar_hit_metadata_for_sequence,
    build_bigram_counts_from_samples,
    count_bin_name,
    evaluate_ar_hits,
    evaluate_ar_hit_group_ablation,
)
from train.checkpoints import (  # noqa: E402
    CheckpointCompatibilityError,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)


def test_handcrafted_repeated_bigrams_produce_expected_mask() -> None:
    x = [1, 2, 1, 2, 3]
    y = [2, 3, 2, 3, 4]
    meta = ar_hit_metadata_for_sequence(x, y)
    assert [m.is_ar for m in meta] == [False, False, True, True, False]
    assert meta[2].most_recent_lag == 2
    assert meta[3].earliest_lag == 2


def test_future_occurrences_do_not_count() -> None:
    x = [7, 1, 7]
    y = [8, 2, 8]
    meta = ar_hit_metadata_for_sequence(x, y)
    assert [m.is_ar for m in meta] == [False, False, True]


def test_labels_are_shifted_next_token_bigrams() -> None:
    tokens = [4, 5, 4, 5]
    x = tokens[:-1]
    y = tokens[1:]
    meta = ar_hit_metadata_for_sequence(x, y)
    assert [m.is_ar for m in meta] == [False, False, True]
    assert meta[2].target_token == 5


def test_record_boundary_resets_unbroken_context_policy() -> None:
    x = [1, 1, 1]
    y = [2, 2, 2]
    meta = ar_hit_metadata_for_sequence(
        x,
        y,
        sample_start_offset=0,
        record_offsets=[0, 2, 4],
    )
    assert meta[0].crosses_record_boundary is False
    assert meta[1].crosses_record_boundary is True
    assert meta[1].is_ar is False
    assert meta[2].is_ar is False
    assert meta[2].context_has_record_boundary is True


def test_training_count_bins_include_endpoints() -> None:
    assert count_bin_name(0) == "count_0"
    assert count_bin_name(1) == "count_1"
    assert count_bin_name(2) == "count_2_5"
    assert count_bin_name(5) == "count_2_5"
    assert count_bin_name(6) == "count_6_20"
    assert count_bin_name(20) == "count_6_20"
    assert count_bin_name(21) == "count_gt20"


def test_bigram_counts_skip_record_crossing_pairs() -> None:
    counts = build_bigram_counts_from_samples(
        [(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]))],
        record_offsets=[0, 2],
        sample_offsets=[0],
    )
    assert counts[(1, 2)] == 1
    assert counts[(2, 3)] == 0
    assert counts[(3, 4)] == 1


def test_empty_and_subthreshold_bins_are_finite_and_not_inferential() -> None:
    acc = ARHitAccumulator(min_inferential_targets=1000)
    acc.add("overall", 1.0)
    acc.add("non_ar", 1.0)
    acc.add("count_0", 1.0)
    metrics = acc.metrics()
    assert metrics["overall/nll"] == pytest.approx(1.0)
    assert metrics["overall/ppl"] == pytest.approx(2.718281828459045)
    assert metrics["count_0/inferential"] is False
    assert metrics["count_1/nll"] is None
    assert metrics["count_1/inferential"] is False


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 3)
        self.proj = torch.nn.Linear(3, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(input_ids))


def test_checkpoint_vocabulary_mismatch_fails_closed(tmp_path: Path) -> None:
    model = _TinyModel(vocab_size=5)
    config = {
        "model": {"implementation": "fake", "vocab_size": 5},
        "training": {"seed": 0, "applied_seed": 0, "torch_initial_seed": 0},
    }
    provenance = {"dataset_digest": "dataset", "tokenizer_digest": "tok_a"}
    payload = build_checkpoint_payload(
        model=model,
        config=config,
        config_fingerprint="cfg",
        dataset_provenance=provenance,
        epoch=1,
        global_step=1,
    )
    path = tmp_path / "final.pt"
    save_checkpoint(payload, path)
    with pytest.raises(CheckpointCompatibilityError, match="tokenizer digest mismatch"):
        load_checkpoint(
            path,
            model=_TinyModel(vocab_size=5),
            expected_model_config=config["model"],
            expected_dataset_digest="dataset",
            expected_tokenizer_digest="tok_b",
        )


class _FakeAblationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.span_ablation_mode = "none"
        self.multiresolution_group_metadata = [{"name": "fine"}, {"name": "coarse"}]

    def set_span_ablation_mode(self, mode: str = "none") -> None:
        self.span_ablation_mode = mode

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
        logits[..., 2] = 5.0
        if self.span_ablation_mode == "fine":
            logits[..., 3] = 7.0
        return logits


def test_group_ablation_restores_model_state_after_ar_eval() -> None:
    model = _FakeAblationModel()
    loader = [(torch.tensor([[1, 1, 1]]), torch.tensor([[2, 2, 2]]))]
    base = {
        "overall/nll": 0.1,
        "overall/ppl": 1.1,
        "ar/nll": 0.1,
        "ar/ppl": 1.1,
        "non_ar/nll": 0.1,
        "non_ar/ppl": 1.1,
    }
    metrics = evaluate_ar_hit_group_ablation(
        model,
        loader,
        torch.device("cpu"),
        base,
        train_bigram_counts={(1, 2): 3},
    )
    assert metrics["ablation/status"] == "ok"
    assert "ablation/fine/overall/delta_nll" in metrics
    assert model.span_ablation_mode == "none"


def test_evaluate_ar_hits_collects_per_sequence_blocks() -> None:
    model = _FakeAblationModel()
    loader = [(torch.tensor([[1, 1, 1], [1, 1, 1]]), torch.tensor([[2, 2, 2], [2, 2, 2]]))]
    metrics = evaluate_ar_hits(
        model,
        loader,
        torch.device("cpu"),
        train_bigram_counts={(1, 2): 3},
        collect_blocks=True,
    )
    blocks = metrics["blocks"]
    assert len(blocks) == 2
    assert [b["seq"] for b in blocks] == [0, 1]
    # sequence [1,1,1]->[2,2,2] with count (1,2)=3: t=0 non-AR, t=1..2 AR
    for block in blocks:
        assert block["ar_targets"] == 2
        assert block["non_ar_targets"] == 1
    total_ar = sum(b["ar_targets"] for b in blocks)
    assert total_ar == metrics["ar/targets"]
    ar_nll = sum(b["ar_nll"] for b in blocks) / total_ar
    assert ar_nll == pytest.approx(metrics["ar/nll"])
    non_ar_nll = sum(b["non_ar_nll"] for b in blocks) / sum(b["non_ar_targets"] for b in blocks)
    assert non_ar_nll == pytest.approx(metrics["non_ar/nll"])


def test_evaluate_ar_hits_blocks_default_off() -> None:
    model = _FakeAblationModel()
    loader = [(torch.tensor([[1, 1, 1]]), torch.tensor([[2, 2, 2]]))]
    metrics = evaluate_ar_hits(
        model,
        loader,
        torch.device("cpu"),
        train_bigram_counts={(1, 2): 3},
    )
    assert "blocks" not in metrics
