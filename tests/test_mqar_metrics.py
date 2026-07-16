from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train.mqar import (  # noqa: E402
    evaluate_mqar,
    evaluate_mqar_group_ablation,
    lag_bin_metrics_from_logits,
    query_metrics_from_logits,
    train_mqar_update_block,
)


def test_query_metrics_count_only_non_ignored_targets() -> None:
    logits = torch.zeros(1, 6, 5)
    labels = torch.full((1, 6), -100)
    labels[0, 2] = 3
    labels[0, 4] = 1
    logits[0, 2, 3] = 10.0
    logits[0, 4, 0] = 10.0
    metrics = query_metrics_from_logits(
        logits,
        labels,
        torch.tensor([[2, 4]]),
    )
    assert metrics["valid_tokens"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["exact_sequence_accuracy"] == 0.0


def test_lag_bin_metrics_report_empty_bins() -> None:
    logits = torch.zeros(1, 3, 7)
    labels = torch.full((1, 3), -100)
    labels[0, 1] = 2
    logits[0, 1, 2] = 5.0
    metrics = lag_bin_metrics_from_logits(
        logits,
        labels,
        torch.tensor([[1]]),
        torch.tensor([[32]]),
    )
    assert metrics["lag/lag_1_32_query_count"] == 1
    assert metrics["lag/lag_1_32_accuracy"] == 1.0
    assert metrics["lag/lag_33_128_query_count"] == 0
    assert metrics["lag/lag_33_128_accuracy"] is None


class _FakeAblationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.span_ablation_mode = "none"
        self.multiresolution_group_metadata = [{"name": "fine"}, {"name": "coarse"}]

    def set_span_ablation_mode(self, mode: str = "none") -> None:
        self.span_ablation_mode = mode

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        logits[..., 1] = 5.0
        if self.span_ablation_mode == "fine":
            logits[..., 2] = 6.0
        return logits


def test_group_ablation_uses_public_hook_and_restores_state() -> None:
    model = _FakeAblationModel()
    loader = [
        {
            "input_ids": torch.tensor([[0, 1, 2]]),
            "labels": torch.tensor([[-100, 1, -100]]),
            "query_positions": torch.tensor([[1]]),
            "lags": torch.tensor([[12]]),
        }
    ]
    base = {"loss": 0.1, "accuracy": 1.0, "lag/lag_1_32_accuracy": 1.0, "lag/lag_1_32_loss": 0.1}
    metrics = evaluate_mqar_group_ablation(model, loader, torch.device("cpu"), base)
    assert metrics["ablation/status"] == "ok"
    assert metrics["ablation/fine_delta_accuracy"] == pytest.approx(1.0)
    assert model.span_ablation_mode == "none"


class _TinyMQARLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 7, d_model: int = 5) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


def _mqar_batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [-100, 2, -100, 3],
            [-100, 3, -100, 4],
            [-100, 4, -100, 5],
            [-100, 5, -100, 6],
        ],
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "query_positions": torch.tensor([[1, 3], [1, 3], [1, 3], [1, 3]], dtype=torch.long),
        "lags": torch.tensor([[4, 8], [4, 8], [4, 8], [4, 8]], dtype=torch.long),
    }


def _slice_batch(batch: dict[str, torch.Tensor], start: int, stop: int) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in batch.items()}


def test_gradient_accumulation_matches_single_full_batch_update() -> None:
    torch.manual_seed(123)
    full_model = _TinyMQARLM()
    accum_model = _TinyMQARLM()
    accum_model.load_state_dict(full_model.state_dict())
    full_opt = torch.optim.SGD(full_model.parameters(), lr=0.2)
    accum_opt = torch.optim.SGD(accum_model.parameters(), lr=0.2)
    batch = _mqar_batch()

    train_mqar_update_block(
        full_model,
        [batch],
        full_opt,
        torch.device("cpu"),
        max_updates=1,
        clip_grad_norm=0.0,
    )
    train_mqar_update_block(
        accum_model,
        [_slice_batch(batch, 0, 2), _slice_batch(batch, 2, 4)],
        accum_opt,
        torch.device("cpu"),
        max_updates=1,
        clip_grad_norm=0.0,
        grad_accum_steps=2,
    )

    for full_param, accum_param in zip(full_model.parameters(), accum_model.parameters()):
        assert torch.allclose(full_param, accum_param, atol=1e-6, rtol=1e-6)


def test_eval_microbatching_preserves_mqar_metrics() -> None:
    torch.manual_seed(321)
    model = _TinyMQARLM()
    batch = _mqar_batch()

    full_metrics = evaluate_mqar(model, [batch], torch.device("cpu"))
    micro_metrics = evaluate_mqar(
        model,
        [batch],
        torch.device("cpu"),
        microbatch_size=2,
    )

    for key in ("loss", "ppl", "accuracy", "valid_tokens", "exact_sequence_accuracy"):
        assert micro_metrics[key] == pytest.approx(full_metrics[key])
    assert micro_metrics["lag/lag_1_32_query_count"] == full_metrics["lag/lag_1_32_query_count"]
    assert micro_metrics["lag/lag_1_32_loss"] == pytest.approx(full_metrics["lag/lag_1_32_loss"])
