from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train.mqar import (  # noqa: E402
    evaluate_mqar_group_ablation,
    lag_bin_metrics_from_logits,
    query_metrics_from_logits,
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
