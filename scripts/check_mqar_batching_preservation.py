#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train.mqar import evaluate_mqar, train_mqar_update_block  # noqa: E402


class TinyMQARLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 7, d_model: int = 5) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


def _batch() -> dict[str, torch.Tensor]:
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
        "query_positions": torch.tensor(
            [[1, 3], [1, 3], [1, 3], [1, 3]],
            dtype=torch.long,
        ),
        "lags": torch.tensor(
            [[4, 8], [4, 8], [4, 8], [4, 8]],
            dtype=torch.long,
        ),
    }


def _slice_batch(
    batch: dict[str, torch.Tensor],
    start: int,
    stop: int,
) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in batch.items()}


def _assert_close(name: str, left: float, right: float, tol: float = 1e-6) -> None:
    if abs(float(left) - float(right)) > tol:
        raise AssertionError(f"{name}: {left} != {right}")


def main() -> None:
    torch.manual_seed(123)
    full_model = TinyMQARLM()
    accum_model = TinyMQARLM()
    accum_model.load_state_dict(full_model.state_dict())
    full_opt = torch.optim.SGD(full_model.parameters(), lr=0.2)
    accum_opt = torch.optim.SGD(accum_model.parameters(), lr=0.2)
    batch = _batch()

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
        if not torch.allclose(full_param, accum_param, atol=1e-6, rtol=1e-6):
            raise AssertionError("accumulated microbatch update differs from full batch")

    torch.manual_seed(321)
    eval_model = TinyMQARLM()
    full_metrics = evaluate_mqar(eval_model, [batch], torch.device("cpu"))
    micro_metrics = evaluate_mqar(
        eval_model,
        [batch],
        torch.device("cpu"),
        microbatch_size=2,
    )
    for key in (
        "loss",
        "ppl",
        "accuracy",
        "valid_tokens",
        "exact_sequence_accuracy",
        "lag/lag_1_32_loss",
        "lag/lag_1_32_accuracy",
        "lag/lag_1_32_query_count",
    ):
        if isinstance(full_metrics[key], float):
            _assert_close(key, full_metrics[key], micro_metrics[key])
        elif full_metrics[key] != micro_metrics[key]:
            raise AssertionError(f"{key}: {full_metrics[key]} != {micro_metrics[key]}")

    print("batching preservation checks passed")


if __name__ == "__main__":
    main()
