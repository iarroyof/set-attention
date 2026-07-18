#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.lca_cmp import build_lca_cmp_datasets  # noqa: E402
from train.lca_cmp import evaluate_lca, train_lca_update_block  # noqa: E402


class TinyLCA(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


def _slice_batch(batch: dict[str, torch.Tensor], start: int, stop: int) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in batch.items()}


def _collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


def main() -> None:
    train, _ = build_lca_cmp_datasets(
        {
            "vocab_size": 128,
            "seq_len": 32,
            "num_train_examples": 4,
            "num_val_examples": 4,
            "batch_size": 4,
            "marker_fraction": 0.12,
            "count_jitter": 2,
            "dataset_seed": 3,
        }
    )
    batch = _collate([train[idx] for idx in range(4)])

    torch.manual_seed(123)
    full_model = TinyLCA()
    accum_model = TinyLCA()
    accum_model.load_state_dict(full_model.state_dict())
    full_opt = torch.optim.SGD(full_model.parameters(), lr=0.2)
    accum_opt = torch.optim.SGD(accum_model.parameters(), lr=0.2)

    train_lca_update_block(
        full_model,
        [batch],
        full_opt,
        torch.device("cpu"),
        max_updates=1,
        clip_grad_norm=0.0,
    )
    train_lca_update_block(
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
    eval_model = TinyLCA()
    full_metrics = evaluate_lca(eval_model, [batch], torch.device("cpu"), vocab_size=128)
    micro_metrics = evaluate_lca(
        eval_model,
        [batch],
        torch.device("cpu"),
        vocab_size=128,
        microbatch_size=2,
    )
    for key in ("loss", "ppl", "accuracy", "valid_tokens", "exact_sequence_accuracy"):
        left = full_metrics[key]
        right = micro_metrics[key]
        if isinstance(left, float):
            atol = 1e-6
            rtol = 1e-6 if key == "ppl" else 0.0
            if abs(left - right) > atol + rtol * max(abs(left), abs(right), 1.0):
                raise AssertionError(f"{key}: {left} != {right}")
        elif left != right:
            raise AssertionError(f"{key}: {left} != {right}")
    print("LCA batching preservation checks passed")


if __name__ == "__main__":
    main()
