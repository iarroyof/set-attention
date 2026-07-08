from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train.metrics_impl import masked_lm_loss_and_counts  # noqa: E402


def test_masked_lm_loss_and_accuracy_match_hand_computation() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [-1.0, 0.0, 3.0],
                [0.0, 2.0, 1.0],
            ]
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[0, -100, 2, 0]])
    loss, count, correct = masked_lm_loss_and_counts(logits, labels)

    expected = F.cross_entropy(
        torch.stack([logits[0, 0], logits[0, 2], logits[0, 3]]),
        torch.tensor([0, 2, 0]),
    )
    assert torch.allclose(loss, expected)
    assert count == 3
    assert correct == 2
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, 1], torch.zeros(3))


def test_all_ignored_batch_is_finite_and_zero() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.full((2, 3), -100)
    loss, count, correct = masked_lm_loss_and_counts(logits, labels)
    assert loss.item() == 0.0
    assert count == 0
    assert correct == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
