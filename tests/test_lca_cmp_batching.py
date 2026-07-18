from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.load import load_config  # noqa: E402
from data.lca_cmp import build_lca_cmp_datasets  # noqa: E402
from train.lca_cmp import evaluate_lca, train_lca_update_block  # noqa: E402


class TinyLCA(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


def _collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


def _slice_batch(batch: dict[str, torch.Tensor], start: int, stop: int) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in batch.items()}


def _lca_batch() -> dict[str, torch.Tensor]:
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
    return _collate([train[idx] for idx in range(4)])


def test_gradient_accumulation_matches_single_full_batch_update() -> None:
    batch = _lca_batch()
    torch.manual_seed(123)
    full_model = TinyLCA()
    accum_model = TinyLCA()
    accum_model.load_state_dict(full_model.state_dict())
    full_opt = torch.optim.SGD(full_model.parameters(), lr=0.2)
    accum_opt = torch.optim.SGD(accum_model.parameters(), lr=0.2)

    full_metrics = train_lca_update_block(
        full_model,
        [batch],
        full_opt,
        torch.device("cpu"),
        max_updates=1,
        clip_grad_norm=0.0,
    )
    accum_metrics = train_lca_update_block(
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
    assert accum_metrics["_optimizer_steps"] == full_metrics["_optimizer_steps"] == 1
    assert accum_metrics["_microbatches_per_optimizer_step"] == 2
    assert accum_metrics["valid_tokens"] == full_metrics["valid_tokens"]
    assert accum_metrics["loss"] == pytest.approx(full_metrics["loss"], abs=1e-6)
    assert accum_metrics["ppl"] == pytest.approx(full_metrics["ppl"], rel=1e-6)
    assert accum_metrics["accuracy"] == pytest.approx(full_metrics["accuracy"], abs=1e-6)


def test_eval_microbatching_preserves_metrics() -> None:
    batch = _lca_batch()
    torch.manual_seed(321)
    model = TinyLCA()

    full_metrics = evaluate_lca(model, [batch], torch.device("cpu"), vocab_size=128)
    micro_metrics = evaluate_lca(
        model,
        [batch],
        torch.device("cpu"),
        vocab_size=128,
        microbatch_size=2,
    )

    for key in ("loss", "accuracy", "exact_sequence_accuracy"):
        assert micro_metrics[key] == pytest.approx(full_metrics[key], abs=1e-6)
    assert micro_metrics["ppl"] == pytest.approx(full_metrics["ppl"], rel=1e-6)
    assert micro_metrics["valid_tokens"] == full_metrics["valid_tokens"]
    for name in ("near_below", "near_above"):
        assert micro_metrics[f"bucket/{name}_count"] == full_metrics[f"bucket/{name}_count"]
        if full_metrics[f"bucket/{name}_count"] > 0:
            assert micro_metrics[f"bucket/{name}_loss"] == pytest.approx(
                full_metrics[f"bucket/{name}_loss"], abs=1e-6
            )
            assert micro_metrics[f"bucket/{name}_accuracy"] == pytest.approx(
                full_metrics[f"bucket/{name}_accuracy"], abs=1e-6
            )


def test_config_defaults_resolve_grad_accum_and_eval_microbatch() -> None:
    cfg = load_config(ROOT / "configs" / "set_dictionary" / "sd9_multiresolution.yaml")
    assert cfg["training"]["grad_accum_steps"] == 1
    assert cfg["training"]["eval_microbatch_size"] is None


def test_lca_smoke_config_explicitly_defaults_to_no_microbatching() -> None:
    cfg = load_config(ROOT / "configs" / "lca_cmp" / "token_smoke.yaml")
    assert cfg["training"]["grad_accum_steps"] == 1
    assert cfg["training"]["eval_microbatch_size"] is None
    assert cfg["data"]["dataset"] == "lca_cmp"
