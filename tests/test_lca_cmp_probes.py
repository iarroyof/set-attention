from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.lca_cmp import (  # noqa: E402
    build_lca_cmp_datasets,
    generate_lca_cmp,
    lca_cmp_fingerprint,
    oracle_count_base,
    special_tokens,
)
from data.mqar import IGNORE_INDEX  # noqa: E402
from train.lca_cmp import train_lca_update_block  # noqa: E402


def _base(**overrides):
    cfg = {"vocab_size": 128, "seq_len": 32, "num_examples": 8, "seed": 11}
    cfg.update(overrides)
    return cfg


def test_prefix_supervision_is_deterministic() -> None:
    first = generate_lca_cmp(_base(supervision="prefix"))
    second = generate_lca_cmp(_base(supervision="prefix"))
    assert lca_cmp_fingerprint(first) == lca_cmp_fingerprint(second)
    assert (first.labels == second.labels).all()


def test_prefix_supervision_shares_inputs_with_endpoint() -> None:
    endpoint = generate_lca_cmp(_base())
    prefix = generate_lca_cmp(_base(supervision="prefix"))
    # Same seed and identical rng consumption: the input stream and marker
    # placement are bit-identical; only the label layout differs.
    assert (endpoint.input_ids == prefix.input_ids).all()
    assert (endpoint.marker_counts == prefix.marker_counts).all()
    assert (endpoint.targets == prefix.targets).all()
    assert (endpoint.labels != prefix.labels).any()


def test_prefix_supervision_label_layout() -> None:
    generated = generate_lca_cmp(_base(supervision="prefix"))
    specials = special_tokens(128)
    # Every position is supervised with an answer token (no IGNORE_INDEX).
    assert (generated.labels != IGNORE_INDEX).all()
    assert set(generated.labels.unique().tolist()) <= {
        specials["answer_false"],
        specials["answer_true"],
    }
    # At the final position the scaled threshold equals the endpoint
    # threshold, so the last-position label matches the endpoint label.
    endpoint = generate_lca_cmp(_base())
    assert (generated.labels[:, -1] == endpoint.labels[:, -1]).all()


def test_prefix_supervision_labels_match_prefix_counts() -> None:
    generated = generate_lca_cmp(_base(supervision="prefix"))
    specials = special_tokens(128)
    seq_len = 32
    context_len = seq_len - 1
    threshold = max(1, min(context_len - 1, round(context_len * 0.08)))
    marker_mask = (generated.input_ids == specials["marker"]).numpy()
    prefix_counts = np.cumsum(marker_mask, axis=1)
    steps = np.arange(1, seq_len + 1, dtype=np.float64)
    thresholds = np.maximum(1, np.rint(threshold * steps / context_len)).astype(np.int64)
    expected = np.where(
        prefix_counts >= thresholds[None, :],
        specials["answer_true"],
        specials["answer_false"],
    )
    assert (generated.labels.numpy() == expected).all()


def test_prefix_supervision_builds_via_data_cfg() -> None:
    train, val = build_lca_cmp_datasets(
        {
            "vocab_size": 128,
            "seq_len": 32,
            "num_train_examples": 8,
            "num_val_examples": 4,
            "dataset_seed": 3,
            "supervision": "prefix",
        }
    )
    assert train.config.supervision == "prefix"
    assert val.config.supervision == "prefix"
    assert (train.generated.labels != IGNORE_INDEX).all()
    assert train.dataset_digest != val.dataset_digest


def test_oracle_count_token_layout() -> None:
    generated = generate_lca_cmp(_base(oracle_count_token=True))
    specials = special_tokens(128)
    base = oracle_count_base(128, 32, 0.08, 16)
    oracle_ids = generated.input_ids[:, -2]
    # The oracle slot carries the reserved id base + (count - low_min); with
    # the default fraction/jitter at this scale low_min is 0.
    assert (oracle_ids == torch.from_numpy(base + generated.marker_counts.numpy())).all()
    # Reserved range is disjoint from specials and noise, and holds no marker.
    assert (oracle_ids >= base).all() and (oracle_ids < 128 - 4).all()
    body = generated.input_ids[:, :-2]
    noise = body[body != specials["marker"]]
    assert (noise < base).all()
    assert (generated.input_ids[:, -2] != specials["marker"]).all()
    # Supervision stays endpoint-style: only the final query position.
    assert (generated.labels[:, :-1] == IGNORE_INDEX).all()
    valid = generated.labels[:, -1]
    assert set(valid.tolist()) <= {specials["answer_false"], specials["answer_true"]}


def test_oracle_count_token_is_deterministic() -> None:
    first = generate_lca_cmp(_base(oracle_count_token=True))
    second = generate_lca_cmp(_base(oracle_count_token=True))
    assert lca_cmp_fingerprint(first) == lca_cmp_fingerprint(second)
    assert (first.input_ids == second.input_ids).all()


def test_oracle_count_token_requires_vocab_headroom() -> None:
    with pytest.raises(ValueError):
        generate_lca_cmp({"vocab_size": 24, "seq_len": 38, "num_examples": 2, "seed": 1, "oracle_count_token": True})


def test_invalid_supervision_rejected() -> None:
    with pytest.raises(ValueError):
        generate_lca_cmp(_base(supervision="everywhere"))


class _TinyLCA(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


def test_train_curve_records_per_update_loss() -> None:
    train, _ = build_lca_cmp_datasets(
        {
            "vocab_size": 128,
            "seq_len": 32,
            "num_train_examples": 4,
            "num_val_examples": 4,
            "batch_size": 4,
            "dataset_seed": 3,
        }
    )
    batch = {key: torch.stack([train[idx][key] for idx in range(4)]) for key in train[0]}
    torch.manual_seed(7)
    model = _TinyLCA()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = train_lca_update_block(
        model,
        [batch, batch, batch],
        optimizer,
        torch.device("cpu"),
        max_updates=3,
        clip_grad_norm=0.0,
        record_curve=True,
    )
    curve = metrics["_curve"]
    assert [update for update, _ in curve] == [1, 2, 3]
    assert all(np.isfinite(loss) for _, loss in curve)
    # The run-mean loss is the token-weighted mean of the per-update losses
    # (identical valid-token counts per update here).
    assert metrics["loss"] == pytest.approx(sum(loss for _, loss in curve) / 3, abs=1e-6)


def test_train_curve_disabled_by_default() -> None:
    train, _ = build_lca_cmp_datasets(
        {
            "vocab_size": 128,
            "seq_len": 32,
            "num_train_examples": 4,
            "num_val_examples": 4,
            "batch_size": 4,
            "dataset_seed": 3,
        }
    )
    batch = {key: torch.stack([train[idx][key] for idx in range(4)]) for key in train[0]}
    torch.manual_seed(7)
    model = _TinyLCA()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = train_lca_update_block(
        model,
        [batch],
        optimizer,
        torch.device("cpu"),
        max_updates=1,
        clip_grad_norm=0.0,
    )
    assert metrics["_curve"] == []
