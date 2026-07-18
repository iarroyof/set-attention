from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.lca_cmp import (  # noqa: E402
    LCACmpDataset,
    build_lca_cmp_datasets,
    generate_lca_cmp,
    lca_cmp_fingerprint,
    split_seed,
)


def _data_cfg(**overrides):
    cfg = {
        "vocab_size": 128,
        "seq_len": 32,
        "num_train_examples": 8,
        "num_val_examples": 4,
        "marker_fraction": 0.12,
        "count_jitter": 2,
        "dataset_seed": 3,
    }
    cfg.update(overrides)
    return cfg


def test_generator_same_seed_is_deterministic() -> None:
    first = generate_lca_cmp({"vocab_size": 128, "seq_len": 32, "num_examples": 8, "seed": 11})
    second = generate_lca_cmp({"vocab_size": 128, "seq_len": 32, "num_examples": 8, "seed": 11})
    assert lca_cmp_fingerprint(first) == lca_cmp_fingerprint(second)
    assert (first.input_ids == second.input_ids).all()
    assert (first.labels == second.labels).all()
    assert (first.marker_counts == second.marker_counts).all()
    assert (first.targets == second.targets).all()


def test_generator_different_seed_differs() -> None:
    first = generate_lca_cmp({"vocab_size": 128, "seq_len": 32, "num_examples": 8, "seed": 11})
    second = generate_lca_cmp({"vocab_size": 128, "seq_len": 32, "num_examples": 8, "seed": 12})
    assert lca_cmp_fingerprint(first) != lca_cmp_fingerprint(second)


def test_dataset_digest_deterministic_and_seed_sensitive() -> None:
    train_a, val_a = build_lca_cmp_datasets(_data_cfg())
    train_b, val_b = build_lca_cmp_datasets(_data_cfg())
    assert train_a.dataset_digest == train_b.dataset_digest
    assert val_a.dataset_digest == val_b.dataset_digest

    train_c, _ = build_lca_cmp_datasets(_data_cfg(dataset_seed=4))
    assert train_a.dataset_digest != train_c.dataset_digest


def test_train_validation_seeds_are_disjoint() -> None:
    train, val = build_lca_cmp_datasets(_data_cfg())
    assert train.config.seed != val.config.seed
    assert train.config.seed == split_seed(3, "train")
    assert val.config.seed == split_seed(3, "validation")


def test_provenance_records_digest_and_counts() -> None:
    train, val = build_lca_cmp_datasets(_data_cfg())
    record = train.provenance()
    assert record["dataset_digest"] == train.dataset_digest
    assert record["query_count"] == 8
    assert record["token_count"] == 8 * 32
    assert record["split"] == "train"


def test_query_label_uses_answer_special_tokens() -> None:
    generated = generate_lca_cmp({"vocab_size": 128, "seq_len": 32, "num_examples": 8, "seed": 5})
    # Only the final query position carries a supervised label.
    assert (generated.labels[:, :-1] == -100).all()
    valid_labels = generated.labels[:, -1]
    assert set(valid_labels.tolist()) <= {126, 127}
    assert (generated.input_ids[:, -1] == 125).all()
