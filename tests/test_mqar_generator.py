from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.mqar import IGNORE_INDEX, MQARConfig, build_mqar_datasets, generate_mqar, split_seed  # noqa: E402


def test_mqar_generator_matches_placement_and_mask_contract() -> None:
    generated = generate_mqar(
        MQARConfig(
            vocab_size=64,
            num_examples=5,
            seq_len=32,
            num_kv_pairs=4,
            seed=123,
            random_non_queries=False,
        )
    )
    half = generated.config.vocab_size // 2
    context = generated.config.num_kv_pairs * 2

    assert generated.input_ids.shape == (5, 32)
    assert generated.labels.shape == (5, 32)
    assert torch.equal(generated.input_ids[:, :context:2], generated.query_keys)
    assert torch.equal(generated.input_ids[:, 1:context:2], generated.query_values)
    assert torch.all((generated.query_keys >= 1) & (generated.query_keys < half))
    assert torch.all((generated.query_values >= half) & (generated.query_values < 64))

    for i in range(len(generated.input_ids)):
        assert len(set(generated.query_keys[i].tolist())) == 4
        assert len(set(generated.query_values[i].tolist())) == 4
        for j in range(4):
            query_pos = int(generated.query_positions[i, j])
            key_pos = int(generated.key_positions[i, j])
            assert generated.input_ids[i, query_pos].item() == generated.query_keys[i, j].item()
            assert generated.labels[i, query_pos].item() == generated.query_values[i, j].item()
            assert generated.lags[i, j].item() == query_pos - key_pos
        valid = generated.labels[i].ne(IGNORE_INDEX)
        assert valid.sum().item() == 4
        assert set(torch.where(valid)[0].tolist()) == set(generated.query_positions[i].tolist())


def test_mqar_seed_reproducibility_and_split_separation() -> None:
    cfg = MQARConfig(vocab_size=64, num_examples=3, seq_len=32, num_kv_pairs=4, seed=7)
    a = generate_mqar(cfg)
    b = generate_mqar(cfg)
    c = generate_mqar(MQARConfig(vocab_size=64, num_examples=3, seq_len=32, num_kv_pairs=4, seed=8))
    assert torch.equal(a.input_ids, b.input_ids)
    assert torch.equal(a.labels, b.labels)
    assert not torch.equal(a.input_ids, c.input_ids)

    train, validation = build_mqar_datasets(
        {
            "seq_len": 32,
            "num_kv_pairs": 4,
            "vocab_size": 64,
            "num_train_examples": 3,
            "num_val_examples": 3,
            "dataset_seed": 11,
        }
    )
    assert train.config.seed == split_seed(11, "train")
    assert validation.config.seed == split_seed(11, "validation")
    assert train.dataset_digest != validation.dataset_digest
