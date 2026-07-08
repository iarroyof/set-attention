from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.ordered_text import (  # noqa: E402
    build_ordered_text_provenance,
    combined_dataset_digest,
    vocabulary_digest,
    vocabulary_tokens,
)
from run_experiment import _make_loader  # noqa: E402
from data import wikitext2  # noqa: E402


def test_ordered_text_offsets_and_digests_are_stable() -> None:
    stoi = {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3, "c": 4}
    vocab = vocabulary_tokens(stoi)
    vocab_sha = vocabulary_digest(vocab)
    records = [["a", "b"], ["c"], ["a", "c", "b"]]
    first = build_ordered_text_provenance(
        dataset="toy",
        split="train",
        records=records,
        seq_len=2,
        vocabulary_sha256=vocab_sha,
    )
    second = build_ordered_text_provenance(
        dataset="toy",
        split="train",
        records=records,
        seq_len=2,
        vocabulary_sha256=vocab_sha,
    )
    assert first == second
    assert first.record_offsets == (0, 2, 3, 6)
    assert first.sample_offsets == (0, 2)
    assert first.token_count == 6

    changed = build_ordered_text_provenance(
        dataset="toy",
        split="train",
        records=[["a", "b"], ["c"], ["b", "c", "a"]],
        seq_len=2,
        vocabulary_sha256=vocab_sha,
    )
    assert changed.dataset_digest != first.dataset_digest
    assert combined_dataset_digest(first, first) != combined_dataset_digest(
        changed,
        changed,
    )


def test_same_loader_seed_reproduces_first_two_batches() -> None:
    values = torch.arange(40)
    dataset = TensorDataset(values, values + 1)
    first = _make_loader(dataset, 4, True, seed=71)
    second = _make_loader(dataset, 4, True, seed=71)
    third = _make_loader(dataset, 4, True, seed=72)

    first_iter = iter(first)
    second_iter = iter(second)
    first_batches = [next(first_iter)[0], next(first_iter)[0]]
    second_batches = [next(second_iter)[0], next(second_iter)[0]]
    third_batch = next(iter(third))[0]

    assert all(
        torch.equal(left, right)
        for left, right in zip(first_batches, second_batches)
    )
    assert not torch.equal(first_batches[0], third_batch)


def test_wikitext_adapter_emits_stable_ordered_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lines = ["alpha beta", "gamma", "alpha delta beta"]
    monkeypatch.setattr(wikitext2, "ensure_hf_cache", lambda _: tmp_path)
    monkeypatch.setattr(
        wikitext2,
        "load_wikitext_lines",
        lambda *args, **kwargs: list(lines),
    )
    first = wikitext2.Wikitext2Dataset(
        split="train",
        seq_len=2,
        cache_root=str(tmp_path),
    )
    second = wikitext2.Wikitext2Dataset(
        split="train",
        seq_len=2,
        cache_root=str(tmp_path),
    )
    assert first.provenance == second.provenance
    assert first.vocabulary_tokens == second.vocabulary_tokens
    assert len(first) == len(first.provenance.sample_offsets)
