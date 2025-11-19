import pytest
pytest.importorskip('tokenizers')
import torch

from set_attention.sets.bank_builders import build_windowed_bank_from_texts
from set_attention.tokenizers.registry import (
    available_tokenizer_types,
    create_tokenizer,
    load_tokenizer,
    save_tokenizer,
)
from set_attention.tokenizers.utils import read_tokenizer_meta


CORPUS = ["the quick brown fox", "the quick blue hare"]


@pytest.mark.parametrize(
    "kind,config",
    [
        ("ausa", {"seed_lengths": (2, 3), "min_freq": 1, "max_len": 12}),
        ("hf_unigram", {"vocab_size": 64, "min_frequency": 1}),
        ("hf_bpe", {"vocab_size": 64, "min_frequency": 1}),
    ],
)
def test_tokenizer_registry_roundtrip(kind, config, tmp_path):
    assert kind in available_tokenizer_types()
    tokenizer = create_tokenizer(kind, config)
    tokenizer.fit(CORPUS)
    ids = tokenizer.encode_text("quick fox")
    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.long
    out_dir = tmp_path / kind
    save_tokenizer(tokenizer, str(out_dir))
    meta = read_tokenizer_meta(str(out_dir))
    assert meta is not None
    assert meta.get("type") == kind

    reloaded = load_tokenizer(str(out_dir))
    re_ids = reloaded.encode_text("quick fox")
    assert re_ids.dtype == torch.long
    assert set(ids.tolist()) == set(re_ids.tolist())
    assert reloaded.vocab_size() == tokenizer.vocab_size()


@pytest.mark.parametrize(
    "kind,config",
    [
        ("ausa", {"seed_lengths": (2, 3), "min_freq": 1, "max_len": 12}),
        ("hf_unigram", {"vocab_size": 64, "min_frequency": 1}),
        ("hf_bpe", {"vocab_size": 64, "min_frequency": 1}),
    ],
)
def test_bank_builder_accepts_tokenizer(kind, config):
    tokenizer = create_tokenizer(kind, config)
    tokenizer.fit(CORPUS)
    bank = build_windowed_bank_from_texts(tokenizer, CORPUS, window=3, stride=2)
    assert isinstance(bank.values, torch.Tensor)
    assert bank.values.dtype == torch.long
    assert bank.set_offsets[-1].item() == bank.values.numel()
    assert bank.seq_offsets[-1].item() > 0
