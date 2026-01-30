from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

from set_attention.data.hf_cache import ensure_hf_cache
from set_attention.data.wikitext import iter_wikitext_lines, load_wikitext_lines, tokenize_lines, load_wikitext_hf_dataset


SPECIAL_TOKENS = ["<pad>", "<unk>"]


def build_vocab(tokens: List[str]) -> Tuple[dict, dict]:
    vocab = list(SPECIAL_TOKENS)
    seen = set(vocab)
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            vocab.append(tok)
    stoi = {tok: idx for idx, tok in enumerate(vocab)}
    itos = {idx: tok for tok, idx in stoi.items()}
    return stoi, itos


class Wikitext2Dataset(Dataset):
    def __init__(
        self,
        split: str,
        seq_len: int,
        limit: int | None = None,
        cache_root: str | None = None,
    ) -> None:
        cache_dir = ensure_hf_cache(cache_root)
        lines = load_wikitext_lines("wikitext2", split, cache_dir, limit=limit)
        tokens = tokenize_lines(lines)
        self.stoi, self.itos = build_vocab(tokens)

        ids = [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]
        self.samples = []
        for start in range(0, max(0, len(ids) - seq_len - 1), seq_len):
            chunk = ids[start : start + seq_len + 1]
            if len(chunk) != seq_len + 1:
                continue
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


def _build_vocab_stream(
    dataset: str,
    split: str,
    cache_dir: Path,
    limit: int | None = None,
) -> Tuple[dict, dict]:
    vocab = list(SPECIAL_TOKENS)
    seen = set(vocab)
    ds = load_wikitext_hf_dataset(dataset, cache_dir, streaming=True)
    for line in iter_wikitext_lines(dataset, split, cache_dir, limit=limit, dataset_obj=ds):
        for tok in line.split():
            if tok not in seen:
                seen.add(tok)
                vocab.append(tok)
    stoi = {tok: idx for idx, tok in enumerate(vocab)}
    itos = {idx: tok for tok, idx in stoi.items()}
    return stoi, itos


class Wikitext2IterableDataset(IterableDataset):
    def __init__(
        self,
        split: str,
        seq_len: int,
        limit: int | None = None,
        cache_root: str | None = None,
    ) -> None:
        self.split = split
        self.seq_len = seq_len
        self.limit = limit
        self.cache_dir = ensure_hf_cache(cache_root)
        self.stoi, self.itos = _build_vocab_stream("wikitext2", split, self.cache_dir, limit=limit)

    def __iter__(self):
        ds = load_wikitext_hf_dataset("wikitext2", self.cache_dir, streaming=True)
        buffer: List[int] = []
        for line in iter_wikitext_lines("wikitext2", self.split, self.cache_dir, limit=self.limit, dataset_obj=ds):
            for tok in line.split():
                buffer.append(self.stoi.get(tok, self.stoi["<unk>"]))
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
