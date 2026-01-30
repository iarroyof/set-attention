from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import os
import torch
from torch.utils.data import Dataset, DataLoader


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _build_vocab(lines: List[List[str]], min_freq: int = 1) -> Tuple[dict, dict]:
    from collections import Counter

    cnt = Counter()
    for toks in lines:
        cnt.update(toks)
    itos = ["<pad>", "<s>", "</s>", "<unk>"]
    for w, c in cnt.items():
        if c >= min_freq:
            itos.append(w)
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, {i: w for w, i in stoi.items()}


def _build_shared_vocab(src_lines: List[List[str]], tgt_lines: List[List[str]], min_freq: int = 1) -> Tuple[dict, dict]:
    from collections import Counter

    cnt = Counter()
    for toks in src_lines:
        cnt.update(toks)
    for toks in tgt_lines:
        cnt.update(toks)
    itos = ["<pad>", "<s>", "</s>", "<unk>"]
    for w, c in cnt.items():
        if c >= min_freq:
            itos.append(w)
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, {i: w for w, i in stoi.items()}


def _tokenize(s: str) -> List[str]:
    return s.split()


def _encode(toks: List[str], stoi: dict, max_len: int) -> torch.LongTensor:
    ids = [stoi.get(t, stoi["<unk>"]) for t in toks]
    ids = [stoi["<s>"]] + ids[: max_len - 2] + [stoi["</s>"]]
    if len(ids) < max_len:
        ids = ids + [stoi["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


@dataclass
class TextPairConfig:
    src_path: str
    tgt_path: str
    max_len: int = 64
    batch_size: int = 32
    min_freq: int = 1


class TextPairDataset(Dataset):
    def __init__(self, cfg: TextPairConfig):
        src = _read_lines(cfg.src_path)
        tgt = _read_lines(cfg.tgt_path)
        assert len(src) == len(tgt), "Mismatched parallel corpora lengths"
        self.pairs = list(zip(src, tgt))
        src_toks = [_tokenize(s) for s in src]
        tgt_toks = [_tokenize(t) for t in tgt]
        self.src_stoi, src_itos = _build_vocab(src_toks, cfg.min_freq)
        self.tgt_stoi, tgt_itos = _build_vocab(tgt_toks, cfg.min_freq)
        self.src_itos = src_itos
        self.tgt_itos = tgt_itos
        self.max_len = cfg.max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i: int):
        s, t = self.pairs[i]
        s_ids = _encode(_tokenize(s), self.src_stoi, self.max_len)
        t_ids = _encode(_tokenize(t), self.tgt_stoi, self.max_len)
        return s_ids, t_ids


def make_textpair_loader(cfg: TextPairConfig) -> DataLoader:
    ds = TextPairDataset(cfg)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)


class InMemoryTextPairDataset(Dataset):
    """Top-level picklable dataset to support DataLoader workers on Windows."""
    def __init__(self, src_texts: List[str], tgt_texts: List[str], max_len: int = 64, min_freq: int = 1):
        assert len(src_texts) == len(tgt_texts)
        self.pairs = list(zip(src_texts, tgt_texts))
        src_toks = [_tokenize(s) for s in src_texts]
        tgt_toks = [_tokenize(t) for t in tgt_texts]
        self.src_stoi, src_itos = _build_vocab(src_toks, min_freq)
        self.tgt_stoi, tgt_itos = _build_vocab(tgt_toks, min_freq)
        self.src_itos = src_itos
        self.tgt_itos = tgt_itos
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i: int):
        s, t = self.pairs[i]
        s_ids = _encode(_tokenize(s), self.src_stoi, self.max_len)
        t_ids = _encode(_tokenize(t), self.tgt_stoi, self.max_len)
        return s_ids, t_ids


class InMemorySharedTextPairDataset(Dataset):
    """In-memory seq2seq dataset with a shared vocabulary."""
    def __init__(self, src_texts: List[str], tgt_texts: List[str], max_len: int = 64, min_freq: int = 1):
        assert len(src_texts) == len(tgt_texts)
        self.pairs = list(zip(src_texts, tgt_texts))
        src_toks = [_tokenize(s) for s in src_texts]
        tgt_toks = [_tokenize(t) for t in tgt_texts]
        self.stoi, itos = _build_shared_vocab(src_toks, tgt_toks, min_freq)
        self.itos = itos
        self.max_len = max_len
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<s>"]
        self.eos_id = self.stoi["</s>"]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i: int):
        s, t = self.pairs[i]
        s_ids = _encode(_tokenize(s), self.stoi, self.max_len)
        t_ids = _encode(_tokenize(t), self.stoi, self.max_len)
        return s_ids, t_ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id):
                continue
            if idx == self.eos_id:
                break
            toks.append(self.itos.get(int(idx), "<unk>"))
        return " ".join(toks)


def make_textpair_loader_from_lists(src_texts: List[str], tgt_texts: List[str], max_len: int = 64, batch_size: int = 32, min_freq: int = 1) -> DataLoader:
    ds = InMemoryTextPairDataset(src_texts, tgt_texts, max_len=max_len, min_freq=min_freq)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
