from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import torch

from set_attention.tokenizers.utils import write_tokenizer_meta

ACTIVE_TOKENIZER_TYPE = "ausa"


@dataclass
class TokenizerConfig:
    modality: str = "string"
    seed_lengths: Tuple[int, ...] = (3, 4, 5)
    min_freq: int = 1
    max_len: int = 16
    grow_k: int = 0  # rounds growth disabled by default
    max_vocab: int = 200_000
    lowercase: bool = True


@dataclass
class VocabEntry:
    id: int
    sym: str
    freq: float
    last_update: int


class ActiveUniverseTokenizer:
    """
    Minimal AUT tokenizer for strings:
      - fit(corpus): builds a set of frequent k-mers across seed_lengths
      - encode_text: returns sorted-unique token ids present in a string
      - save/load: persist state

    This module is intentionally lightweight (no external automata deps).
    """

    def __init__(self, cfg: TokenizerConfig):
        self.cfg = cfg
        self.vocab: Dict[str, VocabEntry] = {}
        self.sym2id: Dict[str, int] = {}
        self.id2sym: Dict[int, str] = {}

    def _now(self) -> int:
        return int(time.time())

    def _extract_kmers(self, text: str) -> List[str]:
        if self.cfg.lowercase:
            text = text.lower()
        toks: List[str] = []
        for L in self.cfg.seed_lengths:
            if L <= 0 or len(text) < L:
                continue
            toks.extend(text[i : i + L] for i in range(0, len(text) - L + 1))
        return toks

    def _add(self, sym: str, freq: float):
        if sym in self.sym2id:
            vid = self.sym2id[sym]
            ve = self.vocab[sym]
            ve.freq += freq
            ve.last_update = self._now()
            self.vocab[sym] = ve
            return
        vid = len(self.id2sym) + 1
        ve = VocabEntry(id=vid, sym=sym, freq=float(freq), last_update=self._now())
        self.vocab[sym] = ve
        self.sym2id[sym] = vid
        self.id2sym[vid] = sym

    def fit(self, corpus: Iterable[str]):
        counts: Dict[str, int] = {}
        for doc in corpus:
            for tok in set(self._extract_kmers(doc)):
                counts[tok] = counts.get(tok, 0) + 1
        items = [(t, c) for t, c in counts.items() if c >= self.cfg.min_freq and len(t) <= self.cfg.max_len]
        items.sort(key=lambda x: (-x[1], len(x[0]), x[0]))
        for t, c in items[: self.cfg.max_vocab]:
            self._add(t, c)

    def encode_text(self, text: str) -> torch.LongTensor:
        ids: List[int] = []
        for tok in set(self._extract_kmers(text)):
            if tok in self.sym2id:
                ids.append(self.sym2id[tok])
        if not ids:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(set(ids)), dtype=torch.long)

    def get_top_landmarks(self, rank: int) -> torch.LongTensor:
        items = sorted(self.vocab.values(), key=lambda ve: (-ve.freq, len(ve.sym), ve.sym))
        return torch.tensor([ve.id for ve in items[: max(1, rank)]], dtype=torch.long)

    def vocab_size(self) -> int:
        return len(self.sym2id) + 1

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "version": "1.0",
            "config": asdict(self.cfg),
            "vocab": [asdict(ve) for _, ve in sorted(self.vocab.items(), key=lambda kv: kv[1].id)],
        }
        with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        write_tokenizer_meta(out_dir, ACTIVE_TOKENIZER_TYPE, meta["config"])

    @staticmethod
    def load(out_dir: str) -> "ActiveUniverseTokenizer":
        with open(os.path.join(out_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = TokenizerConfig(**meta["config"])
        tok = ActiveUniverseTokenizer(cfg)
        for ve in meta["vocab"]:
            entry = VocabEntry(**ve)
            tok.vocab[entry.sym] = entry
            tok.sym2id[entry.sym] = entry.id
            tok.id2sym[entry.id] = entry.sym
        return tok
