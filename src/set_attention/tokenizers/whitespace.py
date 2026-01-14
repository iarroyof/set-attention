from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import torch

from set_attention.tokenizers.utils import write_tokenizer_meta

WHITESPACE_TOKENIZER_TYPE = "whitespace"


@dataclass
class WhitespaceConfig:
    lowercase: bool = True
    min_freq: int = 1
    max_vocab: int = 200_000
    max_len: int = 64


class WhitespaceTokenizer:
    """Simple whitespace tokenizer with a deterministic vocab."""

    def __init__(self, cfg: WhitespaceConfig):
        self.cfg = cfg
        self.sym2id: Dict[str, int] = {}
        self.id2sym: Dict[int, str] = {}

    def fit(self, corpus: Iterable[str]) -> None:
        counts: Dict[str, int] = {}
        for doc in corpus:
            if self.cfg.lowercase:
                doc = doc.lower()
            for tok in doc.split():
                counts[tok] = counts.get(tok, 0) + 1
        items = [(tok, cnt) for tok, cnt in counts.items() if cnt >= self.cfg.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        for tok, _ in items[: self.cfg.max_vocab]:
            if tok in self.sym2id:
                continue
            idx = len(self.id2sym) + 1
            self.sym2id[tok] = idx
            self.id2sym[idx] = tok

    def encode_text(self, text: str) -> torch.LongTensor:
        if self.cfg.lowercase:
            text = text.lower()
        ids: List[int] = []
        for tok in text.split():
            idx = self.sym2id.get(tok)
            if idx is not None:
                ids.append(idx)
        if not ids:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(set(ids)), dtype=torch.long)

    def vocab_size(self) -> int:
        return len(self.sym2id) + 1

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        vocab = [self.id2sym[idx] for idx in sorted(self.id2sym)]
        payload = {"version": "1.0", "config": asdict(self.cfg), "vocab": vocab}
        with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        write_tokenizer_meta(out_dir, WHITESPACE_TOKENIZER_TYPE, payload["config"])

    @staticmethod
    def load(out_dir: str) -> "WhitespaceTokenizer":
        with open(os.path.join(out_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = WhitespaceConfig(**payload["config"])
        tok = WhitespaceTokenizer(cfg)
        for idx, sym in enumerate(payload.get("vocab", []), start=1):
            tok.sym2id[sym] = idx
            tok.id2sym[idx] = sym
        return tok
