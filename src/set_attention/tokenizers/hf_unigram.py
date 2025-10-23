from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import torch

from set_attention.tokenizers.utils import read_tokenizer_meta, write_tokenizer_meta

HF_UNIGRAM_TYPE = "hf_unigram"

try:
    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import Unigram
    from tokenizers.trainers import UnigramTrainer
except ImportError as exc:  # pragma: no cover - surfaced via registry checks
    Tokenizer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _ensure_tokenizers_available() -> None:
    if Tokenizer is None:
        raise ImportError(
            "The 'tokenizers' package is required for HF Unigram tokenization. "
            "Install it via `pip install tokenizers`."
        ) from _IMPORT_ERROR


def _build_normalizer(lowercase: bool):
    parts: List[normalizers.Normalizer] = [normalizers.NFC()]
    if lowercase:
        parts.append(normalizers.Lowercase())
    if len(parts) == 1:
        return parts[0]
    return normalizers.Sequence(parts)


@dataclass
class HFUnigramConfig:
    vocab_size: int = 16000
    min_frequency: int = 2
    special_tokens: Sequence[str] = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")
    unk_token: str = "[UNK]"
    lowercase: bool = True
    show_progress: bool = False


class HFUnigramTokenizer:
    def __init__(self, cfg: HFUnigramConfig, tokenizer: Optional["Tokenizer"] = None):
        _ensure_tokenizers_available()
        self.cfg = cfg
        self._tokenizer = tokenizer or Tokenizer(Unigram())
        self._tokenizer.normalizer = _build_normalizer(cfg.lowercase)
        self._tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self._special_ids: Set[int] = set()
        self._update_special_ids()

    def _update_special_ids(self) -> None:
        if Tokenizer is None:
            return
        specials: Set[int] = set()
        for sym in self.cfg.special_tokens:
            idx = self._tokenizer.token_to_id(sym)
            if idx is not None:
                specials.add(int(idx))
        self._special_ids = specials

    def fit(self, corpus: Iterable[str]) -> None:
        trainer = UnigramTrainer(
            vocab_size=self.cfg.vocab_size,
            unk_token=self.cfg.unk_token,
            special_tokens=list(dict.fromkeys(self.cfg.special_tokens)),
            show_progress=self.cfg.show_progress,
        )
        self._tokenizer.train_from_iterator(corpus, trainer)
        self._update_special_ids()

    def encode_text(self, text: str) -> torch.LongTensor:
        output = self._tokenizer.encode(text)
        unique_ids = sorted({i for i in output.ids if i not in self._special_ids})
        if not unique_ids:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(unique_ids, dtype=torch.long)

    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    def save(self, out_dir: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path / "tokenizer.json"))
        write_tokenizer_meta(out_dir, HF_UNIGRAM_TYPE, asdict(self.cfg))

    @classmethod
    def load(cls, out_dir: str, cfg: Optional[HFUnigramConfig] = None) -> "HFUnigramTokenizer":
        _ensure_tokenizers_available()
        tokenizer = Tokenizer.from_file(str(Path(out_dir) / "tokenizer.json"))
        if cfg is None:
            meta = read_tokenizer_meta(out_dir) or {}
            cfg_dict = meta.get("config")
            cfg = HFUnigramConfig(**cfg_dict) if isinstance(cfg_dict, dict) else HFUnigramConfig()
        inst = cls(cfg, tokenizer=tokenizer)
        inst._update_special_ids()
        return inst
