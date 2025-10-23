from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import torch

from set_attention.tokenizers.utils import read_tokenizer_meta, write_tokenizer_meta

HF_BPE_TYPE = "hf_bpe"

try:
    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
except ImportError as exc:  # pragma: no cover - surfaced via registry checks
    Tokenizer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _ensure_tokenizers_available() -> None:
    if Tokenizer is None:
        raise ImportError(
            "The 'tokenizers' package is required for HF BPE tokenization. "
            "Install it via `pip install tokenizers`."
        ) from _IMPORT_ERROR


def _build_normalizer(lowercase: bool):
    parts = [normalizers.NFC()]
    if lowercase:
        parts.append(normalizers.Lowercase())
    if len(parts) == 1:
        return parts[0]
    return normalizers.Sequence(parts)


@dataclass
class HFBPEConfig:
    vocab_size: int = 16000
    min_frequency: int = 2
    special_tokens: Sequence[str] = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")
    unk_token: str = "[UNK]"
    lowercase: bool = True
    show_progress: bool = False


class HFBPETokenizer:
    def __init__(self, cfg: HFBPEConfig, tokenizer: Optional["Tokenizer"] = None):
        _ensure_tokenizers_available()
        self.cfg = cfg
        self._tokenizer = tokenizer or Tokenizer(BPE(unk_token=cfg.unk_token))
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
        trainer = BpeTrainer(
            vocab_size=self.cfg.vocab_size,
            min_frequency=self.cfg.min_frequency,
            special_tokens=list(dict.fromkeys(self.cfg.special_tokens)),
            show_progress=self.cfg.show_progress,
        )
        self._tokenizer.train_from_iterator(corpus, trainer)
        self._update_special_ids()

    def encode_text(self, text: str) -> torch.LongTensor:
        encoded = self._tokenizer.encode(text)
        unique_ids = sorted({tok_id for tok_id in encoded.ids if tok_id not in self._special_ids})
        if not unique_ids:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(unique_ids, dtype=torch.long)

    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    def save(self, out_dir: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path / "tokenizer.json"))
        write_tokenizer_meta(out_dir, HF_BPE_TYPE, asdict(self.cfg))

    @classmethod
    def load(cls, out_dir: str, cfg: Optional[HFBPEConfig] = None) -> "HFBPETokenizer":
        _ensure_tokenizers_available()
        tokenizer = Tokenizer.from_file(str(Path(out_dir) / "tokenizer.json"))
        if cfg is None:
            meta = read_tokenizer_meta(out_dir) or {}
            cfg_dict = meta.get("config")
            cfg = HFBPEConfig(**cfg_dict) if isinstance(cfg_dict, dict) else HFBPEConfig()
        inst = cls(cfg, tokenizer=tokenizer)
        inst._update_special_ids()
        return inst
