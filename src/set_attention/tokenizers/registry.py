from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Type

import torch

from set_attention.tokenizers.active_tokenizer import (
    ACTIVE_TOKENIZER_TYPE,
    ActiveUniverseTokenizer,
    TokenizerConfig,
)
from set_attention.tokenizers.hf_bpe import HF_BPE_TYPE, HFBPEConfig, HFBPETokenizer
from set_attention.tokenizers.hf_unigram import HF_UNIGRAM_TYPE, HFUnigramConfig, HFUnigramTokenizer
from set_attention.tokenizers.whitespace import WHITESPACE_TOKENIZER_TYPE, WhitespaceConfig, WhitespaceTokenizer
from set_attention.tokenizers.utils import read_tokenizer_meta


class TokenizerProtocol(Protocol):
    def fit(self, corpus: Iterable[str]) -> None:
        ...

    def encode_text(self, text: str) -> torch.LongTensor:
        ...

    def save(self, out_dir: str) -> None:
        ...

    def vocab_size(self) -> int:
        ...


@dataclass(frozen=True)
class _RegistryEntry:
    type_name: str
    constructor: Callable[[Any], TokenizerProtocol]
    loader: Callable[[str, Optional[Dict[str, Any]]], TokenizerProtocol]
    config_cls: Type

    def build(self, cfg_dict: Optional[Dict[str, Any]] = None):
        cfg = self.config_cls(**(cfg_dict or {}))
        return self.constructor(cfg)


def _ausa_loader(path: str, _: Optional[Dict[str, Any]]) -> TokenizerProtocol:
    return ActiveUniverseTokenizer.load(path)


def _hf_unigram_loader(path: str, cfg_dict: Optional[Dict[str, Any]]) -> TokenizerProtocol:
    cfg = HFUnigramConfig(**cfg_dict) if cfg_dict is not None else None
    return HFUnigramTokenizer.load(path, cfg)


def _hf_bpe_loader(path: str, cfg_dict: Optional[Dict[str, Any]]) -> TokenizerProtocol:
    cfg = HFBPEConfig(**cfg_dict) if cfg_dict is not None else None
    return HFBPETokenizer.load(path, cfg)


_REGISTRY: Dict[str, _RegistryEntry] = {
    WHITESPACE_TOKENIZER_TYPE: _RegistryEntry(
        WHITESPACE_TOKENIZER_TYPE,
        constructor=lambda cfg: WhitespaceTokenizer(cfg),
        loader=lambda path, _: WhitespaceTokenizer.load(path),
        config_cls=WhitespaceConfig,
    ),
    ACTIVE_TOKENIZER_TYPE: _RegistryEntry(
        ACTIVE_TOKENIZER_TYPE,
        constructor=lambda cfg: ActiveUniverseTokenizer(cfg),
        loader=_ausa_loader,
        config_cls=TokenizerConfig,
    ),
    HF_UNIGRAM_TYPE: _RegistryEntry(
        HF_UNIGRAM_TYPE,
        constructor=lambda cfg: HFUnigramTokenizer(cfg),
        loader=_hf_unigram_loader,
        config_cls=HFUnigramConfig,
    ),
    HF_BPE_TYPE: _RegistryEntry(
        HF_BPE_TYPE,
        constructor=lambda cfg: HFBPETokenizer(cfg),
        loader=_hf_bpe_loader,
        config_cls=HFBPEConfig,
    ),
}


def available_tokenizer_types() -> Iterable[str]:
    return tuple(_REGISTRY.keys())


def create_tokenizer(kind: str, config: Optional[Dict[str, Any]] = None) -> TokenizerProtocol:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown tokenizer kind '{kind}'. Available: {', '.join(available_tokenizer_types())}")
    entry = _REGISTRY[kind]
    try:
        return entry.build(config)
    except ImportError as exc:  # bubble missing optional deps with context
        raise ImportError(f"Failed to initialize tokenizer '{kind}': {exc}") from exc


def load_tokenizer(path: str, kind: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> TokenizerProtocol:
    detected_cfg: Optional[Dict[str, Any]] = None
    if kind is None:
        meta = read_tokenizer_meta(path)
        if meta:
            kind = meta.get("type", kind)
            if "config" in meta and isinstance(meta["config"], dict):
                detected_cfg = meta["config"]
    if kind is None:
        kind = ACTIVE_TOKENIZER_TYPE
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown tokenizer kind '{kind}' when loading from '{path}'.")
    entry = _REGISTRY[kind]
    cfg_dict = config if config is not None else detected_cfg
    try:
        return entry.loader(path, cfg_dict)
    except ImportError as exc:
        raise ImportError(f"Failed to load tokenizer '{kind}' from '{path}': {exc}") from exc


def save_tokenizer(tokenizer: TokenizerProtocol, out_dir: str) -> None:
    tokenizer.save(out_dir)
