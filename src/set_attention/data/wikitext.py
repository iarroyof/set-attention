from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


_WIKITEXT_CONFIGS = {
    "wikitext2": ("wikitext", "wikitext-2-raw-v1"),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1"),
}


def load_wikitext_lines(
    dataset: str,
    split: str,
    cache_dir: Path,
    limit: Optional[int] = None,
) -> List[str]:
    """Return cleaned text lines from a Wikitext split."""
    if dataset not in _WIKITEXT_CONFIGS:
        raise ValueError(f"Unsupported Wikitext dataset '{dataset}'")
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError("The 'datasets' package is required for Wikitext loading.") from exc

    name, config = _WIKITEXT_CONFIGS[dataset]
    ds = load_dataset(name, config, cache_dir=str(cache_dir))
    dsplit = ds[split]
    lines: List[str] = []
    for item in dsplit:
        text = item.get("text", "")
        if not text:
            continue
        line = text.strip()
        if not line:
            continue
        lines.append(line)
        if limit is not None and len(lines) >= limit:
            break
    return lines


def tokenize_lines(lines: List[str]) -> List[str]:
    tokens: List[str] = []
    for line in lines:
        tokens.extend(line.split())
    return tokens


def chunk_tokens(tokens: List[str], seq_len: int, stride: Optional[int] = None) -> List[List[str]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    stride = stride or seq_len
    stride = max(1, stride)
    sequences: List[List[str]] = []
    for start in range(0, max(0, len(tokens) - seq_len + 1), stride):
        chunk = tokens[start : start + seq_len]
        if len(chunk) == seq_len:
            sequences.append(chunk)
    return sequences


__all__ = ["load_wikitext_lines", "tokenize_lines", "chunk_tokens"]
