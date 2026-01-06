from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple


_WIKITEXT_CONFIGS = {
    "wikitext2": ("wikitext", "wikitext-2-raw-v1"),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1"),
}


def load_wikitext_hf_dataset(
    dataset: str,
    cache_dir: Path,
    *,
    streaming: bool = False,
):
    """Load the HuggingFace dataset for downstream streaming-friendly helpers."""
    if dataset not in _WIKITEXT_CONFIGS:
        raise ValueError(f"Unsupported Wikitext dataset '{dataset}'")
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError("The 'datasets' package is required for Wikitext loading.") from exc

    name, config = _WIKITEXT_CONFIGS[dataset]
    try:
        return load_dataset(name, config, cache_dir=str(cache_dir), streaming=streaming)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {name}:{config} from cache {cache_dir}. "
            "Prefetch the dataset (scripts/prefetch_datasets.py) or allow network access."
        ) from exc


def iter_wikitext_lines(
    dataset: str,
    split: str,
    cache_dir: Path,
    limit: Optional[int] = None,
    dataset_obj=None,
    indices: Optional[List[int]] = None,
) -> Iterator[str]:
    """Yield cleaned text lines lazily from a Wikitext split."""
    if dataset_obj is None:
        dataset_obj = load_wikitext_hf_dataset(dataset, cache_dir)
    if indices is not None and getattr(dataset_obj, "is_streaming", False):
        raise ValueError("Subset indices are not supported with streaming datasets.")
    if split not in dataset_obj:
        raise ValueError(f"Split '{split}' not available for dataset '{dataset}'")
    dsplit = dataset_obj[split]
    if indices is not None:
        dsplit = dsplit.select(indices)
    count = 0
    for item in dsplit:
        text = item.get("text", "")
        if not text:
            continue
        line = text.strip()
        if not line:
            continue
        yield line
        count += 1
        if limit is not None and count >= limit:
            break


def load_wikitext_lines(
    dataset: str,
    split: str,
    cache_dir: Path,
    limit: Optional[int] = None,
    indices: Optional[List[int]] = None,
) -> List[str]:
    """Return cleaned text lines from a Wikitext split."""
    return list(iter_wikitext_lines(dataset, split, cache_dir, limit, indices=indices))


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


__all__ = [
    "chunk_tokens",
    "iter_wikitext_lines",
    "load_wikitext_hf_dataset",
    "load_wikitext_lines",
    "tokenize_lines",
]
