from __future__ import annotations
import os
from typing import List, Tuple, Optional


def _resolve_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    env_cache = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME")
    if cache_dir:
        expanded = os.path.expanduser(cache_dir)
        if os.path.exists(expanded):
            return expanded
        # if provided path missing, fall back to env cache if available
        if env_cache:
            return env_cache
        return expanded
    return env_cache


def load_seq2seq_pairs(
    dataset: str,
    split: str = "train",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None,
    indices: Optional[List[int]] = None,
) -> Tuple[List[str], List[str]]:
    """Download a small, widely used seq2seq dataset via Hugging Face Datasets.

    Supported:
      - 'wmt16_en_ro': English↔Romanian translation (uses 'wmt16', 'ro-en' config).
      - 'wmt16_en_es': English↔Spanish translation (uses 'wmt16', 'es-en' config).
      - 'wmt16_en_fr': English↔French translation (uses 'wmt16', 'fr-en' config).
      - 'cnn_dailymail': summarization ('cnn_dailymail', '3.0.0').

    Returns lists (src_texts, tgt_texts) for the requested split. If datasets
    library is unavailable or download fails, raises ImportError/RuntimeError.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError("HuggingFace 'datasets' package is required to download datasets.") from e

    cache_root = _resolve_cache_dir(cache_dir)
    kwargs = {}
    if cache_root:
        kwargs["cache_dir"] = cache_root
    src_list: List[str] = []
    tgt_list: List[str] = []

    if dataset == "wmt16_en_ro":
        try:
            ds = load_dataset("wmt16", "ro-en", download_mode="reuse_dataset_if_exists", **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load wmt16 ro-en. Dataset may be missing from cache and network download failed. "
                "Prefetch into the shared cache or enable network access. "
                f"cache_dir={cache_root or 'default'} | original error: {exc}"
            ) from exc
        ds_split = ds[split]
        if indices is not None:
            ds_split = ds_split.select(indices)
        for rec in ds_split:
            # translation dict has keys 'ro','en'
            tr = rec.get("translation", {})
            src = tr.get("en", "")
            tgt = tr.get("ro", "")
            if src and tgt:
                src_list.append(src)
                tgt_list.append(tgt)
            if limit is not None and len(src_list) >= limit:
                break
    elif dataset == "wmt16_en_es":
        try:
            ds = load_dataset("wmt16", "es-en", download_mode="reuse_dataset_if_exists", **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load wmt16 es-en. Dataset may be missing from cache and network download failed. "
                "Prefetch into the shared cache or enable network access. "
                f"cache_dir={cache_root or 'default'} | original error: {exc}"
            ) from exc
        ds_split = ds[split]
        if indices is not None:
            ds_split = ds_split.select(indices)
        for rec in ds_split:
            # translation dict has keys 'es','en'
            tr = rec.get("translation", {})
            src = tr.get("en", "")
            tgt = tr.get("es", "")
            if src and tgt:
                src_list.append(src)
                tgt_list.append(tgt)
            if limit is not None and len(src_list) >= limit:
                break
    elif dataset == "wmt16_en_fr":
        try:
            ds = load_dataset("wmt16", "fr-en", download_mode="reuse_dataset_if_exists", **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load wmt16 fr-en. Dataset may be missing from cache and network download failed. "
                "Prefetch into the shared cache or enable network access. "
                f"cache_dir={cache_root or 'default'} | original error: {exc}"
            ) from exc
        ds_split = ds[split]
        if indices is not None:
            ds_split = ds_split.select(indices)
        for rec in ds_split:
            # translation dict has keys 'fr','en'
            tr = rec.get("translation", {})
            src = tr.get("en", "")
            tgt = tr.get("fr", "")
            if src and tgt:
                src_list.append(src)
                tgt_list.append(tgt)
            if limit is not None and len(src_list) >= limit:
                break
    elif dataset == "cnn_dailymail":
        try:
            ds = load_dataset("cnn_dailymail", "3.0.0", download_mode="reuse_dataset_if_exists", **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load cnn_dailymail. Dataset may be missing from cache and network download failed. "
                "Prefetch into the shared cache or enable network access. "
                f"cache_dir={cache_root or 'default'} | original error: {exc}"
            ) from exc
        ds_split = ds[split]
        if indices is not None:
            ds_split = ds_split.select(indices)
        for rec in ds_split:
            src = rec.get("article", "")
            tgt = rec.get("highlights", "")
            if src and tgt:
                src_list.append(src)
                tgt_list.append(tgt)
            if limit is not None and len(src_list) >= limit:
                break
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return src_list, tgt_list
