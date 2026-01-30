from __future__ import annotations
import os
from typing import Iterable, Iterator, List, Tuple, Optional, Callable
import warnings


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
    val_split: float = 0.2,
    split_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Download a small, widely used seq2seq dataset via Hugging Face Datasets.

    Supported:
      - 'wmt14_fr_en': French↔English translation (uses 'wmt/wmt14', 'fr-en' config).
      - 'cnn_dailymail': summarization ('abisee/cnn_dailymail', '3.0.0').
      - 'opus_books_en_fr': OPUS Books en-fr ('opus_books', 'en-fr').

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

    if dataset == "wmt14_fr_en":
        try:
            ds = load_dataset("wmt/wmt14", "fr-en", download_mode="reuse_dataset_if_exists", **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load wmt14 fr-en. Dataset may be missing from cache and network download failed. "
                "Prefetch into the shared cache or enable network access. "
                f"cache_dir={cache_root or 'default'} | original error: {exc}"
            ) from exc
        ds_split = _resolve_split(ds, dataset, split, val_split, split_seed)
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
            ds = load_dataset(
                "abisee/cnn_dailymail", "3.0.0", download_mode="reuse_dataset_if_exists", **kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load cnn_dailymail. Dataset may be missing from cache and network download failed. "
                "Prefetch into the shared cache or enable network access. "
                f"cache_dir={cache_root or 'default'} | original error: {exc}"
            ) from exc
        ds_split = _resolve_split(ds, dataset, split, val_split, split_seed)
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
    elif dataset == "opus_books_en_fr":
        try:
            ds = load_dataset(
                "opus_books", "en-fr", download_mode="reuse_dataset_if_exists", **kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load opus_books en-fr. Dataset may be missing from cache and network download failed. "
                "Prefetch into the shared cache or enable network access. "
                f"cache_dir={cache_root or 'default'} | original error: {exc}"
            ) from exc
        ds_split = _resolve_split(ds, dataset, split, val_split, split_seed)
        if indices is not None:
            ds_split = ds_split.select(indices)
        for rec in ds_split:
            tr = rec.get("translation", {})
            src = tr.get("en", "")
            tgt = tr.get("fr", "")
            if src and tgt:
                src_list.append(src)
                tgt_list.append(tgt)
            if limit is not None and len(src_list) >= limit:
                break
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return src_list, tgt_list


def _resolve_split(ds, dataset: str, split: str, val_split: float, split_seed: int):
    if split in ds and not (split in {"train", "validation"} and "validation" not in ds):
        return ds[split]
    if split in {"train", "validation", "test"} and "train" in ds:
        warnings.warn(
            f"Split '{split}' not found for {dataset}; creating validation split from train "
            f"(test_size={val_split}, seed={split_seed}).",
            RuntimeWarning,
        )
        split_ds = ds["train"].train_test_split(test_size=val_split, seed=split_seed)
        return split_ds["train"] if split == "train" else split_ds["test"]
    warnings.warn(
        f"Split '{split}' not found for {dataset}; falling back to 'train'.",
        RuntimeWarning,
    )
    return ds["train"]


def _stream_filter_by_split(idx: int, split: str, val_split: float, seed: int) -> bool:
    bucket = (idx * 1103515245 + seed) % 1000000
    is_val = bucket < int(val_split * 1000000)
    return is_val if split == "validation" else not is_val


def iter_seq2seq_pairs(
    dataset: str,
    split: str = "train",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None,
    val_split: float = 0.2,
    split_seed: int = 42,
    streaming: bool = True,
) -> Iterator[Tuple[str, str]]:
    """Stream (src, tgt) pairs lazily."""
    os.environ.setdefault("HF_DATASETS_DISABLE_SHARE_MEM", "1")
    os.environ.setdefault("TORCH_DISABLE_SHARED_MEMORY", "1")
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError("HuggingFace 'datasets' package is required to download datasets.") from e

    cache_root = _resolve_cache_dir(cache_dir)
    kwargs = {}
    if cache_root:
        kwargs["cache_dir"] = cache_root

    if dataset == "wmt14_fr_en":
        name, cfg = "wmt/wmt14", "fr-en"
    elif dataset == "cnn_dailymail":
        name, cfg = "abisee/cnn_dailymail", "3.0.0"
    elif dataset == "opus_books_en_fr":
        name, cfg = "opus_books", "en-fr"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    def _emit(ds_obj, use_streaming: bool):
        if use_streaming:
            if split in ds_obj:
                stream = ds_obj[split]
                for idx, rec in enumerate(stream):
                    if dataset == "cnn_dailymail":
                        src = rec.get("article", "")
                        tgt = rec.get("highlights", "")
                    else:
                        tr = rec.get("translation", {})
                        src = tr.get("en", "")
                        tgt = tr.get("fr", "")
                    if src and tgt:
                        yield src, tgt
                    if limit is not None and idx + 1 >= limit:
                        break
                return
            # Fallback: split from train stream deterministically.
            stream = ds_obj["train"]
            kept = 0
            for idx, rec in enumerate(stream):
                if not _stream_filter_by_split(idx, split, val_split, split_seed):
                    continue
                if dataset == "cnn_dailymail":
                    src = rec.get("article", "")
                    tgt = rec.get("highlights", "")
                else:
                    tr = rec.get("translation", {})
                    src = tr.get("en", "")
                    tgt = tr.get("fr", "")
                if src and tgt:
                    yield src, tgt
                    kept += 1
                if limit is not None and kept >= limit:
                    break
            return

        ds_split = _resolve_split(ds_obj, dataset, split, val_split, split_seed)
        for idx, rec in enumerate(ds_split):
            if dataset == "cnn_dailymail":
                src = rec.get("article", "")
                tgt = rec.get("highlights", "")
            else:
                tr = rec.get("translation", {})
                src = tr.get("en", "")
                tgt = tr.get("fr", "")
            if src and tgt:
                yield src, tgt
            if limit is not None and idx + 1 >= limit:
                break

    if streaming:
        try:
            ds = load_dataset(
                name,
                cfg,
                cache_dir=str(cache_root) if cache_root else None,
                streaming=True,
            )
            yield from _emit(ds, True)
            return
        except RuntimeError as exc:
            warnings.warn(
                "Streaming dataset failed to initialize shared memory; falling back to "
                "memory-mapped dataset iteration.",
                RuntimeWarning,
            )

    ds = load_dataset(name, cfg, cache_dir=str(cache_root) if cache_root else None, streaming=False)
    yield from _emit(ds, False)
