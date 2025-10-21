from __future__ import annotations
from typing import List, Tuple, Optional


def load_seq2seq_pairs(dataset: str, split: str = "train", limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Download a small, widely used seq2seq dataset via Hugging Face Datasets.

    Supported:
      - 'wmt16_en_ro': English↔Romanian translation (uses 'wmt16', 'ro-en' config).
      - 'cnn_dailymail': summarization ('cnn_dailymail', '3.0.0').

    Returns lists (src_texts, tgt_texts) for the requested split. If datasets
    library is unavailable or download fails, raises ImportError/RuntimeError.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError("HuggingFace 'datasets' package is required to download datasets.") from e

    src_list: List[str] = []
    tgt_list: List[str] = []

    if dataset == "wmt16_en_ro":
        ds = load_dataset("wmt16", "ro-en")
        ds_split = ds[split]
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
    elif dataset == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0")
        ds_split = ds[split]
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

