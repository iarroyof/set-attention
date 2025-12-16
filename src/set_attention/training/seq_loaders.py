from __future__ import annotations
from typing import Tuple, Optional

from set_attention.experiments.data_nlp import (
    TextPairConfig,
    TextPairDataset,
    InMemoryTextPairDataset,
)
from set_attention.experiments.datasets_hf import load_seq2seq_pairs


def get_seq2seq_datasets(
    dataset: str = "",
    limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    src_path: Optional[str] = None,
    tgt_path: Optional[str] = None,
    demo: bool = False,
    demo_samples: int = 200,
    max_len: int = 64,
):
    """Return (train_dataset, val_dataset) for seq2seq tasks.

    Precedence: dataset -> demo -> local files.
    """
    if dataset:
        src_tr, tgt_tr = load_seq2seq_pairs(dataset, split="train", limit=limit)
        src_va, tgt_va = load_seq2seq_pairs(dataset, split="validation", limit=val_limit or limit)
        train_ds = InMemoryTextPairDataset(src_tr, tgt_tr, max_len=max_len)
        val_ds = InMemoryTextPairDataset(src_va, tgt_va, max_len=max_len)
        return train_ds, val_ds

    if demo:
        import random
        random.seed(0)
        src_demo = []
        tgt_demo = []
        for _ in range(demo_samples):
            L = random.randint(5, 12)
            toks = [str(random.randint(1, 9)) for __ in range(L)]
            src_demo.append(" ".join(toks))
            tgt_demo.append(" ".join(reversed(toks)))
        train_ds = InMemoryTextPairDataset(src_demo, tgt_demo, max_len=max_len)
        val_ds = InMemoryTextPairDataset(src_demo, tgt_demo, max_len=max_len)
        return train_ds, val_ds

    # Local files
    if not src_path or not tgt_path:
        raise ValueError("--src and --tgt required when not using --dataset or --demo")
    cfg = TextPairConfig(src_path=src_path, tgt_path=tgt_path, max_len=max_len)
    train_ds = TextPairDataset(cfg)
    val_ds = TextPairDataset(cfg)
    return train_ds, val_ds
