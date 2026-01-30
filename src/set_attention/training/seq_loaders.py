from __future__ import annotations
from typing import Tuple, Optional

from set_attention.experiments.data_nlp import (
    TextPairConfig,
    TextPairDataset,
    InMemoryTextPairDataset,
    InMemorySharedTextPairDataset,
    StreamingSharedTextPairDataset,
)
from set_attention.experiments.datasets_hf import load_seq2seq_pairs, iter_seq2seq_pairs


def get_seq2seq_datasets(
    dataset: str = "",
    limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    train_indices: Optional[list[int]] = None,
    val_indices: Optional[list[int]] = None,
    src_path: Optional[str] = None,
    tgt_path: Optional[str] = None,
    demo: bool = False,
    demo_samples: int = 200,
    max_len: int = 64,
    cache_dir: Optional[str] = None,
    shared_vocab: bool = True,
    val_split: float = 0.2,
    split_seed: int = 42,
    streaming: bool = True,
):
    """Return (train_dataset, val_dataset) for seq2seq tasks.

    Precedence: dataset -> demo -> local files.
    """
    if not shared_vocab:
        raise NotImplementedError(
            "Separate src/tgt vocab is not implemented yet. "
            "Set model.seq2seq.shared_vocab=true to use shared vocab."
        )

    if dataset:
        if streaming:
            def _train_iter():
                return iter_seq2seq_pairs(
                    dataset,
                    split="train",
                    limit=limit,
                    cache_dir=cache_dir,
                    val_split=val_split,
                    split_seed=split_seed,
                    streaming=streaming,
                )

            def _val_iter():
                return iter_seq2seq_pairs(
                    dataset,
                    split="validation",
                    limit=val_limit or limit,
                    cache_dir=cache_dir,
                    val_split=val_split,
                    split_seed=split_seed,
                    streaming=streaming,
                )

            # Build shared vocab from the train stream.
            from collections import Counter

            cnt = Counter()
            for src, tgt in _train_iter():
                cnt.update(src.split())
                cnt.update(tgt.split())
            itos = ["<pad>", "<s>", "</s>", "<unk>"]
            for w, c in cnt.items():
                if c >= 1:
                    itos.append(w)
            stoi = {w: i for i, w in enumerate(itos)}
            train_ds = StreamingSharedTextPairDataset(_train_iter, stoi, {i: w for w, i in stoi.items()}, max_len=max_len)
            val_ds = StreamingSharedTextPairDataset(_val_iter, stoi, {i: w for w, i in stoi.items()}, max_len=max_len)
            return train_ds, val_ds

        src_tr, tgt_tr = load_seq2seq_pairs(
            dataset,
            split="train",
            limit=limit,
            cache_dir=cache_dir,
            indices=train_indices,
            val_split=val_split,
            split_seed=split_seed,
        )
        src_va, tgt_va = load_seq2seq_pairs(
            dataset,
            split="validation",
            limit=val_limit or limit,
            cache_dir=cache_dir,
            indices=val_indices,
            val_split=val_split,
            split_seed=split_seed,
        )
        train_ds = InMemorySharedTextPairDataset(src_tr, tgt_tr, max_len=max_len)
        val_ds = InMemorySharedTextPairDataset(src_va, tgt_va, max_len=max_len)
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
        if streaming:
            def _demo_iter():
                for s, t in zip(src_demo, tgt_demo):
                    yield s, t
            stoi = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
            itos = {i: w for w, i in stoi.items()}
            train_ds = StreamingSharedTextPairDataset(_demo_iter, stoi, itos, max_len=max_len)
            val_ds = StreamingSharedTextPairDataset(_demo_iter, stoi, itos, max_len=max_len)
            return train_ds, val_ds
        train_ds = InMemorySharedTextPairDataset(src_demo, tgt_demo, max_len=max_len)
        val_ds = InMemorySharedTextPairDataset(src_demo, tgt_demo, max_len=max_len)
        return train_ds, val_ds

    # Local files
    if not src_path or not tgt_path:
        raise ValueError("--src and --tgt required when not using --dataset or --demo")
    cfg = TextPairConfig(src_path=src_path, tgt_path=tgt_path, max_len=max_len)
    train_ds = TextPairDataset(cfg)
    val_ds = TextPairDataset(cfg)
    raise NotImplementedError(
        "Shared vocabulary is required for seq2seq right now. "
        "Local file datasets must be converted to a shared vocab loader."
    )
