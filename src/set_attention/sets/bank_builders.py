from __future__ import annotations
from typing import List, Tuple

import torch

from set_attention.tokenizers.registry import TokenizerProtocol
from set_attention.sets.banked import BankedSetBatch


def build_windowed_bank_from_texts(
    tokenizer: TokenizerProtocol,
    texts: List[str],
    window: int = 5,
    stride: int = 5,
) -> BankedSetBatch:
    """Construct a banked CSR of atom sets from sliding windows over tokenized texts.

    For each text, split on whitespace to tokens, create windows of given length
    and stride, join back to strings, and encode via tokenizer.encode_text.
    """
    all_vals: List[torch.Tensor] = []
    set_offsets = [0]
    seq_offsets = [0]
    nsets = 0
    for text in texts:
        toks = text.split()
        L = len(toks)
        sets_this_seq = 0
        for i in range(0, max(1, L - window + 1), max(1, stride)):
            j = min(L, i + window)
            s = " ".join(toks[i:j])
            ids = tokenizer.encode_text(s)
            all_vals.append(ids)
            set_offsets.append(set_offsets[-1] + ids.numel())
            sets_this_seq += 1
            nsets += 1
        if sets_this_seq == 0:
            # ensure at least one empty set per sequence
            set_offsets.append(set_offsets[-1])
            sets_this_seq = 1
            nsets += 1
        seq_offsets.append(seq_offsets[-1] + sets_this_seq)
    if all_vals:
        values = torch.cat(all_vals, dim=0)
    else:
        values = torch.empty(0, dtype=torch.long)
    return BankedSetBatch(values=values, set_offsets=torch.tensor(set_offsets, dtype=torch.long), seq_offsets=torch.tensor(seq_offsets, dtype=torch.long))


def build_windowed_bank_from_ids(
    sequences: List[torch.Tensor],
    window: int = 8,
    stride: int = 4,
) -> BankedSetBatch:
    """Construct a banked CSR from sequences of integer IDs (1-D tensors)."""
    all_vals: List[torch.Tensor] = []
    set_offsets = [0]
    seq_offsets = [0]
    for seq in sequences:
        seq = seq.to(torch.long)
        L = seq.numel()
        sets_this_seq = 0
        if L == 0:
            set_offsets.append(set_offsets[-1])
            seq_offsets.append(seq_offsets[-1] + 1)
            continue
        step = max(1, stride)
        if window <= 0:
            window = L
        for start in range(0, max(1, L - window + 1), step):
            end = min(L, start + window)
            ids = torch.unique(seq[start:end])
            if ids.numel() == 0:
                continue
            ids, _ = torch.sort(ids)
            all_vals.append(ids)
            set_offsets.append(set_offsets[-1] + ids.numel())
            sets_this_seq += 1
        if sets_this_seq == 0:
            set_offsets.append(set_offsets[-1])
            sets_this_seq = 1
        seq_offsets.append(seq_offsets[-1] + sets_this_seq)
    if all_vals:
        values = torch.cat(all_vals, dim=0)
    else:
        values = torch.empty(0, dtype=torch.long)
    return BankedSetBatch(
        values=values,
        set_offsets=torch.tensor(set_offsets, dtype=torch.long),
        seq_offsets=torch.tensor(seq_offsets, dtype=torch.long),
    )
