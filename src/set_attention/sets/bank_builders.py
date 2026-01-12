from __future__ import annotations
from typing import List

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
    sequences: torch.Tensor | List[torch.Tensor],
    window: int = 8,
    stride: int = 4,
    pad_id: int | None = None,
) -> BankedSetBatch:
    """Construct a banked CSR from token ID sequences.

    Supports either a list of 1-D tensors or a 2-D tensor [N, L].
    When pad_id is provided, PAD tokens are ignored per window.
    """
    all_vals: List[torch.Tensor] = []
    set_offsets = [0]
    seq_offsets = [0]
    step = max(1, stride)

    if isinstance(sequences, torch.Tensor):
        if sequences.ndim == 1:
            seq_iter = [sequences]
        elif sequences.ndim == 2:
            seq_iter = [row for row in sequences]
        else:
            raise ValueError("sequences tensor must be 1-D or 2-D.")
    else:
        seq_iter = sequences

    for seq in seq_iter:
        seq = seq.to(torch.long)
        if pad_id is not None:
            keep = (seq != pad_id).nonzero(as_tuple=False)
            if keep.numel() == 0:
                L = 0
                seq = seq[:0]
            else:
                last = int(keep[-1].item())
                seq = seq[: last + 1]
                L = seq.numel()
        else:
            L = seq.numel()
        sets_this_seq = 0
        if L == 0:
            set_offsets.append(set_offsets[-1])
            seq_offsets.append(seq_offsets[-1] + 1)
            continue
        win = window if window > 0 else L
        for start in range(0, max(1, L - win + 1), step):
            end = min(L, start + win)
            ids = seq[start:end]
            if pad_id is not None:
                ids = ids[ids != pad_id]
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
