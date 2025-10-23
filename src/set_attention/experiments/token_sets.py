from __future__ import annotations
from typing import List, Tuple
import torch

from set_attention.tokenizers.registry import TokenizerProtocol


def build_token_sets_from_texts(
    tokenizer: TokenizerProtocol,
    texts: List[str],
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Encode each text as a set of token IDs (sorted, unique) and collate into
    (values, offsets) CSR-style tensors. This is sequence-level sets; you may
    replicate per token if needed for attention calls.
    """
    ids_all: List[torch.Tensor] = []
    offs = [0]
    for t in texts:
        ids = tokenizer.encode_text(t)
        ids_all.append(ids)
        offs.append(offs[-1] + ids.numel())
    if ids_all:
        vals = torch.cat(ids_all, dim=0)
    else:
        vals = torch.empty(0, dtype=torch.long)
    return vals, torch.tensor(offs, dtype=torch.long)
