from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import torch

from set_attention.experiments.token_sets import build_token_sets_from_texts
from set_attention.tokenizers.registry import TokenizerProtocol
from set_attention.kernels.sketches import MinHasher

SPECIAL_TOKENS = {"<pad>", "<s>", "</s>"}


def encode_sentence(text: str, stoi: dict, max_len: int) -> torch.Tensor:
    tokens = text.split()
    bos = stoi.get("<s>", 0)
    eos = stoi.get("</s>", 1)
    pad = stoi.get("<pad>", 0)
    unk = stoi.get("<unk>", pad)
    ids = [bos]
    ids.extend(stoi.get(tok, unk) for tok in tokens)
    ids.append(eos)
    if len(ids) < max_len:
        ids.extend([pad] * (max_len - len(ids)))
    return torch.tensor(ids[:max_len], dtype=torch.long)


def ids_to_tokens(ids: torch.Tensor, itos: dict) -> List[str]:
    tokens: List[str] = []
    for idx in ids.tolist():
        tok = itos.get(int(idx), "<unk>")
        if tok in SPECIAL_TOKENS:
            continue
        tokens.append(tok)
    return tokens


def ints_to_text(row: torch.Tensor) -> str:
    return " ".join(f"tok{int(v)}" for v in row.tolist())


@dataclass
class TokenSetStore:
    values: torch.Tensor
    offsets: torch.Tensor
    signatures: torch.Tensor

    def gather(self, batch_indices: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vals: List[torch.Tensor] = []
        offs = [0]
        sigs: List[torch.Tensor] = []
        total = 0
        for idx in batch_indices.tolist():
            a = int(self.offsets[idx].item())
            b = int(self.offsets[idx + 1].item())
            chunk = self.values[a:b]
            vals.append(chunk)
            total += chunk.numel()
            offs.append(total)
            sigs.append(self.signatures[idx : idx + 1])
        values = torch.cat(vals, dim=0) if vals else torch.empty(0, dtype=torch.long, device=device)
        offsets = torch.tensor(offs, dtype=torch.long, device=device)
        signatures = torch.cat(sigs, dim=0) if sigs else torch.empty(0, self.signatures.size(1), dtype=torch.long, device=device)
        return values, offsets, signatures


def build_token_set_store(dataset, tokenizer: TokenizerProtocol, k: int, device: torch.device) -> Tuple[TokenSetStore, TokenSetStore]:
    src_texts = [s for (s, _) in dataset.pairs]
    tgt_texts = [t for (_, t) in dataset.pairs]
    src_vals, src_offs = build_token_sets_from_texts(tokenizer, src_texts)
    tgt_vals, tgt_offs = build_token_sets_from_texts(tokenizer, tgt_texts)
    mh = MinHasher(k=k, device=src_vals.device)
    src_sigs = mh.sketch_vec(src_vals, src_offs)
    tgt_sigs = mh.sketch_vec(tgt_vals, tgt_offs)
    src_store = TokenSetStore(src_vals.to(device), src_offs.to(device), src_sigs.to(device))
    tgt_store = TokenSetStore(tgt_vals.to(device), tgt_offs.to(device), tgt_sigs.to(device))
    return src_store, tgt_store


def text_batch_iterator(
    pairs: Sequence[Tuple[str, str]],
    src_stoi: dict,
    tgt_stoi: dict,
    tgt_refs: Sequence[List[str]],
    max_len: int,
    batch_size: int,
    shuffle: bool,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]]:
    indices = list(range(len(pairs)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        src_batch = torch.stack([encode_sentence(pairs[i][0], src_stoi, max_len) for i in batch_idx], dim=0)
        tgt_batch = torch.stack([encode_sentence(pairs[i][1], tgt_stoi, max_len) for i in batch_idx], dim=0)
        refs = [tgt_refs[i] for i in batch_idx]
        yield torch.tensor(batch_idx, dtype=torch.long), src_batch, tgt_batch, refs
