from __future__ import annotations

from typing import Tuple

import torch

from .banked import BankedSetBatch
from .bank_cache import SetBankCache


def gather_bank_batch(
    bank: BankedSetBatch,
    cache: SetBankCache,
    batch_indices: torch.Tensor,
    phi_dynamic: torch.Tensor,
    use_adapter: bool,
    atom_dim: int,
    minhash_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather padded set features for a batch of sequence indices."""
    device = batch_indices.device
    seq_offsets = bank.seq_offsets
    B = batch_indices.numel()
    counts = []
    for idx in batch_indices.tolist():
        a = int(seq_offsets[idx].item())
        c = int(seq_offsets[idx + 1].item())
        counts.append(c - a)
    S_max = max(counts) if counts else 0
    Phi_pad = torch.zeros(B, S_max, atom_dim, device=device)
    Sig_pad = torch.zeros(B, S_max, minhash_k, dtype=torch.long, device=device)
    Size_pad = torch.zeros(B, S_max, dtype=torch.long, device=device)
    mask = torch.zeros(B, S_max, dtype=torch.bool, device=device)
    for b_idx, seq_idx in enumerate(batch_indices.tolist()):
        a = int(seq_offsets[seq_idx].item())
        c = int(seq_offsets[seq_idx + 1].item())
        if c <= a:
            continue
        set_idx = torch.arange(a, c, device=device, dtype=torch.long)
        sig_seq, size_seq = cache.gather_sig_size(set_idx)
        Sig_pad[b_idx, : sig_seq.size(0)] = sig_seq
        Size_pad[b_idx, : size_seq.size(0)] = size_seq
        mask[b_idx, : sig_seq.size(0)] = True
        if use_adapter:
            Phi_seq = cache.compute_phi_for_indices(set_idx, phi_dynamic)
        else:
            Phi_seq = cache.Phi_sets.index_select(0, set_idx)
        Phi_pad[b_idx, : Phi_seq.size(0)] = Phi_seq
    return Phi_pad, Sig_pad, Size_pad, mask
