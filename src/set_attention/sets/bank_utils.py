from __future__ import annotations

from typing import Optional, Tuple

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


def update_coverage_stats(
    mask: torch.Tensor,
    set_idx: torch.Tensor,
    coverage_mask: Optional[torch.Tensor] = None,
) -> Tuple[int, int, Optional[torch.Tensor]]:
    """Accumulate coverage statistics for a batch and optionally mark covered sets.

    Returns the total number of active sets in the batch, the batch sequence count,
    and the mutated coverage mask (if provided).
    """
    if mask.numel() == 0:
        return 0, int(mask.size(0)), coverage_mask
    batch_sets = int(mask.sum().item())
    batch_seqs = int(mask.size(0))
    if coverage_mask is not None and coverage_mask.numel() > 0 and set_idx.numel() > 0:
        coverage_mask[set_idx.detach().cpu().long()] = True
    return batch_sets, batch_seqs, coverage_mask


def pad_segments_from_ptrs(
    data: torch.Tensor,
    ptrs: torch.Tensor,
    *,
    fill_value: float | int = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a concatenated (N, ...) tensor into (B, S_max, ...) using ptr delimiters.

    Args:
        data: Tensor with leading dimension N (total sets).
        ptrs: (B+1,) tensor of int64 offsets delimiting sets per sequence.
        fill_value: Value used to initialize the padded tensor.

    Returns:
        padded: (B, S_max, ...) tensor filled with data rows.
        mask: (B, S_max) bool tensor indicating valid locations.
    """
    B = int(ptrs.numel() - 1)
    if B <= 0:
        padded_shape = (0, 0) + tuple(data.shape[1:])
        padded = data.new_full(padded_shape, fill_value)
        mask = ptrs.new_zeros((0, 0), dtype=torch.bool)
        return padded, mask
    counts = ptrs[1:] - ptrs[:-1]
    S_max = int(counts.max().item()) if counts.numel() > 0 else 0
    padded_shape = (B, S_max) + tuple(data.shape[1:])
    padded = data.new_full(padded_shape, fill_value)
    mask = torch.zeros(B, S_max, dtype=torch.bool, device=data.device)
    if data.shape[0] == 0 or S_max == 0:
        return padded, mask
    seq_ids = torch.repeat_interleave(torch.arange(B, device=data.device, dtype=torch.long), counts)
    base_offsets = ptrs[:-1]
    rel_ids = torch.arange(data.shape[0], device=data.device, dtype=torch.long) - torch.repeat_interleave(base_offsets, counts)
    padded[seq_ids, rel_ids] = data
    mask[seq_ids, rel_ids] = True
    return padded, mask
