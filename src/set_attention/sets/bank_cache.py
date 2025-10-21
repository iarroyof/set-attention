from __future__ import annotations
from typing import Tuple, List

import torch
import torch.nn as nn

from set_attention.kernels.sketches import MinHasher


class SetBankCache:
    """Precompute features for all sets in a bank and provide batch gathers.

    Args:
        phi_z: (V, D) atom feature bank (can be nn.Embedding.weight)
        minhash_k: signature length for MinHash
    """

    def __init__(self, phi_z: torch.Tensor, minhash_k: int = 64):
        self.phi_z = phi_z  # (V, D)
        self.mh = MinHasher(k=minhash_k, device=phi_z.device)
        self.Phi_sets: torch.Tensor | None = None  # (Nsets, D)
        self.Sig_sets: torch.Tensor | None = None  # (Nsets, k)
        self.Size_sets: torch.Tensor | None = None  # (Nsets,)
        # Store raw CSR to enable per-batch pooling for adapters
        self._values: torch.Tensor | None = None
        self._set_offsets: torch.Tensor | None = None
        self._elem_set_ids: torch.Tensor | None = None  # (NNZ,) map each value to its set index

    def precompute(self, values: torch.LongTensor, set_offsets: torch.LongTensor) -> None:
        """Precompute feature-independent data (signatures, sizes).

        A snapshot of Phi_sets is kept for convenience under no_grad, but for
        trainable adapters compute Phi per-batch via compute_phi_for_indices.
        """
        with torch.no_grad():
            device = values.device
            nsets = set_offsets.numel() - 1
            D = self.phi_z.shape[1]
            Phi = torch.zeros(nsets, D, device=device)
            sizes = torch.empty(nsets, dtype=torch.long, device=device)
            for j in range(nsets):
                a = int(set_offsets[j].item())
                c = int(set_offsets[j + 1].item())
                ids = values[a:c]
                sizes[j] = ids.numel()
                if ids.numel() > 0:
                    Phi[j] = self.phi_z.index_select(0, ids).sum(dim=0)
            Sig = self.mh.sketch_vec(values, set_offsets)  # (Nsets, k)
            self.Phi_sets = Phi
            self.Sig_sets = Sig
            self.Size_sets = sizes
            self._values = values
            self._set_offsets = set_offsets
            # Precompute per-element set-id vector for fast GPU pooling
            counts = (set_offsets[1:] - set_offsets[:-1]).to(torch.long)
            if counts.numel() > 0:
                self._elem_set_ids = torch.repeat_interleave(torch.arange(nsets, device=device, dtype=torch.long), counts)
            else:
                self._elem_set_ids = torch.empty(0, dtype=torch.long, device=device)

    def gather(self, seq_offsets: torch.LongTensor, batch_indices: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather concatenated set features for the selected batch sequences.

        Returns:
            Phi_sel: (Nq, D)
            Sig_sel: (Nq, k)
            Size_sel: (Nq,)
            ptrs: (B+1,) pointers delimiting which rows belong to each batch sequence
        """
        assert self.Phi_sets is not None and self.Sig_sets is not None and self.Size_sets is not None
        device = self.Phi_sets.device
        phi_rows: List[torch.Tensor] = []
        sig_rows: List[torch.Tensor] = []
        size_rows: List[torch.Tensor] = []
        ptrs = [0]
        total = 0
        for b in batch_indices.tolist():
            a = int(seq_offsets[b].item())
            c = int(seq_offsets[b + 1].item())
            phi_rows.append(self.Phi_sets[a:c])
            sig_rows.append(self.Sig_sets[a:c])
            size_rows.append(self.Size_sets[a:c])
            total += (c - a)
            ptrs.append(total)
        Phi_sel = torch.cat(phi_rows, dim=0) if phi_rows else torch.empty(0, self.phi_z.shape[1], device=device)
        Sig_sel = torch.cat(sig_rows, dim=0) if sig_rows else torch.empty(0, self.mh.k, dtype=torch.long, device=device)
        Size_sel = torch.cat(size_rows, dim=0) if size_rows else torch.empty(0, dtype=torch.long, device=device)
        return Phi_sel, Sig_sel, Size_sel, torch.tensor(ptrs, dtype=torch.long, device=device)

    def gather_indices(self, seq_offsets: torch.LongTensor, batch_indices: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather set indices for batch sequences and block pointers."""
        assert self._set_offsets is not None
        device = self._set_offsets.device
        idx_rows: List[torch.Tensor] = []
        ptrs = [0]
        total = 0
        for b in batch_indices.tolist():
            a = int(seq_offsets[b].item())
            c = int(seq_offsets[b + 1].item())
            idx_rows.append(torch.arange(a, c, device=device, dtype=torch.long))
            total += (c - a)
            ptrs.append(total)
        set_idx = torch.cat(idx_rows, dim=0) if idx_rows else torch.empty(0, dtype=torch.long, device=device)
        return set_idx, torch.tensor(ptrs, dtype=torch.long, device=device)

    def gather_sig_size(self, set_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.Sig_sets is not None and self.Size_sets is not None
        return self.Sig_sets.index_select(0, set_idx), self.Size_sets.index_select(0, set_idx)

    def compute_phi_for_indices(self, set_idx: torch.Tensor, phi_cur: torch.Tensor) -> torch.Tensor:
        """Compute Phi for selected set indices using current phi_cur (autograd-friendly)."""
        assert self._values is not None and self._set_offsets is not None and self._elem_set_ids is not None
        device = phi_cur.device
        D = phi_cur.shape[1]
        nsel = set_idx.numel()
        out = torch.zeros(nsel, D, device=device, dtype=phi_cur.dtype)
        if nsel == 0:
            return out
        # Build mapping from global set id -> local row id [0..nsel-1], -1 for others
        nsets_total = int(self._set_offsets.numel() - 1)
        mapping = torch.full((nsets_total,), -1, device=device, dtype=torch.long)
        mapping.index_copy_(0, set_idx, torch.arange(nsel, device=device, dtype=torch.long))
        # Select elements whose set is in the selected indices
        local_rows = mapping.index_select(0, self._elem_set_ids)
        mask = local_rows >= 0
        if mask.any():
            row_ids = local_rows[mask]
            atom_ids = self._values[mask]
            out.index_add_(0, row_ids, phi_cur.index_select(0, atom_ids))
        return out
