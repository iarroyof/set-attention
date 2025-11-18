from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

from set_attention.kernels.sketches import MinHasher
from set_attention.universe.pool import UniversePool


class SetFeatureCache:
    """Cache for set-level features backed by a UniversePool.

    Stores the CSR layout describing which universe atoms belong to each set,
    precomputes metadata (sizes, MinHash signatures), and provides differentiable
    pooling into the current atom feature bank.
    """

    def __init__(
        self,
        pool: UniversePool,
        values: torch.LongTensor,
        set_offsets: torch.LongTensor,
        seq_offsets: Optional[torch.LongTensor] = None,
        *,
        minhash: Optional[MinHasher] = None,
    ):
        if values.dtype != torch.long:
            raise ValueError("values must be torch.int64 ids.")
        if set_offsets.dim() != 1:
            raise ValueError("set_offsets must be a 1-D tensor.")
        if values.numel() != set_offsets[-1].item():
            raise ValueError("last set offset must equal number of values.")

        self.pool = pool
        self.values = values.clone()
        self.set_offsets = set_offsets.clone()
        self.seq_offsets = seq_offsets.clone() if seq_offsets is not None else None

        self._values_pos = pool.lookup_positions(self.values)
        counts = (self.set_offsets[1:] - self.set_offsets[:-1]).to(
            device=self.set_offsets.device, dtype=torch.long
        )
        self.size_sets = counts

        nsets = self.size_sets.numel()
        if nsets > 0:
            self._elem_set_ids = torch.repeat_interleave(
                torch.arange(nsets, device=self.set_offsets.device, dtype=torch.long), counts
            )
        else:
            self._elem_set_ids = torch.empty(0, dtype=torch.long, device=self.set_offsets.device)

        self.sig_sets: Optional[torch.LongTensor] = None
        if minhash is not None:
            self.build_minhash(minhash)

    def num_sets(self) -> int:
        return int(self.set_offsets.numel() - 1)

    def num_sequences(self) -> int:
        if self.seq_offsets is None:
            return 0
        return int(self.seq_offsets.numel() - 1)

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "SetFeatureCache":
        device = torch.device(device)
        self.values = self.values.to(device, non_blocking=non_blocking)
        self.set_offsets = self.set_offsets.to(device, non_blocking=non_blocking)
        self._values_pos = self._values_pos.to(device, non_blocking=non_blocking)
        self._elem_set_ids = self._elem_set_ids.to(device, non_blocking=non_blocking)
        self.size_sets = self.size_sets.to(device, non_blocking=non_blocking)
        if self.seq_offsets is not None:
            self.seq_offsets = self.seq_offsets.to(device, non_blocking=non_blocking)
        if self.sig_sets is not None:
            self.sig_sets = self.sig_sets.to(device, non_blocking=non_blocking)
        return self

    def build_minhash(self, minhash: MinHasher) -> None:
        self.sig_sets = minhash.sketch_vec(self.values, self.set_offsets)

    def gather_sig_size(self, set_idx: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sig_sets is None:
            raise RuntimeError("MinHash signatures not built; call build_minhash() first.")
        sig = self.sig_sets.index_select(0, set_idx)
        size = self.size_sets.index_select(0, set_idx)
        return sig, size

    def gather_sequence_sets(self, batch_indices: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.seq_offsets is None:
            raise RuntimeError("seq_offsets is not available for this cache.")
        device = batch_indices.device
        idx_rows = []
        ptrs = [0]
        total = 0
        for b in batch_indices.tolist():
            a = int(self.seq_offsets[b].item())
            c = int(self.seq_offsets[b + 1].item())
            idx_rows.append(torch.arange(a, c, device=device, dtype=torch.long))
            total += c - a
            ptrs.append(total)
        set_idx = torch.cat(idx_rows, dim=0) if idx_rows else torch.empty(0, dtype=torch.long, device=device)
        ptr_tensor = torch.tensor(ptrs, dtype=torch.long, device=device)
        return set_idx, ptr_tensor

    def gather_flat(
        self,
        batch_indices: torch.LongTensor,
        phi_cur: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather concatenated set features and signatures with CSR pointers."""
        if self.seq_offsets is None:
            raise RuntimeError("seq_offsets required to gather per-sequence banks.")
        if self.sig_sets is None:
            raise RuntimeError("MinHash signatures not built; call build_minhash() first.")

        seq_device = self.seq_offsets.device
        batch_indices_seq = batch_indices.to(seq_device)
        set_idx, ptrs = self.gather_sequence_sets(batch_indices_seq)
        ptrs = ptrs.to(phi_cur.device)
        if set_idx.numel() == 0:
            atom_dim = phi_cur.shape[1]
            k = int(self.sig_sets.shape[1])
            empty_phi = torch.zeros(0, atom_dim, device=phi_cur.device, dtype=phi_cur.dtype)
            empty_sig = torch.zeros(0, k, dtype=self.sig_sets.dtype, device=phi_cur.device)
            empty_size = torch.zeros(0, dtype=self.size_sets.dtype, device=phi_cur.device)
            return empty_phi, empty_sig, empty_size, ptrs

        set_idx_dev = set_idx.to(phi_cur.device)
        phi_sets = self.compute_phi_for_indices(set_idx_dev, phi_cur)
        sig_sets = self.sig_sets.index_select(0, set_idx).to(phi_cur.device)
        size_sets = self.size_sets.index_select(0, set_idx).to(phi_cur.device)
        return phi_sets, sig_sets, size_sets, ptrs

    def compute_phi_for_indices(self, set_idx: torch.LongTensor, phi_cur: torch.Tensor) -> torch.Tensor:
        nsel = set_idx.numel()
        if nsel == 0:
            return torch.zeros(0, phi_cur.shape[1], device=phi_cur.device, dtype=phi_cur.dtype)

        if phi_cur.shape[0] != len(self.pool):
            raise ValueError("phi_cur row count must match universe size.")

        device = phi_cur.device
        elem_ids = self._elem_set_ids.to(device)
        atom_pos = self._values_pos.to(device)

        mapping = torch.full((self.num_sets(),), -1, dtype=torch.long, device=device)
        mapping.index_copy_(0, set_idx.to(device), torch.arange(nsel, device=device, dtype=torch.long))

        local_rows = mapping.index_select(0, elem_ids)
        mask = local_rows >= 0
        out = torch.zeros(nsel, phi_cur.shape[1], device=device, dtype=phi_cur.dtype)
        if mask.any():
            row_ids = local_rows[mask]
            atom_rows = atom_pos[mask]
            out.index_add_(0, row_ids, phi_cur.index_select(0, atom_rows))
        return out

    def gather_padded(
        self,
        batch_indices: torch.Tensor,
        phi_cur: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather per-sequence sets and return padded Phi/Sigma/Size tensors plus mask."""
        if self.seq_offsets is None:
            raise RuntimeError("seq_offsets required to gather per-sequence banks.")
        if self.sig_sets is None:
            raise RuntimeError("MinHash signatures not built; call build_minhash() first.")

        device = phi_cur.device
        seq_device = self.seq_offsets.device
        batch_indices_seq = batch_indices.to(seq_device)
        set_idx, ptrs = self.gather_sequence_sets(batch_indices_seq)
        ptrs_list = ptrs.tolist()
        B = int(batch_indices_seq.numel())
        counts = [ptrs_list[i + 1] - ptrs_list[i] for i in range(B)] if B > 0 else []
        S_max = max(counts) if counts else 0
        atom_dim = phi_cur.shape[1]
        minhash_k = int(self.sig_sets.shape[1])

        Phi_pad = torch.zeros(B, S_max, atom_dim, device=device, dtype=phi_cur.dtype)
        Sig_pad = torch.zeros(B, S_max, minhash_k, device=device, dtype=torch.long)
        Size_pad = torch.zeros(B, S_max, device=device, dtype=torch.long)
        mask = torch.zeros(B, S_max, device=device, dtype=torch.bool)

        if set_idx.numel() > 0 and S_max > 0:
            set_idx_dev = set_idx.to(device)
            phi_sets = self.compute_phi_for_indices(set_idx_dev, phi_cur)
            sig_sets = self.sig_sets.index_select(0, set_idx).to(device)
            size_sets = self.size_sets.index_select(0, set_idx).to(device)
            for bi in range(B):
                a = ptrs_list[bi]
                c = ptrs_list[bi + 1]
                if c <= a:
                    continue
                span = c - a
                Phi_pad[bi, :span] = phi_sets[a:c]
                Sig_pad[bi, :span] = sig_sets[a:c]
                Size_pad[bi, :span] = size_sets[a:c]
                mask[bi, :span] = True

        return Phi_pad, Sig_pad, Size_pad, mask

    def sequence_ptrs(self, batch_indices: Iterable[int]) -> torch.Tensor:
        device = self.seq_offsets.device if self.seq_offsets is not None else torch.device("cpu")
        ptrs = [0]
        total = 0
        for b in batch_indices:
            a = int(self.seq_offsets[b].item())
            c = int(self.seq_offsets[b + 1].item())
            total += c - a
            ptrs.append(total)
        return torch.tensor(ptrs, dtype=torch.long, device=device)
