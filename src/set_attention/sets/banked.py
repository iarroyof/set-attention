from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List

import torch


@dataclass
class BankedSetBatch:
    """Two-level ragged (banked) CSR for multiple sets per sequence.

    values: concatenated atom IDs for all sets across all sequences
    set_offsets: CSR pointers over `values` delimiting each set
    seq_offsets: CSR pointers over `set_offsets` delimiting which sets belong to each sequence
    """

    values: torch.LongTensor        # (NNZ,)
    set_offsets: torch.LongTensor   # (Nsets+1,)
    seq_offsets: torch.LongTensor   # (B+1,)

    def to(self, device: torch.device | str) -> "BankedSetBatch":
        self.values = self.values.to(device)
        self.set_offsets = self.set_offsets.to(device)
        self.seq_offsets = self.seq_offsets.to(device)
        return self

    def num_sequences(self) -> int:
        return int(self.seq_offsets.numel() - 1)

    def num_sets(self) -> int:
        return int(self.set_offsets.numel() - 1)

    def sets_for_sequence(self, b: int) -> range:
        a = int(self.seq_offsets[b].item())
        c = int(self.seq_offsets[b + 1].item())
        return range(a, c)

