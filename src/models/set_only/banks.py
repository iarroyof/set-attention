from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass
class Bank:
    set_indices: torch.Tensor  # [m, max_set_size] with -1 padding
    set_sizes: torch.Tensor  # [m]
    token_to_sets: torch.Tensor  # [seq, max_sets_per_token] with -1 padding
    set_positions: torch.Tensor  # [m]

    def pool(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        if token_embeddings.dim() != 3:
            raise ValueError("token_embeddings must be [batch, seq, d]")
        batch, _, d_model = token_embeddings.shape

        indices = self.set_indices.clamp_min(0)
        gathered = token_embeddings[:, indices]  # [batch, m, max_set_size, d]
        mask = (self.set_indices >= 0).unsqueeze(0).unsqueeze(-1)
        summed = (gathered * mask).sum(dim=2)
        denom = self.set_sizes.clamp_min(1).unsqueeze(0).unsqueeze(-1)
        return summed / denom


def num_sets_for_length(seq_len: int, window_size: int, stride: int) -> int:
    """Calculate number of sets created by build_window_bank."""
    if seq_len <= 0:
        return 0
    # Number of window starting positions: range(0, seq_len, stride)
    return math.ceil(seq_len / stride)


def build_window_bank(
    seq_len: int,
    window_size: int,
    stride: int,
    device: torch.device,
) -> Bank:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    starts = list(range(0, seq_len, stride))
    set_indices_list: list[list[int]] = []
    for start in starts:
        end = min(start + window_size, seq_len)
        set_indices_list.append(list(range(start, end)))

    m = len(set_indices_list)
    max_set_size = max(len(s) for s in set_indices_list) if m else 0
    set_indices = torch.full((m, max_set_size), -1, dtype=torch.long, device=device)
    set_sizes = torch.zeros((m,), dtype=torch.long, device=device)

    for j, indices in enumerate(set_indices_list):
        if not indices:
            continue
        set_sizes[j] = len(indices)
        set_indices[j, : len(indices)] = torch.tensor(indices, device=device)

    token_sets: list[list[int]] = [[] for _ in range(seq_len)]
    for j, indices in enumerate(set_indices_list):
        for idx in indices:
            token_sets[idx].append(j)

    max_sets_per_token = max((len(s) for s in token_sets), default=0)
    token_to_sets = torch.full(
        (seq_len, max_sets_per_token), -1, dtype=torch.long, device=device
    )
    for t, sets in enumerate(token_sets):
        if sets:
            token_to_sets[t, : len(sets)] = torch.tensor(sets, device=device)

    set_positions = torch.arange(m, device=device, dtype=torch.long)
    return Bank(
        set_indices=set_indices,
        set_sizes=set_sizes,
        token_to_sets=token_to_sets,
        set_positions=set_positions,
    )
