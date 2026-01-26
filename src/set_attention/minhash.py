from __future__ import annotations

import torch


def minhash_signatures(
    set_indices: torch.Tensor,
    num_hashes: int,
    max_id: int,
    seed: int = 0,
) -> torch.Tensor:
    if set_indices.dim() != 2:
        raise ValueError("set_indices must be [m, max_set_size]")
    m, _ = set_indices.shape
    device = set_indices.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    prime = 2**31 - 1
    a = torch.randint(1, prime, (num_hashes,), generator=gen, device=device)
    b = torch.randint(0, prime, (num_hashes,), generator=gen, device=device)

    ids = set_indices.clone()
    valid = ids >= 0
    ids[~valid] = max_id + 1

    hashed = (a.view(1, -1, 1) * ids.view(m, 1, -1) + b.view(1, -1, 1)) % prime
    hashed = hashed.masked_fill(~valid.view(m, 1, -1), prime)
    sig = hashed.min(dim=-1).values
    return sig
