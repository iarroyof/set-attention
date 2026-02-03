from __future__ import annotations

import torch
import torch.nn.functional as F


def set_diversity_loss(
    set_embeddings: torch.Tensor,
    mode: str = "cosine",
    target_similarity: float = 0.3,
) -> torch.Tensor:
    """Penalize overly similar set embeddings."""
    if set_embeddings.dim() != 3:
        raise ValueError("set_embeddings must be [batch, num_sets, d_model]")
    if mode != "cosine":
        raise NotImplementedError(f"mode={mode}")

    B, M, _ = set_embeddings.shape
    if M <= 1:
        return set_embeddings.new_tensor(0.0)

    normed = F.normalize(set_embeddings, p=2, dim=-1)
    sim = torch.matmul(normed, normed.transpose(-2, -1))
    mask = ~torch.eye(M, dtype=torch.bool, device=sim.device)
    off_diag = sim[:, mask].reshape(B, M, M - 1)
    mean_sim = off_diag.mean()
    loss = F.relu(mean_sim - target_similarity)
    return loss
