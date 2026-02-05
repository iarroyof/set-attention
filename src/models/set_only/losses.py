from __future__ import annotations

import torch
import torch.nn.functional as F


def set_diversity_loss(
    set_embeddings: torch.Tensor,
    mode: str = "exponential",
    target_similarity: float = 0.1,
    set_positions: torch.Tensor = None,
    margin: float = 0.3,
    temperature: float = 0.1,
) -> torch.Tensor:

    if mode == "cross_entropy":
        return set_diversity_cross_entropy_loss(
            set_embeddings=set_embeddings,
            target_similarity=target_similarity
        )
    elif mode == "exponential":
        return set_diversity_exponential_loss(
            set_embeddings=set_embeddings,
            target_similarity=target_similarity
        )
    elif mode == 'position_contrastive':
        return set_diversity_position_contrastive_loss(
            set_embeddings=set_embeddings,
            set_positions=set_positions,
            margin=margin,
            temperature=temperature
         )
    else:
        raise ValueError(f"No {mode} loss mode available or implemented")


def set_diversity_cross_entropy_loss(
    set_embeddings: torch.Tensor,
    target_similarity: float = 0.1,
) -> torch.Tensor:
    """Match similarity distribution to target via cross-entropy."""
    B, M, _ = set_embeddings.shape
    if M < 2:
        return torch.tensor(0.0, device=set_embeddings.device)
    
    # Compute similarities
    normed = F.normalize(set_embeddings, p=2, dim=-1)
    sim = torch.matmul(normed, normed.transpose(-2, -1))
    mask = ~torch.eye(M, dtype=torch.bool, device=sim.device)
    off_diag = sim[:, mask].reshape(B, M * (M - 1))
    
    # Convert to probabilities via softmax
    sim_probs = torch.softmax(off_diag / 0.1, dim=-1)  # temp=0.1 for sharpness
    
    # Target: uniform distribution (all pairs equally dissimilar)
    target_probs = torch.ones_like(sim_probs) / sim_probs.shape[-1]
    
    # Cross-entropy loss
    loss = -(target_probs * torch.log(sim_probs + 1e-8)).sum(dim=-1).mean()
    
    return loss


def set_diversity_exponential_loss(
    set_embeddings: torch.Tensor,
    target_similarity: float = 0.1,
) -> torch.Tensor:
    """Penalize overly similar set embeddings with exponential penalty."""
    B, M, _ = set_embeddings.shape
    if M < 2:
        return torch.tensor(0.0, device=set_embeddings.device)
    
    # Compute pairwise cosine similarity
    normed = F.normalize(set_embeddings, p=2, dim=-1)
    sim = torch.matmul(normed, normed.transpose(-2, -1))
    mask = ~torch.eye(M, dtype=torch.bool, device=sim.device)
    off_diag = sim[:, mask].reshape(B, M, M - 1)
    
    # Exponential penalty: grows super-linearly with similarity
    excess = torch.clamp(off_diag - target_similarity, min=0.0)
    loss = torch.exp(5.0 * excess).mean() - 1.0  # exp(0)=1, so subtract baseline
    
    return loss


def set_diversity_position_contrastive_loss(
    set_embeddings: torch.Tensor,
    set_positions: torch.Tensor = None,  # [num_sets]
    margin: float = 0.3,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrastive loss: nearby sets can be similar, distant sets must differ."""
    B, M, D = set_embeddings.shape
    if M < 2:
        return torch.tensor(0.0, device=set_embeddings.device)
    
    # Compute pairwise cosine similarity
    normed = F.normalize(set_embeddings, p=2, dim=-1)
    sim = torch.matmul(normed, normed.transpose(-2, -1))  # [B, M, M]
    
    # Position-based weighting
    if set_positions is not None:
        # Distance in sequence (0 for adjacent, 1 for max distance)
        pos_delta = torch.abs(set_positions.unsqueeze(0) - set_positions.unsqueeze(1))
        pos_delta = pos_delta.float() / pos_delta.max().clamp(min=1)
        
        # Weight: 0 for adjacent sets, 1 for distant sets
        # Adjacent sets allowed to be similar, distant sets must differ
        weights = pos_delta.unsqueeze(0).expand(B, -1, -1)  # [B, M, M]
    else:
        weights = torch.ones(B, M, M, device=sim.device)
    
    # Mask diagonal
    mask = ~torch.eye(M, dtype=torch.bool, device=sim.device)
    
    # Only penalize when: (1) similarity is high AND (2) sets are distant
    excess_sim = torch.clamp(sim - margin, min=0.0)
    weighted_penalty = (excess_sim * weights)[:, mask].mean()
    
    return weighted_penalty
