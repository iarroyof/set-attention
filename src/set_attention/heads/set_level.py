from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class SetPrototypeBank(nn.Module):
    """Learnable set prototypes over a fixed universe of atoms.

    Each prototype is a Bernoulli membership vector p_j over |U*| (parameterized
    by logits). Atom features are provided by an embedding (atom_emb). The
    prototype feature is Phi(P_j) = p_j @ atom_emb.
    """

    def __init__(self, vocab_size: int, atom_dim: int, num_prototypes: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.atom_dim = atom_dim
        self.num_prototypes = num_prototypes
        self.phi_logits = nn.Parameter(torch.zeros(num_prototypes, vocab_size))

    def forward(self, atom_emb_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # p: (M, |U|)
        p = torch.sigmoid(self.phi_logits)
        # PhiP: (M, D)
        PhiP = p @ atom_emb_weight  # (M, D)
        sizeP = p.sum(dim=-1)  # (M,)
        return PhiP, sizeP


class SetLevelCrossAttentionPrototype(nn.Module):
    """Set-level attention using learned prototypes and token adapters.

    Computes set-level outputs once per example and mixes them to tokens via
    learned gates (token->set adapters).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        atom_dim: int = 128,
        num_prototypes: int = 16,
        gamma: float = 0.3,
        beta: float = 1.0,
        tau: float = 1.0,
        freeze_pool: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.atom_dim = atom_dim
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.tau = float(tau)

        # Atom feature bank (can be frozen to emulate fixed pool)
        self.atom_emb = nn.Embedding(vocab_size, atom_dim)
        if freeze_pool:
            for p in self.atom_emb.parameters():
                p.requires_grad = False

        # Prototypes
        self.proto = SetPrototypeBank(vocab_size, atom_dim, num_prototypes)

        # Per-head linear projections A, B, and value projection V
        self.proj_A = nn.Linear(atom_dim, atom_dim, bias=False)
        self.proj_B = nn.Linear(atom_dim, atom_dim, bias=False)
        self.val_proj = nn.Linear(atom_dim, self.head_dim, bias=True)

        # Token -> head gate and output projection
        self.gate = nn.Linear(d_model, num_heads)
        self.out = nn.Linear(d_model, d_model)

        # Size feature weights
        self.size_w = nn.Parameter(torch.zeros(2))  # weights for [|S_q|, |P_j|]

    def expected_delta(self, set_ids: torch.LongTensor, set_offs: torch.LongTensor, p: torch.Tensor) -> torch.Tensor:
        """Compute E[Delta(S, p)] = |S| + |p|_1 - 2 * sum_{u in S} p_u for all prototypes.

        Args:
            set_ids: (NNZ,) atom ids for each example (concatenated)
            set_offs: (B+1,) CSR offsets
            p: (M, |U|) Bernoulli probs for prototypes
        Returns:
            delta: (B, M) expected symmetric difference sizes
        """
        device = set_ids.device
        B = set_offs.numel() - 1
        M = p.size(0)
        delta = torch.empty(B, M, device=device)
        p_sum = p.sum(dim=-1)  # (M,)
        for b in range(B):
            a, c = int(set_offs[b].item()), int(set_offs[b + 1].item())
            ids_b = set_ids[a:c]
            size_b = ids_b.numel()
            if size_b == 0:
                delta[b] = p_sum  # |S|=0 -> Δ = |p|
            else:
                # sum p_u over ids in this set
                p_sel = p[:, ids_b]  # (M, |S|)
                sum_p_S = p_sel.sum(dim=-1)  # (M,)
                delta[b] = size_b + p_sum - 2.0 * sum_p_S
        return delta

    def set_feature(self, set_ids: torch.LongTensor, set_offs: torch.LongTensor) -> torch.Tensor:
        """Sum embedding of atoms for each set: Phi(S) = sum_u emb[u]. Returns (B, D)."""
        device = set_ids.device
        B = set_offs.numel() - 1
        out = torch.zeros(B, self.atom_dim, device=device)
        emb_w = self.atom_emb.weight
        for b in range(B):
            a, c = int(set_offs[b].item()), int(set_offs[b + 1].item())
            ids_b = set_ids[a:c]
            if ids_b.numel() > 0:
                out[b] = emb_w.index_select(0, ids_b).sum(dim=0)
        return out

    def forward(
        self,
        token_states: torch.Tensor,  # (B,L,D)
        query_ids: torch.LongTensor,
        query_offs: torch.LongTensor,
        key_ids: Optional[torch.LongTensor] = None,
        key_offs: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        B, L, D = token_states.shape
        # Compute features for query sets
        PhiQ = self.set_feature(query_ids, query_offs)  # (B, atom_dim)
        sizeQ = (query_offs[1:] - query_offs[:-1]).to(token_states.device).float()  # (B,)

        # Compute prototype features once
        PhiZ = self.atom_emb.weight  # (|U|, atom_dim)
        PhiP, sizeP = self.proto(PhiZ)  # (M, atom_dim), (M,)

        # Expected Δ between query set and prototypes (B,M)
        p = torch.sigmoid(self.proto.phi_logits)  # (M,|U|)
        DeltaQP = self.expected_delta(query_ids, query_offs, p)  # (B,M)

        # Scoring per head; for simplicity, share projections across heads and split value later
        AQ = self.proj_A(PhiQ)  # (B, atom_dim)
        BP = self.proj_B(PhiP)  # (M, atom_dim)
        # Inner products: (B,M)
        dot = AQ @ BP.T
        size_term = self.size_w[0] * sizeQ.unsqueeze(1) + self.size_w[1] * sizeP.unsqueeze(0)
        scores = -self.gamma * DeltaQP + self.beta * dot + size_term
        attn = torch.softmax(scores / max(1e-6, self.tau), dim=1)  # (B,M)

        # Values from prototypes: project to head_dim and sum with weights, per head we reuse same value projection
        Vp = self.val_proj(PhiP)  # (M, head_dim)
        # For h heads, tile per head and reshape
        Z = attn @ Vp  # (B, head_dim)
        Z = Z.unsqueeze(1).expand(B, self.num_heads, self.head_dim)  # (B,H,Dh)

        # Token adapters: gates from token states -> heads
        gates = torch.softmax(self.gate(token_states), dim=-1)  # (B,L,H)
        # Mix: out_tok[b,l,:] = concat_h (g_blh * Z_bh)
        Z_exp = Z.unsqueeze(1)  # (B,1,H,Dh)
        mixed = (gates.unsqueeze(-1) * Z_exp).reshape(B, L, self.d_model)
        return self.out(mixed)

