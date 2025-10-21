from __future__ import annotations
import torch


class MinHasher(torch.nn.Module):
    """Simple MinHash signatures for integer IDs (CPU/GPU).

    Uses k hash functions h(x) = (a*x + b) mod p, with a,b random in [1,p).
    """

    def __init__(self, k: int = 128, prime: int = (1 << 61) - 1, device=None):
        super().__init__()
        self.k = int(k)
        self.p = int(prime)
        gen = torch.Generator(device=device)
        a = torch.randint(1, self.p - 1, (k,), generator=gen, device=device, dtype=torch.long)
        b = torch.randint(0, self.p - 1, (k,), generator=gen, device=device, dtype=torch.long)
        self.register_buffer("a", a)
        self.register_buffer("b", b)

    def _hash(self, ids: torch.LongTensor) -> torch.LongTensor:
        return (self.a[:, None] * ids[None, :] + self.b[:, None]) % self.p

    def sketch(self, ids: torch.LongTensor, offsets: torch.LongTensor) -> torch.Tensor:
        B = offsets.numel() - 1
        sigs = torch.empty(B, self.k, dtype=torch.long, device=ids.device)
        for i in range(B):
            s = torch.unique(ids[offsets[i] : offsets[i + 1]])
            if s.numel() == 0:
                sigs[i].fill_(self.p - 1)
            else:
                H = self._hash(s)
                sigs[i] = H.min(dim=1).values
        return sigs

    def sketch_vec(self, ids: torch.LongTensor, offsets: torch.LongTensor) -> torch.Tensor:
        """Vectorized MinHash over multiple sets.

        Computes signatures for all sets using segment-wise minimum without
        Python loops over sets. Loops over k hash functions only.

        Args:
            ids: (NNZ,) int64 concatenated ids
            offsets: (S+1,) CSR offsets
        Returns:
            sigs: (S, k) int64 signatures
        """
        device = ids.device
        S = offsets.numel() - 1
        if S <= 0:
            return torch.empty(0, self.k, dtype=torch.long, device=device)
        # Build segment ids per element
        seg_ids = torch.empty_like(ids)
        # Vectorized fill of segment ids from offsets
        # Note: This still uses a small loop over sets but avoids Python hashing
        start = 0
        for s in range(S):
            a, b = int(offsets[s].item()), int(offsets[s + 1].item())
            if b > a:
                seg_ids[a:b] = s
        sigs = torch.empty(S, self.k, dtype=torch.long, device=device)
        for j in range(self.k):
            h = (self.a[j] * ids + self.b[j]) % self.p  # (NNZ,)
            # segment-wise min
            # initialize with large value
            out = torch.full((S,), self.p - 1, dtype=torch.long, device=device)
            out = out.scatter_reduce(0, seg_ids, h, reduce="amin", include_self=True)
            sigs[:, j] = out
        return sigs

    @staticmethod
    def jaccard_from_signatures(sigA: torch.Tensor, sigB: torch.Tensor) -> torch.Tensor:
        nA, k = sigA.shape
        nB = sigB.shape[0]
        blk = max(1, 1_000_000 // max(k, 1))
        out = torch.empty(nA, nB, dtype=torch.float32, device=sigA.device)
        for i in range(0, nA, blk):
            i2 = min(nA, i + blk)
            Ablk = sigA[i:i2, None, :]
            eq = (Ablk == sigB[None, :, :])
            out[i:i2] = eq.float().mean(dim=-1)
        return out


def symdiff_from_jaccard(jacc: torch.Tensor, sizeA: torch.Tensor, sizeB: torch.Tensor) -> torch.Tensor:
    SA = sizeA[:, None].float()
    SB = sizeB[None, :].float()
    inter = (jacc / (1.0 + jacc + 1e-8)) * (SA + SB)
    delta = SA + SB - 2.0 * inter
    return delta.clamp_min(0.0)
