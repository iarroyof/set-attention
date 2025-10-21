from __future__ import annotations
import torch


def _sizes_from_offsets(offsets: torch.LongTensor) -> torch.Tensor:
    return (offsets[1:] - offsets[:-1]).to(torch.int64)


def symdiff_exact(idsA, offsA, idsB, offsB):
    """Exact |A Δ B| for all pairs using two-pointer intersection.
    Assumes sorted unique ids within each set.
    Returns: (nA, nB) int32 tensor.
    """
    nA, nB = offsA.numel() - 1, offsB.numel() - 1
    out = torch.empty(nA, nB, dtype=torch.int32, device=idsA.device)
    for i in range(nA):
        A = idsA[offsA[i] : offsA[i + 1]]
        aN = A.numel()
        for j in range(nB):
            B = idsB[offsB[j] : offsB[j + 1]]
            bN = B.numel()
            ia = ib = inter = 0
            while ia < aN and ib < bN:
                av = int(A[ia].item())
                bv = int(B[ib].item())
                if av == bv:
                    inter += 1
                    ia += 1
                    ib += 1
                elif av < bv:
                    ia += 1
                else:
                    ib += 1
            out[i, j] = aN + bN - 2 * inter
    return out


def delta_rbf_exact(idsA, offsA, idsB, offsB, gamma: float) -> torch.Tensor:
    D = symdiff_exact(idsA, offsA, idsB, offsB).to(torch.float32)
    return torch.exp(-float(gamma) * D)

