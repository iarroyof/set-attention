from __future__ import annotations
import math
from typing import Tuple
import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=pred.device)
    for t, p in zip(target.view(-1), pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


@torch.no_grad()
def ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """Expected Calibration Error for binary classification.
    probs: (N,) predicted probability of class 1
    targets: (N,) {0,1}
    """
    device = probs.device
    bins = torch.linspace(0, 1, steps=n_bins + 1, device=device)
    ece = torch.tensor(0.0, device=device)
    N = probs.numel()
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1]) if i < n_bins - 1 else (probs >= bins[i]) & (probs <= bins[i + 1])
        if mask.any():
            acc = (targets[mask] == (probs[mask] >= 0.5)).float().mean()
            conf = probs[mask].mean()
            ece += (mask.float().mean()) * (acc - conf).abs()
    return ece


@torch.no_grad()
def binary_roc_auc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    # Simple trapezoidal ROC-AUC (CPU-safe fallback)
    # Sort by decreasing score
    scores = probs.detach().cpu()
    y = targets.detach().cpu().long()
    order = torch.argsort(scores, descending=True)
    y = y[order]
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    tps = torch.cumsum((y == 1).int(), dim=0)
    fps = torch.cumsum((y == 0).int(), dim=0)
    tpr = tps.float() / P
    fpr = fps.float() / N
    # Integrate via trapezoid over unique fpr
    fpr_u, idx = torch.unique_consecutive(fpr, return_inverse=False, return_counts=False, dim=0), None
    # interpolate tpr at unique fpr using the last occurrence
    # (since we sorted scores, tpr monotonically increases)
    # build mapping fpr -> max tpr at that point
    tpr_u = torch.zeros_like(fpr_u)
    j = 0
    for i in range(fpr.shape[0]):
        if i == 0 or fpr[i] != fpr[i - 1]:
            tpr_u[j] = tpr[i]
            j += 1
        else:
            tpr_u[j - 1] = tpr[i]
    auc = torch.trapz(tpr_u, fpr_u).item()
    return float(auc)


@torch.no_grad()
def binary_pr_auc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    scores = probs.detach().cpu()
    y = targets.detach().cpu().long()
    order = torch.argsort(scores, descending=True)
    y = y[order]
    P = int((y == 1).sum())
    if P == 0:
        return float("nan")
    tps = torch.cumsum((y == 1).int(), dim=0).float()
    fps = torch.cumsum((y == 0).int(), dim=0).float()
    prec = tps / (tps + fps + 1e-8)
    rec = tps / P
    # enforce starting at rec=0, prec=1
    rec = torch.cat([torch.tensor([0.0]), rec])
    prec = torch.cat([torch.tensor([1.0]), prec])
    auc = torch.trapz(prec, rec).item()
    return float(auc)


@torch.no_grad()
def chamfer_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    # a: (N, L, D), b: (M, L, D); compare first min(N,M)
    n = min(a.shape[0], b.shape[0])
    A = a[:n].reshape(n, -1)
    B = b[:n].reshape(n, -1)
    d = torch.cdist(A, B, p=2)
    return float(d.min(dim=1).values.mean() + d.min(dim=0).values.mean())


@torch.no_grad()
def one_nn_two_sample(x: torch.Tensor, y: torch.Tensor) -> float:
    # 1-NN two-sample test accuracy (should be ~0.5 if indistinguishable)
    X = torch.cat([x, y], dim=0)
    device = X.device
    labels = torch.cat([
        torch.zeros(x.size(0), device=device),
        torch.ones(y.size(0), device=device)
    ]).long()
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    nn_idx = torch.argmin(D, dim=1)
    nn_label = labels[nn_idx]
    acc = (nn_label == labels).float().mean().item()
    return float(acc)
