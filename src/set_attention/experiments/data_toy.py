from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


@dataclass
class ToySeqConfig:
    n_samples: int = 1000
    seq_len: int = 32
    vocab_size: int = 64
    motif: Tuple[int, int, int] = (7, 11, 13)
    batch_size: int = 64
    val_frac: float = 0.2
    seed: int = 1337


def make_toy_sequence_classification(cfg: ToySeqConfig):
    g = torch.Generator().manual_seed(cfg.seed)
    X = torch.randint(0, cfg.vocab_size, (cfg.n_samples, cfg.seq_len), generator=g)
    y = torch.zeros(cfg.n_samples, dtype=torch.long)
    # positive class if motif tokens appear anywhere (set-membership based)
    motif_set = set(cfg.motif)
    for i in range(cfg.n_samples):
        tokens = set(X[i].tolist())
        if motif_set.issubset(tokens):
            y[i] = 1
        else:
            y[i] = 0
    # rebalance by flipping labels randomly to approach balance if needed
    # (keep simple; synthetic randomness usually yields decent balance)
    ds = TensorDataset(X, y)
    n_val = int(cfg.val_frac * cfg.n_samples)
    n_train = cfg.n_samples - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


@dataclass
class ToyDiffConfig:
    n_samples: int = 1000
    seq_len: int = 16
    dim: int = 8
    n_modes: int = 4
    batch_size: int = 64
    val_frac: float = 0.2
    seed: int = 2024


def make_toy_continuous_sequences(cfg: ToyDiffConfig):
    g = torch.Generator().manual_seed(cfg.seed)
    # Mixture of modes in R^dim per position; sequences share mode assignment to induce structure
    means = torch.randn(cfg.n_modes, cfg.dim, generator=g) * 2.0
    cov = torch.eye(cfg.dim)

    X = torch.empty(cfg.n_samples, cfg.seq_len, cfg.dim)
    for i in range(cfg.n_samples):
        mode = torch.randint(0, cfg.n_modes, (1,), generator=g).item()
        noise = torch.randn(cfg.seq_len, cfg.dim, generator=g)
        X[i] = means[mode] + 0.3 * noise

    ds = TensorDataset(X)
    n_val = int(cfg.val_frac * cfg.n_samples)
    n_train = cfg.n_samples - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader

