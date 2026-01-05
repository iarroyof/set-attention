"""
Lightweight helpers to estimate VRAM use and mark benchmark runs as skipped/oom.
The goal is to avoid obvious OOMs in scaling sweeps without changing core logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

PRECISION_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}


def bytes_from_gb(gb: float) -> int:
    return int(float(gb) * (1024**3))


def vram_budget_bytes(gpu_vram_gb: float, safety: float = 0.92) -> int:
    return int(bytes_from_gb(gpu_vram_gb) * float(safety))


def _dtype_bytes(precision: str) -> int:
    return PRECISION_BYTES.get(precision, 4)


def estimate_dense_attn_bytes(batch: int, seq_len: int, d_model: int, nhead: int, precision: str, backward: bool = True) -> int:
    # dominant term is attention score matrix
    bytes_per = _dtype_bytes(precision)
    scores = batch * nhead * seq_len * seq_len * bytes_per
    # activations + softmax + V matmul
    factor = 6.0 if backward else 3.0
    return int(scores * factor + batch * seq_len * d_model * bytes_per * 2)


def estimate_ska_bytes(batch: int, seq_len: int, window: int, stride: int, nhead: int, precision: str, backward: bool = True) -> int:
    # crude upper bound: sets per seq ~ seq_len/stride, atoms per set ~ window
    bytes_per = _dtype_bytes(precision)
    sets = max(1, seq_len // max(1, stride))
    # emulate set-set interactions
    scores = batch * nhead * sets * sets * bytes_per
    factor = 4.0 if backward else 2.0
    return int(scores * factor + batch * sets * window * bytes_per * 2)


@dataclass
class SkipDecision:
    skip: bool
    reason: str = ""


def should_skip_dense(batch: int, seq_len: int, d_model: int, nhead: int, precision: str, gpu_vram_gb: float) -> SkipDecision:
    if gpu_vram_gb <= 0:
        return SkipDecision(False, "")
    need = estimate_dense_attn_bytes(batch, seq_len, d_model, nhead, precision, backward=True)
    budget = vram_budget_bytes(gpu_vram_gb)
    if need > budget:
        return SkipDecision(True, f"est_dense_bytes={need/1e9:.2f}GB>budget={budget/1e9:.2f}GB")
    return SkipDecision(False, "")


def should_skip_ska(batch: int, seq_len: int, window: int, stride: int, nhead: int, precision: str, gpu_vram_gb: float) -> SkipDecision:
    if gpu_vram_gb <= 0:
        return SkipDecision(False, "")
    need = estimate_ska_bytes(batch, seq_len, window, stride, nhead, precision, backward=True)
    budget = vram_budget_bytes(gpu_vram_gb)
    if need > budget:
        return SkipDecision(True, f"est_ska_bytes={need/1e9:.2f}GB>budget={budget/1e9:.2f}GB")
    return SkipDecision(False, "")
