from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


@dataclass
class UniverseStats:
    size: int
    has_phi: bool
    has_minhash: bool
    device: torch.device
    metadata: Dict[str, Any]


class UniversePool:
    """Container for the Active-Universe atom features and signatures.

    Stores the canonical atom ids (``U_ids``), optional atom features (``phi_bank``),
    and optional MinHash signatures (``mh_sigs``). Utility methods cover device
    transfers, lookups, and serialization. The pool assumes that ids are unique.
    """

    def __init__(
        self,
        U_ids: torch.LongTensor,
        phi_bank: Optional[torch.Tensor] = None,
        mh_sigs: Optional[torch.LongTensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if U_ids.dim() != 1:
            raise ValueError("U_ids must be a 1-D tensor of atom ids.")
        self.U_ids = U_ids.clone().long()
        self.metadata: Dict[str, Any] = metadata.copy() if metadata else {}

        if phi_bank is not None:
            if phi_bank.shape[0] != self.U_ids.numel():
                raise ValueError("phi_bank must have the same number of rows as U_ids.")
            self.phi_bank = phi_bank.clone()
        else:
            self.phi_bank = None

        if mh_sigs is not None:
            if mh_sigs.shape[0] != self.U_ids.numel():
                raise ValueError("mh_sigs must have the same number of rows as U_ids.")
            self.mh_sigs = mh_sigs.clone().long()
        else:
            self.mh_sigs = None

        self._build_lookup()

    def _build_lookup(self) -> None:
        # Build searchsorted helpers for fast id -> position lookup.
        sorted_ids, perm = torch.sort(self.U_ids)
        if sorted_ids.unique_consecutive().numel() != sorted_ids.numel():
            raise ValueError("U_ids contain duplicate values.")
        self._sorted_ids = sorted_ids
        self._sorted_to_orig = perm
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=torch.long)
        self._orig_to_sorted = inv_perm

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.U_ids.numel())

    @property
    def device(self) -> torch.device:
        return self.U_ids.device

    @property
    def dim(self) -> Optional[int]:
        return int(self.phi_bank.shape[1]) if self.phi_bank is not None else None

    def stats(self) -> UniverseStats:
        return UniverseStats(
            size=len(self),
            has_phi=self.phi_bank is not None,
            has_minhash=self.mh_sigs is not None,
            device=self.device,
            metadata=self.metadata,
        )

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "UniversePool":
        device = torch.device(device)
        self.U_ids = self.U_ids.to(device, non_blocking=non_blocking)
        self._sorted_ids = self._sorted_ids.to(device, non_blocking=non_blocking)
        self._sorted_to_orig = self._sorted_to_orig.to(device, non_blocking=non_blocking)
        self._orig_to_sorted = self._orig_to_sorted.to(device, non_blocking=non_blocking)
        if self.phi_bank is not None:
            self.phi_bank = self.phi_bank.to(device, non_blocking=non_blocking)
        if self.mh_sigs is not None:
            self.mh_sigs = self.mh_sigs.to(device, non_blocking=non_blocking)
        return self

    def cpu(self) -> "UniversePool":  # pragma: no cover - simple wrapper
        return self.to(torch.device("cpu"))

    def clone(self) -> "UniversePool":
        return UniversePool(
            self.U_ids.clone(),
            phi_bank=self.phi_bank.clone() if self.phi_bank is not None else None,
            mh_sigs=self.mh_sigs.clone() if self.mh_sigs is not None else None,
            metadata=self.metadata.copy(),
        )

    def lookup_positions(self, ids: torch.LongTensor, *, check: bool = True) -> torch.LongTensor:
        """Map atom ids to row positions in the pool.

        Args:
            ids: Tensor of atom ids to locate.
            check: verify that all ids exist, raising KeyError if a miss occurs.
        """
        ids = ids.to(self._sorted_ids.device)
        idx_sorted = torch.searchsorted(self._sorted_ids, ids)
        if check:
            valid = (idx_sorted >= 0) & (idx_sorted < self._sorted_ids.numel())
            if not torch.all(valid):
                missing = ids[~valid].unique()
                raise KeyError(f"Some ids are outside the known universe: {missing.tolist()}")
            matched = self._sorted_ids.index_select(0, idx_sorted.clamp_max(self._sorted_ids.numel() - 1))
            if not torch.equal(matched, ids):
                missing = ids[matched != ids].unique()
                raise KeyError(f"Ids not found in universe: {missing.tolist()}")
        return self._sorted_to_orig.index_select(0, idx_sorted)

    def get_phi(self, ids: torch.LongTensor) -> torch.Tensor:
        if self.phi_bank is None:
            raise RuntimeError("phi_bank is not available in this UniversePool.")
        positions = self.lookup_positions(ids)
        return self.phi_bank.index_select(0, positions)

    def get_minhash(self, ids: torch.LongTensor) -> torch.LongTensor:
        if self.mh_sigs is None:
            raise RuntimeError("mh_sigs is not available in this UniversePool.")
        positions = self.lookup_positions(ids)
        return self.mh_sigs.index_select(0, positions)

    def update_phi(self, phi_bank: torch.Tensor) -> None:
        if phi_bank.shape[0] != self.U_ids.numel():
            raise ValueError("New phi_bank must have matching number of rows.")
        self.phi_bank = phi_bank.clone()

    def update_minhash(self, mh_sigs: torch.LongTensor) -> None:
        if mh_sigs.shape[0] != self.U_ids.numel():
            raise ValueError("New mh_sigs must have matching number of rows.")
        self.mh_sigs = mh_sigs.clone().long()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        blob = {
            "U_ids": self.U_ids.cpu(),
            "phi_bank": self.phi_bank.cpu() if self.phi_bank is not None else None,
            "mh_sigs": self.mh_sigs.cpu() if self.mh_sigs is not None else None,
            "metadata": self.metadata,
        }
        torch.save(blob, path)

    @classmethod
    def load(cls, path: str | Path, map_location: Optional[str | torch.device] = None) -> "UniversePool":
        path = Path(path)
        blob = torch.load(path, map_location=map_location)
        return cls(
            blob["U_ids"],
            phi_bank=blob.get("phi_bank"),
            mh_sigs=blob.get("mh_sigs"),
            metadata=blob.get("metadata"),
        )

    def log_summary(self, prefix: str = "[UniversePool]") -> str:
        stats = self.stats()
        meta_keys = ",".join(sorted(stats.metadata.keys()))
        return (
            f"{prefix} size={stats.size} device={stats.device} "
            f"phi={'yes' if stats.has_phi else 'no'} "
            f"minhash={'yes' if stats.has_minhash else 'no'} "
            f"metadata={{{meta_keys}}}"
        )

    @classmethod
    def from_ids(
        cls,
        ids: Iterable[int],
        *,
        phi_bank: Optional[torch.Tensor] = None,
        mh_sigs: Optional[torch.LongTensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device | str] = None,
    ) -> "UniversePool":
        ids_tensor = torch.tensor(list(ids), dtype=torch.long, device=device or "cpu")
        return cls(ids_tensor, phi_bank=phi_bank, mh_sigs=mh_sigs, metadata=metadata)

