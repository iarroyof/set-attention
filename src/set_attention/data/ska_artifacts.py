"""Serializable packs for SKA banks and routing signatures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from set_attention.sets.banked import BankedSetBatch
from set_attention.universe import SetFeatureCache, UniversePool


@dataclass
class BankPack:
    values: torch.LongTensor
    set_offsets: torch.LongTensor
    seq_offsets: torch.LongTensor
    universe_ids: torch.LongTensor

    def to_dict(self) -> dict:
        return {
            "values": self.values.cpu(),
            "set_offsets": self.set_offsets.cpu(),
            "seq_offsets": self.seq_offsets.cpu(),
            "universe_ids": self.universe_ids.cpu(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BankPack":
        return cls(
            values=payload["values"],
            set_offsets=payload["set_offsets"],
            seq_offsets=payload["seq_offsets"],
            universe_ids=payload["universe_ids"],
        )

    @classmethod
    def from_banked_batch(cls, bank: BankedSetBatch, universe_ids: torch.LongTensor) -> "BankPack":
        return cls(
            values=bank.values.detach().cpu(),
            set_offsets=bank.set_offsets.detach().cpu(),
            seq_offsets=bank.seq_offsets.detach().cpu(),
            universe_ids=universe_ids.detach().cpu(),
        )


@dataclass
class RoutingPack:
    sig_sets: torch.LongTensor
    size_sets: torch.LongTensor

    def to_dict(self) -> dict:
        return {
            "sig_sets": self.sig_sets.cpu(),
            "size_sets": self.size_sets.cpu(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RoutingPack":
        return cls(sig_sets=payload["sig_sets"], size_sets=payload["size_sets"])


def save_bank_pack(path: Path, pack: BankPack) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack.to_dict(), path)


def load_bank_pack(path: Path) -> BankPack:
    payload = torch.load(path, map_location="cpu")
    return BankPack.from_dict(payload)


def save_routing_pack(path: Path, pack: RoutingPack) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack.to_dict(), path)


def load_routing_pack(path: Path) -> RoutingPack:
    payload = torch.load(path, map_location="cpu")
    return RoutingPack.from_dict(payload)


def cache_from_packs(
    pool: UniversePool,
    bank: BankPack,
    routing: Optional[RoutingPack] = None,
) -> SetFeatureCache:
    cache = SetFeatureCache(pool, bank.values, bank.set_offsets, bank.seq_offsets, minhash=None)
    if routing is not None:
        if routing.size_sets is not None:
            cache.size_sets = routing.size_sets.to(cache.size_sets.device)
        if routing.sig_sets is not None:
            cache.sig_sets = routing.sig_sets.to(cache.values.device)
    return cache


__all__ = [
    "BankPack",
    "RoutingPack",
    "cache_from_packs",
    "load_bank_pack",
    "load_routing_pack",
    "save_bank_pack",
    "save_routing_pack",
]
