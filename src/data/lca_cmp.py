from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from data.mqar import IGNORE_INDEX


@dataclass(frozen=True)
class LCACmpConfig:
    vocab_size: int = 8192
    num_examples: int = 100_000
    seq_len: int = 2048
    seed: int = 0
    marker_fraction: float = 0.08
    count_jitter: int = 16
    random_noise: bool = True
    split: str = "train"
    supervision: str = "endpoint"
    oracle_count_token: bool = False


@dataclass(frozen=True)
class LCACmpGenerated:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_positions: torch.Tensor
    lags: torch.Tensor
    marker_counts: torch.Tensor
    targets: torch.Tensor
    config: LCACmpConfig


def split_seed(base_seed: int, split: str) -> int:
    offsets = {
        "train": 400_000,
        "calibration": 500_000,
        "validation": 500_000,
        "test": 600_000,
    }
    if split not in offsets:
        raise ValueError(f"unknown LCA split {split!r}; expected one of {sorted(offsets)}")
    return int(base_seed) + offsets[split]


def _special_tokens(vocab_size: int) -> dict[str, int]:
    return {
        "marker": int(vocab_size) - 4,
        "query": int(vocab_size) - 3,
        "answer_false": int(vocab_size) - 2,
        "answer_true": int(vocab_size) - 1,
    }


def special_tokens(vocab_size: int) -> dict[str, int]:
    return _special_tokens(vocab_size)


def _validate_config(cfg: LCACmpConfig) -> None:
    if cfg.vocab_size < 16:
        raise ValueError("LCA vocab_size must be at least 16")
    if cfg.seq_len < 16:
        raise ValueError("LCA seq_len must be at least 16")
    if cfg.num_examples <= 0:
        raise ValueError("LCA num_examples must be positive")
    if not (0.0 < cfg.marker_fraction < 0.5):
        raise ValueError("LCA marker_fraction must be in (0, 0.5)")
    if cfg.count_jitter <= 0:
        raise ValueError("LCA count_jitter must be positive")
    if cfg.supervision not in {"endpoint", "prefix"}:
        raise ValueError("LCA supervision must be 'endpoint' or 'prefix'")
    if cfg.oracle_count_token and oracle_count_base(
        cfg.vocab_size, cfg.seq_len, cfg.marker_fraction, cfg.count_jitter
    ) <= 1:
        raise ValueError("LCA vocab_size too small to reserve the oracle count-token range")


def oracle_count_base(
    vocab_size: int,
    seq_len: int,
    marker_fraction: float = 0.08,
    count_jitter: int = 16,
) -> int:
    """First token id of the reserved oracle count range.

    The generator draws the marker count from a window of
    ``span = high_max - low_min + 1`` realizable values (the low/high jitter
    ranges around the bucket threshold). The oracle encodes the true count as
    the dedicated id ``oracle_count_base + (count - low_min)``; the span-wide
    range sits directly below the four special tokens and disjoint from the
    noise-token range. The endpoint bucket is recoverable from the oracle id
    alone (``offset >= threshold - low_min``).
    """
    context_len = int(seq_len) - 1
    threshold = max(1, min(context_len - 1, round(context_len * float(marker_fraction))))
    jitter = max(1, int(count_jitter))
    low_min = max(0, threshold - jitter)
    high_max = min(context_len, threshold + jitter)
    span = high_max - low_min + 1
    return int(vocab_size) - 4 - span


def generate_lca_cmp(config: LCACmpConfig | Mapping[str, Any]) -> LCACmpGenerated:
    cfg = config if isinstance(config, LCACmpConfig) else LCACmpConfig(**dict(config))
    _validate_config(cfg)

    rng = np.random.default_rng(int(cfg.seed))
    vocab_size = int(cfg.vocab_size)
    seq_len = int(cfg.seq_len)
    num_examples = int(cfg.num_examples)
    context_len = seq_len - 1
    threshold = max(1, min(context_len - 1, round(context_len * float(cfg.marker_fraction))))
    jitter = max(1, int(cfg.count_jitter))
    low_min = max(0, threshold - jitter)
    low_max = max(low_min, threshold - 1)
    high_min = threshold
    high_max = min(context_len, threshold + jitter)
    if high_min > high_max:
        raise ValueError("LCA threshold/count_jitter leaves no positive count range")

    specials = _special_tokens(vocab_size)
    oracle_base = (
        oracle_count_base(vocab_size, seq_len, cfg.marker_fraction, cfg.count_jitter)
        if cfg.oracle_count_token
        else None
    )
    noise_high = oracle_base if oracle_base is not None else vocab_size - 4
    if cfg.random_noise:
        examples = rng.integers(1, noise_high, size=(num_examples, seq_len), dtype=np.int64)
    else:
        examples = np.zeros((num_examples, seq_len), dtype=np.int64)
    examples[:, -1] = specials["query"]

    # With the oracle enabled, position seq_len-2 is reserved for the count
    # token, so markers are placed only in the first context_len-1 positions.
    marker_span = context_len - 1 if oracle_base is not None else context_len
    high_max = min(high_max, marker_span)
    if high_min > high_max:
        raise ValueError("LCA oracle marker span leaves no positive count range")

    labels_np = np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64)
    targets = rng.integers(0, 2, size=num_examples, dtype=np.int64)
    marker_counts = np.zeros(num_examples, dtype=np.int64)
    for idx, target in enumerate(targets.tolist()):
        if target:
            count = int(rng.integers(high_min, high_max + 1))
        else:
            count = int(rng.integers(low_min, low_max + 1)) if low_max >= low_min else 0
        marker_counts[idx] = count
        if count > 0:
            positions = rng.choice(marker_span, size=count, replace=False)
            examples[idx, positions] = specials["marker"]
        labels_np[idx, -1] = specials["answer_true"] if target else specials["answer_false"]

    if cfg.supervision == "prefix":
        # Every position t predicts the bucket of the marker count inside the
        # prefix [0..t]. The per-position threshold is the endpoint threshold
        # scaled by the covered context fraction, so the class balance and the
        # count_jitter margin match the endpoint task at every horizon; at the
        # final position this reduces exactly to the endpoint label.
        marker_mask = examples == specials["marker"]
        prefix_counts = np.cumsum(marker_mask, axis=1)
        steps = np.arange(1, seq_len + 1, dtype=np.float64)
        thresholds = np.maximum(1, np.rint(threshold * steps / context_len)).astype(np.int64)
        labels_np = np.where(
            prefix_counts >= thresholds[None, :],
            specials["answer_true"],
            specials["answer_false"],
        ).astype(np.int64)

    if oracle_base is not None:
        examples[:, -2] = oracle_base + (marker_counts - low_min)

    query_positions = np.full((num_examples, 1), seq_len - 1, dtype=np.int64)
    lags = np.full((num_examples, 1), context_len, dtype=np.int64)

    return LCACmpGenerated(
        input_ids=torch.from_numpy(examples).long(),
        labels=torch.from_numpy(labels_np).long(),
        query_positions=torch.from_numpy(query_positions).long(),
        lags=torch.from_numpy(lags).long(),
        marker_counts=torch.from_numpy(marker_counts).long(),
        targets=torch.from_numpy(targets).long(),
        config=cfg,
    )


def lca_cmp_fingerprint(generated: LCACmpGenerated) -> str:
    hasher = hashlib.sha256()
    for tensor in (
        generated.input_ids,
        generated.labels,
        generated.query_positions,
        generated.lags,
        generated.marker_counts,
        generated.targets,
    ):
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    hasher.update(json.dumps(asdict(generated.config), sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def _int_sequence_digest(values: list[int] | tuple[int, ...]) -> str:
    payload = json.dumps([int(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LCACmpDataset(Dataset):
    def __init__(self, config: LCACmpConfig | Mapping[str, Any]) -> None:
        self.generated = generate_lca_cmp(config)
        self.config = self.generated.config
        self.dataset_digest = lca_cmp_fingerprint(self.generated)

    def __len__(self) -> int:
        return int(self.generated.input_ids.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.config.vocab_size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.generated.input_ids[idx],
            "labels": self.generated.labels[idx],
            "query_positions": self.generated.query_positions[idx],
            "lags": self.generated.lags[idx],
            "marker_counts": self.generated.marker_counts[idx],
            "targets": self.generated.targets[idx],
        }

    def provenance(self) -> dict[str, Any]:
        record_offsets = [idx * int(self.config.seq_len) for idx in range(int(self.config.num_examples) + 1)]
        return {
            "split": self.config.split,
            "seed": int(self.config.seed),
            "num_examples": int(self.config.num_examples),
            "seq_len": int(self.config.seq_len),
            "marker_fraction": float(self.config.marker_fraction),
            "count_jitter": int(self.config.count_jitter),
            "supervision": str(self.config.supervision),
            "oracle_count_token": bool(self.config.oracle_count_token),
            "dataset_digest": self.dataset_digest,
            "token_count": int(self.config.num_examples * self.config.seq_len),
            "record_offsets_digest": _int_sequence_digest(record_offsets),
            "query_count": int(self.config.num_examples),
        }


def lca_cmp_provenance(train: LCACmpDataset, validation: LCACmpDataset) -> dict[str, Any]:
    specials = _special_tokens(train.config.vocab_size)
    tokenizer_payload = {
        "type": "synthetic_count_threshold",
        "vocab_size": train.config.vocab_size,
        "noise_range": [1, train.config.vocab_size - 5],
        "specials": specials,
        "ignore_index": IGNORE_INDEX,
    }
    tokenizer_digest = hashlib.sha256(
        json.dumps(tokenizer_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dataset_digest = hashlib.sha256(
        json.dumps(
            {
                "train": train.dataset_digest,
                "validation": validation.dataset_digest,
                "generator": "lca_cmp_threshold_count_v1",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "dataset": "lca_cmp",
        "dataset_digest": dataset_digest,
        "tokenizer": "lca_cmp_threshold_count_v1",
        "tokenizer_digest": tokenizer_digest,
        "train": train.provenance(),
        "validation": validation.provenance(),
    }


def build_lca_cmp_datasets(data_cfg: Mapping[str, Any]) -> tuple[LCACmpDataset, LCACmpDataset]:
    base_seed = int(data_cfg.get("dataset_seed", data_cfg.get("seed", 0)))
    common = {
        "vocab_size": int(data_cfg.get("vocab_size", 8192)),
        "seq_len": int(data_cfg["seq_len"]),
        "marker_fraction": float(data_cfg.get("marker_fraction", 0.08)),
        "count_jitter": int(data_cfg.get("count_jitter", 16)),
        "random_noise": bool(data_cfg.get("random_noise", True)),
        "supervision": str(data_cfg.get("supervision", "endpoint")),
        "oracle_count_token": bool(data_cfg.get("oracle_count_token", False)),
    }
    train = LCACmpDataset(
        LCACmpConfig(
            **common,
            num_examples=int(data_cfg.get("num_train_examples", data_cfg.get("limit", 100_000))),
            seed=int(data_cfg.get("train_seed", split_seed(base_seed, "train"))),
            split="train",
        )
    )
    validation = LCACmpDataset(
        LCACmpConfig(
            **common,
            num_examples=int(data_cfg.get("num_val_examples", data_cfg.get("val_limit", 3_000))),
            seed=int(data_cfg.get("validation_seed", split_seed(base_seed, "validation"))),
            split="validation",
        )
    )
    if train.config.seed == validation.config.seed:
        raise ValueError("LCA train and validation applied seeds must be disjoint")
    return train, validation
