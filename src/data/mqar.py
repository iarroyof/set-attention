from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100
DEFAULT_LAG_BINS: tuple[tuple[str, int, int], ...] = (
    ("lag_1_32", 1, 32),
    ("lag_33_128", 33, 128),
    ("lag_129_512", 129, 512),
    ("lag_513_1024", 513, 1024),
    ("lag_1025_2047", 1025, 2047),
)


@dataclass(frozen=True)
class MQARConfig:
    vocab_size: int = 8192
    num_examples: int = 100_000
    seq_len: int = 512
    num_kv_pairs: int = 64
    seed: int = 0
    power_a: float = 0.01
    random_non_queries: bool = True
    split: str = "train"


@dataclass(frozen=True)
class MQARGenerated:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_positions: torch.Tensor
    key_positions: torch.Tensor
    lags: torch.Tensor
    query_keys: torch.Tensor
    query_values: torch.Tensor
    config: MQARConfig


def split_seed(base_seed: int, split: str) -> int:
    offsets = {
        "train": 100_000,
        "calibration": 200_000,
        "validation": 200_000,
        "test": 300_000,
    }
    if split not in offsets:
        raise ValueError(f"unknown MQAR split {split!r}; expected one of {sorted(offsets)}")
    return int(base_seed) + offsets[split]


def _validate_config(cfg: MQARConfig) -> None:
    if cfg.seq_len % 2 != 0:
        raise ValueError("MQAR seq_len must be even")
    if cfg.vocab_size <= cfg.seq_len:
        raise ValueError("MQAR vocab_size must be greater than seq_len")
    if cfg.vocab_size < 4:
        raise ValueError("MQAR vocab_size must be at least 4")
    if cfg.num_kv_pairs <= 0:
        raise ValueError("MQAR num_kv_pairs must be positive")
    if cfg.num_kv_pairs * 4 > cfg.seq_len:
        raise ValueError("MQAR requires num_kv_pairs * 4 <= seq_len")
    if cfg.num_examples <= 0:
        raise ValueError("MQAR num_examples must be positive")
    if cfg.power_a <= 0.0:
        raise ValueError("MQAR power_a must be positive")


def power_law_gap_probs(space: int, power_a: float = 0.01) -> np.ndarray:
    if space <= 0:
        raise ValueError("gap sampling space must be positive")
    values = np.arange(1, int(space) + 1, dtype=np.float64)
    probs = float(power_a) * values ** (float(power_a) - 1.0)
    return probs / probs.sum()


def generate_mqar(config: MQARConfig | Mapping[str, Any]) -> MQARGenerated:
    cfg = config if isinstance(config, MQARConfig) else MQARConfig(**dict(config))
    _validate_config(cfg)

    rng = np.random.default_rng(int(cfg.seed))
    vocab_size = int(cfg.vocab_size)
    num_examples = int(cfg.num_examples)
    num_kv_pairs = int(cfg.num_kv_pairs)
    seq_len = int(cfg.seq_len)
    context_size = num_kv_pairs * 2

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size, dtype=np.int64)
    value_choices = np.arange(key_vocab_size, vocab_size, dtype=np.int64)
    if num_kv_pairs > len(key_choices) or num_kv_pairs > len(value_choices):
        raise ValueError("MQAR num_kv_pairs exceeds available unique key/value tokens")

    keys = np.stack(
        [rng.choice(key_choices, size=num_kv_pairs, replace=False) for _ in range(num_examples)],
        axis=0,
    )
    values = np.stack(
        [rng.choice(value_choices, size=num_kv_pairs, replace=False) for _ in range(num_examples)],
        axis=0,
    )

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    space = (seq_len - context_size) // 2
    probs = power_law_gap_probs(space, cfg.power_a)
    gap_choices = np.arange(space, dtype=np.int64)
    gaps = np.stack(
        [rng.choice(gap_choices, size=num_kv_pairs, replace=False, p=probs) for _ in range(num_examples)],
        axis=0,
    )

    queries = np.zeros((num_examples, seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    full_labels = np.full((num_examples, seq_len + 1), IGNORE_INDEX, dtype=np.int64)
    label_positions_unshifted = gaps * 2 + context_size + 1
    np.put_along_axis(full_labels, label_positions_unshifted, values=values, axis=1)

    input_ids_np = examples[:, :-1].copy()
    labels_np = full_labels[:, 1:].copy()
    if cfg.random_non_queries:
        zero_mask = input_ids_np == 0
        distractors = rng.integers(0, vocab_size, size=input_ids_np.shape, dtype=np.int64)
        input_ids_np[zero_mask] = distractors[zero_mask]

    query_positions = gaps * 2 + context_size
    key_positions = np.broadcast_to(
        np.arange(0, context_size, 2, dtype=np.int64),
        (num_examples, num_kv_pairs),
    ).copy()
    lags = query_positions - key_positions

    return MQARGenerated(
        input_ids=torch.from_numpy(input_ids_np).long(),
        labels=torch.from_numpy(labels_np).long(),
        query_positions=torch.from_numpy(query_positions).long(),
        key_positions=torch.from_numpy(key_positions).long(),
        lags=torch.from_numpy(lags).long(),
        query_keys=torch.from_numpy(keys).long(),
        query_values=torch.from_numpy(values).long(),
        config=cfg,
    )


def mqar_fingerprint(generated: MQARGenerated) -> str:
    hasher = hashlib.sha256()
    for tensor in (
        generated.input_ids,
        generated.labels,
        generated.query_positions,
        generated.key_positions,
        generated.lags,
    ):
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    hasher.update(json.dumps(asdict(generated.config), sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def _int_sequence_digest(values: list[int] | tuple[int, ...]) -> str:
    payload = json.dumps([int(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mqar_provenance(train: "MQARDataset", validation: "MQARDataset") -> dict[str, Any]:
    train_digest = train.dataset_digest
    validation_digest = validation.dataset_digest
    tokenizer_payload = {
        "type": "synthetic_disjoint_halves",
        "vocab_size": train.config.vocab_size,
        "key_range": [1, train.config.vocab_size // 2 - 1],
        "value_range": [train.config.vocab_size // 2, train.config.vocab_size - 1],
        "ignore_index": IGNORE_INDEX,
    }
    tokenizer_digest = hashlib.sha256(
        json.dumps(tokenizer_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dataset_digest = hashlib.sha256(
        json.dumps(
            {
                "train": train_digest,
                "validation": validation_digest,
                "generator": "zoology_multiquery_ar_compatible_v1",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "dataset": "mqar",
        "dataset_digest": dataset_digest,
        "tokenizer": "mqar_synthetic_disjoint_halves_v1",
        "tokenizer_digest": tokenizer_digest,
        "train": train.provenance(),
        "validation": validation.provenance(),
    }


class MQARDataset(Dataset):
    def __init__(self, config: MQARConfig | Mapping[str, Any]) -> None:
        self.generated = generate_mqar(config)
        self.config = self.generated.config
        self.dataset_digest = mqar_fingerprint(self.generated)

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
            "key_positions": self.generated.key_positions[idx],
            "lags": self.generated.lags[idx],
            "query_keys": self.generated.query_keys[idx],
            "query_values": self.generated.query_values[idx],
        }

    def provenance(self) -> dict[str, Any]:
        record_offsets = [idx * int(self.config.seq_len) for idx in range(int(self.config.num_examples) + 1)]
        query_offsets = self.generated.query_positions.reshape(-1).tolist()
        return {
            "split": self.config.split,
            "seed": int(self.config.seed),
            "num_examples": int(self.config.num_examples),
            "seq_len": int(self.config.seq_len),
            "num_kv_pairs": int(self.config.num_kv_pairs),
            "power_a": float(self.config.power_a),
            "dataset_digest": self.dataset_digest,
            "token_count": int(self.config.num_examples * self.config.seq_len),
            "record_offsets_digest": _int_sequence_digest(record_offsets),
            "sample_offsets_digest": _int_sequence_digest(query_offsets),
            "query_count": int(self.config.num_examples * self.config.num_kv_pairs),
        }


def build_mqar_datasets(data_cfg: Mapping[str, Any]) -> tuple[MQARDataset, MQARDataset]:
    base_seed = int(data_cfg.get("dataset_seed", data_cfg.get("seed", 0)))
    common = {
        "vocab_size": int(data_cfg.get("vocab_size", 8192)),
        "seq_len": int(data_cfg["seq_len"]),
        "num_kv_pairs": int(data_cfg["num_kv_pairs"]),
        "power_a": float(data_cfg.get("power_a", 0.01)),
        "random_non_queries": bool(data_cfg.get("random_non_queries", True)),
    }
    train = MQARDataset(
        MQARConfig(
            **common,
            num_examples=int(data_cfg.get("num_train_examples", data_cfg.get("limit", 100_000))),
            seed=int(data_cfg.get("train_seed", split_seed(base_seed, "train"))),
            split="train",
        )
    )
    validation = MQARDataset(
        MQARConfig(
            **common,
            num_examples=int(data_cfg.get("num_val_examples", data_cfg.get("val_limit", 3_000))),
            seed=int(data_cfg.get("validation_seed", split_seed(base_seed, "validation"))),
            split="validation",
        )
    )
    if train.config.seed == validation.config.seed:
        raise ValueError("MQAR train and validation applied seeds must be disjoint")
    return train, validation
