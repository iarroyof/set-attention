from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


TOKENIZER_NAME = "whitespace_first_seen_v1"
RECORD_POLICY = "nonempty_source_record_v1"


def _stable_digest(payload: object) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def vocabulary_tokens(stoi: dict[str, int]) -> list[str]:
    if not stoi:
        raise ValueError("vocabulary must not be empty")
    ordered: list[str | None] = [None] * len(stoi)
    for token, index in stoi.items():
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("vocabulary indices must be integers")
        if index < 0 or index >= len(ordered) or ordered[index] is not None:
            raise ValueError("vocabulary indices must be unique and contiguous")
        ordered[index] = str(token)
    if any(token is None for token in ordered):
        raise ValueError("vocabulary indices must be contiguous")
    return [str(token) for token in ordered]


def vocabulary_digest(tokens: Sequence[str]) -> str:
    return _stable_digest(
        {
            "tokenizer": TOKENIZER_NAME,
            "tokens": list(tokens),
        }
    )


@dataclass(frozen=True)
class OrderedTextProvenance:
    dataset: str
    split: str
    tokenizer: str
    record_policy: str
    token_count: int
    record_offsets: tuple[int, ...]
    sample_offsets: tuple[int, ...]
    ordered_token_digest: str
    record_offsets_digest: str
    sample_offsets_digest: str
    vocabulary_digest: str
    dataset_digest: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["record_offsets"] = list(self.record_offsets)
        payload["sample_offsets"] = list(self.sample_offsets)
        return payload


class OrderedTextProvenanceBuilder:
    def __init__(self) -> None:
        self._ordered_hasher = hashlib.sha256()
        self._token_count = 0
        self._record_offsets = [0]

    @staticmethod
    def _update_token(hasher: "hashlib._Hash", token: str) -> None:
        encoded = token.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)

    def add_record(self, tokens: Iterable[str]) -> None:
        added = 0
        for token in tokens:
            self._update_token(self._ordered_hasher, str(token))
            self._token_count += 1
            added += 1
        if added:
            self._record_offsets.append(self._token_count)

    def finalize(
        self,
        *,
        dataset: str,
        split: str,
        seq_len: int,
        vocabulary_sha256: str,
    ) -> OrderedTextProvenance:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        sample_offsets = tuple(
            range(0, max(0, self._token_count - seq_len), seq_len)
        )
        record_offsets = tuple(self._record_offsets)
        ordered_digest = self._ordered_hasher.hexdigest()
        record_digest = _stable_digest(record_offsets)
        sample_digest = _stable_digest(sample_offsets)
        identity = {
            "dataset": dataset,
            "split": split,
            "tokenizer": TOKENIZER_NAME,
            "record_policy": RECORD_POLICY,
            "token_count": self._token_count,
            "ordered_token_digest": ordered_digest,
            "record_offsets_digest": record_digest,
            "sample_offsets_digest": sample_digest,
            "vocabulary_digest": vocabulary_sha256,
        }
        return OrderedTextProvenance(
            dataset=dataset,
            split=split,
            tokenizer=TOKENIZER_NAME,
            record_policy=RECORD_POLICY,
            token_count=self._token_count,
            record_offsets=record_offsets,
            sample_offsets=sample_offsets,
            ordered_token_digest=ordered_digest,
            record_offsets_digest=record_digest,
            sample_offsets_digest=sample_digest,
            vocabulary_digest=vocabulary_sha256,
            dataset_digest=_stable_digest(identity),
        )


def build_ordered_text_provenance(
    *,
    dataset: str,
    split: str,
    records: Iterable[Iterable[str]],
    seq_len: int,
    vocabulary_sha256: str,
) -> OrderedTextProvenance:
    builder = OrderedTextProvenanceBuilder()
    for record in records:
        builder.add_record(record)
    return builder.finalize(
        dataset=dataset,
        split=split,
        seq_len=seq_len,
        vocabulary_sha256=vocabulary_sha256,
    )


def combined_dataset_digest(
    train: OrderedTextProvenance,
    validation: OrderedTextProvenance,
) -> str:
    return _stable_digest(
        {
            "train": train.dataset_digest,
            "validation": validation.dataset_digest,
            "vocabulary": train.vocabulary_digest,
            "tokenizer": train.tokenizer,
        }
    )


def dataset_provenance_bundle(train_dataset, validation_dataset) -> dict[str, object]:
    train = getattr(train_dataset, "provenance", None)
    validation = getattr(validation_dataset, "provenance", None)
    vocab = getattr(train_dataset, "vocabulary_tokens", None)
    if not isinstance(train, OrderedTextProvenance):
        raise ValueError("training dataset does not expose ordered provenance")
    if not isinstance(validation, OrderedTextProvenance):
        raise ValueError("validation dataset does not expose ordered provenance")
    if not isinstance(vocab, list) or not all(isinstance(token, str) for token in vocab):
        raise ValueError("training dataset does not expose an ordered vocabulary")
    if train.vocabulary_digest != validation.vocabulary_digest:
        raise ValueError("training and validation vocabulary digests differ")
    return {
        "dataset": train.dataset,
        "dataset_digest": combined_dataset_digest(train, validation),
        "tokenizer": train.tokenizer,
        "tokenizer_digest": train.vocabulary_digest,
        "vocabulary": list(vocab),
        "train": train.to_dict(),
        "validation": validation.to_dict(),
    }


__all__ = [
    "OrderedTextProvenance",
    "OrderedTextProvenanceBuilder",
    "build_ordered_text_provenance",
    "combined_dataset_digest",
    "dataset_provenance_bundle",
    "vocabulary_digest",
    "vocabulary_tokens",
]
