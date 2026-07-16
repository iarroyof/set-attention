from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # torch is optional for pure metadata tests and syntax checks.
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


COUNT_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("count_0", 0, 0),
    ("count_1", 1, 1),
    ("count_2_5", 2, 5),
    ("count_6_20", 6, 20),
    ("count_gt20", 21, None),
)

LAG_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("lag_1_32", 1, 32),
    ("lag_33_128", 33, 128),
    ("lag_129_512", 129, 512),
    ("lag_513_1024", 513, 1024),
    ("lag_1025_plus", 1025, None),
)

MIN_INFERENTIAL_TARGETS = 1000
IGNORE_INDEX = -100


@dataclass(frozen=True)
class ARHitMetadata:
    is_ar: bool
    target_token: int
    most_recent_lag: int | None
    earliest_lag: int | None
    train_bigram_count: int
    count_bin: str
    lag_bin: str
    chunk_start: bool
    crosses_record_boundary: bool
    context_has_record_boundary: bool


def count_bin_name(count: int) -> str:
    for name, lo, hi in COUNT_BINS:
        if count >= lo and (hi is None or count <= hi):
            return name
    raise ValueError(f"negative bigram count: {count}")


def lag_bin_name(lag: int | None) -> str:
    if lag is None:
        return "non_ar"
    for name, lo, hi in LAG_BINS:
        if lag >= lo and (hi is None or lag <= hi):
            return name
    raise ValueError(f"invalid lag: {lag}")


def _record_boundary_set(record_offsets: Iterable[int] | None) -> set[int]:
    if record_offsets is None:
        return set()
    return {int(offset) for offset in record_offsets if int(offset) > 0}


def _crosses_boundary(global_pos: int, record_boundaries: set[int]) -> bool:
    return int(global_pos) + 1 in record_boundaries


def _context_has_boundary(start_offset: int, target_pos: int, record_boundaries: set[int]) -> bool:
    if not record_boundaries:
        return False
    lo = int(start_offset) + 1
    hi = int(start_offset) + int(target_pos) + 1
    return any(lo <= boundary <= hi for boundary in record_boundaries)


def ar_hit_metadata_for_sequence(
    input_ids: Sequence[int],
    labels: Sequence[int],
    *,
    train_bigram_counts: Mapping[tuple[int, int], int] | None = None,
    sample_start_offset: int = 0,
    record_offsets: Iterable[int] | None = None,
    reset_at_record_boundaries: bool = True,
    ignore_index: int = IGNORE_INDEX,
) -> list[ARHitMetadata]:
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have the same length")
    counts = train_bigram_counts or {}
    record_boundaries = _record_boundary_set(record_offsets)
    prior_positions: dict[tuple[int, int], list[int]] = defaultdict(list)
    out: list[ARHitMetadata] = []
    for t, (current_token, target_token) in enumerate(zip(input_ids, labels)):
        global_pos = int(sample_start_offset) + t
        target = int(target_token)
        if target == int(ignore_index):
            out.append(
                ARHitMetadata(
                    is_ar=False,
                    target_token=target,
                    most_recent_lag=None,
                    earliest_lag=None,
                    train_bigram_count=0,
                    count_bin="ignored",
                    lag_bin="ignored",
                    chunk_start=t == 0,
                    crosses_record_boundary=False,
                    context_has_record_boundary=_context_has_boundary(
                        sample_start_offset,
                        t,
                        record_boundaries,
                    ),
                )
            )
            continue
        boundary_cross = _crosses_boundary(global_pos, record_boundaries)
        bigram = (int(current_token), target)
        positions = prior_positions.get(bigram, [])
        is_ar = (not boundary_cross) and len(positions) > 0
        most_recent_lag = (t - positions[-1]) if is_ar else None
        earliest_lag = (t - positions[0]) if is_ar else None
        train_count = int(counts.get(bigram, 0))
        out.append(
            ARHitMetadata(
                is_ar=is_ar,
                target_token=target,
                most_recent_lag=most_recent_lag,
                earliest_lag=earliest_lag,
                train_bigram_count=train_count,
                count_bin=count_bin_name(train_count),
                lag_bin=lag_bin_name(most_recent_lag),
                chunk_start=t == 0,
                crosses_record_boundary=boundary_cross,
                context_has_record_boundary=_context_has_boundary(
                    sample_start_offset,
                    t,
                    record_boundaries,
                ),
            )
        )
        if reset_at_record_boundaries and boundary_cross:
            prior_positions.clear()
        else:
            prior_positions[bigram].append(t)
    return out


def build_bigram_counts_from_samples(
    samples: Iterable[tuple[Any, Any]],
    *,
    record_offsets: Iterable[int] | None = None,
    sample_offsets: Sequence[int] | None = None,
) -> Counter[tuple[int, int]]:
    record_boundaries = _record_boundary_set(record_offsets)
    counts: Counter[tuple[int, int]] = Counter()
    for sample_index, (input_ids, labels) in enumerate(samples):
        xs = [int(v) for v in _to_list(input_ids)]
        ys = [int(v) for v in _to_list(labels)]
        if len(xs) != len(ys):
            raise ValueError("sample input_ids and labels lengths differ")
        start = int(sample_offsets[sample_index]) if sample_offsets is not None and sample_index < len(sample_offsets) else 0
        for t, (x, y) in enumerate(zip(xs, ys)):
            if y == IGNORE_INDEX:
                continue
            if _crosses_boundary(start + t, record_boundaries):
                continue
            counts[(x, y)] += 1
    return counts


def build_bigram_counts_from_dataset(dataset: Any) -> Counter[tuple[int, int]]:
    provenance = getattr(dataset, "provenance", None)
    record_offsets = getattr(provenance, "record_offsets", None)
    sample_offsets = getattr(provenance, "sample_offsets", None)
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return build_bigram_counts_from_samples(
            samples,
            record_offsets=record_offsets,
            sample_offsets=sample_offsets,
        )
    return build_bigram_counts_from_samples(dataset, record_offsets=record_offsets)


def _to_list(value: Any) -> list[int]:
    if torch is not None and torch.is_tensor(value):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in value]


def empty_metric_row(name: str) -> dict[str, Any]:
    return {
        f"{name}/nll": None,
        f"{name}/ppl": None,
        f"{name}/targets": 0,
        f"{name}/inferential": False,
    }


class ARHitAccumulator:
    def __init__(self, *, min_inferential_targets: int = MIN_INFERENTIAL_TARGETS) -> None:
        self.min_inferential_targets = int(min_inferential_targets)
        self.loss_sums: Counter[str] = Counter()
        self.counts: Counter[str] = Counter()

    def add(self, key: str, nll: float) -> None:
        if not math.isfinite(float(nll)):
            raise ValueError(f"non-finite NLL for {key}: {nll!r}")
        self.loss_sums[str(key)] += float(nll)
        self.counts[str(key)] += 1

    def metrics(self) -> dict[str, Any]:
        keys = ["overall", "ar", "non_ar"]
        keys.extend(name for name, _, _ in COUNT_BINS)
        keys.extend(name for name, _, _ in LAG_BINS)
        out: dict[str, Any] = {}
        for key in keys:
            n = int(self.counts.get(key, 0))
            if n == 0:
                out.update(empty_metric_row(key))
                continue
            nll = float(self.loss_sums[key]) / n
            out[f"{key}/nll"] = nll
            out[f"{key}/ppl"] = math.exp(nll)
            out[f"{key}/targets"] = n
            out[f"{key}/inferential"] = n >= self.min_inferential_targets
        ar_n = int(self.counts.get("ar", 0))
        total_n = int(self.counts.get("overall", 0))
        out["ar/target_fraction"] = ar_n / total_n if total_n else 0.0
        return out


def _batch_sample_offsets(batch_start: int, batch_size: int, dataset: Any) -> list[int]:
    provenance = getattr(dataset, "provenance", None)
    sample_offsets = getattr(provenance, "sample_offsets", None)
    if sample_offsets is None:
        return [0] * batch_size
    return [int(sample_offsets[batch_start + i]) for i in range(batch_size)]


def _record_offsets(dataset: Any) -> tuple[int, ...] | None:
    provenance = getattr(dataset, "provenance", None)
    offsets = getattr(provenance, "record_offsets", None)
    if offsets is None:
        return None
    return tuple(int(v) for v in offsets)


def evaluate_ar_hits(
    model: Any,
    dataloader: Any,
    device: Any,
    *,
    train_bigram_counts: Mapping[tuple[int, int], int],
    min_inferential_targets: int = MIN_INFERENTIAL_TARGETS,
) -> dict[str, Any]:
    if torch is None or F is None:
        raise RuntimeError("evaluate_ar_hits requires torch")
    model.eval()
    accumulator = ARHitAccumulator(min_inferential_targets=min_inferential_targets)
    dataset = getattr(dataloader, "dataset", None)
    records = _record_offsets(dataset)
    seen_samples = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            per_token_nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(labels)
            starts = _batch_sample_offsets(seen_samples, int(input_ids.shape[0]), dataset)
            for b in range(int(input_ids.shape[0])):
                metadata = ar_hit_metadata_for_sequence(
                    _to_list(input_ids[b]),
                    _to_list(labels[b]),
                    train_bigram_counts=train_bigram_counts,
                    sample_start_offset=starts[b],
                    record_offsets=records,
                )
                for t, meta in enumerate(metadata):
                    if meta.count_bin == "ignored":
                        continue
                    nll = float(per_token_nll[b, t].detach().cpu().item())
                    accumulator.add("overall", nll)
                    accumulator.add("ar" if meta.is_ar else "non_ar", nll)
                    accumulator.add(meta.count_bin, nll)
                    if meta.is_ar:
                        accumulator.add(meta.lag_bin, nll)
            seen_samples += int(input_ids.shape[0])
    return accumulator.metrics()


def evaluate_ar_hit_group_ablation(
    model: Any,
    dataloader: Any,
    device: Any,
    base_metrics: Mapping[str, Any],
    *,
    train_bigram_counts: Mapping[tuple[int, int], int],
    min_inferential_targets: int = MIN_INFERENTIAL_TARGETS,
) -> dict[str, Any]:
    if not hasattr(model, "set_span_ablation_mode"):
        return {"ablation/status": "missing_set_span_ablation_mode_hook"}
    groups = [str(m["name"]) for m in getattr(model, "multiresolution_group_metadata", [])]
    wanted = [name for name in ("fine", "coarse") if name in groups]
    if not wanted:
        return {"ablation/status": "missing_fine_coarse_groups"}
    previous = str(getattr(model, "span_ablation_mode", "none"))
    out: dict[str, Any] = {"ablation/status": "ok"}
    try:
        for group in wanted:
            model.set_span_ablation_mode(group)
            metrics = evaluate_ar_hits(
                model,
                dataloader,
                device,
                train_bigram_counts=train_bigram_counts,
                min_inferential_targets=min_inferential_targets,
            )
            for key, value in metrics.items():
                out[f"ablation/{group}/{key}"] = value
            for bucket in ("overall", "ar", "non_ar"):
                base = base_metrics.get(f"{bucket}/nll")
                ablated = metrics.get(f"{bucket}/nll")
                if base is not None and ablated is not None:
                    out[f"ablation/{group}/{bucket}/delta_nll"] = float(ablated) - float(base)
                    out[f"ablation/{group}/{bucket}/delta_ppl"] = float(metrics[f"{bucket}/ppl"]) - float(base_metrics[f"{bucket}/ppl"])
    finally:
        model.set_span_ablation_mode(previous)
    return out


def write_metrics_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_metrics_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in columns})


__all__ = [
    "ARHitAccumulator",
    "ARHitMetadata",
    "COUNT_BINS",
    "LAG_BINS",
    "MIN_INFERENTIAL_TARGETS",
    "ar_hit_metadata_for_sequence",
    "build_bigram_counts_from_dataset",
    "build_bigram_counts_from_samples",
    "count_bin_name",
    "evaluate_ar_hit_group_ablation",
    "evaluate_ar_hits",
    "lag_bin_name",
    "write_metrics_csv",
    "write_metrics_json",
]
