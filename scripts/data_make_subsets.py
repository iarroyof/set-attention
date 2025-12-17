#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from set_attention.data.wikitext import _WIKITEXT_CONFIGS, load_wikitext_hf_dataset


def _load_documents(dataset: str, split: str, cache_dir: Path) -> List[str]:
    ds = load_wikitext_hf_dataset(dataset, cache_dir)
    if split not in ds:
        raise ValueError(f"Split '{split}' not available for dataset '{dataset}'.")
    docs = []
    for item in ds[split]:
        text = item.get("text", "")
        if not text:
            continue
        line = text.strip()
        if line:
            docs.append(line)
    if not docs:
        raise RuntimeError(f"No documents found for {dataset}:{split}")
    return docs


def _token_lengths(docs: List[str]) -> Tuple[List[int], int]:
    lengths = [len(doc.split()) for doc in docs]
    total_tokens = sum(lengths)
    if total_tokens == 0:
        raise RuntimeError("No tokens found in corpus.")
    return lengths, total_tokens


def _bucket_indices(lengths: List[int]) -> Dict[int, List[int]]:
    arr = np.array(lengths, dtype=np.int64)
    quantiles = np.quantile(arr, np.linspace(0, 1, 11))
    buckets: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, length in enumerate(lengths):
        bucket = int(np.searchsorted(quantiles, length, side="right")) - 1
        bucket = min(max(bucket, 0), 9)
        buckets[bucket].append(idx)
    return buckets


def _assign_targets(lengths: List[int], buckets: Dict[int, List[int]], budget_tokens: int) -> Dict[int, int]:
    total_tokens = sum(lengths[idx] for idx in range(len(lengths)))
    bucket_tokens = {
        b: sum(lengths[idx] for idx in indices) for b, indices in buckets.items()
    }
    targets: Dict[int, int] = {}
    remaining = budget_tokens
    for b in range(10):
        frac = bucket_tokens[b] / total_tokens if total_tokens > 0 else 0.0
        tokens = min(bucket_tokens[b], int(round(budget_tokens * frac)))
        targets[b] = tokens
        remaining -= tokens
    # distribute remainder greedily
    b = 0
    while remaining > 0 and b < 10:
        spare = bucket_tokens[b] - targets[b]
        if spare > 0:
            add = min(spare, remaining)
            targets[b] += add
            remaining -= add
        b += 1
    return targets


def _sample_subset(
    lengths: List[int],
    buckets: Dict[int, List[int]],
    budget_tokens: int,
    seed: int,
) -> Tuple[List[int], int]:
    rng = random.Random(seed)
    bucket_targets = _assign_targets(lengths, buckets, budget_tokens)
    selected: List[int] = []
    used_tokens = 0
    for b in range(10):
        indices = buckets[b][:]
        rng.shuffle(indices)
        target = bucket_targets.get(b, 0)
        for idx in indices:
            if target <= 0 or used_tokens >= budget_tokens:
                break
            selected.append(idx)
            used_tokens += lengths[idx]
            target -= lengths[idx]
        bucket_targets[b] = target
        if used_tokens >= budget_tokens:
            break
    if used_tokens < budget_tokens:
        all_indices = list(range(len(lengths)))
        rng.shuffle(all_indices)
        for idx in all_indices:
            if idx in selected:
                continue
            selected.append(idx)
            used_tokens += lengths[idx]
            if used_tokens >= budget_tokens:
                break
    return selected, used_tokens


def _sequence_stats(lengths: List[int], window: int, stride: int) -> float:
    if window <= 0 or stride <= 0:
        return 0.0
    counts = []
    for L in lengths:
        if L < window:
            counts.append(0)
        else:
            steps = 1 + max(0, math.floor((L - window) / stride))
            counts.append(steps)
    return float(sum(counts) / max(1, len(counts)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build token-budget subsets for Wikitext datasets.")
    parser.add_argument("--dataset", choices=list(_WIKITEXT_CONFIGS.keys()), required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--budgets",
        nargs="+",
        required=True,
        help="Token budgets per subset. Values <=1 are treated as fractions; >1 as absolute token counts.",
    )
    parser.add_argument("--cache-dir", type=str, default="~/.cache/set-attention/hf_datasets")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    docs = _load_documents(args.dataset, args.split, cache_dir)
    lengths, total_tokens = _token_lengths(docs)
    buckets = _bucket_indices(lengths)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw_budget in args.budgets:
        budget_value = float(raw_budget)
        token_budget = (
            int(total_tokens * budget_value) if 0 < budget_value <= 1.0 else int(budget_value)
        )
        token_budget = max(1, min(token_budget, total_tokens))
        subset_indices, actual_tokens = _sample_subset(lengths, buckets, token_budget, args.seed)
        subset_lengths = [lengths[idx] for idx in subset_indices]
        avg_seq_sets = _sequence_stats(subset_lengths, args.window, args.stride)
        frac = actual_tokens / total_tokens if total_tokens > 0 else 0.0
        label = (
            f"{int(budget_value*100):02d}pct"
            if 0 < budget_value <= 1.0
            else f"{token_budget}_tokens"
        )
        payload = {
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "total_tokens": total_tokens,
            "target_tokens": token_budget,
            "actual_tokens": actual_tokens,
            "actual_fraction": frac,
            "num_documents": len(subset_indices),
            "avg_tokens_per_doc": sum(subset_lengths) / max(1, len(subset_lengths)),
            "avg_sets_per_seq_estimate": avg_seq_sets,
            "indices": subset_indices,
        }
        out_path = out_dir / f"{args.dataset}_{args.split}_{label}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(
            f"[subset] {label}: docs={len(subset_indices)} tokens={actual_tokens} "
            f"({frac*100:.2f}% of corpus) avg_sets≈{avg_seq_sets:.2f}"
        )


if __name__ == "__main__":
    main()
