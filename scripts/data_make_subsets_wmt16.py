#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from typing import List

from set_attention.data.hf_cache import ensure_hf_cache


def _load_wmt16_train(cache_dir: Path):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise ImportError("HuggingFace 'datasets' package is required for WMT16 subsets.") from exc
    return load_dataset("wmt16", "ro-en", download_mode="reuse_dataset_if_exists", cache_dir=str(cache_dir))["train"]


def _sample_indices(total: int, pct: float, seed: int) -> List[int]:
    count = max(1, int(round(total * pct / 100.0)))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic percentage subsets for WMT16 En-Ro train split.")
    parser.add_argument("--pct", nargs="+", type=float, required=True, help="Percentages to sample (e.g., 10 25 50).")
    parser.add_argument("--output-dir", type=str, default="subsets")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="HF datasets cache root; empty uses HF_HOME/HF_DATASETS_CACHE.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = ensure_hf_cache(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_train = _load_wmt16_train(cache_dir)
    total = len(ds_train)
    if total <= 0:
        raise RuntimeError("WMT16 train split is empty.")

    for raw in args.pct:
        pct = float(raw)
        if pct <= 0 or pct >= 100:
            raise ValueError(f"--pct must be in (0,100); got {pct}")
        indices = _sample_indices(total, pct, args.seed)
        label = f"{int(round(pct))}pct"
        out_path = out_dir / f"wmt16_en_ro_train_{label}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[subset] exists; skipping {out_path}")
            continue
        payload = {
            "dataset": "wmt16_en_ro",
            "split": "train",
            "pct": pct,
            "seed": args.seed,
            "total": total,
            "count": len(indices),
            "indices": indices,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"[subset] wrote {out_path} (count={len(indices)}/{total})")


if __name__ == "__main__":
    main()
