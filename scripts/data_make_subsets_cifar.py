#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List

from set_attention.data import resolve_data_root


def _load_cifar10_len(root: Path) -> int:
    try:
        import torchvision
        from torchvision import transforms as T
    except Exception as exc:
        raise ImportError("torchvision is required for CIFAR-10 subsets.") from exc
    transform = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
    return len(ds)


def _sample_indices(total: int, pct: float, seed: int) -> List[int]:
    count = max(1, int(round(total * pct / 100.0)))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic percentage subsets for CIFAR-10 train split.")
    parser.add_argument("--pct", nargs="+", type=float, required=True, help="Percentages to sample (e.g., 10 25 50).")
    parser.add_argument("--output-dir", type=str, default="subsets")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data-root", type=str, default="", help="Root data directory (defaults to set-attention cache).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    vision_root = data_root / "vision" / "cifar10"
    vision_root.mkdir(parents=True, exist_ok=True)
    total = _load_cifar10_len(vision_root)
    if total <= 0:
        raise RuntimeError("CIFAR-10 train split is empty.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw in args.pct:
        pct = float(raw)
        if pct <= 0 or pct >= 100:
            raise ValueError(f"--pct must be in (0,100); got {pct}")
        indices = _sample_indices(total, pct, args.seed)
        label = f"{int(round(pct))}pct"
        out_path = out_dir / f"cifar10_train_{label}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[subset] exists; skipping {out_path}")
            continue
        payload = {
            "dataset": "cifar10",
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
