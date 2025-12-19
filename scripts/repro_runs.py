import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate benchmark CSVs into summary stats.")
    ap.add_argument(
        "--input", nargs="+", default=[], help="Input benchmark CSV files (default: all under out/benchmarks)."
    )
    ap.add_argument(
        "--output",
        type=str,
        default="out/benchmarks/bench_summary.csv",
        help="Output summary CSV path.",
    )
    return ap.parse_args()


def load_rows(paths: List[Path]) -> List[Dict[str, str]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
    return rows


def split_columns(row: Dict[str, str]):
    metrics = {}
    keys = {}
    for k, v in row.items():
        if not v:
            keys[k] = ""
            continue
        val = v
        if not isinstance(val, (str, bytes)):
            keys[k] = str(val)
            continue
        try:
            metrics[k] = float(val)
        except ValueError:
            keys[k] = val
    return keys, metrics


def group_rows(rows: List[Dict[str, str]]):
    groups = defaultdict(list)
    for row in rows:
        keys, metrics = split_columns(row)
        key_tuple = tuple(sorted(keys.items()))
        groups[key_tuple].append(metrics)
    return groups


def aggregate_metrics(group):
    if not group:
        return {}
    metrics = {}
    for k in group[0].keys():
        values = [g[k] for g in group if k in g]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        metrics[k] = {"mean": mean, "std": var ** 0.5, "n": len(values)}
    return metrics


def write_summary(groups, output_path: Path):
    rows = []
    for key_tuple, metrics_list in groups.items():
        key_dict = dict(key_tuple)
        metrics = aggregate_metrics(metrics_list)
        row = dict(key_dict)
        for metric_name, stats in metrics.items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
            row[f"{metric_name}_n"] = stats["n"]
        rows.append(row)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    input_paths = [Path(p) for p in args.input]
    if not input_paths:
        input_paths = list(Path("out/benchmarks").glob("*.csv"))
    rows = load_rows(input_paths)
    if not rows:
        print("No rows found; nothing to aggregate.")
        return
    groups = group_rows(rows)
    write_summary(groups, Path(args.output))
    print(f"Aggregated {len(rows)} rows into {len(groups)} groups -> {args.output}")


if __name__ == "__main__":
    main()
