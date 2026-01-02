import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

FORCE_STR_COLUMNS = {
    "seed",
    "rep",
    "run_uid",
    "device",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "git_sha",
}

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


def detect_numeric_columns(rows: List[Dict[str, str]]) -> Tuple[set, List[str]]:
    if not rows:
        return set(), []
    header = list(rows[0].keys())
    numeric_cols = set()
    for col in header:
        if col in FORCE_STR_COLUMNS:
            continue
        values = [row[col] for row in rows if row.get(col) not in ("", None)]
        if not values:
            continue
        is_numeric = True
        for v in values:
            try:
                float(v)
            except (ValueError, TypeError):
                is_numeric = False
                break
        if is_numeric:
            numeric_cols.add(col)
    return numeric_cols, header


def group_rows(rows: List[Dict[str, str]], numeric_cols: set, group_cols: List[str]):
    groups = defaultdict(list)
    for row in rows:
        key_tuple = tuple((col, row.get(col, "")) for col in group_cols)
        metrics = {}
        for col in numeric_cols:
            val = row.get(col, "")
            if val in ("", None):
                continue
            try:
                metrics[col] = float(val)
            except (ValueError, TypeError):
                continue
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


def write_summary(groups, output_path: Path, group_cols: List[str], numeric_order: List[str]):
    rows = []
    metric_names = set()
    for key_tuple, metrics_list in groups.items():
        key_dict = dict(key_tuple)
        metrics = aggregate_metrics(metrics_list)
        metric_names.update(metrics.keys())
        rows.append((key_dict, metrics))
    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["note"])
            writer.writeheader()
            writer.writerow({"note": "no_data"})
        return
    key_cols = [col for col in group_cols if col]
    metric_cols = [col for col in numeric_order if col in metric_names]
    fieldnames = key_cols + [f"{m}_mean" for m in metric_cols] + [f"{m}_std" for m in metric_cols] + [
        f"{m}_n" for m in metric_cols
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key_dict, metrics in rows:
            row = {col: key_dict.get(col, "") for col in key_cols}
            for metric_name, stats in metrics.items():
                row[f"{metric_name}_mean"] = stats["mean"]
                row[f"{metric_name}_std"] = stats["std"]
                row[f"{metric_name}_n"] = stats["n"]
            writer.writerow(row)


def main():
    args = parse_args()
    input_paths = []
    if args.input:
        for pattern in args.input:
            # Expand glob patterns even when passed in quotes
            input_paths.extend(Path().glob(pattern))
    if not input_paths:
        input_paths = list(Path("out/benchmarks").glob("*.csv"))
    rows = load_rows(input_paths)
    if not rows:
        write_summary({}, Path(args.output), [], [])
        print("No rows found; nothing to aggregate.")
        return
    numeric_cols, header = detect_numeric_columns(rows)
    numeric_cols -= FORCE_STR_COLUMNS
    group_cols = [col for col in header if col not in numeric_cols and col not in FORCE_STR_COLUMNS]
    groups = group_rows(rows, numeric_cols, group_cols)
    write_summary(groups, Path(args.output), group_cols, [col for col in header if col in numeric_cols])
    print(f"Aggregated {len(rows)} rows into {len(groups)} groups -> {args.output}")


if __name__ == "__main__":
    main()
