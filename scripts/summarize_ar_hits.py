#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REGISTERED_ROWS = ("token", "b0", "b25", "b100")
REGISTERED_SEEDS = (0, 1, 2)
COUNT_BINS = ("count_0", "count_1", "count_2_5", "count_6_20", "count_gt20")
LAG_BINS = ("lag_1_32", "lag_33_128", "lag_129_512", "lag_513_1024", "lag_1025_plus")
NONFINITE_RE = re.compile(r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", re.IGNORECASE)


class ARHitSummarizerError(ValueError):
    pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "NA", "None"}:
        raise ARHitSummarizerError(f"missing required numeric field {key}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ARHitSummarizerError(f"non-finite value in {key}: {value!r}")
    return parsed


def _int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value in {"", "NA", "None"}:
        raise ARHitSummarizerError(f"missing required integer field {key}")
    parsed = int(float(value))
    if str(parsed) != str(value).strip() and not str(value).strip().endswith(".0"):
        raise ARHitSummarizerError(f"malformed integer field {key}: {value!r}")
    return parsed


def validate_row(row: dict[str, str], *, require_registered_matrix: bool = True) -> dict[str, Any]:
    blob = json.dumps(row, sort_keys=True)
    if NONFINITE_RE.search(blob):
        raise ARHitSummarizerError("row contains word-boundary NaN/Inf")
    if row.get("task") != "natural_ar_hits" or row.get("dataset") != "wikitext2":
        raise ARHitSummarizerError("row is not a natural AR-hit metric row")
    if row.get("model.backend") != "exact":
        raise ARHitSummarizerError("wrong backend; expected exact")
    for key in (
        "checkpoint_sha256",
        "config_fingerprint",
        "model_config_digest",
        "data.dataset_digest",
        "data.tokenizer_digest",
        "val/overall/nll",
        "val/overall/ppl",
        "val/overall/targets",
        "val/ar/nll",
        "val/ar/targets",
        "val/non_ar/nll",
        "val/non_ar/targets",
    ):
        if row.get(key, "NA") in {"", "NA", "None"}:
            raise ARHitSummarizerError(f"metadata incomplete: {key}")
    label = row.get("row", "")
    if require_registered_matrix and label not in REGISTERED_ROWS:
        raise ARHitSummarizerError(f"unknown registered row {label!r}")
    seed = _int(row, "seed")
    if require_registered_matrix and seed not in REGISTERED_SEEDS:
        raise ARHitSummarizerError(f"malformed seed {seed}; expected one of {REGISTERED_SEEDS}")
    out = {
        "row": label,
        "seed": seed,
        "overall_nll": _float(row, "val/overall/nll"),
        "ar_nll": _float(row, "val/ar/nll"),
        "non_ar_nll": _float(row, "val/non_ar/nll"),
        "overall_targets": _int(row, "val/overall/targets"),
        "ar_targets": _int(row, "val/ar/targets"),
        "non_ar_targets": _int(row, "val/non_ar/targets"),
        "ar_fraction": _float(row, "val/ar/target_fraction"),
    }
    for bin_name in COUNT_BINS:
        count = _int(row, f"val/{bin_name}/targets")
        out[f"{bin_name}_targets"] = count
        out[f"{bin_name}_inferential"] = row.get(f"val/{bin_name}/inferential") == "True"
        if count > 0:
            out[f"{bin_name}_nll"] = _float(row, f"val/{bin_name}/nll")
    for bin_name in LAG_BINS:
        count = _int(row, f"val/{bin_name}/targets")
        out[f"{bin_name}_targets"] = count
        out[f"{bin_name}_inferential"] = row.get(f"val/{bin_name}/inferential") == "True"
        if count > 0:
            out[f"{bin_name}_nll"] = _float(row, f"val/{bin_name}/nll")
    return out


def summarize(rows: list[dict[str, Any]], *, require_registered_matrix: bool = True) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["row"]].append(row)
    if require_registered_matrix:
        missing = []
        for row_name in REGISTERED_ROWS:
            seeds = sorted(row["seed"] for row in grouped.get(row_name, []))
            if seeds != list(REGISTERED_SEEDS):
                missing.append(f"{row_name}: seeds {seeds}, expected {list(REGISTERED_SEEDS)}")
        if missing:
            raise ARHitSummarizerError("incomplete registered matrix: " + "; ".join(missing))
    out = []
    for row_name, group_rows in sorted(grouped.items()):
        n = len(group_rows)
        ar_nlls = [r["ar_nll"] for r in group_rows]
        non_ar_nlls = [r["non_ar_nll"] for r in group_rows]
        out.append(
            {
                "row": row_name,
                "n": n,
                "overall_nll_mean": mean(r["overall_nll"] for r in group_rows),
                "ar_nll_mean": mean(ar_nlls),
                "ar_nll_sd": stdev(ar_nlls) if n > 1 else 0.0,
                "non_ar_nll_mean": mean(non_ar_nlls),
                "non_ar_nll_sd": stdev(non_ar_nlls) if n > 1 else 0.0,
                "ar_targets_total": sum(int(r["ar_targets"]) for r in group_rows),
                "non_ar_targets_total": sum(int(r["non_ar_targets"]) for r in group_rows),
                "has_inferential_ar_bin": any(
                    bool(r.get(f"{name}_inferential", False))
                    for r in group_rows
                    for name in COUNT_BINS
                    if name != "count_0"
                ),
            }
        )
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize natural AR-hit rows")
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out/ar_hits/summary.tsv"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for path in args.csv:
        rows.extend(_read_csv(path))
    require_registered = not args.allow_incomplete
    validated = [validate_row(row, require_registered_matrix=require_registered) for row in rows]
    summary = summarize(validated, require_registered_matrix=require_registered)
    write_tsv(args.out, summary)
    print(f"validated_rows={len(validated)} summary={args.out}")


if __name__ == "__main__":
    main()
