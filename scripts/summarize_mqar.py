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


ROWS = ("token", "b0", "b25", "b50", "b75", "b100")
SEEDS = (0, 1, 2)
LAG_BINS = ("lag_1_32", "lag_33_128", "lag_129_512", "lag_513_1024", "lag_1025_2047")
NONFINITE_RE = re.compile(r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", re.IGNORECASE)


class MQARSummarizerError(ValueError):
    pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "NA", "None"}:
        raise MQARSummarizerError(f"missing required numeric field {key}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise MQARSummarizerError(f"non-finite value in {key}: {value!r}")
    return parsed


def _int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value in {"", "NA", "None"}:
        raise MQARSummarizerError(f"missing required integer field {key}")
    parsed = int(float(value))
    if str(parsed) != str(value).strip() and not str(value).strip().endswith(".0"):
        raise MQARSummarizerError(f"malformed integer field {key}: {value!r}")
    return parsed


def row_label(row: dict[str, str]) -> str:
    impl = row.get("model.implementation")
    if impl == "baseline_token":
        return "token"
    groups = row.get("model.multiresolution.groups", "")
    if "fine" not in groups and "coarse" not in groups:
        raise MQARSummarizerError("set row missing multiresolution group metadata")
    fine = re.search(r"'name': 'fine'.*?'num_heads': ([0-9]+)", groups)
    coarse = re.search(r"'name': 'coarse'.*?'num_heads': ([0-9]+)", groups)
    fine_heads = int(fine.group(1)) if fine else 0
    coarse_heads = int(coarse.group(1)) if coarse else 0
    mapping = {(8, 0): "b0", (6, 2): "b25", (4, 4): "b50", (2, 6): "b75", (0, 8): "b100"}
    try:
        return mapping[(fine_heads, coarse_heads)]
    except KeyError as exc:
        raise MQARSummarizerError(f"unregistered fine/coarse head split {(fine_heads, coarse_heads)}") from exc


def validate_row(row: dict[str, str], *, allow_incomplete: bool = False) -> dict[str, Any]:
    blob = json.dumps(row, sort_keys=True)
    if NONFINITE_RE.search(blob):
        raise MQARSummarizerError("row contains word-boundary NaN/Inf")
    if row.get("task") != "mqar" or row.get("dataset") != "mqar":
        raise MQARSummarizerError("row is not an MQAR metric row")
    stage = row.get("stage", "")
    if "smoke" in stage.lower() or "limited" in stage.lower():
        raise MQARSummarizerError("smoke/limited rows are rejected")
    if row.get("model.backend") != "exact":
        raise MQARSummarizerError("wrong backend; expected exact")
    if _int(row, "data.seq_len") != 2048:
        raise MQARSummarizerError("wrong seq_len; expected 2048")
    if _int(row, "data.num_kv_pairs") != 256:
        raise MQARSummarizerError("wrong num_kv_pairs; expected 256")
    if _int(row, "data.num_train_examples") < 100000:
        raise MQARSummarizerError("limited train example count")
    if _int(row, "data.num_val_examples") < 3000:
        raise MQARSummarizerError("limited validation example count")
    if _int(row, "training.max_updates") <= 1:
        raise MQARSummarizerError("preflight/smoke update count is not summarizable")
    seed = _int(row, "training.seed")
    if seed not in SEEDS:
        raise MQARSummarizerError(f"malformed seed {seed}; expected one of {SEEDS}")
    label = row.get("data.mqar_row") or row.get("mqar.row") or row_label(row)
    if label not in ROWS:
        raise MQARSummarizerError(f"unknown registered row {label!r}")
    for key in (
        "config_fingerprint",
        "data.dataset_digest",
        "data.tokenizer_digest",
        "val/loss",
        "val/accuracy",
        "val/valid_tokens",
        "val/exact_sequence_accuracy",
    ):
        if row.get(key, "NA") in {"", "NA", "None"}:
            raise MQARSummarizerError(f"metadata incomplete: {key}")
    for lag_bin in LAG_BINS:
        count_key = f"val/lag/{lag_bin}_query_count"
        if row.get(count_key, "NA") in {"", "NA", "None"}:
            raise MQARSummarizerError(f"missing lag-bin count {count_key}")
        count = _int(row, count_key)
        if count > 0:
            _float(row, f"val/lag/{lag_bin}_accuracy")
    return {
        "row": label,
        "seed": seed,
        "loss": _float(row, "val/loss"),
        "accuracy": _float(row, "val/accuracy"),
        "exact_sequence_accuracy": _float(row, "val/exact_sequence_accuracy"),
        "query_count": _int(row, "val/valid_tokens"),
    }


def summarize(validated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        grouped[row["row"]].append(row)
    missing = []
    for row_name in ROWS:
        seeds = sorted(row["seed"] for row in grouped.get(row_name, []))
        if seeds != list(SEEDS):
            missing.append(f"{row_name}: seeds {seeds}, expected {list(SEEDS)}")
    if missing:
        raise MQARSummarizerError("incomplete registered matrix: " + "; ".join(missing))
    out = []
    for row_name in ROWS:
        rows = grouped[row_name]
        acc = [r["accuracy"] for r in rows]
        loss = [r["loss"] for r in rows]
        exact = [r["exact_sequence_accuracy"] for r in rows]
        out.append(
            {
                "row": row_name,
                "n": len(rows),
                "accuracy_mean": mean(acc),
                "accuracy_sd": stdev(acc),
                "loss_mean": mean(loss),
                "loss_sd": stdev(loss),
                "exact_sequence_accuracy_mean": mean(exact),
                "query_count_total": sum(int(r["query_count"]) for r in rows),
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
    parser = argparse.ArgumentParser(description="Validate and summarize registered MRP-3 MQAR rows")
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out/mqar/summary.tsv"))
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for path in args.csv:
        rows.extend(_read_csv(path))
    validated = [validate_row(row) for row in rows]
    summary = summarize(validated)
    write_tsv(args.out, summary)
    print(f"validated_rows={len(validated)} summary={args.out}")


if __name__ == "__main__":
    main()
