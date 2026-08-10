#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REGISTERED_ROWS = ("token", "b0", "b25", "b100")
REGISTERED_SEEDS = (0, 1, 2)
GATE_STAR_ROW = "b25"
GATE_ENDPOINT_ROWS = ("b0", "b100")
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


def load_blocks(payloads: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Extract per-sequence NLL blocks keyed by (row, seed) from eval payloads."""
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for payload in payloads:
        row_meta = payload.get("row", {})
        label = str(row_meta.get("row", ""))
        seed = int(float(str(row_meta.get("seed", ""))))
        blocks = payload.get("blocks")
        if not blocks:
            raise ARHitSummarizerError(f"payload for {label} seed {seed} carries no per-sequence blocks")
        key = (label, seed)
        if key in out:
            raise ARHitSummarizerError(f"duplicate blocks for {label} seed {seed}")
        norm = []
        for block in blocks:
            norm.append(
                {
                    "ar_nll": float(block["ar_nll"]),
                    "ar_targets": int(block["ar_targets"]),
                    "non_ar_nll": float(block["non_ar_nll"]),
                    "non_ar_targets": int(block["non_ar_targets"]),
                }
            )
        out[key] = norm
    return out


def _count_weighted_nll(blocks: list[dict[str, Any]], idxs: list[int], prefix: str) -> float | None:
    nll = sum(blocks[i][f"{prefix}_nll"] for i in idxs)
    n = sum(blocks[i][f"{prefix}_targets"] for i in idxs)
    if n == 0:
        return None
    return nll / n


def _paired_seed_stats(
    blocks_by: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    star: str,
    endpoint: str,
    seeds: tuple[int, ...],
    draws: dict[int, list[int]],
) -> tuple[float, float] | None:
    """Mean-across-seeds paired AR difference and AR-vs-nonAR DiD for one draw set."""
    ar_diffs: list[float] = []
    dids: list[float] = []
    for seed in seeds:
        star_blocks = blocks_by[(star, seed)]
        end_blocks = blocks_by[(endpoint, seed)]
        idx = draws[seed]
        ar_s = _count_weighted_nll(star_blocks, idx, "ar")
        ar_e = _count_weighted_nll(end_blocks, idx, "ar")
        na_s = _count_weighted_nll(star_blocks, idx, "non_ar")
        na_e = _count_weighted_nll(end_blocks, idx, "non_ar")
        if None in (ar_s, ar_e, na_s, na_e):
            return None
        ar_diff = float(ar_s) - float(ar_e)
        ar_diffs.append(ar_diff)
        dids.append(ar_diff - (float(na_s) - float(na_e)))
    return mean(ar_diffs), mean(dids)


def _percentile_ci(samples: list[float], level: float = 0.95) -> tuple[float, float]:
    ordered = sorted(samples)
    n = len(ordered)
    tail = (1.0 - level) / 2.0
    lo = ordered[min(n - 1, max(0, int(tail * n)))]
    hi = ordered[min(n - 1, max(0, int((1.0 - tail) * n)))]
    return lo, hi


def paired_bootstrap_gate(
    blocks_by: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    star: str = GATE_STAR_ROW,
    endpoints: tuple[str, ...] = GATE_ENDPOINT_ROWS,
    seeds: tuple[int, ...] = REGISTERED_SEEDS,
    resamples: int = 10000,
    rng_seed: int = 13,
) -> dict[str, Any]:
    """Sequence-block bootstrap for the registered support gate.

    Conditions per endpoint (sequences resampled with replacement within each
    seed; the same draw is applied to both rows, preserving pairing):

    2. star minus endpoint paired AR NLL, 95% CI strictly below zero;
    3. difference-in-differences (AR diff minus non-AR diff), 95% CI strictly
       below zero.
    """
    for key in [(star, s) for s in seeds] + [(e, s) for e in endpoints for s in seeds]:
        if key not in blocks_by:
            raise ARHitSummarizerError(f"missing blocks for {key[0]} seed {key[1]}")
    rng = random.Random(rng_seed)
    gate: dict[str, Any] = {
        "star": star,
        "seeds": list(seeds),
        "resamples": int(resamples),
        "rng_seed": int(rng_seed),
        "endpoints": {},
    }
    for endpoint in endpoints:
        base_draws = {s: list(range(len(blocks_by[(star, s)]))) for s in seeds}
        base = _paired_seed_stats(blocks_by, star=star, endpoint=endpoint, seeds=seeds, draws=base_draws)
        if base is None:
            raise ARHitSummarizerError(f"degenerate blocks for {star} vs {endpoint}")
        boot_ar: list[float] = []
        boot_did: list[float] = []
        attempts = 0
        while len(boot_ar) < resamples:
            attempts += 1
            if attempts > resamples * 10:
                raise ARHitSummarizerError("too many degenerate bootstrap resamples")
            draws = {
                s: [rng.randrange(len(blocks_by[(star, s)])) for _ in range(len(blocks_by[(star, s)]))]
                for s in seeds
            }
            stats = _paired_seed_stats(blocks_by, star=star, endpoint=endpoint, seeds=seeds, draws=draws)
            if stats is None:
                continue
            boot_ar.append(stats[0])
            boot_did.append(stats[1])
        ar_lo, ar_hi = _percentile_ci(boot_ar)
        did_lo, did_hi = _percentile_ci(boot_did)
        gate["endpoints"][endpoint] = {
            "ar_diff_point": base[0],
            "ar_diff_ci_lo": ar_lo,
            "ar_diff_ci_hi": ar_hi,
            "cond2_pass": ar_hi < 0.0,
            "did_point": base[1],
            "did_ci_lo": did_lo,
            "did_ci_hi": did_hi,
            "cond3_pass": did_hi < 0.0,
        }
    gate["cond2_pass"] = all(gate["endpoints"][e]["cond2_pass"] for e in endpoints)
    gate["cond3_pass"] = all(gate["endpoints"][e]["cond3_pass"] for e in endpoints)
    gate["supportive"] = bool(gate["cond2_pass"] and gate["cond3_pass"])
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize natural AR-hit rows")
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out/ar_hits/summary.tsv"))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--blocks-json",
        nargs="*",
        type=Path,
        default=None,
        help="eval JSONs carrying per-sequence blocks; enables the bootstrap support gate",
    )
    parser.add_argument("--gate-out", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for path in args.csv:
        rows.extend(_read_csv(path))
    require_registered = not args.allow_incomplete
    validated = [validate_row(row, require_registered_matrix=require_registered) for row in rows]
    summary = summarize(validated, require_registered_matrix=require_registered)
    write_tsv(args.out, summary)
    print(f"validated_rows={len(validated)} summary={args.out}")
    if args.blocks_json:
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.blocks_json]
        blocks_by = load_blocks(payloads)
        gate = paired_bootstrap_gate(
            blocks_by,
            resamples=int(args.bootstrap),
            rng_seed=int(args.bootstrap_seed),
        )
        if args.gate_out is not None:
            args.gate_out.parent.mkdir(parents=True, exist_ok=True)
            args.gate_out.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for endpoint, entry in sorted(gate["endpoints"].items()):
            print(
                f"gate vs {endpoint}: ar_diff={entry['ar_diff_point']:+.4f} "
                f"CI[{entry['ar_diff_ci_lo']:+.4f},{entry['ar_diff_ci_hi']:+.4f}] "
                f"cond2={'PASS' if entry['cond2_pass'] else 'FAIL'}; "
                f"did={entry['did_point']:+.4f} "
                f"CI[{entry['did_ci_lo']:+.4f},{entry['did_ci_hi']:+.4f}] "
                f"cond3={'PASS' if entry['cond3_pass'] else 'FAIL'}"
            )
        print(f"gate_supportive={gate['supportive']} gate_out={args.gate_out}")


if __name__ == "__main__":
    main()
