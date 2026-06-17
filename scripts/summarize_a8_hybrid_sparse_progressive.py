#!/usr/bin/env python3
"""Validate and summarize A8 hybrid sparse progressive artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out" / "paper_mechanisms" / "a8_hybrid_sparse_progressive"
LOG_ROOT = ROOT / "logs" / "a8_hybrid_sparse_progressive"
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"

ALL_RUNS = TABLES / "a8_hybrid_sparse_progressive_all_runs.tsv"
SUMMARY = TABLES / "a8_hybrid_sparse_progressive_summary.tsv"
MANIFEST = CHECKS / "a8_hybrid_sparse_progressive_manifest.json"

EXPECTED: dict[str, str] = {
    "TTSSSS": "4:2;4:2;8:4;8:4",
    "TSTSTS": "4:2;4:2;8:4",
    "TTTTSS": "4:2;8:4",
}
SEEDS = (0, 1, 2)
EPOCHS = 10
NONFINITE_RE = re.compile(r"(?<![A-Za-z0-9_])(?:nan|NaN|-inf|inf)(?![A-Za-z0-9_])")

NUMERIC_KEYS = [
    "train/loss",
    "val/loss",
    "train/ppl",
    "val/ppl",
    "train/peak_vram_mib",
    "train/time_per_epoch_s",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        out = float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{key} is not numeric in {row.get('run_name', '<unknown>')}: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{key} is not finite in {row.get('run_name', '<unknown>')}: {value!r}")
    return out


def find_csv(pattern: str, seed: int) -> Path:
    slug = f"a8_hybrid_sparse_{pattern}_D384_FF1536_L512_lr1e-4_seed{seed}"
    matches = sorted(RAW.glob(f"**/{slug}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one CSV for {slug}, found {len(matches)}")
    return matches[0]


def validate_metadata(meta: dict[str, object], pattern: str, seed: int) -> None:
    required = {
        "model.implementation": "hybrid_token_set",
        "model.attention_family": "sparse",
        "model.backend": "local_band",
        "model.backend_params.radius": 4,
        "model.hybrid.pattern": pattern,
        "model.set_causality_mode": "strict_past",
        "resolved.output_residual_mode": "empty_only",
        "resolved.hybrid_pattern": pattern,
        "resolved.hybrid_set_topologies": EXPECTED[pattern],
        "data.seq_len": 512,
        "data.batch_size": 16,
        "training.epochs": EPOCHS,
        "training.seed": seed,
        "resolved.d_phi": 384,
        "resolved.set_state_dim": 384,
    }
    for key, expected in required.items():
        observed = meta.get(key)
        if observed != expected:
            raise ValueError(f"{pattern} seed {seed}: {key} expected {expected!r}, observed {observed!r}")
    lr = float(meta.get("training.lr"))
    if not math.isclose(lr, 1e-4, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{pattern} seed {seed}: training.lr expected 1e-4, observed {meta.get('training.lr')!r}")


def summarize(values: list[float]) -> dict[str, float]:
    n = len(values)
    sd = stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": mean(values), "std": sd, "ci95": ci95}


def scan_logs() -> list[str]:
    failures: list[str] = []
    if not LOG_ROOT.exists():
        failures.append(f"missing log root: {LOG_ROOT}")
        return failures
    bad_terms = ("out of memory", "traceback", "wandb step", "permission denied")
    for path in sorted(LOG_ROOT.glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        lower = text.lower()
        for term in bad_terms:
            if term in lower:
                failures.append(f"{path.relative_to(ROOT)} contains {term!r}")
        if NONFINITE_RE.search(text):
            failures.append(f"{path.relative_to(ROOT)} contains standalone nonfinite token")
    return failures


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    all_rows: list[dict[str, object]] = []

    status_path = RAW / "a8_hybrid_sparse_progressive_status.tsv"
    if not status_path.exists():
        failures.append(f"missing status TSV: {status_path}")
    else:
        status_rows = list(csv.DictReader(status_path.open(encoding="utf-8"), delimiter="\t"))
        if len(status_rows) != 9:
            failures.append(f"expected 9 status rows, found {len(status_rows)}")
        bad = [r for r in status_rows if r.get("exit_code") != "0"]
        if bad:
            failures.append(f"nonzero exits: {bad}")

    failures.extend(scan_logs())

    for pattern in EXPECTED:
        for seed in SEEDS:
            try:
                csv_path = find_csv(pattern, seed)
                json_path = csv_path.with_suffix(".json")
                if not json_path.exists():
                    raise FileNotFoundError(f"missing JSON for {csv_path}")
                rows = read_csv(csv_path)
                if len(rows) != EPOCHS:
                    raise ValueError(f"{csv_path}: expected {EPOCHS} epochs, found {len(rows)}")
                last = rows[-1]
                if int(float(last.get("epoch", "nan"))) != EPOCHS:
                    raise ValueError(f"{csv_path}: final epoch is {last.get('epoch')!r}, expected {EPOCHS}")
                for key in NUMERIC_KEYS:
                    as_float(last, key)
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                validate_metadata(meta, pattern, seed)
                all_rows.append(
                    {
                        "family": "hybrid_sparse_local_band",
                        "pattern": pattern,
                        "backend": "local_band",
                        "seed": seed,
                        "L": 512,
                        "D": 384,
                        "d_ff": 1536,
                        "set_topologies": EXPECTED[pattern],
                        "epochs": EPOCHS,
                        "lr": meta.get("training.lr"),
                        "final_train_ppl": as_float(last, "train/ppl"),
                        "final_val_ppl": as_float(last, "val/ppl"),
                        "peak_vram_mib": as_float(last, "train/peak_vram_mib"),
                        "time_per_epoch_s": as_float(last, "train/time_per_epoch_s"),
                        "csv_path": str(csv_path.relative_to(ROOT)),
                        "json_path": str(json_path.relative_to(ROOT)),
                        "csv_sha256": sha256(csv_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(str(exc))

    fields = [
        "family",
        "pattern",
        "backend",
        "seed",
        "L",
        "D",
        "d_ff",
        "set_topologies",
        "epochs",
        "lr",
        "final_train_ppl",
        "final_val_ppl",
        "peak_vram_mib",
        "time_per_epoch_s",
        "csv_path",
        "json_path",
        "csv_sha256",
    ]
    with ALL_RUNS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows: list[dict[str, object]] = []
    for pattern in EXPECTED:
        group = [r for r in all_rows if r["pattern"] == pattern]
        if not group:
            continue
        val = summarize([float(r["final_val_ppl"]) for r in group])
        train = summarize([float(r["final_train_ppl"]) for r in group])
        vram = summarize([float(r["peak_vram_mib"]) for r in group])
        time_s = summarize([float(r["time_per_epoch_s"]) for r in group])
        summary_rows.append(
            {
                "family": "hybrid_sparse_local_band",
                "pattern": pattern,
                "backend": "local_band",
                "n": len(group),
                "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda r: int(r["seed"]))),
                "L": 512,
                "D": 384,
                "d_ff": 1536,
                "set_topologies": EXPECTED[pattern],
                "mean_final_val_ppl": val["mean"],
                "std_final_val_ppl": val["std"],
                "ci95_final_val_ppl": val["ci95"],
                "mean_final_train_ppl": train["mean"],
                "std_final_train_ppl": train["std"],
                "mean_peak_vram_mib": vram["mean"],
                "std_peak_vram_mib": vram["std"],
                "mean_time_per_epoch_s": time_s["mean"],
                "std_time_per_epoch_s": time_s["std"],
            }
        )

    with SUMMARY.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "family",
            "pattern",
            "backend",
            "n",
            "seeds",
            "L",
            "D",
            "d_ff",
            "set_topologies",
            "mean_final_val_ppl",
            "std_final_val_ppl",
            "ci95_final_val_ppl",
            "mean_final_train_ppl",
            "std_final_train_ppl",
            "mean_peak_vram_mib",
            "std_peak_vram_mib",
            "mean_time_per_epoch_s",
            "std_time_per_epoch_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "phase": "A8 hybrid sparse progressive",
        "status": "pass" if not failures else "fail",
        "expected_runs": 9,
        "validated_runs": len(all_rows),
        "failures": failures,
        "all_runs_tsv": str(ALL_RUNS.relative_to(ROOT)),
        "summary_tsv": str(SUMMARY.relative_to(ROOT)),
        "raw_root": str(RAW.relative_to(ROOT)),
        "families": summary_rows,
        "seed_extension_policy": "Seeds 3,4 are pending follow-up approval.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
