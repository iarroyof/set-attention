#!/usr/bin/env python3
"""Validate and summarize the A4.1 long-context smoke (L=2048, dense baseline + dense SKA).

Design note: Smoke test only -- 1 seed, 10 epochs each. Purpose is to verify:
  (a) L=2048 fits in GPU memory on RTX 4090 without OOM,
  (b) both models converge (finite losses throughout),
  (c) SKA strict_past causality and window/stride are correctly wired at L=2048.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out" / "paper_mechanisms" / "a41_smoke"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"
LOG_ROOT = ROOT / "logs" / "a41_smoke"

LR = "1e-4"
LR_TAG = "1e-4"
SEQ_LEN = 2048
WINDOW = 16
STRIDE = 8
SEED = 0
M = (SEQ_LEN - WINDOW) // STRIDE + 1  # 255

RESOLVED_KEYS_SKA = [
    "resolved.d_phi",
    "resolved.adapter_type",
    "resolved.router_min_temp",
    "resolved.pooling_alpha",
    "resolved.hash_seed",
    "resolved.hash_normalize",
    "resolved.hash_num_bins",
]
SET_ONLY_KEYS = [
    "model.set_causality_mode",
    "model.pooling.alpha",
    "model.feature_params.hash_seed",
    "model.router.min_temp",
]


@dataclass(frozen=True)
class ExpectedRun:
    impl: str          # "baseline_token" or "set_only"
    slug: str          # "baseline_dense" or "set_dense"
    config: str

    @property
    def group(self) -> str:
        if self.impl == "baseline_token":
            return f"a41_smoke_baseline_dense_L{SEQ_LEN}"
        return f"a41_smoke_set_dense_L{SEQ_LEN}"

    @property
    def name(self) -> str:
        if self.impl == "baseline_token":
            return f"a41_baseline_dense_D384_FF1536_L{SEQ_LEN}_lr{LR_TAG}_seed{SEED}"
        return f"a41_set_dense_D384_FF1536_L{SEQ_LEN}_w{WINDOW}_s{STRIDE}_lr{LR_TAG}_seed{SEED}"

    @property
    def csv_path(self) -> Path:
        return RAW / self.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


EXPECTED_RUNS = [
    ExpectedRun("baseline_token", "baseline_dense", "configs/a4_long_context/baseline_dense_lc.yaml"),
    ExpectedRun("set_only",       "set_dense",      "configs/a4_long_context/set_dense_lc.yaml"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        val = row.get(key)
        if val not in {None, "", "NA", "None"}:
            return val
    return "NA"


def float_or_none(raw: object) -> float | None:
    if raw in {None, "", "NA", "None"}:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def finite_csv(rows: list[dict[str, str]]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for i, row in enumerate(rows, 1):
        for key, raw in row.items():
            if isinstance(raw, str) and raw.strip().lower() in {"nan", "inf", "-inf"}:
                bad.append(f"row {i}: {key}={raw}")
    return not bad, bad


def scan_logs() -> list[str]:
    """Scan A4.1 log files for OOM, errors, and numeric anomalies.

    Uses word-boundary regex for nan/inf to avoid false positives from
    normal English words that contain 'nan' as a substring.
    """
    if not LOG_ROOT.exists():
        return [f"missing log root: {LOG_ROOT}"]
    substr_patterns = ["OOM", "out of memory", "Traceback", "RuntimeError", "ValueError"]
    token_re = re.compile(r"(?<![A-Za-z0-9_])(?:nan|NaN|-inf|inf)(?![A-Za-z0-9_])")
    failures: list[str] = []
    for path in sorted(LOG_ROOT.glob("*.log")):
        text = path.read_text(errors="replace")
        matched = False
        for pattern in substr_patterns:
            if pattern in text:
                failures.append(f"{path.relative_to(ROOT)} contains {pattern!r}")
                matched = True
                break
        if not matched:
            m = token_re.search(text)
            if m:
                failures.append(
                    f"{path.relative_to(ROOT)} contains standalone token {m.group()!r}"
                )
    return failures


def validate_run(run: ExpectedRun) -> dict[str, str]:
    if not run.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {run.csv_path}")
    if not run.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {run.json_path}")

    rows = read_rows(run.csv_path)
    if len(rows) != 10:
        raise ValueError(f"{run.csv_path} has {len(rows)} rows, expected 10")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 11)):
        raise ValueError(f"{run.csv_path} epochs are not 1..10: {epochs}")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite CSV values in {run.csv_path}: {bad[:5]}")

    meta = json.loads(run.json_path.read_text())

    last = rows[-1]
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl"]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{run.csv_path} missing finite final {key}")

    # SKA-specific checks
    if run.impl == "set_only":
        missing = [k for k in RESOLVED_KEYS_SKA + SET_ONLY_KEYS if k not in meta]
        if missing:
            raise ValueError(f"{run.json_path} missing metadata keys: {missing}")
        if meta.get("model.set_causality_mode") != "strict_past":
            raise ValueError(f"{run.json_path} is not strict_past")
        if int(meta.get("model.window_size", 0)) != WINDOW:
            raise ValueError(f"{run.json_path} wrong window_size (expected {WINDOW})")
        if int(meta.get("model.stride", 0)) != STRIDE:
            raise ValueError(f"{run.json_path} wrong stride (expected {STRIDE})")
        candidate_mean = value(last, "ausa/candidate_count_mean", "ausa/router_candidate_count_struct_mean")
        if float_or_none(candidate_mean) is None:
            raise ValueError(f"{run.csv_path} missing finite candidate-count mean")

    peak_vram = value(last, "train/peak_vram_mib")

    return {
        "phase": "A4.1",
        "impl": run.impl,
        "slug": run.slug,
        "seed": str(SEED),
        "lr": LR,
        "D": "384",
        "d_ff": "1536",
        "L": str(SEQ_LEN),
        "w": str(WINDOW) if run.impl == "set_only" else "NA",
        "s": str(STRIDE) if run.impl == "set_only" else "NA",
        "M": str(M) if run.impl == "set_only" else "NA",
        "config": run.config,
        "csv_path": str(run.csv_path.relative_to(ROOT)),
        "json_path": str(run.json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_train_loss": value(last, "train/loss"),
        "final_val_loss": value(last, "val/loss"),
        "final_train_ppl": value(last, "train/ppl"),
        "final_val_ppl": value(last, "val/ppl"),
        "time_per_epoch_s": value(last, "train/time_per_epoch_s"),
        "peak_vram_mib": peak_vram,
        "candidate_count_mean": (
            value(last, "ausa/candidate_count_mean", "ausa/router_candidate_count_struct_mean")
            if run.impl == "set_only" else "NA"
        ),
        "set_causality_mode": (
            str(meta.get("model.set_causality_mode", "NA"))
            if run.impl == "set_only" else "NA"
        ),
        "pooling_alpha": str(meta.get("resolved.pooling_alpha", "NA")),
        "router_min_temp": str(meta.get("resolved.router_min_temp", "NA")),
        "source_csv_sha256": sha256(run.csv_path),
    }


def write_tsv(path: Path, rows: Iterable[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_info() -> dict:
    def run(cmd: list[str]) -> dict:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip().splitlines(),
            "stderr": proc.stderr.strip().splitlines(),
        }
    return {
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def markdown_audit(rows: list[dict[str, str]], failures: list[str]) -> str:
    prelaunch_path = AUDIT / "A4_1_smoke_prelaunch.json"
    prelaunch = json.loads(prelaunch_path.read_text()) if prelaunch_path.exists() else {}
    lines = [
        "# A4.1 Long-Context Smoke Audit",
        "",
        "Status: PASS" if not failures else "Status: FAIL",
        "",
        "## Scope",
        "",
        f"- Sequence length: `L={SEQ_LEN}`.",
        "- Models: dense baseline (baseline_token) + dense SKA (set_only, w=16, s=8).",
        "- 1 seed (seed=0), 10 epochs.",
        f"- M (set tokens at L=2048): `M={M}`.",
        "- batch_size=16 on single RTX 4090 (~15.6 GB peak fp32 -- within 24 GB budget).",
        "- Purpose: verify OOM-free execution and finite convergence at L=2048.",
        "",
        "## Commands / Scripts",
        "",
        "- `bash scripts/run_a41_smoke.sh`",
        "- `python scripts/summarize_a41_smoke.py`",
        "",
        "## Prelaunch State",
        "",
        f"- Branch: `{(prelaunch.get('branch') or {}).get('stdout', ['?'])[0]}`",
        f"- HEAD: `{(prelaunch.get('head') or {}).get('stdout', ['?'])[0]}`",
        f"- A3.3 manifest: `{prelaunch.get('a3_3_manifest_status', '?')}` with "
        f"`{prelaunch.get('a3_3_validated_runs', '?')}` / "
        f"`{prelaunch.get('a3_3_expected_runs', '?')}` runs.",
        f"- A3.3 handoff: `{prelaunch.get('a3_3_audit_status_line', '?')}`",
        "",
        "## Failures / Retries",
        "",
    ]
    if failures:
        for f in failures:
            lines.append(f"- {f}")
    else:
        lines.append("- None.")
    lines += ["", "## Run Artifacts", ""]
    all_cols = [
        "slug", "impl", "seed", "lr", "L", "w", "s", "M", "rows",
        "final_val_loss", "final_val_ppl", "peak_vram_mib",
        "time_per_epoch_s", "candidate_count_mean", "set_causality_mode",
        "config", "csv_path", "source_csv_sha256",
    ]
    lines.append("| " + " | ".join(all_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in all_cols) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(c, "NA") for c in all_cols) + " |")
    lines += [
        "",
        "## Generated Artifacts",
        "",
        "- `out/paper_integrated_evidence/tables/a41_smoke_all_runs.tsv`",
        "- `out/paper_integrated_evidence/checks/a41_smoke_manifest.json`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    import sys

    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    print(f"Expected {len(EXPECTED_RUNS)} runs.")

    validated: list[dict[str, str]] = []
    failures: list[str] = []

    for run in EXPECTED_RUNS:
        try:
            validated.append(validate_run(run))
        except (FileNotFoundError, ValueError) as exc:
            failures.append(str(exc))

    log_failures = scan_logs()
    failures.extend(log_failures)

    print(f"Validated: {len(validated)} / {len(EXPECTED_RUNS)}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)

    all_cols = list(validated[0].keys()) if validated else []
    write_tsv(TABLES / "a41_smoke_all_runs.tsv", validated, all_cols)

    manifest = {
        "status": "pass" if not failures else "fail",
        "expected_runs": len(EXPECTED_RUNS),
        "validated_runs": len(validated),
        "failures": failures,
        "config": {
            "D": 384,
            "d_ff": 1536,
            "L": SEQ_LEN,
            "w": WINDOW,
            "s": STRIDE,
            "M": M,
            "seed": SEED,
            "epochs": 10,
            "batch_size": 16,
            "set_causality_mode": "strict_past",
        },
        "source_csvs": [
            {"path": r["csv_path"], "sha256": r["source_csv_sha256"]}
            for r in validated
        ],
        "generated": [
            {
                "path": str((TABLES / "a41_smoke_all_runs.tsv").relative_to(ROOT)),
                "sha256": sha256(TABLES / "a41_smoke_all_runs.tsv"),
            },
        ],
        "git": git_info(),
    }
    (CHECKS / "a41_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    audit_text = markdown_audit(validated, failures)
    (AUDIT / "A4_1_smoke.md").write_text(audit_text)

    print(f"status: {manifest['status']}")
    print(f"validated_runs: {manifest['validated_runs']} / expected: {manifest['expected_runs']}")
    print(json.dumps({"status": manifest["status"], "validated_runs": manifest["validated_runs"]}, indent=2))

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
