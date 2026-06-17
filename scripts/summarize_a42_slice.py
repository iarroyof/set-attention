#!/usr/bin/env python3
"""Validate and summarize the A4.2 long-context family slice (L=2048, w=16, s=8).

Design note: LR-norm headline reference family (D=384, d_ff=1536) at L=2048.
Families: baseline_token (dense), set_dense (exact), set_sparse (local_band),
          set_linear (landmark).  3 seeds each.  12 total runs.
BATCH=4 confirmed safe in A4.1.  M=255 for all SKA variants at L=2048, w=16, s=8.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out" / "paper_mechanisms" / "a42_slice"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"
LOG_ROOT = ROOT / "logs" / "a42_slice"

LR = "1e-4"
LR_TAG = "1e-4"   # bash ${LR//./p} on "1e-4" (no dot) → unchanged
SEQ_LEN = 2048
WINDOW = 16
STRIDE = 8
SEEDS = [0, 1, 2]
M = (SEQ_LEN - WINDOW) // STRIDE + 1          # 255
LANDMARK_COUNT = max(round(0.25 * M), 2)       # 64

# Metadata keys expected in the run JSON sidecar for SKA variants
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

# (family_slug, impl, backend, config, is_set)
FAMILIES: list[tuple[str, str, str, str, bool]] = [
    ("baseline_token", "baseline_token", "dense",      "configs/a4_long_context/baseline_dense_lc.yaml", False),
    ("set_dense",      "set_only",       "exact",      "configs/a4_long_context/set_dense_lc.yaml",      True),
    ("set_sparse",     "set_only",       "local_band", "configs/a4_long_context/set_sparse_lc.yaml",     True),
    ("set_linear",     "set_only",       "landmark",   "configs/a4_long_context/set_linear_lc.yaml",     True),
]


@dataclass(frozen=True)
class ExpectedRun:
    family_slug: str
    impl: str
    backend: str
    config: str
    is_set: bool
    seed: int

    @property
    def group(self) -> str:
        return f"a42_slice_{self.family_slug}_D384_FF1536_L{SEQ_LEN}"

    @property
    def name(self) -> str:
        lr_tag = LR_TAG.replace(".", "p")   # "1e-4" has no dot → unchanged
        if self.is_set:
            return (
                f"a42_{self.family_slug}_D384_FF1536_L{SEQ_LEN}"
                f"_w{WINDOW}_s{STRIDE}_lr{lr_tag}_seed{self.seed}"
            )
        return f"a42_{self.family_slug}_D384_FF1536_L{SEQ_LEN}_lr{lr_tag}_seed{self.seed}"

    @property
    def csv_path(self) -> Path:
        return RAW / self.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")

    @property
    def landmark_count(self) -> str:
        return str(LANDMARK_COUNT) if self.backend == "landmark" else "NA"


def expected_runs() -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []
    for slug, impl, backend, config, is_set in FAMILIES:
        for seed in SEEDS:
            runs.append(ExpectedRun(slug, impl, backend, config, is_set, seed))
    return runs


EXPECTED_RUNS = expected_runs()


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
    """Scan A4.2 log files for OOM, errors, and numeric anomalies.

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

    if run.is_set:
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
        "phase": "A4.2",
        "impl": run.impl,
        "backend": run.backend,
        "family_slug": run.family_slug,
        "seed": str(run.seed),
        "lr": LR,
        "D": "384",
        "d_ff": "1536",
        "L": str(SEQ_LEN),
        "w": str(WINDOW) if run.is_set else "NA",
        "s": str(STRIDE) if run.is_set else "NA",
        "M": str(M) if run.is_set else "NA",
        "landmark_count": run.landmark_count,
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
            if run.is_set else "NA"
        ),
        "set_causality_mode": (
            str(meta.get("model.set_causality_mode", "NA"))
            if run.is_set else "NA"
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


def ppl_summary(validated: list[dict[str, str]]) -> dict[str, dict]:
    """Per-family mean/std final_val_ppl across seeds."""
    by_family: dict[str, list[float]] = {}
    for row in validated:
        fam = row["family_slug"]
        v = float_or_none(row["final_val_ppl"])
        if v is not None:
            by_family.setdefault(fam, []).append(v)
    out: dict[str, dict] = {}
    for fam, vals in by_family.items():
        out[fam] = {
            "n": len(vals),
            "mean_val_ppl": round(mean(vals), 3) if vals else None,
            "std_val_ppl": round(pstdev(vals), 3) if len(vals) > 1 else None,
        }
    return out


def markdown_audit(validated: list[dict[str, str]], failures: list[str]) -> str:
    prelaunch_path = AUDIT / "A4_2_slice_prelaunch.json"
    prelaunch = json.loads(prelaunch_path.read_text()) if prelaunch_path.exists() else {}
    n_total = len(EXPECTED_RUNS)
    status = "PASS" if not failures else "FAIL"
    lines = [
        "# A4.2 Long-Context Family Slice Audit",
        "",
        f"Status: {status}",
        "",
        "## Scope",
        "",
        f"- Sequence length: `L={SEQ_LEN}`.",
        "- LR-norm headline reference: D=384, d_ff=1536, w=16, s=8.",
        "- Families: baseline_token (dense), set_dense (exact), set_sparse (local_band), set_linear (landmark).",
        f"- Seeds: {SEEDS}.",
        f"- Total runs: {n_total} ({len(FAMILIES)} families × {len(SEEDS)} seeds).",
        f"- M (set tokens at L={SEQ_LEN}, w={WINDOW}, s={STRIDE}): `M={M}`.",
        f"- Landmark count (coverage=0.25): `{LANDMARK_COUNT}`.",
        "- batch_size=4 (confirmed OOM-free in A4.1 for fp32 dense at L=2048).",
        "",
        "## Commands / Scripts",
        "",
        "- `bash scripts/run_a42_slice.sh`",
        "- `python scripts/summarize_a42_slice.py`",
        "",
        "## Prelaunch State",
        "",
        f"- Branch: `{(prelaunch.get('branch') or {}).get('stdout', ['?'])[0]}`",
        f"- HEAD: `{(prelaunch.get('head') or {}).get('stdout', ['?'])[0]}`",
        f"- A4.1 manifest: `{prelaunch.get('a4_1_manifest_status', '?')}` "
        f"with `{prelaunch.get('a4_1_validated_runs', '?')}` / "
        f"`{prelaunch.get('a4_1_expected_runs', '?')}` runs.",
        f"- A4.1 handoff: `{prelaunch.get('a4_1_audit_status_line', '?')}`",
        "",
        "## Failures / Retries",
        "",
    ]
    if failures:
        for f in failures:
            lines.append(f"- {f}")
    else:
        lines.append("- None.")

    # Per-family summary
    summary = ppl_summary(validated)
    lines += ["", "## Per-Family val_ppl Summary (mean ± std over seeds)", ""]
    lines.append("| family | n | mean_val_ppl | std_val_ppl |")
    lines.append("| --- | --- | --- | --- |")
    for slug, _, _, _, _ in FAMILIES:
        s = summary.get(slug, {})
        lines.append(
            f"| {slug} | {s.get('n', 0)} | "
            f"{s.get('mean_val_ppl', 'NA')} | {s.get('std_val_ppl', 'NA')} |"
        )

    all_cols = [
        "family_slug", "impl", "backend", "seed", "lr", "L", "w", "s", "M",
        "landmark_count", "rows", "final_val_loss", "final_val_ppl",
        "peak_vram_mib", "time_per_epoch_s",
        "candidate_count_mean", "set_causality_mode",
        "config", "csv_path", "source_csv_sha256",
    ]
    lines += ["", "## Run Artifacts", ""]
    lines.append("| " + " | ".join(all_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in all_cols) + " |")
    for row in validated:
        lines.append("| " + " | ".join(row.get(c, "NA") for c in all_cols) + " |")
    lines += [
        "",
        "## Generated Artifacts",
        "",
        "- `out/paper_integrated_evidence/tables/a42_slice_all_runs.tsv`",
        "- `out/paper_integrated_evidence/checks/a42_slice_manifest.json`",
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
    write_tsv(TABLES / "a42_slice_all_runs.tsv", validated, all_cols)

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
            "landmark_count": LANDMARK_COUNT,
            "seeds": SEEDS,
            "epochs": 10,
            "batch_size": 4,
            "set_causality_mode": "strict_past",
            "families": [f[0] for f in FAMILIES],
        },
        "ppl_summary": ppl_summary(validated),
        "source_csvs": [
            {"path": r["csv_path"], "sha256": r["source_csv_sha256"]}
            for r in validated
        ],
        "generated": [
            {
                "path": str((TABLES / "a42_slice_all_runs.tsv").relative_to(ROOT)),
                "sha256": sha256(TABLES / "a42_slice_all_runs.tsv"),
            },
        ],
        "git": git_info(),
    }
    (CHECKS / "a42_slice_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    audit_text = markdown_audit(validated, failures)
    (AUDIT / "A4_2_slice.md").write_text(audit_text)

    print(f"status: {manifest['status']}")
    print(f"validated_runs: {manifest['validated_runs']} / expected: {manifest['expected_runs']}")
    print(json.dumps({"status": manifest["status"], "validated_runs": manifest["validated_runs"]}, indent=2))

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
