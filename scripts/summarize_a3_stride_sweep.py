#!/usr/bin/env python3
"""Validate and summarize the A3.3 stride sweep (fixed w=16, vary s in {4,8,12,16}).

Design note: stride confounds M = floor((L-w)/s)+1, so this sweep is a
demoted complement captioned as confounded by M. Endpoint strides (4,16) use
seeds {0,1,2}; interior strides (8,12) use seed 0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out" / "paper_mechanisms" / "a3_stride_sweep"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"
LOG_ROOT = ROOT / "logs" / "a3_stride_sweep"

LR = "1e-4"
LR_TAG = "1e-4"
WINDOW = 16
SEQ_LEN = 512
STRIDES = [4, 8, 12, 16]
FAMILIES = [
    ("dense_exact",      "Set Dense",  "exact",      "configs/paper_complements/family_dense_exact.yaml"),
    ("sparse_local_band","Set Sparse", "local_band",  "configs/paper_complements/family_sparse_local_band.yaml"),
    ("linear_landmark",  "Set Linear", "landmark",    "configs/paper_complements/family_linear_landmark.yaml"),
]
RESOLVED_KEYS = [
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
    "model.pooling.learnable_alpha",
    "model.feature_params.hash_seed",
    "model.feature_params.normalize",
    "model.feature_params.num_bins",
    "model.router.min_temp",
]
RANGE_KEYS = [
    "ausa/router_entropy_norm",
    "ausa/router_top1_gap_norm",
    "ausa/router_top1_weight",
]


def seeds_for_stride(stride: int) -> list[int]:
    return [0, 1, 2] if stride in {4, 16} else [0]


@dataclass(frozen=True)
class ExpectedRun:
    family_slug: str
    family: str
    backend: str
    config: str
    stride: int
    seed: int

    @property
    def m(self) -> int:
        return (SEQ_LEN - WINDOW) // self.stride + 1

    @property
    def landmark_count(self) -> str:
        if self.backend != "landmark":
            return "NA"
        return str(max(round(0.25 * self.m), 2))

    @property
    def group(self) -> str:
        return f"a3_stride_sweep_{self.family_slug}_D384_FF1536_w{WINDOW}"

    @property
    def name(self) -> str:
        lr_tag = LR_TAG.replace(".", "p")
        return (
            f"a3_stride_{self.family_slug}_D384_FF1536_w{WINDOW}_s{self.stride}_"
            f"lr{lr_tag}_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return RAW / self.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


def expected_runs() -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []
    for slug, family, backend, config in FAMILIES:
        for stride in STRIDES:
            for seed in seeds_for_stride(stride):
                runs.append(ExpectedRun(slug, family, backend, config, stride, seed))
    return runs


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
    """Scan A3.3 log files for numeric anomalies and error patterns.

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
        if "WARNING" in text and "step" in text.lower() and "wandb" in text.lower():
            failures.append(f"{path.relative_to(ROOT)} contains W&B step warning")
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
    missing = [key for key in RESOLVED_KEYS + SET_ONLY_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{run.json_path} missing metadata keys: {missing}")
    if meta.get("model.set_causality_mode") != "strict_past":
        raise ValueError(f"{run.json_path} is not strict_past")
    if int(meta.get("model.window_size")) != WINDOW:
        raise ValueError(f"{run.json_path} has wrong window_size (expected {WINDOW})")
    if int(meta.get("model.stride")) != run.stride:
        raise ValueError(f"{run.json_path} has wrong stride (expected {run.stride})")
    if run.backend == "landmark":
        coverage = float_or_none(meta.get("model.backend_params.landmark_coverage"))
        resolved_coverage = float_or_none(meta.get("resolved.landmark_coverage"))
        count = str(meta.get("resolved.landmark_count", "NA"))
        if coverage != 0.25 or resolved_coverage != 0.25:
            raise ValueError(f"{run.json_path} does not record landmark_coverage=0.25")
        if count != run.landmark_count:
            raise ValueError(
                f"{run.json_path} landmark_count={count}, expected {run.landmark_count}"
            )

    last = rows[-1]
    candidate_mean = value(last, "ausa/candidate_count_mean", "ausa/router_candidate_count_struct_mean")
    candidate_max  = value(last, "ausa/candidate_count_max",  "ausa/router_candidate_count_struct_max")
    if float_or_none(candidate_mean) is None:
        raise ValueError(f"{run.csv_path} missing finite candidate-count mean")
    if float_or_none(candidate_max) is None:
        raise ValueError(f"{run.csv_path} missing finite candidate-count max")
    for key in RANGE_KEYS:
        val = float_or_none(last.get(key))
        if val is not None and not (-1e-6 <= val <= 1.000001):
            raise ValueError(f"{run.csv_path} has out-of-range {key}={val}")
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl"]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{run.csv_path} missing finite final {key}")

    return {
        "phase": "A3.3",
        "family_slug": run.family_slug,
        "family": run.family,
        "backend": run.backend,
        "seed": str(run.seed),
        "lr": LR,
        "D": "384",
        "d_ff": "1536",
        "w": str(WINDOW),
        "s": str(run.stride),
        "M": str(run.m),
        "expected_landmark_count": run.landmark_count,
        "config": run.config,
        "csv_path": str(run.csv_path.relative_to(ROOT)),
        "json_path": str(run.json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_train_loss": value(last, "train/loss"),
        "final_val_loss": value(last, "val/loss"),
        "final_train_ppl": value(last, "train/ppl"),
        "final_val_ppl": value(last, "val/ppl"),
        "time_per_epoch_s": value(last, "train/time_per_epoch_s"),
        "peak_vram_mib": value(last, "train/peak_vram_mib"),
        "candidate_count_mean": candidate_mean,
        "candidate_count_max": candidate_max,
        "router_entropy_norm": value(last, "ausa/router_entropy_norm"),
        "router_top1_weight": value(last, "ausa/router_top1_weight"),
        "router_top1_gap_norm": value(last, "ausa/router_top1_gap_norm"),
        "resolved_d_phi": str(meta.get("resolved.d_phi", "NA")),
        "resolved_adapter_type": str(meta.get("resolved.adapter_type", "NA")),
        "pooling_alpha": str(meta.get("resolved.pooling_alpha", "NA")),
        "hash_seed": str(meta.get("resolved.hash_seed", "NA")),
        "hash_normalize": str(meta.get("resolved.hash_normalize", "NA")),
        "hash_num_bins": str(meta.get("resolved.hash_num_bins", "NA")),
        "router_min_temp": str(meta.get("resolved.router_min_temp", "NA")),
        "landmark_coverage": str(meta.get("resolved.landmark_coverage", "NA")),
        "landmark_count": str(meta.get("resolved.landmark_count", "NA")),
        "source_csv_sha256": sha256(run.csv_path),
    }


def write_tsv(path: Path, rows: Iterable[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["family_slug"], row["backend"], row["s"])].append(row)
    out: list[dict[str, str]] = []
    for (family, slug, backend, stride), group in sorted(grouped.items()):
        ppls = [float(row["final_val_ppl"]) for row in group]
        cand = [float(row["candidate_count_mean"]) for row in group]
        out.append({
            "phase": "A3.3",
            "family": family,
            "family_slug": slug,
            "backend": backend,
            "w": str(WINDOW),
            "s": stride,
            "M": group[0]["M"],
            "runs": str(len(group)),
            "seeds": ",".join(sorted(row["seed"] for row in group)),
            "val_ppl_mean": f"{mean(ppls):.6f}",
            "val_ppl_std": f"{pstdev(ppls):.6f}" if len(ppls) > 1 else "0.000000",
            "val_ppl_min": f"{min(ppls):.6f}",
            "val_ppl_max": f"{max(ppls):.6f}",
            "candidate_count_mean": f"{mean(cand):.6f}",
            "candidate_count_min": f"{min(cand):.6f}",
            "candidate_count_max": f"{max(cand):.6f}",
        })
    return out


def validate_candidate_variation(rows: list[dict[str, str]]) -> list[str]:
    """Warn if candidate-count mean does not vary across strides (it must, since M changes)."""
    failures: list[str] = []
    for slug, family, _, _ in FAMILIES:
        seed0 = [r for r in rows if r["family_slug"] == slug and r["seed"] == "0"]
        by_stride = {int(r["s"]): float(r["candidate_count_mean"]) for r in seed0}
        values = [by_stride[s] for s in STRIDES if s in by_stride]
        if len(values) < 2:
            failures.append(f"{family}: insufficient seed0 rows to check candidate variation")
            continue
        if len({round(v, 8) for v in values}) < 2:
            failures.append(f"{family}: candidate-count mean did not vary with stride (confound check)")
    return failures


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


def markdown_audit(rows: list[dict[str, str]], summary: list[dict[str, str]], failures: list[str]) -> str:
    prelaunch_path = AUDIT / "A3_3_stride_sweep_prelaunch.json"
    prelaunch = json.loads(prelaunch_path.read_text()) if prelaunch_path.exists() else {}
    lines = [
        "# A3.3 Stride Sweep Audit",
        "",
        "Status: PASS" if not failures else "Status: FAIL",
        "",
        "## Scope",
        "",
        f"- Fixed window: `w={WINDOW}`.",
        "- Strides: `s in {4, 8, 12, 16}`.",
        "- Families: SetDense/exact, SetSparse/local_band, SetLinear/landmark.",
        "- Seeds: endpoint strides (4, 16) use seeds 0,1,2; interior strides (8, 12) use seed 0.",
        "- **Demoted complement**: M changes with s (confounding). Caption accordingly.",
        "  M values: s=4→M=125, s=8→M=63, s=12→M=42, s=16→M=32.",
        "- A2.2/A2.3 remain the locked s=8 LR-normalized headline/family grid; not overridden.",
        "",
        "## Commands / Scripts",
        "",
        "- `bash scripts/run_a3_stride_sweep.sh`",
        "- `python scripts/summarize_a3_stride_sweep.py`",
        "",
        "## Prelaunch State",
        "",
        f"- Branch: `{(prelaunch.get('branch') or {}).get('stdout', ['?'])[0]}`",
        f"- HEAD: `{(prelaunch.get('head') or {}).get('stdout', ['?'])[0]}`",
        f"- A3.2 manifest: `{prelaunch.get('a3_2_manifest_status', '?')}` with "
        f"`{prelaunch.get('a3_2_validated_runs', '?')}` / "
        f"`{prelaunch.get('a3_2_expected_runs', '?')}` runs.",
        f"- A3.2 handoff: `{prelaunch.get('a3_2_audit_status_line', '?')}`",
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
        "family", "backend", "seed", "lr", "w", "s", "M", "rows",
        "final_val_ppl", "time_per_epoch_s", "candidate_count_mean",
        "config", "csv_path", "source_csv_sha256",
    ]
    lines.append("| " + " | ".join(all_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in all_cols) + " |")
    for row in sorted(rows, key=lambda r: (r["family"], int(r["s"]), r["seed"])):
        lines.append("| " + " | ".join(row.get(c, "NA") for c in all_cols) + " |")
    lines += ["", "## Summary", ""]
    sum_cols = [
        "family", "backend", "w", "s", "M", "runs", "seeds",
        "val_ppl_mean", "val_ppl_std", "candidate_count_mean",
    ]
    lines.append("| " + " | ".join(sum_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in sum_cols) + " |")
    for row in summary:
        lines.append("| " + " | ".join(str(row.get(c, "NA")) for c in sum_cols) + " |")
    lines += [
        "",
        "## Generated Artifacts",
        "",
        "- `out/paper_integrated_evidence/tables/a3_stride_sweep_all_runs.tsv`",
        "- `out/paper_integrated_evidence/tables/a3_stride_sweep_summary.tsv`",
        "- `out/paper_integrated_evidence/checks/a3_stride_sweep_manifest.json`",
        "",
        "## Note on Confounding",
        "",
        "Stride and M are not independently controlled here. "
        "Interpret this sweep as showing how ppl and candidate-count jointly vary "
        "as the number of sets (M) increases with decreasing stride at fixed window. "
        "Do not caption this as an isolated stride effect.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    import sys

    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    runs = expected_runs()
    print(f"Expected {len(runs)} runs.")

    validated: list[dict[str, str]] = []
    failures: list[str] = []

    for run in runs:
        try:
            validated.append(validate_run(run))
        except (FileNotFoundError, ValueError) as exc:
            failures.append(str(exc))

    log_failures = scan_logs()
    failures.extend(log_failures)

    cand_failures = validate_candidate_variation(validated)
    failures.extend(cand_failures)

    print(f"Validated: {len(validated)} / {len(runs)}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)

    all_cols = list(validated[0].keys()) if validated else []
    write_tsv(TABLES / "a3_stride_sweep_all_runs.tsv", validated, all_cols)
    summary = summarize(validated)
    sum_cols = [
        "phase", "family", "family_slug", "backend", "w", "s", "M", "runs", "seeds",
        "val_ppl_mean", "val_ppl_std", "val_ppl_min", "val_ppl_max",
        "candidate_count_mean", "candidate_count_min", "candidate_count_max",
    ]
    write_tsv(TABLES / "a3_stride_sweep_summary.tsv", summary, sum_cols)

    manifest = {
        "status": "pass" if not failures else "fail",
        "expected_runs": len(runs),
        "validated_runs": len(validated),
        "summary_rows": len(summary),
        "failures": failures,
        "config": {
            "D": 384,
            "d_ff": 1536,
            "L": SEQ_LEN,
            "w": WINDOW,
            "strides": STRIDES,
            "M_by_stride": {str(s): (SEQ_LEN - WINDOW) // s + 1 for s in STRIDES},
            "endpoint_strides": [4, 16],
            "interior_strides": [8, 12],
            "set_causality_mode": "strict_past",
            "landmark_coverage": 0.25,
        },
        "source_csvs": [
            {"path": r["csv_path"], "sha256": r["source_csv_sha256"]}
            for r in validated
        ],
        "generated": [
            {
                "path": str((TABLES / "a3_stride_sweep_all_runs.tsv").relative_to(ROOT)),
                "sha256": sha256(TABLES / "a3_stride_sweep_all_runs.tsv"),
            },
            {
                "path": str((TABLES / "a3_stride_sweep_summary.tsv").relative_to(ROOT)),
                "sha256": sha256(TABLES / "a3_stride_sweep_summary.tsv"),
            },
        ],
        "git": git_info(),
    }
    (CHECKS / "a3_stride_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    audit_text = markdown_audit(validated, summary, failures)
    (AUDIT / "A3_3_stride_sweep.md").write_text(audit_text)

    print(f"status: {manifest['status']}")
    print(f"validated_runs: {manifest['validated_runs']} / expected: {manifest['expected_runs']}")
    print(json.dumps({"status": manifest["status"], "validated_runs": manifest["validated_runs"]}, indent=2))

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
