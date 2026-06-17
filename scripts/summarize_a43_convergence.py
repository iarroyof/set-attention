#!/usr/bin/env python3
"""Validate and summarize the A4.3 30-epoch convergence panel."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out" / "paper_mechanisms" / "a43_convergence"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"
LOG_ROOT = ROOT / "logs" / "a43_convergence"

SEQ_LEN = 512
WINDOW = 16
STRIDE = 8
M = (SEQ_LEN - WINDOW) // STRIDE + 1
SEED = 0
EPOCHS = 30
LR = "1e-4"

RESOLVED_SET_KEYS = [
    "resolved.d_phi",
    "resolved.adapter_type",
    "resolved.router_min_temp",
    "resolved.pooling_alpha",
    "resolved.hash_seed",
    "resolved.hash_normalize",
    "resolved.hash_num_bins",
]


@dataclass(frozen=True)
class ExpectedRun:
    family_slug: str
    family: str
    impl: str
    attention_family: str
    backend: str
    config: str
    is_set: bool

    @property
    def group(self) -> str:
        return f"a43_convergence_{self.family_slug}_D384_FF1536_L{SEQ_LEN}"

    @property
    def name(self) -> str:
        lr_tag = LR.replace(".", "p")
        return (
            f"a43_{self.family_slug}_D384_FF1536_L{SEQ_LEN}_w{WINDOW}_s{STRIDE}_"
            f"lr{lr_tag}_seed{SEED}_ep{EPOCHS}"
        )

    @property
    def csv_path(self) -> Path:
        return RAW / self.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")

    @property
    def landmark_count(self) -> str:
        if self.backend != "landmark":
            return "NA"
        if self.is_set:
            return str(max(round(0.25 * M), 2))
        return str(max(round(0.25 * SEQ_LEN), 2))


RUNS = [
    ExpectedRun(
        "baseline_dense_exact",
        "Baseline Dense",
        "baseline_token",
        "dense",
        "exact",
        "configs/paper_lr_norm/baseline_dense_exact.yaml",
        False,
    ),
    ExpectedRun(
        "baseline_sparse_local_band",
        "Baseline Sparse",
        "baseline_token",
        "sparse",
        "local_band",
        "configs/paper_lr_norm/baseline_sparse_local_band.yaml",
        False,
    ),
    ExpectedRun(
        "baseline_linear_landmark",
        "Baseline Linear",
        "baseline_token",
        "linear",
        "landmark",
        "configs/paper_lr_norm/baseline_linear_landmark.yaml",
        False,
    ),
    ExpectedRun(
        "set_dense_exact",
        "Set Dense",
        "set_only",
        "dense",
        "exact",
        "configs/paper_lr_norm/set_dense_exact.yaml",
        True,
    ),
    ExpectedRun(
        "set_sparse_local_band",
        "Set Sparse",
        "set_only",
        "sparse",
        "local_band",
        "configs/paper_lr_norm/set_sparse_local_band.yaml",
        True,
    ),
    ExpectedRun(
        "set_linear_landmark",
        "Set Linear",
        "set_only",
        "linear",
        "landmark",
        "configs/paper_lr_norm/set_linear_landmark.yaml",
        True,
    ),
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
    if raw in {None, "", "NA", "None", "nan"}:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def finite_csv(rows: list[dict[str, str]]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for i, row in enumerate(rows, 1):
        for key, raw in row.items():
            if isinstance(raw, str) and raw.strip().lower() in {"nan", "inf", "-inf"}:
                bad.append(f"row {i}: {key}={raw}")
    return not bad, bad


def run(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }


def scan_logs() -> list[str]:
    if not LOG_ROOT.exists():
        return [f"missing log root: {LOG_ROOT.relative_to(ROOT)}"]
    substr_patterns = [
        "OOM",
        "out of memory",
        "Traceback",
        "RuntimeError",
        "ValueError",
        "User provided step",
        "less than current step",
        "non-monotonic",
        "W&B step",
    ]
    token_re = re.compile(r"(?<![A-Za-z0-9_])(?:nan|NaN|-inf|inf)(?![A-Za-z0-9_])")
    failures: list[str] = []
    for path in sorted(LOG_ROOT.glob("*.log")):
        text = path.read_text(errors="replace")
        for pattern in substr_patterns:
            if pattern in text:
                failures.append(f"{path.relative_to(ROOT)} contains {pattern!r}")
                break
        else:
            match = token_re.search(text)
            if match:
                failures.append(
                    f"{path.relative_to(ROOT)} contains standalone token {match.group()!r}"
                )
    return failures


def validate_run(spec: ExpectedRun) -> dict[str, str]:
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {spec.csv_path}")
    if not spec.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {spec.json_path}")
    rows = read_rows(spec.csv_path)
    if len(rows) < EPOCHS:
        raise ValueError(f"{spec.csv_path} has {len(rows)} rows, expected >= {EPOCHS}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs[:EPOCHS] != list(range(1, EPOCHS + 1)):
        raise ValueError(f"{spec.csv_path} first epochs are not 1..{EPOCHS}: {epochs[:5]}...")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite CSV values in {spec.csv_path}: {bad[:5]}")
    meta = json.loads(spec.json_path.read_text())
    if meta.get("model.implementation") != spec.impl:
        raise ValueError(f"{spec.json_path} wrong implementation")
    if meta.get("model.attention_family") != spec.attention_family:
        raise ValueError(f"{spec.json_path} wrong attention_family")
    if meta.get("model.backend") != spec.backend:
        raise ValueError(f"{spec.json_path} wrong backend")
    if int(meta.get("data.seq_len")) != SEQ_LEN:
        raise ValueError(f"{spec.json_path} wrong data.seq_len")
    if int(meta.get("model.max_seq_len")) != SEQ_LEN:
        raise ValueError(f"{spec.json_path} wrong model.max_seq_len")
    if spec.is_set:
        missing = [key for key in RESOLVED_SET_KEYS if key not in meta]
        if missing:
            raise ValueError(f"{spec.json_path} missing set resolved metadata: {missing}")
        if meta.get("model.set_causality_mode") != "strict_past":
            raise ValueError(f"{spec.json_path} is not strict_past")
        if int(meta.get("model.window_size")) != WINDOW:
            raise ValueError(f"{spec.json_path} wrong model.window_size")
        if int(meta.get("model.stride")) != STRIDE:
            raise ValueError(f"{spec.json_path} wrong model.stride")
    if spec.backend == "local_band":
        if str(meta.get("model.backend_params.radius")) != "4":
            raise ValueError(f"{spec.json_path} does not record local_band radius=4")
    if spec.backend == "landmark":
        coverage = float_or_none(meta.get("model.backend_params.landmark_coverage"))
        resolved_coverage = float_or_none(meta.get("resolved.landmark_coverage"))
        count = str(meta.get("resolved.landmark_count", "NA"))
        if coverage != 0.25 or resolved_coverage != 0.25:
            raise ValueError(f"{spec.json_path} does not record landmark_coverage=0.25")
        if count != spec.landmark_count:
            raise ValueError(
                f"{spec.json_path} landmark_count={count}, expected {spec.landmark_count}"
            )
    last = rows[-1]
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl"]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{spec.csv_path} missing finite final {key}")
    return {
        "phase": "A4.3",
        "family_slug": spec.family_slug,
        "family": spec.family,
        "implementation": spec.impl,
        "attention_family": spec.attention_family,
        "backend": spec.backend,
        "seed": str(SEED),
        "lr": LR,
        "D": "384",
        "d_ff": "1536",
        "L": str(SEQ_LEN),
        "w": str(WINDOW),
        "s": str(STRIDE),
        "M": str(M) if spec.is_set else "NA",
        "config": spec.config,
        "csv_path": str(spec.csv_path.relative_to(ROOT)),
        "json_path": str(spec.json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_epoch": str(epochs[-1]),
        "final_train_loss": value(last, "train/loss"),
        "final_val_loss": value(last, "val/loss"),
        "final_train_ppl": value(last, "train/ppl"),
        "final_val_ppl": value(last, "val/ppl"),
        "time_per_epoch_s": value(last, "train/time_per_epoch_s"),
        "peak_vram_mib": value(last, "train/peak_vram_mib"),
        "resolved_d_phi": str(meta.get("resolved.d_phi", "NA")),
        "resolved_adapter_type": str(meta.get("resolved.adapter_type", "NA")),
        "pooling_alpha": str(meta.get("resolved.pooling_alpha", "NA")),
        "hash_seed": str(meta.get("resolved.hash_seed", "NA")),
        "hash_normalize": str(meta.get("resolved.hash_normalize", "NA")),
        "hash_num_bins": str(meta.get("resolved.hash_num_bins", "NA")),
        "router_min_temp": str(meta.get("resolved.router_min_temp", "NA")),
        "landmark_coverage": str(meta.get("resolved.landmark_coverage", "NA")),
        "landmark_count": str(meta.get("resolved.landmark_count", "NA")),
        "source_csv_sha256": sha256(spec.csv_path),
    }


def write_tsv(path: Path, rows: Iterable[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_audit(rows: list[dict[str, str]], failures: list[str]) -> str:
    lines = [
        "# A4.3 Convergence Panel",
        "",
        "Status: PASS" if not failures else "Status: FAIL",
        "",
        f"Expected runs: {len(RUNS)}",
        f"Validated runs: {len(rows)}",
        "",
        "## Provenance",
        "",
        f"- Branch: `{run(['git', 'branch', '--show-current'])['stdout'][0]}`",
        f"- HEAD: `{run(['git', 'rev-parse', 'HEAD'])['stdout'][0]}`",
        f"- Dirty entries: {len(run(['git', 'status', '--short'])['stdout'])}",
        "- Seed: `0`",
        "- LR selection: all families use `1e-4`, selected by lowest mean A2/A2.4 val PPL across seeds.",
        "",
        "## Validation",
        "",
    ]
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- All expected CSV/JSON artifacts exist and completed at least 30 epochs.")
        lines.append("- CSV metrics are finite.")
        lines.append("- Logs contain no OOM, traceback, standalone nan/inf, or W&B step warnings.")
        lines.append("- Set-only runs record strict_past with w=16, s=8, M=63.")
        lines.append("- Landmark runs record landmark_coverage=0.25 and resolved landmark_count.")
    lines.extend(["", "## Summary Artifacts", ""])
    for path in [
        "out/paper_integrated_evidence/tables/a4_convergence_all_runs.tsv",
        "out/paper_integrated_evidence/tables/a4_convergence_summary.tsv",
        "out/paper_integrated_evidence/checks/a4_convergence_manifest.json",
    ]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Results", ""])
    for row in rows:
        lines.append(
            f"- {row['family_slug']}: final val PPL {row['final_val_ppl']}, "
            f"train PPL {row['final_train_ppl']}, epoch {row['final_epoch']}, "
            f"`{row['csv_path']}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    for path in [TABLES, CHECKS, AUDIT]:
        path.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    rows: list[dict[str, str]] = []
    for spec in RUNS:
        try:
            rows.append(validate_run(spec))
        except Exception as exc:  # noqa: BLE001 - collect all artifact failures.
            failures.append(str(exc))
    failures.extend(scan_logs())

    columns = [
        "phase", "family_slug", "family", "implementation", "attention_family", "backend",
        "seed", "lr", "D", "d_ff", "L", "w", "s", "M", "config", "csv_path",
        "json_path", "rows", "final_epoch", "final_train_loss", "final_val_loss",
        "final_train_ppl", "final_val_ppl", "time_per_epoch_s", "peak_vram_mib",
        "resolved_d_phi", "resolved_adapter_type", "pooling_alpha", "hash_seed",
        "hash_normalize", "hash_num_bins", "router_min_temp", "landmark_coverage",
        "landmark_count", "source_csv_sha256",
    ]
    write_tsv(TABLES / "a4_convergence_all_runs.tsv", rows, columns)

    summary_rows = []
    for row in rows:
        summary_rows.append({
            "phase": row["phase"],
            "family_slug": row["family_slug"],
            "family": row["family"],
            "implementation": row["implementation"],
            "backend": row["backend"],
            "lr": row["lr"],
            "seed": row["seed"],
            "final_epoch": row["final_epoch"],
            "final_val_ppl": row["final_val_ppl"],
            "final_train_ppl": row["final_train_ppl"],
            "time_per_epoch_s": row["time_per_epoch_s"],
        })
    if rows:
        vals = [float(row["final_val_ppl"]) for row in rows]
        summary_rows.append({
            "phase": "A4.3",
            "family_slug": "panel_mean",
            "family": "Panel mean",
            "implementation": "mixed",
            "backend": "mixed",
            "lr": LR,
            "seed": str(SEED),
            "final_epoch": str(EPOCHS),
            "final_val_ppl": f"{mean(vals):.6f}",
            "final_train_ppl": "NA",
            "time_per_epoch_s": "NA",
        })
    write_tsv(
        TABLES / "a4_convergence_summary.tsv",
        summary_rows,
        [
            "phase", "family_slug", "family", "implementation", "backend", "lr",
            "seed", "final_epoch", "final_val_ppl", "final_train_ppl", "time_per_epoch_s",
        ],
    )

    generated = [
        "out/paper_integrated_evidence/tables/a4_convergence_all_runs.tsv",
        "out/paper_integrated_evidence/tables/a4_convergence_summary.tsv",
    ]
    manifest = {
        "status": "pass" if not failures else "fail",
        "expected_runs": len(RUNS),
        "validated_runs": len(rows),
        "failures": failures,
        "source_csvs": [
            {"path": row["csv_path"], "sha256": row["source_csv_sha256"]}
            for row in rows
        ],
        "generated": [
            {"path": path, "sha256": sha256(ROOT / path)}
            for path in generated
        ],
    }
    (CHECKS / "a4_convergence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (AUDIT / "A4_3_convergence.md").write_text(markdown_audit(rows, failures))
    print(json.dumps({
        "status": manifest["status"],
        "expected_runs": manifest["expected_runs"],
        "validated_runs": manifest["validated_runs"],
        "failures": failures[:10],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
