#!/usr/bin/env python3
"""Validate and summarize the A6.1 d_phi capacity sweep."""

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


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out" / "paper_mechanisms" / "a6_dphi_capacity"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"
LOG_ROOT = ROOT / "logs" / "a6_dphi_capacity"

SEQ_LEN = 512
WINDOW = 16
STRIDE = 8
M = (SEQ_LEN - WINDOW) // STRIDE + 1
EPOCHS = 10
LR = "1e-4"
DPHIS = [384, 512, 768]
SEEDS = [0, 1, 2]

RESOLVED_KEYS = [
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
    slug: str
    family: str
    backend: str
    config: str
    d_phi: int
    seed: int

    @property
    def group(self) -> str:
        return f"a6_dphi_capacity_{self.slug}_D384_FF1536"

    @property
    def name(self) -> str:
        return (
            f"a6_dphi_{self.slug}_D384_FF1536_dphi{self.d_phi}_w{WINDOW}_s{STRIDE}_"
            f"lr{LR.replace('.', 'p')}_seed{self.seed}"
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
        return str(max(round(0.25 * M), 2))


RUN_FAMILIES = [
    ("set_dense_exact", "Set Dense", "exact", "configs/paper_lr_norm/set_dense_exact.yaml"),
    (
        "set_sparse_local_band",
        "Set Sparse",
        "local_band",
        "configs/paper_lr_norm/set_sparse_local_band.yaml",
    ),
    (
        "set_linear_landmark",
        "Set Linear",
        "landmark",
        "configs/paper_lr_norm/set_linear_landmark.yaml",
    ),
]


def expected_runs() -> list[ExpectedRun]:
    return [
        ExpectedRun(slug, family, backend, config, d_phi, seed)
        for slug, family, backend, config in RUN_FAMILIES
        for d_phi in DPHIS
        for seed in SEEDS
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


def float_or_none(raw: object) -> float | None:
    if raw in {None, "", "NA", "None", "nan"}:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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


def validate_run(run_spec: ExpectedRun) -> dict[str, str]:
    if not run_spec.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {run_spec.csv_path}")
    if not run_spec.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {run_spec.json_path}")

    rows = read_rows(run_spec.csv_path)
    if len(rows) != EPOCHS:
        raise ValueError(f"{run_spec.csv_path} has {len(rows)} rows, expected {EPOCHS}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, EPOCHS + 1)):
        raise ValueError(f"{run_spec.csv_path} epochs are not 1..{EPOCHS}: {epochs}")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite CSV values in {run_spec.csv_path}: {bad[:5]}")

    meta = json.loads(run_spec.json_path.read_text())
    missing = [key for key in RESOLVED_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{run_spec.json_path} missing metadata keys: {missing}")
    checks = {
        "model.implementation": "set_only",
        "model.set_causality_mode": "strict_past",
        "model.d_model": 384,
        "model.dim_feedforward": 1536,
        "model.max_seq_len": SEQ_LEN,
        "model.window_size": WINDOW,
        "model.stride": STRIDE,
        "model.d_phi": run_spec.d_phi,
        "resolved.d_phi": run_spec.d_phi,
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if str(actual) != str(expected):
            raise ValueError(f"{run_spec.json_path} has {key}={actual!r}, expected {expected!r}")
    if run_spec.backend == "landmark":
        if str(meta.get("model.backend_params.landmark_coverage")) != "0.25":
            raise ValueError(f"{run_spec.json_path} missing landmark_coverage=0.25")
        if str(meta.get("resolved.landmark_count")) != run_spec.landmark_count:
            raise ValueError(
                f"{run_spec.json_path} landmark_count={meta.get('resolved.landmark_count')}, "
                f"expected {run_spec.landmark_count}"
            )

    last = rows[-1]
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl"]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{run_spec.csv_path} missing finite final {key}")
    for key in [
        "ausa/candidate_count_mean",
        "ausa/candidate_count_max",
        "ausa/router_entropy_norm",
        "ausa/router_top1_weight",
    ]:
        value = float_or_none(last.get(key))
        if value is None:
            raise ValueError(f"{run_spec.csv_path} missing finite {key}")
        if key.endswith(("entropy_norm", "top1_weight")) and not (-1e-6 <= value <= 1.000001):
            raise ValueError(f"{run_spec.csv_path} has out-of-range {key}={value}")

    return {
        "phase": "A6.1",
        "family_slug": run_spec.slug,
        "family": run_spec.family,
        "backend": run_spec.backend,
        "seed": str(run_spec.seed),
        "lr": LR,
        "D": "384",
        "d_ff": "1536",
        "L": str(SEQ_LEN),
        "w": str(WINDOW),
        "s": str(STRIDE),
        "M": str(M),
        "d_phi": str(run_spec.d_phi),
        "config": run_spec.config,
        "csv_path": str(run_spec.csv_path.relative_to(ROOT)),
        "json_path": str(run_spec.json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_train_loss": last["train/loss"],
        "final_val_loss": last["val/loss"],
        "final_train_ppl": last["train/ppl"],
        "final_val_ppl": last["val/ppl"],
        "time_per_epoch_s": last.get("time/epoch_s", "NA"),
        "peak_vram_mib": last.get("system/peak_vram_allocated_mib", "NA"),
        "resolved_d_phi": str(meta.get("resolved.d_phi")),
        "resolved_adapter_type": str(meta.get("resolved.adapter_type")),
        "pooling_alpha": str(meta.get("resolved.pooling_alpha")),
        "hash_seed": str(meta.get("resolved.hash_seed")),
        "hash_normalize": str(meta.get("resolved.hash_normalize")),
        "hash_num_bins": str(meta.get("resolved.hash_num_bins")),
        "router_min_temp": str(meta.get("resolved.router_min_temp")),
        "landmark_coverage": str(meta.get("resolved.landmark_coverage", "NA")),
        "landmark_count": str(meta.get("resolved.landmark_count", "NA")),
        "candidate_count_mean": last.get("ausa/candidate_count_mean", "NA"),
        "candidate_count_max": last.get("ausa/candidate_count_max", "NA"),
        "source_csv_sha256": sha256(run_spec.csv_path),
    }


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["backend"], row["d_phi"])].append(row)

    summary: list[dict[str, str]] = []
    for (family, backend, d_phi), group_rows in sorted(grouped.items()):
        ppls = [float(row["final_val_ppl"]) for row in group_rows]
        losses = [float(row["final_val_loss"]) for row in group_rows]
        summary.append({
            "phase": "A6.1",
            "family": family,
            "backend": backend,
            "d_phi": d_phi,
            "lr": LR,
            "n": str(len(group_rows)),
            "mean_final_val_ppl": f"{mean(ppls):.6f}",
            "std_final_val_ppl": f"{pstdev(ppls):.6f}" if len(ppls) > 1 else "0.000000",
            "min_final_val_ppl": f"{min(ppls):.6f}",
            "max_final_val_ppl": f"{max(ppls):.6f}",
            "mean_final_val_loss": f"{mean(losses):.6f}",
        })
    return summary


def write_audit(manifest: dict[str, object], summary_rows: list[dict[str, str]]) -> None:
    best_by_family: dict[str, dict[str, str]] = {}
    for row in summary_rows:
        current = best_by_family.get(row["family"])
        if current is None or float(row["mean_final_val_ppl"]) < float(current["mean_final_val_ppl"]):
            best_by_family[row["family"]] = row

    lines = [
        "# A6.1 d_phi Capacity Sweep",
        "",
        f"Status: {manifest['status'].upper()}",
        "",
        "## Scope",
        "",
        "A6.1 tests whether increasing SKA interface capacity via `model.d_phi` improves "
        "performance while holding token model width fixed.",
        "",
        "Fixed setup: D=384, d_ff=1536, L=512, w=16, s=8, M=63, strict_past, "
        "10 epochs, seeds 0/1/2, LR=1e-4. Linear uses landmark_coverage=0.25.",
        "",
        "## Summary",
        "",
        "| family | backend | d_phi | n | mean val PPL | std | min | max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['family']} | {row['backend']} | {row['d_phi']} | {row['n']} | "
            f"{row['mean_final_val_ppl']} | {row['std_final_val_ppl']} | "
            f"{row['min_final_val_ppl']} | {row['max_final_val_ppl']} |"
        )
    lines.extend([
        "",
        "## Best d_phi by Family",
        "",
        "| family | backend | best d_phi | mean val PPL |",
        "| --- | --- | ---: | ---: |",
    ])
    for family, row in sorted(best_by_family.items()):
        lines.append(
            f"| {family} | {row['backend']} | {row['d_phi']} | {row['mean_final_val_ppl']} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- All runs TSV: `out/paper_integrated_evidence/tables/a6_dphi_capacity_all_runs.tsv`",
        "- Summary TSV: `out/paper_integrated_evidence/tables/a6_dphi_capacity_summary.tsv`",
        "- Manifest: `out/paper_integrated_evidence/checks/a6_dphi_capacity_manifest.json`",
        "",
        "## Validation",
        "",
        f"- Expected runs: {manifest['expected_runs']}",
        f"- Validated runs: {manifest['validated_runs']}",
        f"- Log failures: {len(manifest.get('log_failures', []))}",
        f"- Failures: {len(manifest.get('failures', []))}",
    ])
    if manifest.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in manifest["failures"])
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "A6_1_dphi_capacity.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    rows: list[dict[str, str]] = []
    for run_spec in expected_runs():
        try:
            rows.append(validate_run(run_spec))
        except Exception as exc:  # noqa: BLE001
            failures.append(str(exc))

    log_failures = scan_logs()
    failures.extend(log_failures)

    all_runs_path = TABLES / "a6_dphi_capacity_all_runs.tsv"
    summary_path = TABLES / "a6_dphi_capacity_summary.tsv"
    manifest_path = CHECKS / "a6_dphi_capacity_manifest.json"

    all_fields = [
        "phase", "family_slug", "family", "backend", "seed", "lr", "D", "d_ff", "L",
        "w", "s", "M", "d_phi", "config", "csv_path", "json_path", "rows",
        "final_train_loss", "final_val_loss", "final_train_ppl", "final_val_ppl",
        "time_per_epoch_s", "peak_vram_mib", "resolved_d_phi", "resolved_adapter_type",
        "pooling_alpha", "hash_seed", "hash_normalize", "hash_num_bins",
        "router_min_temp", "landmark_coverage", "landmark_count",
        "candidate_count_mean", "candidate_count_max", "source_csv_sha256",
    ]
    if rows:
        write_tsv(all_runs_path, rows, all_fields)
        summary_rows = summarize(rows)
        write_tsv(
            summary_path,
            summary_rows,
            [
                "phase", "family", "backend", "d_phi", "lr", "n",
                "mean_final_val_ppl", "std_final_val_ppl", "min_final_val_ppl",
                "max_final_val_ppl", "mean_final_val_loss",
            ],
        )
    else:
        summary_rows = []

    manifest = {
        "status": "pass" if not failures and len(rows) == len(expected_runs()) else "fail",
        "phase": "A6.1",
        "expected_runs": len(expected_runs()),
        "validated_runs": len(rows),
        "failures": failures,
        "log_failures": log_failures,
        "all_runs_tsv": str(all_runs_path.relative_to(ROOT)),
        "summary_tsv": str(summary_path.relative_to(ROOT)),
        "all_runs_sha256": sha256(all_runs_path) if all_runs_path.exists() else None,
        "summary_sha256": sha256(summary_path) if summary_path.exists() else None,
        "git": {
            "branch": run(["git", "branch", "--show-current"]),
            "head": run(["git", "rev-parse", "HEAD"]),
            "status_short": run(["git", "status", "--short"]),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_audit(manifest, summary_rows)
    print(json.dumps({
        "status": manifest["status"],
        "validated_runs": manifest["validated_runs"],
        "expected_runs": manifest["expected_runs"],
        "failures": failures[:10],
        "summary": str(summary_path.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
    }, indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
