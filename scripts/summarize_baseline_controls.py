#!/usr/bin/env python3
"""Validate and summarize v2.7 matched token-backend baseline controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"

FAMILIES = [
    (
        "baseline_sparse_local_band",
        "Baseline Sparse",
        "sparse",
        "local_band",
        "configs/paper_lr_norm/baseline_sparse_local_band.yaml",
    ),
    (
        "baseline_linear_landmark",
        "Baseline Linear",
        "linear",
        "landmark",
        "configs/paper_lr_norm/baseline_linear_landmark.yaml",
    ),
]


@dataclass(frozen=True)
class ExpectedRun:
    phase: str
    family_slug: str
    family: str
    attention_family: str
    backend: str
    config: str
    seed: int
    lr: str
    seq_len: int
    window: int
    stride: int
    raw_root: Path
    group_prefix: str
    name_prefix: str
    d_model: int = 384
    d_ff: int = 1536

    @property
    def m(self) -> int:
        return ((self.seq_len - self.window) // self.stride) + 1

    @property
    def landmark_count(self) -> str:
        if self.backend != "landmark":
            return "NA"
        return str(max(round(0.25 * self.seq_len), 2))

    @property
    def group(self) -> str:
        if self.phase == "A2.4":
            return f"{self.group_prefix}_{self.family_slug}_D384_FF1536"
        if self.phase == "A3.1-control":
            return f"{self.group_prefix}_{self.family_slug}_D384_FF1536_s{self.stride}"
        return f"{self.group_prefix}_{self.family_slug}_D384_FF1536_L{self.seq_len}"

    @property
    def name(self) -> str:
        lr_tag = self.lr.replace(".", "p")
        if self.phase == "A2.4":
            return (
                f"{self.name_prefix}_{self.family_slug}_D384_FF1536_L{self.seq_len}"
                f"_w{self.window}_s{self.stride}_lr{lr_tag}_seed{self.seed}"
            )
        if self.phase == "A3.1-control":
            return (
                f"{self.name_prefix}_{self.family_slug}_D384_FF1536_w{self.window}"
                f"_s{self.stride}_lr{lr_tag}_seed{self.seed}"
            )
        return (
            f"{self.name_prefix}_{self.family_slug}_D384_FF1536_L{self.seq_len}"
            f"_w{self.window}_s{self.stride}_lr{lr_tag}_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return self.raw_root / self.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


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


def scan_logs(log_root: Path) -> list[str]:
    if not log_root.exists():
        return [f"missing log root: {log_root.relative_to(ROOT)}"]
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
    for path in sorted(log_root.glob("*.log")):
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


def seeds_for_window(window: int) -> list[int]:
    return [0, 1, 2] if window in {6, 16, 24} else [0]


def expected_runs(phase: str) -> list[ExpectedRun]:
    if phase == "a2":
        raw_root = ROOT / "out" / "paper_lr_norm" / "baseline_controls_A2_D384_FF1536"
        runs: list[ExpectedRun] = []
        for slug, family, attn_family, backend, config in FAMILIES:
            for seed in [0, 1, 2]:
                for lr in ["1e-4", "2e-4", "3e-4", "5e-4", "7e-4"]:
                    runs.append(
                        ExpectedRun(
                            "A2.4", slug, family, attn_family, backend, config, seed, lr,
                            512, 16, 8, raw_root, "a2_baseline_controls", "a2_controls",
                        )
                    )
        return runs
    if phase == "a31":
        raw_root = ROOT / "out" / "paper_mechanisms" / "a3_window_baseline_controls"
        runs = []
        for slug, family, attn_family, backend, _config in FAMILIES:
            config = f"configs/paper_complements/{slug}.yaml"
            for window in [6, 8, 12, 16, 20, 24]:
                for seed in seeds_for_window(window):
                    runs.append(
                        ExpectedRun(
                            "A3.1-control", slug, family, attn_family, backend, config, seed,
                            "1e-4", 512, window, 4, raw_root,
                            "a3_window_baseline_controls", "a3_window_controls",
                        )
                    )
        return runs
    if phase == "a42":
        raw_root = ROOT / "out" / "paper_mechanisms" / "a42_baseline_controls"
        configs = {
            "baseline_sparse_local_band": "configs/a4_long_context/baseline_sparse_lc.yaml",
            "baseline_linear_landmark": "configs/a4_long_context/baseline_linear_lc.yaml",
        }
        runs = []
        for slug, family, attn_family, backend, _config in FAMILIES:
            for seed in [0, 1, 2]:
                runs.append(
                    ExpectedRun(
                        "A4.2-control", slug, family, attn_family, backend, configs[slug], seed,
                        "1e-4", 2048, 16, 8, raw_root,
                        "a42_baseline_controls", "a42_controls",
                    )
                )
        return runs
    raise ValueError(f"unknown phase: {phase}")


def paths_for_phase(phase: str) -> dict[str, Path]:
    table_names = {
        "a2": ("a2_baseline_controls_all_runs.tsv", "a2_baseline_controls_summary.tsv"),
        "a31": (
            "a3_window_baseline_controls_all_runs.tsv",
            "a3_window_baseline_controls_summary.tsv",
        ),
        "a42": (
            "a4_long_context_baseline_controls_all_runs.tsv",
            "a4_long_context_baseline_controls_summary.tsv",
        ),
    }
    manifest_names = {
        "a2": "a2_baseline_controls_manifest.json",
        "a31": "a3_window_baseline_controls_manifest.json",
        "a42": "a4_long_context_baseline_controls_manifest.json",
    }
    audit_names = {
        "a2": "A2_4_baseline_controls.md",
        "a31": "A3_1_baseline_controls.md",
        "a42": "A4_2_baseline_controls.md",
    }
    log_roots = {
        "a2": ROOT / "logs" / "a2_baseline_controls",
        "a31": ROOT / "logs" / "a3_window_baseline_controls",
        "a42": ROOT / "logs" / "a42_baseline_controls",
    }
    all_name, summary_name = table_names[phase]
    return {
        "all": TABLES / all_name,
        "summary": TABLES / summary_name,
        "manifest": CHECKS / manifest_names[phase],
        "audit": AUDIT / audit_names[phase],
        "logs": log_roots[phase],
    }


def validate_run(run_spec: ExpectedRun) -> dict[str, str]:
    if not run_spec.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {run_spec.csv_path}")
    if not run_spec.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {run_spec.json_path}")
    rows = read_rows(run_spec.csv_path)
    if len(rows) != 10:
        raise ValueError(f"{run_spec.csv_path} has {len(rows)} rows, expected 10")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 11)):
        raise ValueError(f"{run_spec.csv_path} epochs are not 1..10: {epochs}")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite CSV values in {run_spec.csv_path}: {bad[:5]}")
    meta = json.loads(run_spec.json_path.read_text())
    if meta.get("model.implementation") != "baseline_token":
        raise ValueError(f"{run_spec.json_path} is not a baseline_token run")
    if meta.get("model.attention_family") != run_spec.attention_family:
        raise ValueError(f"{run_spec.json_path} has wrong attention_family")
    if meta.get("model.backend") != run_spec.backend:
        raise ValueError(f"{run_spec.json_path} has wrong backend")
    if str(meta.get("model.causal")).lower() != "true":
        raise ValueError(f"{run_spec.json_path} is not causal")
    if int(meta.get("data.seq_len")) != run_spec.seq_len:
        raise ValueError(f"{run_spec.json_path} has wrong seq_len")
    if int(meta.get("model.max_seq_len")) != run_spec.seq_len:
        raise ValueError(f"{run_spec.json_path} has wrong max_seq_len")
    if run_spec.backend == "local_band":
        if str(meta.get("model.backend_params.radius")) != "4":
            raise ValueError(f"{run_spec.json_path} does not record local_band radius=4")
    if run_spec.backend == "landmark":
        coverage = float_or_none(meta.get("model.backend_params.landmark_coverage"))
        resolved_coverage = float_or_none(meta.get("resolved.landmark_coverage"))
        count = str(meta.get("resolved.landmark_count", "NA"))
        if coverage != 0.25 or resolved_coverage != 0.25:
            raise ValueError(f"{run_spec.json_path} does not record landmark_coverage=0.25")
        if count != run_spec.landmark_count:
            raise ValueError(
                f"{run_spec.json_path} landmark_count={count}, expected {run_spec.landmark_count}"
            )
    last = rows[-1]
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl"]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{run_spec.csv_path} missing finite final {key}")
    return {
        "phase": run_spec.phase,
        "family_slug": run_spec.family_slug,
        "family": run_spec.family,
        "attention_family": run_spec.attention_family,
        "backend": run_spec.backend,
        "seed": str(run_spec.seed),
        "lr": run_spec.lr,
        "D": str(run_spec.d_model),
        "d_ff": str(run_spec.d_ff),
        "L": str(run_spec.seq_len),
        "w": str(run_spec.window),
        "s": str(run_spec.stride),
        "M": str(run_spec.m),
        "config": run_spec.config,
        "csv_path": str(run_spec.csv_path.relative_to(ROOT)),
        "json_path": str(run_spec.json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_train_loss": value(last, "train/loss"),
        "final_val_loss": value(last, "val/loss"),
        "final_train_ppl": value(last, "train/ppl"),
        "final_val_ppl": value(last, "val/ppl"),
        "time_per_epoch_s": value(last, "train/time_per_epoch_s"),
        "peak_vram_mib": value(last, "train/peak_vram_mib"),
        "landmark_coverage": str(meta.get("resolved.landmark_coverage", "NA")),
        "landmark_count": str(meta.get("resolved.landmark_count", "NA")),
        "source_csv_sha256": sha256(run_spec.csv_path),
    }


def write_tsv(path: Path, rows: Iterable[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], phase: str) -> list[dict[str, str]]:
    if phase == "a2":
        keys = ["phase", "family", "attention_family", "backend", "lr"]
    elif phase == "a31":
        keys = ["phase", "family", "attention_family", "backend", "w", "s", "M"]
    else:
        keys = ["phase", "family", "attention_family", "backend", "L", "w", "s", "M"]
    groups = sorted({tuple(row[key] for key in keys) for row in rows})
    out: list[dict[str, str]] = []
    for group in groups:
        members = [row for row in rows if tuple(row[key] for key in keys) == group]
        ppls = [float(row["final_val_ppl"]) for row in members]
        vals = dict(zip(keys, group))
        vals.update({
            "n": str(len(members)),
            "mean_final_val_ppl": f"{mean(ppls):.6f}",
            "std_final_val_ppl": f"{pstdev(ppls):.6f}" if len(ppls) > 1 else "0.000000",
            "min_final_val_ppl": f"{min(ppls):.6f}",
            "max_final_val_ppl": f"{max(ppls):.6f}",
        })
        out.append(vals)
    return out


def markdown_audit(
    phase: str,
    rows: list[dict[str, str]],
    failures: list[str],
    paths: dict[str, Path],
    log_failures: list[str],
) -> str:
    phase_names = {
        "a2": "A2.4 Baseline Controls",
        "a31": "A3.1 Baseline Controls",
        "a42": "A4.2 Baseline Controls",
    }
    lines = [
        f"# {phase_names[phase]}",
        "",
        "Status: PASS" if not failures and not log_failures else "Status: FAIL",
        "",
        f"Expected runs: {len(expected_runs(phase))}",
        f"Validated runs: {len(rows)}",
        "",
        "## Provenance",
        "",
        f"- Branch: `{run(['git', 'branch', '--show-current'])['stdout'][0]}`",
        f"- HEAD: `{run(['git', 'rev-parse', 'HEAD'])['stdout'][0]}`",
        f"- Dirty entries: {len(run(['git', 'status', '--short'])['stdout'])}",
        "",
        "## Validation",
        "",
    ]
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- All expected CSV/JSON artifacts exist and completed 10 epochs.")
        lines.append("- CSV metrics are finite.")
        lines.append("- Logs contain no OOM, traceback, standalone nan/inf, or W&B step warnings.")
        lines.append("- Baseline sparse records causal local_band radius=4.")
        lines.append("- Baseline linear records landmark_coverage=0.25 and resolved landmark_count.")
    lines.extend(["", "## Summary Artifacts", ""])
    for key in ["all", "summary", "manifest"]:
        lines.append(f"- `{paths[key].relative_to(ROOT)}`")
    lines.extend(["", "## Run Artifacts", ""])
    for row in rows:
        lines.append(
            f"- {row['phase']} {row['family_slug']} seed={row['seed']} lr={row['lr']} "
            f"w={row['w']} L={row['L']} val_ppl={row['final_val_ppl']} "
            f"`{row['csv_path']}` sha256={row['source_csv_sha256']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a2", "a31", "a42"], required=True)
    args = parser.parse_args()

    for path in [TABLES, CHECKS, AUDIT]:
        path.mkdir(parents=True, exist_ok=True)

    paths = paths_for_phase(args.phase)
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for run_spec in expected_runs(args.phase):
        try:
            rows.append(validate_run(run_spec))
        except Exception as exc:  # noqa: BLE001 - collect all artifact failures.
            failures.append(str(exc))
    log_failures = scan_logs(paths["logs"])
    failures.extend(log_failures)

    all_columns = [
        "phase", "family_slug", "family", "attention_family", "backend", "seed", "lr",
        "D", "d_ff", "L", "w", "s", "M", "config", "csv_path", "json_path", "rows",
        "final_train_loss", "final_val_loss", "final_train_ppl", "final_val_ppl",
        "time_per_epoch_s", "peak_vram_mib", "landmark_coverage", "landmark_count",
        "source_csv_sha256",
    ]
    summary_rows = summarize(rows, args.phase) if rows else []
    summary_columns = sorted({key for row in summary_rows for key in row.keys()})
    write_tsv(paths["all"], rows, all_columns)
    write_tsv(paths["summary"], summary_rows, summary_columns)

    generated = [paths["all"], paths["summary"]]
    manifest = {
        "status": "pass" if not failures else "fail",
        "phase": args.phase,
        "expected_runs": len(expected_runs(args.phase)),
        "validated_runs": len(rows),
        "failures": failures,
        "source_csvs": [
            {"path": row["csv_path"], "sha256": row["source_csv_sha256"]}
            for row in rows
        ],
        "generated": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in generated
        ],
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    paths["audit"].write_text(markdown_audit(args.phase, rows, failures, paths, log_failures))
    print(json.dumps({
        "status": manifest["status"],
        "phase": args.phase,
        "expected_runs": manifest["expected_runs"],
        "validated_runs": manifest["validated_runs"],
        "failures": failures[:10],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
