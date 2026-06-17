#!/usr/bin/env python3
"""Validate and summarize the A3.2 pooling-temperature sweep."""

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
RAW = ROOT / "out" / "paper_mechanisms" / "a3_pooltau_sweep"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"
LOG_ROOTS = [
    ROOT / "logs" / "a3_pooltau_sweep",
    ROOT / "logs" / "a3_pooltau_sweep_high_tau",
]

LR = "1e-4"
LR_TAG = "1e-4"
SEQ_LEN = 512
WINDOW = 16
STRIDE = 4
EXPECTED_M = ((SEQ_LEN - WINDOW) // STRIDE) + 1
TAUS = ["0.05", "0.1", "0.2", "0.5", "0.95"]
SEEDS = [0, 1, 2]
FAMILIES = [
    ("dense_exact", "Set Dense", "exact", "configs/paper_complements/family_dense_exact.yaml"),
    ("sparse_local_band", "Set Sparse", "local_band", "configs/paper_complements/family_sparse_local_band.yaml"),
    ("linear_landmark", "Set Linear", "landmark", "configs/paper_complements/family_linear_landmark.yaml"),
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


@dataclass(frozen=True)
class ExpectedRun:
    family_slug: str
    family: str
    backend: str
    config: str
    tau: str
    seed: int

    @property
    def landmark_count(self) -> str:
        if self.backend != "landmark":
            return "NA"
        return str(max(round(0.25 * EXPECTED_M), 2))

    @property
    def group(self) -> str:
        return f"a3_pooltau_sweep_{self.family_slug}_D384_FF1536_s4_w16"

    @property
    def name(self) -> str:
        tau_tag = self.tau.replace(".", "p")
        return (
            f"a3_pooltau_{self.family_slug}_D384_FF1536_w16_s4_tau{tau_tag}_"
            f"lr{LR_TAG.replace('.', 'p')}_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return RAW / self.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


def expected_runs() -> list[ExpectedRun]:
    return [
        ExpectedRun(slug, family, backend, config, tau, seed)
        for slug, family, backend, config in FAMILIES
        for tau in TAUS
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
        raise ValueError(f"{run.json_path} has wrong window_size")
    if int(meta.get("model.stride")) != STRIDE:
        raise ValueError(f"{run.json_path} has wrong stride")
    tau = float_or_none(meta.get("model.pooling.tau"))
    if tau is None or abs(tau - float(run.tau)) > 1e-12:
        raise ValueError(f"{run.json_path} has wrong model.pooling.tau={tau}")
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
    candidate_mean = value(
        last,
        "ausa/candidate_count_mean",
        "ausa/router_candidate_count_struct_mean",
    )
    candidate_max = value(
        last,
        "ausa/candidate_count_max",
        "ausa/router_candidate_count_struct_max",
    )
    if float_or_none(candidate_mean) is None:
        raise ValueError(f"{run.csv_path} missing finite candidate-count mean")
    if float_or_none(candidate_max) is None:
        raise ValueError(f"{run.csv_path} missing finite candidate-count max")
    for key in RANGE_KEYS:
        val = float_or_none(last.get(key))
        if val is not None and not (-1e-6 <= val <= 1.000001):
            raise ValueError(f"{run.csv_path} has out-of-range {key}={val}")
    for key in [
        "train/loss",
        "val/loss",
        "train/ppl",
        "val/ppl",
        "ausa/pooling_neff_ratio",
        "ausa/pooling_effective_support",
    ]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{run.csv_path} missing finite final {key}")

    return {
        "phase": "A3.2",
        "family_slug": run.family_slug,
        "family": run.family,
        "backend": run.backend,
        "seed": str(run.seed),
        "lr": LR,
        "D": "384",
        "d_ff": "1536",
        "w": str(WINDOW),
        "s": str(STRIDE),
        "M": str(EXPECTED_M),
        "tau_pool": run.tau,
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
        "pooling_neff_ratio": value(last, "ausa/pooling_neff_ratio"),
        "pooling_neff_l2": value(last, "ausa/pooling_neff_l2"),
        "pooling_effective_support": value(last, "ausa/pooling_effective_support"),
        "grad_ratio_pool_rho_p": value(last, "ausa/grad_ratio_pool_rho_p"),
        "grad_ratio_set_stack_rho_a": value(last, "ausa/grad_ratio_set_stack_rho_a"),
        "grad_ratio_total_rho_pa": value(last, "ausa/grad_ratio_total_rho_pa"),
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
        grouped[(row["family"], row["family_slug"], row["backend"], row["tau_pool"])].append(row)
    out: list[dict[str, str]] = []
    for (family, slug, backend, tau), group in sorted(grouped.items()):
        ppls = [float(row["final_val_ppl"]) for row in group]
        neff = [float(row["pooling_neff_ratio"]) for row in group]
        support = [float(row["pooling_effective_support"]) for row in group]
        cand = [float(row["candidate_count_mean"]) for row in group]
        times = [float(row["time_per_epoch_s"]) for row in group if row["time_per_epoch_s"] != "NA"]
        out.append({
            "phase": "A3.2",
            "family": family,
            "family_slug": slug,
            "backend": backend,
            "tau_pool": tau,
            "w": str(WINDOW),
            "s": str(STRIDE),
            "M": str(EXPECTED_M),
            "runs": str(len(group)),
            "seeds": ",".join(sorted(row["seed"] for row in group)),
            "val_ppl_mean": f"{mean(ppls):.6f}",
            "val_ppl_std": f"{pstdev(ppls):.6f}" if len(ppls) > 1 else "0.000000",
            "val_ppl_min": f"{min(ppls):.6f}",
            "val_ppl_max": f"{max(ppls):.6f}",
            "pooling_neff_ratio_mean": f"{mean(neff):.6f}",
            "pooling_neff_ratio_std": f"{pstdev(neff):.6f}" if len(neff) > 1 else "0.000000",
            "pooling_effective_support_mean": f"{mean(support):.6f}",
            "pooling_effective_support_std": (
                f"{pstdev(support):.6f}" if len(support) > 1 else "0.000000"
            ),
            "candidate_count_mean": f"{mean(cand):.6f}",
            "time_per_epoch_s_mean": f"{mean(times):.6f}" if times else "NA",
        })
    return out


def scan_logs() -> list[str]:
    log_roots = [root for root in LOG_ROOTS if root.exists()]
    if not log_roots:
        return [f"missing log roots: {', '.join(str(root) for root in LOG_ROOTS)}"]
    # Unambiguous error substrings — safe to match anywhere in log text.
    substr_patterns = ["OOM", "out of memory", "Traceback", "RuntimeError", "ValueError"]
    # Standalone numeric-token pattern: match nan/NaN/inf/-inf only when NOT
    # adjacent to alphanumeric chars or underscores (avoids "channel", "planning", etc.).
    token_re = re.compile(r"(?<![A-Za-z0-9_])(?:nan|NaN|-inf|inf)(?![A-Za-z0-9_])")
    failures: list[str] = []
    for root in log_roots:
        for path in sorted(root.glob("*.log")):
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


def run(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }


def markdown_audit(rows: list[dict[str, str]], summary: list[dict[str, str]], failures: list[str]) -> str:
    prelaunch_path = AUDIT / "A3_2_pooltau_sweep_prelaunch.json"
    prelaunch = json.loads(prelaunch_path.read_text()) if prelaunch_path.exists() else {}
    lines = [
        "# A3.2 Pooling-Temperature Sweep Audit",
        "",
        "Status: PASS" if not failures else "Status: FAIL",
        "",
        "## Scope",
        "",
        "- Families: SetDense/exact, SetSparse/local_band, SetLinear/landmark.",
        f"- Pooling temperatures: `tau_pool in {{{', '.join(TAUS)}}}`.",
        "- Seeds: `0,1,2` for every family/tau pair.",
        "- Anchor topology reference: `D=384,d_ff=1536,L=512,w=16,s=4`, strict-past, post-T1 `M=125`.",
        "- Linear uses landmark backend with `landmark_coverage=0.25`, so `resolved.landmark_count=31`.",
        "- A2.2/A2.3 remain the locked `s=8` LR-normalized headline/family grid; they were not rerun or overridden.",
        "",
        "## Commands / Scripts",
        "",
        "- `bash scripts/run_a3_pooltau_sweep.sh`",
        "- `bash scripts/run_a3_pooltau_high_tau_extension.sh`",
        "- `python scripts/summarize_a3_pooltau_sweep.py`",
        "",
        "## Prelaunch State",
        "",
        f"- Branch: `{_first(prelaunch.get('branch'))}`",
        f"- HEAD: `{_first(prelaunch.get('head'))}`",
        f"- A3.1 manifest: `{prelaunch.get('a3_1_manifest_status')}` with `{prelaunch.get('a3_1_validated_runs')}` / `{prelaunch.get('a3_1_expected_runs')}` runs.",
        f"- A3.1 audit: `{prelaunch.get('a3_1_audit_status_line')}`",
        "",
        "## Failures / Retries",
        "",
    ]
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- None.")
    lines.extend(["", "## Run Artifacts", ""])
    artifact_cols = [
        "family", "backend", "tau_pool", "seed", "lr", "w", "s", "M", "rows",
        "final_val_ppl", "time_per_epoch_s", "pooling_neff_ratio",
        "pooling_effective_support", "candidate_count_mean", "config",
        "csv_path", "source_csv_sha256",
    ]
    lines.append("| " + " | ".join(artifact_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(artifact_cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "NA")) for col in artifact_cols) + " |")
    lines.extend(["", "## Summary", ""])
    summary_cols = [
        "family", "backend", "tau_pool", "runs", "seeds",
        "val_ppl_mean", "val_ppl_std", "pooling_neff_ratio_mean",
        "pooling_neff_ratio_std", "pooling_effective_support_mean",
    ]
    lines.append("| " + " | ".join(summary_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(summary_cols)) + " |")
    for row in summary:
        lines.append("| " + " | ".join(str(row.get(col, "NA")) for col in summary_cols) + " |")
    lines.extend([
        "",
        "## Generated Artifacts",
        "",
        "- `out/paper_integrated_evidence/tables/a3_pooltau_sweep_all_runs.tsv`",
        "- `out/paper_integrated_evidence/tables/a3_pooltau_sweep_summary.tsv`",
        "- `out/paper_integrated_evidence/checks/a3_pooltau_sweep_manifest.json`",
        "",
        "## Recommendation For B5",
        "",
        "Use `a3_pooltau_sweep_summary.tsv` for error-bar plots/tables over pooling temperature and `a3_pooltau_sweep_all_runs.tsv` for provenance.",
        "",
    ])
    return "\n".join(lines)


def _first(obj: object) -> str:
    if isinstance(obj, dict):
        stdout = obj.get("stdout")
        if isinstance(stdout, list) and stdout:
            return str(stdout[0])
    return "NA"


def main() -> None:
    for path in [TABLES, CHECKS, AUDIT]:
        path.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    rows: list[dict[str, str]] = []
    for item in expected_runs():
        try:
            rows.append(validate_run(item))
        except Exception as exc:  # noqa: BLE001 - collect every bad artifact.
            failures.append(str(exc))
    failures.extend(scan_logs())

    rows.sort(key=lambda r: (r["family_slug"], float(r["tau_pool"]), int(r["seed"])))
    summary = summarize(rows) if rows else []
    if len(summary) != len(FAMILIES) * len(TAUS):
        failures.append(f"summary has {len(summary)} rows, expected {len(FAMILIES) * len(TAUS)}")

    columns = [
        "phase", "family_slug", "family", "backend", "seed", "lr", "D", "d_ff",
        "w", "s", "M", "tau_pool", "expected_landmark_count", "config",
        "csv_path", "json_path", "rows", "final_train_loss", "final_val_loss",
        "final_train_ppl", "final_val_ppl", "time_per_epoch_s", "peak_vram_mib",
        "candidate_count_mean", "candidate_count_max", "router_entropy_norm",
        "router_top1_weight", "router_top1_gap_norm", "pooling_neff_ratio",
        "pooling_neff_l2", "pooling_effective_support", "grad_ratio_pool_rho_p",
        "grad_ratio_set_stack_rho_a", "grad_ratio_total_rho_pa", "resolved_d_phi",
        "resolved_adapter_type", "pooling_alpha", "hash_seed", "hash_normalize",
        "hash_num_bins", "router_min_temp", "landmark_coverage", "landmark_count",
        "source_csv_sha256",
    ]
    summary_cols = [
        "phase", "family", "family_slug", "backend", "tau_pool", "w", "s", "M",
        "runs", "seeds", "val_ppl_mean", "val_ppl_std", "val_ppl_min",
        "val_ppl_max", "pooling_neff_ratio_mean", "pooling_neff_ratio_std",
        "pooling_effective_support_mean", "pooling_effective_support_std",
        "candidate_count_mean", "time_per_epoch_s_mean",
    ]
    all_runs_path = TABLES / "a3_pooltau_sweep_all_runs.tsv"
    summary_path = TABLES / "a3_pooltau_sweep_summary.tsv"
    write_tsv(all_runs_path, rows, columns)
    write_tsv(summary_path, summary, summary_cols)

    generated_paths = [
        "out/paper_integrated_evidence/tables/a3_pooltau_sweep_all_runs.tsv",
        "out/paper_integrated_evidence/tables/a3_pooltau_sweep_summary.tsv",
    ]
    manifest = {
        "status": "pass" if not failures else "fail",
        "expected_runs": len(expected_runs()),
        "validated_runs": len(rows),
        "summary_rows": len(summary),
        "failures": failures,
        "config": {
            "D": 384,
            "d_ff": 1536,
            "L": SEQ_LEN,
            "w": WINDOW,
            "s": STRIDE,
            "M": EXPECTED_M,
            "tau_pool": TAUS,
            "seeds": SEEDS,
            "set_causality_mode": "strict_past",
            "landmark_coverage": 0.25,
        },
        "source_csvs": [
            {"path": row["csv_path"], "sha256": row["source_csv_sha256"]}
            for row in rows
        ],
        "generated": [
            {"path": path, "sha256": sha256(ROOT / path)}
            for path in generated_paths
        ],
        "git": {
            "branch": run(["git", "branch", "--show-current"]),
            "head": run(["git", "rev-parse", "HEAD"]),
            "status_short": run(["git", "status", "--short"]),
        },
    }
    manifest_path = CHECKS / "a3_pooltau_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (AUDIT / "A3_2_pooltau_sweep.md").write_text(markdown_audit(rows, summary, failures))

    if failures:
        raise SystemExit("\n".join(failures[:20]))
    print(json.dumps({"status": "pass", "validated_runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
