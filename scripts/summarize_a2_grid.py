#!/usr/bin/env python3
"""Validate and summarize the A2 LR-normalized multi-seed grid."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PAPER_LR = ROOT / "out" / "paper_lr_norm"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"

LRS = ["1e-4", "2e-4", "3e-4", "5e-4", "7e-4"]
SEEDS = ["0", "1", "2"]
SPECS = [
    ("D384_FF1536", "384", "1536"),
    ("D512_FF2048", "512", "2048"),
    ("D384_FF3072", "384", "3072"),
    ("D512_FF1024", "512", "1024"),
]

RESOLVED_KEYS = [
    "resolved.d_phi",
    "resolved.adapter_type",
    "resolved.router_min_temp",
    "resolved.pooling_alpha",
    "resolved.hash_seed",
    "resolved.hash_normalize",
    "resolved.hash_num_bins",
    "resolved.landmark_coverage",
    "resolved.landmark_count",
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


@dataclass(frozen=True)
class ExpectedRun:
    phase: str
    group: str
    name: str
    config: str
    family: str
    backend: str
    seed: str
    lr: str
    d_model: str
    d_ff: str
    window: str
    stride: str

    @property
    def csv_path(self) -> Path:
        return PAPER_LR / self.group / f"{self.name}.csv"

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


def f(row: dict[str, str], key: str) -> float | None:
    val = row.get(key, "NA")
    if val in {"", "NA", "None", "nan"}:
        return None
    return float(val)


def fmt(val: float | None, digits: int = 6) -> str:
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def s(value: object) -> str:
    return str(value)


def lr_tag(lr: str) -> str:
    return lr.replace(".", "p")


def expected_runs() -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []
    for spec, d_model, d_ff in SPECS:
        group = f"paper_lr_norm_headline_A2_{spec}"
        for seed in SEEDS:
            for lr in LRS:
                tag = lr_tag(lr)
                runs.append(
                    ExpectedRun(
                        phase="A2.2",
                        group=group,
                        name=f"paper_lrnorm_baseline_{spec}_lr{tag}_seed{seed}",
                        config="configs/paper_lr_norm/baseline_dense_exact.yaml",
                        family="Baseline token",
                        backend="exact",
                        seed=seed,
                        lr=lr,
                        d_model=d_model,
                        d_ff=d_ff,
                        window="NA",
                        stride="NA",
                    )
                )
                runs.append(
                    ExpectedRun(
                        phase="A2.2",
                        group=group,
                        name=f"paper_lrnorm_set_dense_{spec}_lr{tag}_seed{seed}",
                        config="configs/paper_lr_norm/set_dense_exact.yaml",
                        family="Set Dense",
                        backend="exact",
                        seed=seed,
                        lr=lr,
                        d_model=d_model,
                        d_ff=d_ff,
                        window="16",
                        stride="8",
                    )
                )
    family_group = "paper_lr_norm_family_A2_D384_FF1536"
    for seed in SEEDS:
        for lr in LRS:
            tag = lr_tag(lr)
            runs.append(
                ExpectedRun(
                    phase="A2.3",
                    group=family_group,
                    name=f"paper_lrnorm_set_sparse_D384_FF1536_lr{tag}_seed{seed}",
                    config="configs/paper_lr_norm/set_sparse_local_band.yaml",
                    family="Set Sparse",
                    backend="local_band",
                    seed=seed,
                    lr=lr,
                    d_model="384",
                    d_ff="1536",
                    window="16",
                    stride="8",
                )
            )
            runs.append(
                ExpectedRun(
                    phase="A2.3",
                    group=family_group,
                    name=f"paper_lrnorm_set_linear_D384_FF1536_lr{tag}_seed{seed}",
                    config="configs/paper_lr_norm/set_linear_landmark.yaml",
                    family="Set Linear",
                    backend="landmark",
                    seed=seed,
                    lr=lr,
                    d_model="384",
                    d_ff="1536",
                    window="16",
                    stride="8",
                )
            )
    anchor_group = "paper_lr_norm_anchor_A2_D384_FF1536_s4"
    for seed in SEEDS:
        runs.append(
            ExpectedRun(
                phase="A2.1",
                group=anchor_group,
                name=f"paper_lrnorm_anchor_set_dense_D384_FF1536_s4_lr1e-4_seed{seed}",
                config="configs/paper_lr_norm/set_dense_exact.yaml",
                family="Set Dense Anchor",
                backend="exact",
                seed=seed,
                lr="1e-4",
                d_model="384",
                d_ff="1536",
                window="16",
                stride="4",
            )
        )
    return runs


def finite_csv(rows: list[dict[str, str]]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for i, row in enumerate(rows, 1):
        for key, value in row.items():
            if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "-inf"}:
                bad.append(f"row {i}: {key}={value}")
    return not bad, bad


def validate_run(run: ExpectedRun) -> dict[str, str | int | float | bool]:
    if not run.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {run.csv_path}")
    if not run.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {run.json_path}")
    rows = read_rows(run.csv_path)
    if len(rows) != 10:
        raise ValueError(f"{run.csv_path} has {len(rows)} rows, expected 10")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite CSV values in {run.csv_path}: {bad[:5]}")
    meta = json.loads(run.json_path.read_text())
    missing_resolved = [key for key in RESOLVED_KEYS if key not in meta]
    if missing_resolved:
        raise ValueError(f"{run.json_path} missing resolved metadata keys: {missing_resolved}")
    if run.family.startswith("Set"):
        missing_set = [key for key in SET_ONLY_KEYS if key not in meta]
        if missing_set:
            raise ValueError(f"{run.json_path} missing set-only metadata keys: {missing_set}")
        if meta.get("model.set_causality_mode") != "strict_past":
            raise ValueError(f"{run.json_path} is not strict_past")
    if run.backend == "landmark":
        if str(meta.get("model.backend_params.landmark_coverage")) not in {"0.25", "0.250000"}:
            raise ValueError(f"{run.json_path} does not record landmark_coverage=0.25")
        if str(meta.get("resolved.landmark_coverage")) in {"", "NA", "None"}:
            raise ValueError(f"{run.json_path} missing resolved.landmark_coverage")
        if str(meta.get("resolved.landmark_count")) in {"", "NA", "None"}:
            raise ValueError(f"{run.json_path} missing resolved.landmark_count")

    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 11)):
        raise ValueError(f"{run.csv_path} epochs are not 1..10: {epochs}")
    last = rows[-1]
    return {
        "phase": run.phase,
        "family": run.family,
        "backend": run.backend,
        "seed": run.seed,
        "lr": run.lr,
        "D": run.d_model,
        "d_ff": run.d_ff,
        "w": run.window,
        "s": run.stride,
        "M": "125" if run.stride == "4" else ("63" if run.stride == "8" else "NA"),
        "config": run.config,
        "csv_path": str(run.csv_path.relative_to(ROOT)),
        "json_path": str(run.json_path.relative_to(ROOT)),
        "rows": len(rows),
        "final_train_loss": s(last.get("train/loss", "NA")),
        "final_val_loss": s(last.get("val/loss", "NA")),
        "final_train_ppl": s(last.get("train/ppl", "NA")),
        "final_val_ppl": s(last.get("val/ppl", "NA")),
        "time_per_epoch_s": s(last.get("train/time_per_epoch_s", "NA")),
        "peak_vram_mib": s(last.get("train/peak_vram_mib", "NA")),
        "resolved_d_phi": s(meta.get("resolved.d_phi", "NA")),
        "resolved_adapter_type": s(meta.get("resolved.adapter_type", "NA")),
        "pooling_alpha": s(meta.get("resolved.pooling_alpha", "NA")),
        "hash_seed": s(meta.get("resolved.hash_seed", "NA")),
        "hash_normalize": s(meta.get("resolved.hash_normalize", "NA")),
        "hash_num_bins": s(meta.get("resolved.hash_num_bins", "NA")),
        "router_min_temp": s(meta.get("resolved.router_min_temp", "NA")),
        "landmark_coverage": s(meta.get("resolved.landmark_coverage", "NA")),
        "landmark_count": s(meta.get("resolved.landmark_count", "NA")),
        "source_csv_sha256": sha256(run.csv_path),
    }


def write_tsv(path: Path, rows: Iterable[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows: list[dict], family_filter: set[str] | None = None) -> list[dict]:
    selected = []
    groups = sorted({
        (r["D"], r["d_ff"], r["family"])
        for r in rows
        if family_filter is None or r["family"] in family_filter
    })
    for d_model, d_ff, family in groups:
        candidates = [
            r for r in rows
            if r["D"] == d_model and r["d_ff"] == d_ff and r["family"] == family
        ]
        selected.append(min(candidates, key=lambda r: float(r["final_val_ppl"])))
    return selected


def markdown_handoff(rows: list[dict], failures: list[str]) -> str:
    lines = [
        "# A2 Grid Handoff",
        "",
        "Status: PASS" if not failures else "Status: FAIL",
        "",
        f"Expected runs: {len(expected_runs())}",
        f"Validated runs: {len(rows)}",
        "",
        "## Failures / Retries",
        "",
    ]
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- None.")
    lines.extend(["", "## Run Artifacts", ""])
    columns = [
        "phase", "family", "backend", "seed", "lr", "D", "d_ff", "w", "s", "M",
        "rows", "final_val_ppl", "time_per_epoch_s", "config", "csv_path", "source_csv_sha256",
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "NA")) for col in columns) + " |")
    lines.append("")
    lines.append("## Summary Artifacts")
    lines.append("")
    for path in [
        "out/paper_integrated_evidence/tables/a2_anchor_stability.tsv",
        "out/paper_integrated_evidence/tables/a2_lrnorm_headline_all_runs.tsv",
        "out/paper_integrated_evidence/tables/a2_lrnorm_headline_best_by_pair.tsv",
        "out/paper_integrated_evidence/tables/a2_lrnorm_family_slice_all_runs.tsv",
        "out/paper_integrated_evidence/tables/a2_lrnorm_family_best_by_family.tsv",
        "out/paper_integrated_evidence/checks/a2_grid_manifest.json",
    ]:
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    for path in [TABLES, CHECKS, AUDIT]:
        path.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    rows: list[dict] = []
    for run in expected_runs():
        try:
            rows.append(validate_run(run))
        except Exception as exc:  # noqa: BLE001 - report every missing/bad artifact.
            failures.append(str(exc))

    if failures:
        (AUDIT / "A2_grid_handoff.md").write_text(markdown_handoff(rows, failures))
        raise SystemExit("\n".join(failures[:20]))

    handoff_cols = [
        "phase", "family", "backend", "seed", "lr", "D", "d_ff", "w", "s", "M",
        "config", "csv_path", "json_path", "rows", "final_train_loss", "final_val_loss",
        "final_train_ppl", "final_val_ppl", "time_per_epoch_s", "peak_vram_mib",
        "resolved_d_phi", "resolved_adapter_type", "pooling_alpha", "hash_seed",
        "hash_normalize", "hash_num_bins", "router_min_temp", "landmark_coverage",
        "landmark_count", "source_csv_sha256",
    ]
    write_tsv(TABLES / "a2_grid_all_runs.tsv", rows, handoff_cols)

    anchor = [r for r in rows if r["phase"] == "A2.1"]
    headline = [r for r in rows if r["phase"] == "A2.2"]
    family_slice = [
        r for r in rows
        if (
            r["phase"] == "A2.3"
            or (r["phase"] == "A2.2" and r["D"] == "384" and r["d_ff"] == "1536")
        )
    ]
    write_tsv(TABLES / "a2_anchor_stability.tsv", anchor, handoff_cols)
    write_tsv(TABLES / "a2_lrnorm_headline_all_runs.tsv", headline, handoff_cols)
    write_tsv(TABLES / "a2_lrnorm_headline_best_by_pair.tsv", best_rows(headline), handoff_cols)
    write_tsv(TABLES / "a2_lrnorm_family_slice_all_runs.tsv", family_slice, handoff_cols)
    write_tsv(TABLES / "a2_lrnorm_family_best_by_family.tsv", best_rows(family_slice), handoff_cols)

    manifest_paths = [
        "out/paper_integrated_evidence/tables/a2_grid_all_runs.tsv",
        "out/paper_integrated_evidence/tables/a2_anchor_stability.tsv",
        "out/paper_integrated_evidence/tables/a2_lrnorm_headline_all_runs.tsv",
        "out/paper_integrated_evidence/tables/a2_lrnorm_headline_best_by_pair.tsv",
        "out/paper_integrated_evidence/tables/a2_lrnorm_family_slice_all_runs.tsv",
        "out/paper_integrated_evidence/tables/a2_lrnorm_family_best_by_family.tsv",
    ]
    manifest = {
        "status": "pass",
        "expected_runs": len(expected_runs()),
        "validated_runs": len(rows),
        "source_csvs": [
            {"path": row["csv_path"], "sha256": row["source_csv_sha256"]}
            for row in rows
        ],
        "generated": [
            {"path": path, "sha256": sha256(ROOT / path)}
            for path in manifest_paths
        ],
    }
    (CHECKS / "a2_grid_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (AUDIT / "A2_grid_handoff.md").write_text(markdown_handoff(rows, failures))
    print(json.dumps({"status": "pass", "validated_runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
