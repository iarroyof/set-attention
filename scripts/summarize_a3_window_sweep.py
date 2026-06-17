#!/usr/bin/env python3
"""Validate and summarize the A3.1 fixed-stride window-size sweep."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out" / "paper_mechanisms" / "a3_window_sweep"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"

LR = "1e-4"
LR_TAG = "1e-4"
STRIDE = 4
SEQ_LEN = 512
WINDOWS = [6, 8, 12, 16, 20, 24]
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
    window: int
    seed: int

    @property
    def m(self) -> int:
        return ((SEQ_LEN - self.window) // STRIDE) + 1

    @property
    def landmark_count(self) -> str:
        if self.backend != "landmark":
            return "NA"
        return str(max(round(0.25 * self.m), 2))

    @property
    def group(self) -> str:
        return f"a3_window_sweep_{self.family_slug}_D384_FF1536_s4"

    @property
    def name(self) -> str:
        return (
            f"a3_window_{self.family_slug}_D384_FF1536_w{self.window}_s4_"
            f"lr{LR_TAG.replace('.', 'p')}_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return RAW / self.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


def seeds_for_window(window: int) -> list[int]:
    return [0, 1, 2] if window in {6, 16, 24} else [0]


def expected_runs() -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []
    for slug, family, backend, config in FAMILIES:
        for window in WINDOWS:
            for seed in seeds_for_window(window):
                runs.append(ExpectedRun(slug, family, backend, config, window, seed))
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
    if int(meta.get("model.window_size")) != run.window:
        raise ValueError(f"{run.json_path} has wrong window_size")
    if int(meta.get("model.stride")) != STRIDE:
        raise ValueError(f"{run.json_path} has wrong stride")
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
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl"]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{run.csv_path} missing finite final {key}")

    return {
        "phase": "A3.1",
        "family_slug": run.family_slug,
        "family": run.family,
        "backend": run.backend,
        "seed": str(run.seed),
        "lr": LR,
        "D": "384",
        "d_ff": "1536",
        "w": str(run.window),
        "s": str(STRIDE),
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
        grouped[(row["family"], row["family_slug"], row["backend"], row["w"])].append(row)
    out: list[dict[str, str]] = []
    for (family, slug, backend, window), group in sorted(grouped.items()):
        ppls = [float(row["final_val_ppl"]) for row in group]
        cand = [float(row["candidate_count_mean"]) for row in group]
        out.append({
            "phase": "A3.1",
            "family": family,
            "family_slug": slug,
            "backend": backend,
            "w": window,
            "s": str(STRIDE),
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
    failures: list[str] = []
    for slug, family, _, _ in FAMILIES:
        seed0 = [
            row for row in rows
            if row["family_slug"] == slug and row["seed"] == "0"
        ]
        by_window = {
            int(row["w"]): float(row["candidate_count_mean"])
            for row in seed0
        }
        values = [by_window[w] for w in WINDOWS if w in by_window]
        if len(values) != len(WINDOWS):
            failures.append(f"{family}: missing seed0 candidate-count windows")
            continue
        if len({round(v, 8) for v in values}) < 2:
            failures.append(f"{family}: candidate-count mean did not vary with window")
        if any(b < a - 1e-6 for a, b in zip(values, values[1:])):
            failures.append(f"{family}: candidate-count mean was not nondecreasing: {by_window}")
    return failures


def markdown_audit(rows: list[dict[str, str]], summary: list[dict[str, str]], failures: list[str]) -> str:
    prelaunch_path = AUDIT / "A3_1_window_sweep_prelaunch.json"
    prelaunch = json.loads(prelaunch_path.read_text()) if prelaunch_path.exists() else {}
    lines = [
        "# A3.1 Window-Size Sweep Audit",
        "",
        "Status: PASS" if not failures else "Status: FAIL",
        "",
        "## Scope",
        "",
        "- Fixed stride: `s=4`.",
        "- Windows: `w in {6, 8, 12, 16, 20, 24}`.",
        "- Families: SetDense/exact, SetSparse/local_band, SetLinear/landmark.",
        "- Seeds: endpoints/reference `w in {6,16,24}` use seeds 0,1,2; interiors use seed 0.",
        "- A2.2/A2.3 remain the locked `s=8` LR-normalized headline/family grid; they were not rerun or overridden.",
        "",
        "## Commands / Scripts",
        "",
        "- `bash scripts/run_a3_window_sweep.sh`",
        "- `python scripts/summarize_a3_window_sweep.py`",
        "",
        "## Prelaunch State",
        "",
        f"- Branch: `{_first(prelaunch.get('branch'))}`",
        f"- HEAD: `{_first(prelaunch.get('head'))}`",
        f"- A2 manifest: `{prelaunch.get('a2_manifest_status')}` with `{prelaunch.get('a2_manifest_validated_runs')}` / `{prelaunch.get('a2_manifest_expected_runs')}` runs.",
        f"- A2 handoff: `{prelaunch.get('a2_handoff_status_line')}`",
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
        "family", "backend", "seed", "lr", "w", "s", "M", "rows",
        "final_val_ppl", "time_per_epoch_s", "candidate_count_mean",
        "config", "csv_path", "source_csv_sha256",
    ]
    lines.append("| " + " | ".join(artifact_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(artifact_cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "NA")) for col in artifact_cols) + " |")
    lines.extend(["", "## Summary", ""])
    summary_cols = [
        "family", "backend", "w", "s", "M", "runs", "seeds",
        "val_ppl_mean", "val_ppl_std", "candidate_count_mean",
    ]
    lines.append("| " + " | ".join(summary_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(summary_cols)) + " |")
    for row in summary:
        lines.append("| " + " | ".join(str(row.get(col, "NA")) for col in summary_cols) + " |")
    lines.extend([
        "",
        "## Generated Artifacts",
        "",
        "- `out/paper_integrated_evidence/tables/a3_window_sweep_all_runs.tsv`",
        "- `out/paper_integrated_evidence/tables/a3_window_sweep_summary.tsv`",
        "- `out/paper_integrated_evidence/checks/a3_window_sweep_manifest.json`",
        "",
        "## Recommendation For Figure 1 / B5",
        "",
        "Use `a3_window_sweep_summary.tsv` for the fixed-stride candidate-count mechanism figure and `a3_window_sweep_all_runs.tsv` for provenance/error bars.",
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
    for run in expected_runs():
        try:
            rows.append(validate_run(run))
        except Exception as exc:  # noqa: BLE001 - collect every bad artifact.
            failures.append(str(exc))

    if not failures:
        failures.extend(validate_candidate_variation(rows))

    rows.sort(key=lambda r: (r["family_slug"], int(r["w"]), int(r["seed"])))
    summary = summarize(rows) if rows else []

    columns = [
        "phase", "family_slug", "family", "backend", "seed", "lr", "D", "d_ff",
        "w", "s", "M", "expected_landmark_count", "config", "csv_path",
        "json_path", "rows", "final_train_loss", "final_val_loss",
        "final_train_ppl", "final_val_ppl", "time_per_epoch_s", "peak_vram_mib",
        "candidate_count_mean", "candidate_count_max", "router_entropy_norm",
        "router_top1_weight", "router_top1_gap_norm", "resolved_d_phi",
        "resolved_adapter_type", "pooling_alpha", "hash_seed", "hash_normalize",
        "hash_num_bins", "router_min_temp", "landmark_coverage", "landmark_count",
        "source_csv_sha256",
    ]
    summary_cols = [
        "phase", "family", "family_slug", "backend", "w", "s", "M", "runs",
        "seeds", "val_ppl_mean", "val_ppl_std", "val_ppl_min", "val_ppl_max",
        "candidate_count_mean", "candidate_count_min", "candidate_count_max",
    ]
    write_tsv(TABLES / "a3_window_sweep_all_runs.tsv", rows, columns)
    write_tsv(TABLES / "a3_window_sweep_summary.tsv", summary, summary_cols)

    generated_paths = [
        "out/paper_integrated_evidence/tables/a3_window_sweep_all_runs.tsv",
        "out/paper_integrated_evidence/tables/a3_window_sweep_summary.tsv",
    ]
    manifest = {
        "status": "pass" if not failures else "fail",
        "expected_runs": len(expected_runs()),
        "validated_runs": len(rows),
        "failures": failures,
        "source_csvs": [
            {"path": row["csv_path"], "sha256": row["source_csv_sha256"]}
            for row in rows
        ],
        "generated": [
            {"path": path, "sha256": sha256(ROOT / path)}
            for path in generated_paths
        ],
    }
    manifest_path = CHECKS / "a3_window_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (AUDIT / "A3_1_window_sweep.md").write_text(markdown_audit(rows, summary, failures))

    if failures:
        raise SystemExit("\n".join(failures[:20]))
    print(json.dumps({"status": "pass", "validated_runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
