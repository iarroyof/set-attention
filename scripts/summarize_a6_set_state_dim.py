#!/usr/bin/env python3
"""Validate and summarize the A6.2 explicit set_state_dim sweep."""

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
RAW = ROOT / "out" / "paper_mechanisms" / "a6_set_state_width"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"
LOG_ROOT = ROOT / "logs" / "a6_set_state_width"

SEQ_LEN = 512
WINDOW = 16
STRIDE = 8
M = (SEQ_LEN - WINDOW) // STRIDE + 1
EPOCHS = 10
LR = "1e-4"
D_MODEL = 384
D_FF = 1536
DPHI = 384
SET_STATE_DIMS = [384, 512, 768]
NEW_SET_STATE_DIMS = [512, 768]
SEEDS = [0, 1, 2]


@dataclass(frozen=True)
class Family:
    slug: str
    family: str
    impl: str
    attention_family: str
    backend: str
    config: str
    is_set: bool


FAMILIES = [
    Family("baseline_dense_exact", "Baseline Dense", "baseline_token", "dense", "exact", "configs/paper_lr_norm/baseline_dense_exact.yaml", False),
    Family("baseline_sparse_local_band", "Baseline Sparse", "baseline_token", "sparse", "local_band", "configs/paper_lr_norm/baseline_sparse_local_band.yaml", False),
    Family("baseline_linear_landmark", "Baseline Linear", "baseline_token", "linear", "landmark", "configs/paper_lr_norm/baseline_linear_landmark.yaml", False),
    Family("set_dense_exact", "Set Dense", "set_only", "dense", "exact", "configs/paper_lr_norm/set_dense_exact.yaml", True),
    Family("set_sparse_local_band", "Set Sparse", "set_only", "sparse", "local_band", "configs/paper_lr_norm/set_sparse_local_band.yaml", True),
    Family("set_linear_landmark", "Set Linear", "set_only", "linear", "landmark", "configs/paper_lr_norm/set_linear_landmark.yaml", True),
]
SET_FAMILIES = [f for f in FAMILIES if f.is_set]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_delimited(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t" if path.suffix == ".tsv" else ","))


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


def family_by_slug(slug: str) -> Family:
    for family in FAMILIES:
        if family.slug == slug:
            return family
    raise KeyError(slug)


def normalize_family_slug(row: dict[str, str]) -> str | None:
    slug = row.get("family_slug")
    if slug:
        return slug
    family = row.get("family")
    backend = row.get("backend")
    if family == "Baseline token" and backend == "exact":
        return "baseline_dense_exact"
    if family == "Set Dense" and backend == "exact":
        return "set_dense_exact"
    if family == "Set Sparse" and backend == "local_band":
        return "set_sparse_local_band"
    if family == "Set Linear" and backend == "landmark":
        return "set_linear_landmark"
    return None


def anchor_row(row: dict[str, str], source: str) -> dict[str, str] | None:
    slug = normalize_family_slug(row)
    if slug is None:
        return None
    family = family_by_slug(slug)
    if row.get("D") != str(D_MODEL) or row.get("d_ff") != str(D_FF):
        return None
    if row.get("lr") != LR or row.get("seed") not in {"0", "1", "2"}:
        return None
    if family.is_set:
        resolved_d_phi = row.get("resolved_d_phi", row.get("resolved.d_phi", "NA"))
        if str(resolved_d_phi) not in {str(DPHI), "NA"}:
            return None
        set_state_dim = str(D_MODEL)
    else:
        set_state_dim = "NA"
    return {
        "phase": "A6.2-anchor",
        "source": source,
        "family_slug": slug,
        "family": family.family,
        "implementation": family.impl,
        "attention_family": family.attention_family,
        "backend": family.backend,
        "seed": row["seed"],
        "lr": LR,
        "D": str(D_MODEL),
        "d_ff": str(D_FF),
        "L": row.get("L", str(SEQ_LEN)) if row.get("L", "NA") != "NA" else str(SEQ_LEN),
        "w": row.get("w", str(WINDOW)) if family.is_set else "NA",
        "s": row.get("s", str(STRIDE)) if family.is_set else "NA",
        "M": row.get("M", str(M)) if family.is_set else "NA",
        "d_phi": str(DPHI) if family.is_set else "NA",
        "set_state_dim": set_state_dim,
        "config": family.config,
        "csv_path": row["csv_path"],
        "json_path": row["json_path"],
        "rows": row["rows"],
        "final_train_loss": row["final_train_loss"],
        "final_val_loss": row["final_val_loss"],
        "final_train_ppl": row["final_train_ppl"],
        "final_val_ppl": row["final_val_ppl"],
        "time_per_epoch_s": row.get("time_per_epoch_s", "NA"),
        "peak_vram_mib": row.get("peak_vram_mib", "NA"),
        "resolved_d_phi": str(DPHI) if family.is_set else "NA",
        "resolved_set_state_dim": set_state_dim,
        "resolved_adapter_type": row.get("resolved_adapter_type", "NA"),
        "landmark_coverage": row.get("landmark_coverage", "NA"),
        "landmark_count": row.get("landmark_count", "NA"),
        "source_csv_sha256": row.get("source_csv_sha256", "NA"),
    }


def reused_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    family_slice = read_delimited(TABLES / "a2_lrnorm_family_slice_all_runs.tsv")
    controls = read_delimited(TABLES / "a2_baseline_controls_all_runs.tsv")
    a6_1 = read_delimited(TABLES / "a6_dphi_capacity_all_runs.tsv")

    for row in family_slice:
        slug = normalize_family_slug(row)
        if slug == "baseline_dense_exact":
            item = anchor_row(row, "reused-A2")
            if item is not None:
                rows.append(item)
    for row in controls:
        slug = normalize_family_slug(row)
        if slug in {"baseline_sparse_local_band", "baseline_linear_landmark"}:
            item = anchor_row(row, "reused-A2.4")
            if item is not None:
                rows.append(item)
    for row in a6_1:
        slug = normalize_family_slug(row)
        if slug in {"set_dense_exact", "set_sparse_local_band", "set_linear_landmark"} and row.get("d_phi") == str(DPHI):
            item = anchor_row(row, "reused-A6.1")
            if item is not None:
                rows.append(item)
    return rows


def new_run_path(family: Family, set_state_dim: int, seed: int) -> tuple[Path, Path]:
    group = f"a6_set_state_dim_{family.slug}_D{D_MODEL}_FF{D_FF}"
    name = (
        f"a6_setstate_{family.slug}_D{D_MODEL}_FF{D_FF}_setdim{set_state_dim}_"
        f"dphi{DPHI}_w{WINDOW}_s{STRIDE}_lr{LR.replace('.', 'p')}_seed{seed}"
    )
    csv_path = RAW / group / f"{name}.csv"
    return csv_path, csv_path.with_suffix(".json")


def landmark_count(family: Family) -> str:
    if family.backend != "landmark":
        return "NA"
    return str(max(round(0.25 * M), 2))


def validate_new_run(family: Family, set_state_dim: int, seed: int) -> dict[str, str]:
    csv_path, json_path = new_run_path(family, set_state_dim, seed)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {csv_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"missing JSON: {json_path}")
    rows = read_delimited(csv_path)
    if len(rows) != EPOCHS:
        raise ValueError(f"{csv_path} has {len(rows)} rows, expected {EPOCHS}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, EPOCHS + 1)):
        raise ValueError(f"{csv_path} epochs are not 1..{EPOCHS}: {epochs}")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite CSV values in {csv_path}: {bad[:5]}")

    meta = json.loads(json_path.read_text())
    checks = {
        "model.implementation": "set_only",
        "model.d_model": D_MODEL,
        "model.dim_feedforward": D_FF,
        "model.max_seq_len": SEQ_LEN,
        "model.window_size": WINDOW,
        "model.stride": STRIDE,
        "model.set_causality_mode": "strict_past",
        "model.d_phi": DPHI,
        "model.set_state_dim": set_state_dim,
        "resolved.d_phi": DPHI,
        "resolved.set_state_dim": set_state_dim,
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if str(actual) != str(expected):
            raise ValueError(f"{json_path} has {key}={actual!r}, expected {expected!r}")
    if family.backend == "landmark":
        if str(meta.get("model.backend_params.landmark_coverage")) != "0.25":
            raise ValueError(f"{json_path} missing landmark_coverage=0.25")
        if str(meta.get("resolved.landmark_coverage")) != "0.25":
            raise ValueError(f"{json_path} missing resolved.landmark_coverage=0.25")
        if str(meta.get("resolved.landmark_count")) != landmark_count(family):
            raise ValueError(
                f"{json_path} landmark_count={meta.get('resolved.landmark_count')}, "
                f"expected {landmark_count(family)}"
            )

    last = rows[-1]
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl"]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{csv_path} missing finite final {key}")

    return {
        "phase": "A6.2-new",
        "source": "new",
        "family_slug": family.slug,
        "family": family.family,
        "implementation": family.impl,
        "attention_family": family.attention_family,
        "backend": family.backend,
        "seed": str(seed),
        "lr": LR,
        "D": str(D_MODEL),
        "d_ff": str(D_FF),
        "L": str(SEQ_LEN),
        "w": str(WINDOW),
        "s": str(STRIDE),
        "M": str(M),
        "d_phi": str(DPHI),
        "set_state_dim": str(set_state_dim),
        "config": family.config,
        "csv_path": str(csv_path.relative_to(ROOT)),
        "json_path": str(json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_train_loss": last["train/loss"],
        "final_val_loss": last["val/loss"],
        "final_train_ppl": last["train/ppl"],
        "final_val_ppl": last["val/ppl"],
        "time_per_epoch_s": last.get("time/epoch_s", "NA"),
        "peak_vram_mib": last.get("system/peak_vram_allocated_mib", "NA"),
        "resolved_d_phi": str(meta.get("resolved.d_phi", "NA")),
        "resolved_set_state_dim": str(meta.get("resolved.set_state_dim", "NA")),
        "resolved_adapter_type": str(meta.get("resolved.adapter_type", "NA")),
        "landmark_coverage": str(meta.get("resolved.landmark_coverage", "NA")),
        "landmark_count": str(meta.get("resolved.landmark_count", "NA")),
        "source_csv_sha256": sha256(csv_path),
    }


def validate_new_rows() -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for family in SET_FAMILIES:
        for set_state_dim in NEW_SET_STATE_DIMS:
            for seed in SEEDS:
                try:
                    rows.append(validate_new_run(family, set_state_dim, seed))
                except Exception as exc:  # noqa: BLE001
                    failures.append(str(exc))
    return rows, failures


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["family_slug"], row["family"], row["set_state_dim"])
        grouped[key].append(row)

    anchor_mean: dict[str, float] = {}
    for (slug, _family, set_state_dim), group_rows in grouped.items():
        if set_state_dim in {"384", "NA"}:
            anchor_mean[slug] = mean(float(row["final_val_ppl"]) for row in group_rows)

    summary: list[dict[str, str]] = []
    for (slug, family, set_state_dim), group_rows in sorted(grouped.items()):
        ppls = [float(row["final_val_ppl"]) for row in group_rows]
        train_ppls = [float(row["final_train_ppl"]) for row in group_rows]
        times = [float_or_none(row["time_per_epoch_s"]) for row in group_rows]
        vrams = [float_or_none(row["peak_vram_mib"]) for row in group_rows]
        times_f = [v for v in times if v is not None]
        vrams_f = [v for v in vrams if v is not None]
        mean_ppl = mean(ppls)
        delta = mean_ppl - anchor_mean.get(slug, mean_ppl)
        rel = (delta / anchor_mean[slug] * 100.0) if slug in anchor_mean else 0.0
        summary.append({
            "phase": "A6.2",
            "family_slug": slug,
            "family": family,
            "backend": group_rows[0]["backend"],
            "D": str(D_MODEL),
            "d_ff": str(D_FF),
            "d_phi": group_rows[0]["d_phi"],
            "set_state_dim": set_state_dim,
            "lr": LR,
            "n": str(len(group_rows)),
            "mean_final_val_ppl": f"{mean_ppl:.6f}",
            "std_final_val_ppl": f"{pstdev(ppls):.6f}" if len(ppls) > 1 else "0.000000",
            "mean_final_train_ppl": f"{mean(train_ppls):.6f}",
            "std_final_train_ppl": (
                f"{pstdev(train_ppls):.6f}" if len(train_ppls) > 1 else "0.000000"
            ),
            "mean_peak_vram_mib": f"{mean(vrams_f):.6f}" if vrams_f else "NA",
            "mean_time_per_epoch_s": f"{mean(times_f):.6f}" if times_f else "NA",
            "delta_val_ppl_vs_anchor": f"{delta:.6f}",
            "pct_delta_val_ppl_vs_anchor": f"{rel:.6f}",
        })
    return summary


def write_audit(manifest: dict[str, object], summary_rows: list[dict[str, str]]) -> None:
    best: dict[str, dict[str, str]] = {}
    for row in summary_rows:
        if row["set_state_dim"] == "NA":
            continue
        cur = best.get(row["family"])
        if cur is None or float(row["mean_final_val_ppl"]) < float(cur["mean_final_val_ppl"]):
            best[row["family"]] = row

    lines = [
        "# A6.2 Explicit Set-State Dimensionality Sweep",
        "",
        f"Status: {manifest['status'].upper()}",
        "",
        "## Scope",
        "",
        "A6.2 explicitly tests `model.set_state_dim`, the width of pooled set states, "
        "set-attention blocks, set backend value states, and routed set context before "
        "projection back to token width. Token width is held fixed at D=384 and "
        "`d_phi` is held fixed at 384.",
        "",
        "Matched token controls are reused from validated A2/A2.4 artifacts. "
        "SKA `set_state_dim=384` rows are reused from A6.1 `d_phi=384` artifacts. "
        "New runs cover SKA `set_state_dim` in {512,768}.",
        "",
        "## Summary",
        "",
        "| family | backend | set_state_dim | n | mean val PPL | std | delta vs anchor | mean VRAM MiB | sec/epoch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['family']} | {row['backend']} | {row['set_state_dim']} | {row['n']} | "
            f"{row['mean_final_val_ppl']} | {row['std_final_val_ppl']} | "
            f"{row['delta_val_ppl_vs_anchor']} | {row['mean_peak_vram_mib']} | "
            f"{row['mean_time_per_epoch_s']} |"
        )
    lines.extend(["", "## Best SKA set_state_dim", "", "| family | backend | best set_state_dim | mean val PPL |", "| --- | --- | ---: | ---: |"])
    for family, row in sorted(best.items()):
        lines.append(
            f"| {family} | {row['backend']} | {row['set_state_dim']} | {row['mean_final_val_ppl']} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- All runs TSV: `out/paper_integrated_evidence/tables/a6_set_state_width_all_runs.tsv`",
        "- Summary TSV: `out/paper_integrated_evidence/tables/a6_set_state_width_summary.tsv`",
        "- Manifest: `out/paper_integrated_evidence/checks/a6_set_state_width_manifest.json`",
        "",
        "## Validation",
        "",
        f"- Total expected rows: {manifest['expected_runs']}",
        f"- Total validated rows: {manifest['validated_runs']}",
        f"- New expected runs: {manifest['expected_new_runs']}",
        f"- New validated runs: {manifest['validated_new_runs']}",
        f"- Reused rows: {manifest['reused_rows']}",
        f"- Log failures: {len(manifest.get('log_failures', []))}",
        f"- Failures: {len(manifest.get('failures', []))}",
    ])
    if manifest.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in manifest["failures"])
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "A6_2_set_state_width.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    reused = reused_rows()
    new, new_failures = validate_new_rows()
    failures.extend(new_failures)
    log_failures = scan_logs()
    failures.extend(log_failures)

    expected_reused = 18
    expected_new = len(SET_FAMILIES) * len(NEW_SET_STATE_DIMS) * len(SEEDS)
    expected_total = expected_reused + expected_new
    if len(reused) != expected_reused:
        failures.append(f"reused row count {len(reused)} != expected {expected_reused}")
    rows = reused + new

    all_runs_path = TABLES / "a6_set_state_width_all_runs.tsv"
    summary_path = TABLES / "a6_set_state_width_summary.tsv"
    manifest_path = CHECKS / "a6_set_state_width_manifest.json"
    all_fields = [
        "phase", "source", "family_slug", "family", "implementation", "attention_family",
        "backend", "seed", "lr", "D", "d_ff", "L", "w", "s", "M", "d_phi",
        "set_state_dim", "config", "csv_path", "json_path", "rows", "final_train_loss",
        "final_val_loss", "final_train_ppl", "final_val_ppl", "time_per_epoch_s",
        "peak_vram_mib", "resolved_d_phi", "resolved_set_state_dim",
        "resolved_adapter_type", "landmark_coverage", "landmark_count", "source_csv_sha256",
    ]
    if rows:
        write_tsv(all_runs_path, rows, all_fields)
        summary_rows = summarize(rows)
        write_tsv(
            summary_path,
            summary_rows,
            [
                "phase", "family_slug", "family", "backend", "D", "d_ff", "d_phi",
                "set_state_dim", "lr", "n", "mean_final_val_ppl",
                "std_final_val_ppl", "mean_final_train_ppl", "std_final_train_ppl",
                "mean_peak_vram_mib", "mean_time_per_epoch_s",
                "delta_val_ppl_vs_anchor", "pct_delta_val_ppl_vs_anchor",
            ],
        )
    else:
        summary_rows = []

    manifest = {
        "status": "pass" if not failures and len(rows) == expected_total else "fail",
        "phase": "A6.2",
        "expected_runs": expected_total,
        "validated_runs": len(rows),
        "expected_new_runs": expected_new,
        "validated_new_runs": len(new),
        "reused_rows": len(reused),
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
        "validated_new_runs": manifest["validated_new_runs"],
        "expected_new_runs": manifest["expected_new_runs"],
        "failures": failures[:10],
        "summary": str(summary_path.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
    }, indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
