#!/usr/bin/env python3
"""Validate and summarize SD-9.5 mechanism-probe artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"
AUDIT = ROOT / "audit"


SHORT_VARIANTS = {
    "mixed": {"label": "mixed-25", "fine_heads": 6, "coarse_heads": 2},
    "all_fine": {"label": "all-fine", "fine_heads": 8, "coarse_heads": 0},
    "all_coarse": {"label": "all-coarse", "fine_heads": 0, "coarse_heads": 8},
}
SCALE_VARIANTS = {
    "mixed": {"label": "mixed-65", "fine_heads": 3, "coarse_heads": 5},
    "all_fine": {"label": "all-fine", "fine_heads": 8, "coarse_heads": 0},
    "all_coarse": {"label": "all-coarse", "fine_heads": 0, "coarse_heads": 8},
}
METRIC_KEYS = [
    "val/ppl",
    "train/peak_vram_mib",
    "val/span_ablation_delta_ppl",
    "val/span_ablation_fine_delta_ppl",
    "val/span_ablation_coarse_delta_ppl",
    "val/effective_range_fine",
    "val/effective_range_coarse",
    "val/routing_entropy_fine",
    "val/routing_entropy_coarse",
    "val/routing_top1_fine",
    "val/routing_top1_coarse",
    "val/loss_early_freq",
    "val/loss_early_rare",
    "val/loss_late_freq",
    "val/loss_late_rare",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["short", "scale", "all"],
        default="all",
        help="Artifact subset to validate.",
    )
    parser.add_argument(
        "--allow-failed-scale",
        action="store_true",
        help="Record nonzero scale rows, e.g. OOM, instead of failing validation.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def as_float(row: dict[str, str], key: str) -> float | str:
    raw = row.get(key, "")
    if raw in {"", "NA", None}:
        return "NA"
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{key} is nonfinite: {raw!r}")
    return value


def mean_std(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "NA", "NA"
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def find_csv(root: Path, prefix: str, role: str, variant: str, seed: int) -> Path | None:
    matches = sorted(root.glob(f"**/{prefix}_{role}_{variant}_*_seed{seed}.csv"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"expected one CSV for {role}/{variant}/seed{seed}, found {len(matches)}")
    return matches[0]


def scan_logs(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    token_re = re.compile(r"(?<![A-Za-z0-9_])(?:nan|NaN|-inf|inf)(?![A-Za-z0-9_])")
    for root in paths:
        if not root.exists():
            failures.append(f"missing log root: {root.relative_to(ROOT)}")
            continue
        for path in sorted(root.glob("*.log")):
            text = path.read_text(errors="replace")
            if "out of memory" in text.lower() or "cuda oom" in text.lower():
                failures.append(f"{path.relative_to(ROOT)} contains OOM")
                continue
            for pattern in ("Traceback", "RuntimeError", "ValueError"):
                if pattern in text:
                    failures.append(f"{path.relative_to(ROOT)} contains {pattern}")
                    break
            else:
                match = token_re.search(text)
                if match:
                    failures.append(f"{path.relative_to(ROOT)} contains {match.group()!r}")
    return failures


def require_meta(meta: dict[str, object], key: str, expected: object) -> None:
    actual = meta.get(key)
    if isinstance(expected, bool):
        ok = str(actual) == str(expected)
    elif isinstance(expected, (int, float)):
        ok = math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
    else:
        ok = str(actual) == str(expected)
    if not ok:
        raise ValueError(f"{key} expected {expected!r}, got {actual!r}")


def validate_meta(
    json_path: Path,
    *,
    role: str,
    variant: str,
    seed: int,
    seq_len: int,
    batch_size: int,
    backend: str,
    coverage: float | str,
    epochs: int,
) -> dict[str, object]:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    checks = {
        "model.implementation": "set_only",
        "model.backend": backend,
        "model.output_residual_mode": "anchor_span",
        "resolved.output_residual_mode": "anchor_span",
        "model.anchor.enabled": False,
        "resolved.anchor_enabled": False,
        "model.token_mlp.enabled": False,
        "model.candidate_fiber": "endpoint_window",
        "resolved.candidate_fiber": "endpoint_window",
        "model.multiresolution.enabled": True,
        "resolved.multiresolution_enabled": True,
        "model.multivector_basis.enabled": False,
        "model.d_model": 384,
        "model.dim_feedforward": 1536,
        "model.num_layers": 6,
        "model.num_heads": 8,
        "model.d_phi": 384,
        "model.set_state_dim": 384,
        "model.max_seq_len": seq_len,
        "data.seq_len": seq_len,
        "data.batch_size": batch_size,
        "training.epochs": epochs,
        "training.lr": 0.0001,
        "training.seed": seed,
    }
    if backend == "landmark":
        checks["model.backend_params.landmark_coverage"] = coverage
        checks["resolved.landmark_coverage"] = coverage
    for key, expected in checks.items():
        require_meta(meta, key, expected)
    groups = meta.get("resolved.multiresolution_groups", [])
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{json_path}: missing resolved.multiresolution_groups")
    expected = SHORT_VARIANTS if role == "short" else SCALE_VARIANTS
    exp = expected[variant]
    found = {str(g.get("name")): g for g in groups if isinstance(g, dict)}
    if exp["fine_heads"] and "fine" not in found:
        raise ValueError(f"{json_path}: missing fine group")
    if exp["coarse_heads"] and "coarse" not in found:
        raise ValueError(f"{json_path}: missing coarse group")
    if "fine" in found:
        require_group(found["fine"], 2, 1, int(exp["fine_heads"]))
    if "coarse" in found:
        require_group(found["coarse"], 4, 2, int(exp["coarse_heads"]))
    return meta


def require_group(group: dict[str, object], window: int, stride: int, heads: int) -> None:
    if int(group.get("window_size", -1)) != window:
        raise ValueError(f"{group.get('name')}: window mismatch")
    if int(group.get("stride", -1)) != stride:
        raise ValueError(f"{group.get('name')}: stride mismatch")
    if int(group.get("num_heads", -1)) != heads:
        raise ValueError(f"{group.get('name')}: head-count mismatch")


def collect_run(
    *,
    root: Path,
    prefix: str,
    role: str,
    variant: str,
    seed: int,
    seq_len: int,
    batch_size: int,
    backend: str,
    coverage: float | str,
    epochs: int,
    allow_missing: bool,
    status_by_key: dict[tuple[str, int], dict[str, str]],
) -> dict[str, object]:
    status = status_by_key.get((variant, seed), {})
    rc = status.get("exit_code", "")
    if allow_missing and rc not in {"", "0"}:
        return {
            "context": role,
            "variant": variant,
            "label": (SHORT_VARIANTS if role == "short" else SCALE_VARIANTS)[variant]["label"],
            "seed": seed,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "backend": backend,
            "landmark_coverage": coverage,
            "status": "failed",
            "exit_code": rc,
            "csv_path": status.get("csv", ""),
            "failure": "OOM_or_nonzero_exit",
        }
    csv_path = find_csv(root, prefix, role, variant, seed)
    if csv_path is None:
        raise FileNotFoundError(f"missing CSV for {role}/{variant}/seed{seed}")
    rows = read_csv(csv_path)
    if len(rows) != epochs:
        raise ValueError(f"{csv_path}: expected {epochs} epochs, found {len(rows)}")
    final = rows[-1]
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"missing JSON: {json_path}")
    validate_meta(
        json_path,
        role=role,
        variant=variant,
        seed=seed,
        seq_len=seq_len,
        batch_size=batch_size,
        backend=backend,
        coverage=coverage,
        epochs=epochs,
    )
    row: dict[str, object] = {
        "context": role,
        "variant": variant,
        "label": (SHORT_VARIANTS if role == "short" else SCALE_VARIANTS)[variant]["label"],
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "backend": backend,
        "landmark_coverage": coverage,
        "status": "ok",
        "exit_code": rc or "0",
        "csv_path": str(csv_path.relative_to(ROOT)),
        "failure": "",
    }
    for key in METRIC_KEYS:
        row[key.replace("/", "_")] = as_float(final, key)
    return row


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[(row["context"], row["seq_len"], row["variant"])].append(row)
    out = []
    for (context, seq_len, variant), group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "context": context,
            "seq_len": seq_len,
            "variant": variant,
            "n": len(group),
        }
        for key in [k.replace("/", "_") for k in METRIC_KEYS]:
            vals = [float(row[key]) for row in group if row.get(key) not in {"NA", "", None}]
            avg, sd = mean_std(vals)
            summary[f"mean_{key}"] = avg
            summary[f"std_{key}"] = sd
        out.append(summary)
    return out


def scale_gaps(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for seq_len in sorted({row["seq_len"] for row in summary_rows if row["context"] == "scale"}):
        by_variant = {
            row["variant"]: row
            for row in summary_rows
            if row["context"] == "scale" and row["seq_len"] == seq_len
        }
        if not {"mixed", "all_fine"}.issubset(by_variant):
            continue
        mixed = float(by_variant["mixed"]["mean_train_peak_vram_mib"])
        fine = float(by_variant["all_fine"]["mean_train_peak_vram_mib"])
        out.append(
            {
                "seq_len": seq_len,
                "all_fine_minus_mixed_vram_mib": fine - mixed,
                "mixed_vram_mib": mixed,
                "all_fine_vram_mib": fine,
            }
        )
    if len(out) >= 2:
        out[-1]["gap_trend_vs_previous"] = (
            "grows"
            if float(out[-1]["all_fine_minus_mixed_vram_mib"])
            > float(out[-2]["all_fine_minus_mixed_vram_mib"])
            else "does_not_grow"
        )
    return out


def discover_scale_lengths() -> list[int]:
    lengths = {16384, 32768}
    root = ROOT / "out" / "paper_mechanisms"
    for path in root.glob("sd9_5_scaleL_L*"):
        match = re.fullmatch(r"sd9_5_scaleL_L(\d+)(?:_smoke)?", path.name)
        if match:
            lengths.add(int(match.group(1)))
    return sorted(lengths)


def write_audit(
    *,
    run_rows: list[dict[str, object]],
    short_summary: list[dict[str, object]],
    scale_summary: list[dict[str, object]],
    gaps: list[dict[str, object]],
    log_failures: list[str],
) -> None:
    mixed = next((r for r in short_summary if r["context"] == "short" and r["variant"] == "mixed"), None)
    lines = [
        "# SD-9.5 Mechanism Probes",
        "",
        "Guards: CE-only, `anchor.enabled=false`, `token_mlp.enabled=false`, `candidate_fiber=endpoint_window`, `output_residual_mode=anchor_span`, multiresolution only.",
        "",
        "## Short Mechanism Attribution",
    ]
    if mixed:
        lines.extend(
            [
                f"- Mixed short PPL: `{mixed['mean_val_ppl']}`.",
                f"- Fine ablation ΔPPL: `{mixed['mean_val_span_ablation_fine_delta_ppl']}`.",
                f"- Coarse ablation ΔPPL: `{mixed['mean_val_span_ablation_coarse_delta_ppl']}`.",
                f"- Effective range fine/coarse: `{mixed['mean_val_effective_range_fine']}` / `{mixed['mean_val_effective_range_coarse']}`.",
                f"- Routing entropy fine/coarse: `{mixed['mean_val_routing_entropy_fine']}` / `{mixed['mean_val_routing_entropy_coarse']}`.",
                f"- Routing top-1 fine/coarse: `{mixed['mean_val_routing_top1_fine']}` / `{mixed['mean_val_routing_top1_coarse']}`.",
            ]
        )
    lines.extend(["", "## Scale-L Sweep"])
    if scale_summary:
        for row in scale_summary:
            lines.append(
                f"- L={row['seq_len']} {row['variant']}: PPL `{row['mean_val_ppl']}`, peak VRAM `{row['mean_train_peak_vram_mib']}` MiB."
            )
    else:
        lines.append("- No completed scale rows summarized yet.")
    failed_scale = [
        row for row in run_rows if row.get("context") == "scale" and row.get("status") != "ok"
    ]
    if failed_scale:
        lines.append("")
        lines.append("Failed scale rows recorded under the fixed landmark/batch-1 guard:")
        for row in failed_scale:
            lines.append(
                f"- L={row['seq_len']} {row['variant']}: exit `{row.get('exit_code', '')}` ({row.get('failure', '')})."
            )
    if gaps:
        lines.append("")
        lines.append("## VRAM Gap Trend")
        for row in gaps:
            line = f"- L={row['seq_len']}: all-fine minus mixed = `{row['all_fine_minus_mixed_vram_mib']}` MiB."
            if "gap_trend_vs_previous" in row:
                line += f" Trend: `{row['gap_trend_vs_previous']}`."
            lines.append(line)
    lines.extend(["", "## Validation"])
    if log_failures:
        lines.append("- Log scan findings:")
        lines.extend(f"  - {item}" for item in log_failures)
    else:
        lines.append("- `scan_logs()` found no nan/inf/traceback/OOM markers in summarized logs.")
    (AUDIT / "SD_9_5_probes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    log_roots: list[Path] = []

    if args.mode in {"short", "all"}:
        root = ROOT / "out" / "paper_mechanisms" / "sd9_5_probes_short"
        status_rows = read_status(root / "sd9_5_short_status.tsv")
        status_by_key = {(r["variant"], int(r["seed"])): r for r in status_rows if r.get("seed")}
        for variant in SHORT_VARIANTS:
            for seed in [0, 1, 2]:
                run_rows.append(
                    collect_run(
                        root=root,
                        prefix="sd9_5",
                        role="short",
                        variant=variant,
                        seed=seed,
                        seq_len=512,
                        batch_size=16,
                        backend="exact",
                        coverage="NA",
                        epochs=10,
                        allow_missing=False,
                        status_by_key=status_by_key,
                    )
                )
        log_roots.append(ROOT / "logs" / "sd9_5_probes_short")

    if args.mode in {"scale", "all"}:
        for seq_len in discover_scale_lengths():
            root = ROOT / "out" / "paper_mechanisms" / f"sd9_5_scaleL_L{seq_len}"
            log_root = ROOT / "logs" / f"sd9_5_scaleL_L{seq_len}"
            if seq_len == 32768 and not (root / "sd9_5_scale_status.tsv").exists():
                smoke_root = ROOT / "out" / "paper_mechanisms" / f"sd9_5_scaleL_L{seq_len}_smoke"
                if (smoke_root / "sd9_5_scale_status.tsv").exists():
                    root = smoke_root
                    log_root = ROOT / "logs" / f"sd9_5_scaleL_L{seq_len}_smoke"
            status_rows = read_status(root / "sd9_5_scale_status.tsv")
            status_by_key = {(r["variant"], int(r["seed"])): r for r in status_rows if r.get("seed")}
            variants = list(SCALE_VARIANTS)
            if args.allow_failed_scale and status_by_key:
                variants = [variant for (variant, _seed) in status_by_key]
            for variant in variants:
                run_rows.append(
                    collect_run(
                        root=root,
                        prefix="sd9_5",
                        role="scale",
                        variant=variant,
                        seed=0,
                        seq_len=seq_len,
                        batch_size=1,
                        backend="landmark",
                        coverage=0.25,
                        epochs=10,
                        allow_missing=args.allow_failed_scale,
                        status_by_key=status_by_key,
                    )
                )
            log_roots.append(log_root)

    summary_rows = summarize(run_rows)
    short_summary = [r for r in summary_rows if r["context"] == "short"]
    scale_summary = [r for r in summary_rows if r["context"] == "scale"]
    gaps = scale_gaps(summary_rows)
    log_failures = scan_logs(log_roots)
    hard_log_failures = [
        item for item in log_failures if "OOM" not in item or not args.allow_failed_scale
    ]
    write_tsv(TABLES / "sd9_5_probes_runs.tsv", run_rows)
    write_tsv(TABLES / "sd9_5_probes_summary.tsv", summary_rows)
    write_tsv(TABLES / "sd9_5_scaleL_vram_gaps.tsv", gaps)
    manifest = {
        "phase": "SD-9.5",
        "runs": len(run_rows),
        "ok_runs": sum(1 for row in run_rows if row.get("status") == "ok"),
        "failed_runs": sum(1 for row in run_rows if row.get("status") != "ok"),
        "log_findings": log_failures,
        "validation_passed": not hard_log_failures,
        "tables": [
            "out/paper_integrated_evidence/tables/sd9_5_probes_runs.tsv",
            "out/paper_integrated_evidence/tables/sd9_5_probes_summary.tsv",
            "out/paper_integrated_evidence/tables/sd9_5_scaleL_vram_gaps.tsv",
        ],
    }
    (CHECKS / "sd9_5_probes_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_audit(
        run_rows=run_rows,
        short_summary=short_summary,
        scale_summary=scale_summary,
        gaps=gaps,
        log_failures=log_failures,
    )
    if hard_log_failures:
        raise SystemExit("SD-9.5 validation failed; see manifest and audit")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
