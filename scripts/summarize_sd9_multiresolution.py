#!/usr/bin/env python3
"""Validate and summarize SD-9 multi-resolution frontier artifacts."""

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
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"
AUDIT = ROOT / "audit"


@dataclass(frozen=True)
class RoleSpec:
    role: str
    raw: Path
    logs: Path
    seq_len: int
    batch_size: int
    backend: str
    attention_family: str
    coverage: float | str
    epochs: int
    seeds: list[int]


def role_specs(mode: str) -> list[RoleSpec]:
    suffix = "_smoke" if mode == "smoke" else ""
    epochs = 1 if mode == "smoke" else 10
    seeds = [0] if mode == "smoke" else [0, 1, 2]
    return [
        RoleSpec(
            role="short",
            raw=ROOT / "out" / "paper_mechanisms" / f"sd9_multiresolution_short{suffix}",
            logs=ROOT / "logs" / f"sd9_multiresolution_short{suffix}",
            seq_len=512,
            batch_size=16,
            backend="exact",
            attention_family="dense",
            coverage="NA",
            epochs=epochs,
            seeds=seeds,
        ),
        RoleSpec(
            role="long",
            raw=ROOT / "out" / "paper_mechanisms" / f"sd9_multiresolution_long{suffix}",
            logs=ROOT / "logs" / f"sd9_multiresolution_long{suffix}",
            seq_len=8192,
            batch_size=1,
            backend="landmark",
            attention_family="linear",
            coverage=0.25,
            epochs=epochs,
            seeds=seeds,
        ),
    ]


def variants_for(role: str, mode: str) -> list[dict[str, object]]:
    if mode == "smoke":
        return [
            {
                "variant": "mixed",
                "label": "mixed-25" if role == "short" else "mixed-65",
                "fine_heads": 6 if role == "short" else 3,
                "coarse_heads": 2 if role == "short" else 5,
                "groups": [("fine", 2, 1), ("coarse", 4, 2)],
            }
        ]
    return [
        {
            "variant": "mixed",
            "label": "mixed-25" if role == "short" else "mixed-65",
            "fine_heads": 6 if role == "short" else 3,
            "coarse_heads": 2 if role == "short" else 5,
            "groups": [("fine", 2, 1), ("coarse", 4, 2)],
        },
        {
            "variant": "all_fine",
            "label": "all-fine",
            "fine_heads": 8,
            "coarse_heads": 0,
            "groups": [("fine", 2, 1)],
        },
        {
            "variant": "all_coarse",
            "label": "all-coarse",
            "fine_heads": 0,
            "coarse_heads": 8,
            "groups": [("coarse", 4, 2)],
        },
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        value = float(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{key} is not numeric in {row.get('run_name', '<unknown>')}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite in {row.get('run_name', '<unknown>')}: {raw!r}")
    return value


def meta_value(meta: dict[str, object], key: str) -> object:
    return meta.get(key)


def meta_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return str(actual) == str(expected)
    if isinstance(expected, (int, float)):
        try:
            value = float(actual)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and math.isclose(value, float(expected), rel_tol=1e-9, abs_tol=1e-12)
    return str(actual) == str(expected)


def find_csv(spec: RoleSpec, variant: str, seed: int) -> Path:
    pattern = f"sd9_{spec.role}_{variant}_*_seed{seed}.csv"
    matches = sorted(spec.raw.glob(f"**/{pattern}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{spec.role} {variant} seed {seed}: expected one CSV for {pattern}, found {len(matches)}"
        )
    return matches[0]


def scan_logs(log_roots: list[Path]) -> list[str]:
    substr_patterns = ["OOM", "out of memory", "Traceback", "RuntimeError", "ValueError"]
    token_re = re.compile(r"(?<![A-Za-z0-9_])(?:nan|NaN|-inf|inf)(?![A-Za-z0-9_])")
    failures: list[str] = []
    for log_root in log_roots:
        if not log_root.exists():
            failures.append(f"missing log root: {log_root.relative_to(ROOT)}")
            continue
        for path in sorted(log_root.glob("*.log")):
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


def summarize(values: list[float]) -> dict[str, float]:
    sd = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean(values),
        "std": sd,
        "ci95": 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0,
    }


def validate_group_metadata(meta: dict[str, object], expected: dict[str, object], seq_len: int, backend: str) -> tuple[int | str, int | str]:
    groups = meta.get("resolved.multiresolution_groups")
    if not isinstance(groups, list):
        raise ValueError(f"{meta.get('run_name')}: resolved.multiresolution_groups is not a list")
    if len(groups) != len(expected["groups"]):  # type: ignore[arg-type]
        raise ValueError(f"{meta.get('run_name')}: unexpected multiresolution group count {len(groups)}")
    by_name = {str(group.get("name")): group for group in groups if isinstance(group, dict)}
    m_fine: int | str = "NA"
    m_coarse: int | str = "NA"
    for name, w, s in expected["groups"]:  # type: ignore[union-attr]
        if name not in by_name:
            raise ValueError(f"{meta.get('run_name')}: missing group {name}")
        group = by_name[name]
        expected_m = ((seq_len - w) // s) + 1
        checks = {
            "window_size": w,
            "stride": s,
            "M": expected_m,
        }
        for key, value in checks.items():
            if not meta_matches(group.get(key), value):
                raise ValueError(
                    f"{meta.get('run_name')}: group {name} {key} expected {value}, got {group.get(key)}"
                )
        if backend == "landmark":
            expected_lm = min(max(round(0.25 * expected_m), 2), expected_m)
            if not meta_matches(group.get("landmark_count"), expected_lm):
                raise ValueError(
                    f"{meta.get('run_name')}: group {name} landmark_count expected {expected_lm}, got {group.get('landmark_count')}"
                )
        if name == "fine":
            m_fine = expected_m
        if name == "coarse":
            m_coarse = expected_m
    return m_fine, m_coarse


def validate_run(spec: RoleSpec, variant: dict[str, object], seed: int) -> dict[str, object]:
    csv_path = find_csv(spec, str(variant["variant"]), seed)
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"missing JSON: {json_path}")
    rows = read_csv(csv_path)
    if len(rows) != spec.epochs:
        raise ValueError(f"{csv_path}: expected {spec.epochs} epochs, found {len(rows)}")
    final = rows[-1]
    if int(float(final["epoch"])) != spec.epochs:
        raise ValueError(f"{csv_path}: final epoch expected {spec.epochs}, got {final['epoch']}")
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl", "train/peak_vram_mib"]:
        as_float(final, key)
    for row in rows:
        for key, raw in row.items():
            if isinstance(raw, str) and raw.strip().lower() in {"nan", "inf", "-inf"}:
                raise ValueError(f"{csv_path}: {key} has nonfinite token {raw!r}")

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    checks = {
        "model.implementation": "set_only",
        "model.attention_family": spec.attention_family,
        "model.backend": spec.backend,
        "model.set_causality_mode": "strict_past",
        "model.output_residual_mode": "anchor_span",
        "model.anchor.enabled": False,
        "model.anchor.teacher.enabled": False,
        "model.token_mlp.enabled": False,
        "model.candidate_fiber": "endpoint_window",
        "model.set_diversity.lambda_div": 0.0,
        "model.multivector_basis.enabled": False,
        "model.multivector_basis.r": 1,
        "model.multiresolution.enabled": True,
        "model.d_model": 384,
        "model.dim_feedforward": 1536,
        "model.num_layers": 6,
        "model.num_heads": 8,
        "model.max_seq_len": spec.seq_len,
        "model.d_phi": 384,
        "model.set_state_dim": 384,
        "data.batch_size": spec.batch_size,
        "data.seq_len": spec.seq_len,
        "training.epochs": spec.epochs,
        "training.lr": 0.0001,
        "training.seed": seed,
        "resolved.output_residual_mode": "anchor_span",
        "resolved.anchor_enabled": False,
        "resolved.candidate_fiber": "endpoint_window",
        "resolved.multiresolution_enabled": True,
    }
    if spec.backend == "landmark":
        checks["model.backend_params.landmark_coverage"] = 0.25
        checks["resolved.landmark_coverage"] = 0.25
    for key, expected in checks.items():
        actual = meta_value(meta, key)
        if not meta_matches(actual, expected):
            raise ValueError(f"{csv_path}: {key} expected {expected!r}, got {actual!r}")

    m_fine, m_coarse = validate_group_metadata(meta, variant, spec.seq_len, spec.backend)
    coarse_heads = int(variant["coarse_heads"])
    fine_heads = int(variant["fine_heads"])
    return {
        "context": spec.role,
        "variant": variant["variant"],
        "label": variant["label"],
        "seed": seed,
        "backend": spec.backend,
        "landmark_coverage": spec.coverage,
        "seq_len": spec.seq_len,
        "batch_size": spec.batch_size,
        "epochs": spec.epochs,
        "fine_heads": fine_heads,
        "coarse_heads": coarse_heads,
        "blur_fraction": coarse_heads / 8.0,
        "fine_w": 2 if fine_heads else "NA",
        "fine_s": 1 if fine_heads else "NA",
        "coarse_w": 4 if coarse_heads else "NA",
        "coarse_s": 2 if coarse_heads else "NA",
        "M_fine": m_fine,
        "M_coarse": m_coarse,
        "final_train_ppl": as_float(final, "train/ppl"),
        "final_val_ppl": as_float(final, "val/ppl"),
        "peak_vram_mib": as_float(final, "train/peak_vram_mib"),
        "span_ablation_delta_ppl": (
            as_float(final, "val/span_ablation_delta_ppl")
            if final.get("val/span_ablation_delta_ppl") not in {None, "", "NA"}
            else "NA"
        ),
        "router_entropy": final.get("ausa/router_entropy_mean", "NA"),
        "router_top1_share": final.get("ausa/router_top1_share_mean", "NA"),
        "pooling_alpha": meta.get("resolved.pooling_alpha", "NA"),
        "csv_path": str(csv_path.relative_to(ROOT)),
        "json_path": str(json_path.relative_to(ROOT)),
        "csv_sha256": sha256(csv_path),
    }


def pareto_verdict(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    verdicts: list[dict[str, object]] = []
    for context in sorted({str(row["context"]) for row in summary_rows}):
        rows = {str(row["variant"]): row for row in summary_rows if row["context"] == context}
        if not {"mixed", "all_fine", "all_coarse"}.issubset(rows):
            continue
        mixed = rows["mixed"]
        fine = rows["all_fine"]
        coarse = rows["all_coarse"]
        p = float(mixed["blur_fraction"])
        interp_ppl = (1.0 - p) * float(fine["mean_val_ppl"]) + p * float(coarse["mean_val_ppl"])
        interp_vram = (1.0 - p) * float(fine["mean_peak_vram_mib"]) + p * float(coarse["mean_peak_vram_mib"])
        ppl_gain = interp_ppl - float(mixed["mean_val_ppl"])
        vram_gain = interp_vram - float(mixed["mean_peak_vram_mib"])
        verdicts.append(
            {
                "context": context,
                "mixed_label": mixed["label"],
                "blur_fraction": p,
                "mixed_mean_val_ppl": mixed["mean_val_ppl"],
                "interp_val_ppl": interp_ppl,
                "delta_ppl_vs_interp": -ppl_gain,
                "mixed_mean_peak_vram_mib": mixed["mean_peak_vram_mib"],
                "interp_peak_vram_mib": interp_vram,
                "delta_vram_vs_interp_mib": -vram_gain,
                "pareto_better_than_interpolation": ppl_gain > 0.0 and vram_gain > 0.0,
            }
        )
    return verdicts


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--role", choices=["all", "short", "long"], default="all")
    args = parser.parse_args()

    specs = role_specs(args.mode)
    if args.role != "all":
        specs = [spec for spec in specs if spec.role == args.role]

    failures: list[str] = []
    run_rows: list[dict[str, object]] = []
    for spec in specs:
        status_path = spec.raw / f"sd9_multiresolution_{spec.role}_status.tsv"
        if not status_path.exists():
            failures.append(f"missing status TSV: {status_path.relative_to(ROOT)}")
        else:
            status_rows = list(csv.DictReader(status_path.open(encoding="utf-8"), delimiter="\t"))
            expected_count = len(variants_for(spec.role, args.mode)) * len(spec.seeds)
            if len(status_rows) != expected_count:
                failures.append(
                    f"{status_path.relative_to(ROOT)} expected {expected_count} rows, found {len(status_rows)}"
                )
            bad = [row for row in status_rows if row.get("exit_code") != "0"]
            if bad:
                failures.append(f"{status_path.relative_to(ROOT)} nonzero exits: {bad}")
        for variant in variants_for(spec.role, args.mode):
            for seed in spec.seeds:
                try:
                    run_rows.append(validate_run(spec, variant, seed))
                except Exception as exc:  # noqa: BLE001
                    failures.append(str(exc))

    failures.extend(scan_logs([spec.logs for spec in specs]))

    suffix = "_smoke" if args.mode == "smoke" else ""
    role_suffix = "" if args.role == "all" else f"_{args.role}"
    run_table = TABLES / f"sd9_multiresolution{suffix}{role_suffix}_runs.tsv"
    summary_table = TABLES / f"sd9_multiresolution{suffix}{role_suffix}_summary.tsv"
    verdict_table = TABLES / f"sd9_multiresolution{suffix}{role_suffix}_verdict.tsv"
    manifest_path = CHECKS / f"sd9_multiresolution{suffix}{role_suffix}_manifest.json"
    audit_path = AUDIT / ("SD_9_multiresolution_smoke.md" if args.mode == "smoke" else "SD_9_multiresolution.md")

    write_tsv(run_table, run_rows)
    summary_rows: list[dict[str, object]] = []
    for key in sorted({(row["context"], row["variant"]) for row in run_rows}):
        context, variant = key
        group = [row for row in run_rows if row["context"] == context and row["variant"] == variant]
        val = summarize([float(row["final_val_ppl"]) for row in group])
        train = summarize([float(row["final_train_ppl"]) for row in group])
        vram = summarize([float(row["peak_vram_mib"]) for row in group])
        span_values = [
            float(row["span_ablation_delta_ppl"])
            for row in group
            if row["span_ablation_delta_ppl"] != "NA"
        ]
        span = summarize(span_values) if span_values else {"mean": "NA", "std": "NA", "ci95": "NA"}
        first = group[0]
        summary_rows.append(
            {
                "context": context,
                "variant": variant,
                "label": first["label"],
                "n": len(group),
                "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda r: int(r["seed"]))),
                "backend": first["backend"],
                "landmark_coverage": first["landmark_coverage"],
                "seq_len": first["seq_len"],
                "batch_size": first["batch_size"],
                "fine_heads": first["fine_heads"],
                "coarse_heads": first["coarse_heads"],
                "blur_fraction": first["blur_fraction"],
                "M_fine": first["M_fine"],
                "M_coarse": first["M_coarse"],
                "mean_val_ppl": val["mean"],
                "std_val_ppl": val["std"],
                "ci95_val_ppl": val["ci95"],
                "mean_train_ppl": train["mean"],
                "std_train_ppl": train["std"],
                "mean_peak_vram_mib": vram["mean"],
                "std_peak_vram_mib": vram["std"],
                "mean_span_ablation_delta_ppl": span["mean"],
                "std_span_ablation_delta_ppl": span["std"],
            }
        )
    write_tsv(summary_table, summary_rows)
    verdict_rows = pareto_verdict(summary_rows)
    write_tsv(verdict_table, verdict_rows)

    manifest = {
        "status": "PASS" if not failures else "FAIL",
        "mode": args.mode,
        "role": args.role,
        "validated_runs": len(run_rows),
        "failures": failures,
        "run_table": str(run_table.relative_to(ROOT)),
        "summary_table": str(summary_table.relative_to(ROOT)),
        "verdict_table": str(verdict_table.relative_to(ROOT)),
        "audit": str(audit_path.relative_to(ROOT)),
        "git_head": run(["git", "rev-parse", "HEAD"]),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# SD-9 Multi-Resolution Frontier Test",
        "",
        f"Status: {'PASS' if not failures else 'FAIL'}",
        f"Mode: `{args.mode}`",
        "",
        "## Contract",
        "",
        "- Short: L=512, batch=16, dense exact backend on blue-demon.",
        "- Long: L=8192, batch=1, landmark backend with coverage 0.25 on lizmark.",
        "- CE-only, anchor disabled, token MLP disabled, `output_residual_mode=anchor_span`, `candidate_fiber=endpoint_window`.",
        "- Reference only: token baseline 781.1 at L=512; A8.3 L=8192 landmark set 2181.3 / token baseline 1048.4.",
        "",
        "## Verdicts",
        "",
    ]
    if verdict_rows:
        for row in verdict_rows:
            lines.append(
                "- {context}: {label} Pareto-better-than-interpolation = `{verdict}` "
                "(delta PPL vs interpolation {dppl:.4f}, delta VRAM {dvram:.2f} MiB).".format(
                    context=row["context"],
                    label=row["mixed_label"],
                    verdict=row["pareto_better_than_interpolation"],
                    dppl=float(row["delta_ppl_vs_interp"]),
                    dvram=float(row["delta_vram_vs_interp_mib"]),
                )
            )
    else:
        lines.append("- Full Pareto verdicts require mixed, all-fine, and all-coarse rows.")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Runs: `{run_table.relative_to(ROOT)}`",
        f"- Summary: `{summary_table.relative_to(ROOT)}`",
        f"- Verdict: `{verdict_table.relative_to(ROOT)}`",
        f"- Manifest: `{manifest_path.relative_to(ROOT)}`",
    ]
    if failures:
        lines += ["", "## Failures", ""]
        lines.extend(f"- {failure}" for failure in failures)
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failures:
        raise SystemExit("SD-9 validation failed; see manifest and audit")


if __name__ == "__main__":
    main()
