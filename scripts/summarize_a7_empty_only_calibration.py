#!/usr/bin/env python3
"""Validate and summarize A7 empty_only calibration runs."""

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
RAW = ROOT / "out" / "paper_mechanisms" / "a7_empty_only_calibration"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"
LOG_ROOTS = [
    ROOT / "logs" / "a7_empty_only_calibration",
    ROOT / "logs" / "a7_candidate2_extension",
]
BASELINE_SUMMARY = TABLES / "a2_lrnorm_headline_all_runs.tsv"

SEQ_LEN = 512
D_MODEL = 384
D_FF = 1536
EPOCHS = 10
LR = "1e-4"
BASELINE_SEEDS = [0, 1, 2]
SPECS = [
    (1, 1, [0, 1, 2]),
    (2, 1, [0, 1, 2]),
    (3, 1, [0, 1, 2]),
    (2, 2, [0, 1, 2]),
    (4, 2, [0, 1, 2]),
    (8, 4, [0, 1, 2]),
    (16, 8, [0, 1, 2]),
    (32, 16, [0, 1, 2]),
]

RESOLVED_KEYS = [
    "resolved.d_phi",
    "resolved.set_state_dim",
    "resolved.adapter_type",
    "resolved.router_min_temp",
    "resolved.pooling_alpha",
    "resolved.hash_seed",
    "resolved.hash_normalize",
    "resolved.hash_num_bins",
    "resolved.output_residual_mode",
]


@dataclass(frozen=True)
class ExpectedSetRun:
    w: int
    s: int
    seed: int

    @property
    def m(self) -> int:
        return ((SEQ_LEN - self.w) // self.s) + 1

    @property
    def group(self) -> str:
        return "a7_empty_only_calibration_set_dense_D384_FF1536"

    @property
    def name(self) -> str:
        return (
            f"a7_empty_set_dense_D384_FF1536_L{SEQ_LEN}_w{self.w}_s{self.s}_"
            f"M{self.m}_lr{LR.replace('.', 'p')}_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return RAW / self.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


def expected_set_runs() -> list[ExpectedSetRun]:
    return [ExpectedSetRun(w, s, seed) for w, s, seeds in SPECS for seed in seeds]


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


def run(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }


def finite_csv(rows: list[dict[str, str]]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for i, row in enumerate(rows, 1):
        for key, raw in row.items():
            if isinstance(raw, str) and raw.strip().lower() in {"nan", "inf", "-inf"}:
                bad.append(f"row {i}: {key}={raw}")
    return not bad, bad


def scan_logs() -> list[str]:
    log_roots = [root for root in LOG_ROOTS if root.exists()]
    if not log_roots:
        return [f"missing log roots: {', '.join(str(root.relative_to(ROOT)) for root in LOG_ROOTS)}"]
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
    for root in log_roots:
        for path in sorted(root.glob("*.log")):
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


def load_baselines() -> list[dict[str, str]]:
    if not BASELINE_SUMMARY.exists():
        raise FileNotFoundError(f"missing baseline summary: {BASELINE_SUMMARY}")
    out: list[dict[str, str]] = []
    with BASELINE_SUMMARY.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if (
                row.get("family") == "Baseline token"
                and row.get("backend") == "exact"
                and row.get("D") == str(D_MODEL)
                and row.get("d_ff") == str(D_FF)
                and row.get("lr") == LR
                and row.get("seed") in {str(s) for s in BASELINE_SEEDS}
            ):
                csv_path = ROOT / row["csv_path"]
                json_path = ROOT / row["json_path"]
                if not csv_path.exists():
                    raise FileNotFoundError(f"missing baseline CSV: {csv_path}")
                if not json_path.exists():
                    raise FileNotFoundError(f"missing baseline JSON: {json_path}")
                rows = read_rows(csv_path)
                if len(rows) != EPOCHS:
                    raise ValueError(f"{csv_path} has {len(rows)} rows, expected {EPOCHS}")
                ok, bad = finite_csv(rows)
                if not ok:
                    raise ValueError(f"non-finite baseline CSV values in {csv_path}: {bad[:5]}")
                last = rows[-1]
                out.append({
                    "phase": "A2.2_reused_baseline",
                    "family_slug": "baseline_dense_exact",
                    "family": "Baseline Dense",
                    "backend": "exact",
                    "seed": row["seed"],
                    "lr": row["lr"],
                    "D": row["D"],
                    "d_ff": row["d_ff"],
                    "L": str(SEQ_LEN),
                    "w": "NA",
                    "s": "NA",
                    "M": "NA",
                    "M_over_L": "NA",
                    "L_over_M": "NA",
                    "empty_fraction": "NA",
                    "output_residual_mode": "NA",
                    "config": row["config"],
                    "csv_path": row["csv_path"],
                    "json_path": row["json_path"],
                    "rows": str(len(rows)),
                    "final_train_loss": last["train/loss"],
                    "final_val_loss": last["val/loss"],
                    "final_train_ppl": last["train/ppl"],
                    "final_val_ppl": last["val/ppl"],
                    "time_per_epoch_s": last.get("train/time_per_epoch_s", row.get("time_per_epoch_s", "NA")),
                    "peak_vram_mib": last.get("train/peak_vram_mib", row.get("peak_vram_mib", "NA")),
                    "candidate_count_mean": "NA",
                    "candidate_count_max": "NA",
                    "router_entropy_norm": "NA",
                    "router_top1_weight": "NA",
                    "source_csv_sha256": sha256(csv_path),
                    "source_json_sha256": sha256(json_path),
                    "comparison_provenance": "reused from A2 dense LR-normalized headline baseline",
                })
    seeds = sorted(int(r["seed"]) for r in out)
    if seeds != BASELINE_SEEDS:
        raise ValueError(f"baseline seeds mismatch: {seeds}, expected {BASELINE_SEEDS}")
    return sorted(out, key=lambda r: int(r["seed"]))


def validate_set_run(spec: ExpectedSetRun) -> dict[str, str]:
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {spec.csv_path}")
    if not spec.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {spec.json_path}")
    rows = read_rows(spec.csv_path)
    if len(rows) != EPOCHS:
        raise ValueError(f"{spec.csv_path} has {len(rows)} rows, expected {EPOCHS}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, EPOCHS + 1)):
        raise ValueError(f"{spec.csv_path} epochs are not 1..{EPOCHS}: {epochs}")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite CSV values in {spec.csv_path}: {bad[:5]}")

    meta = json.loads(spec.json_path.read_text())
    missing = [key for key in RESOLVED_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{spec.json_path} missing metadata keys: {missing}")
    checks = {
        "model.implementation": "set_only",
        "model.backend": "exact",
        "model.set_causality_mode": "strict_past",
        "model.output_residual_mode": "empty_only",
        "model.d_model": D_MODEL,
        "model.dim_feedforward": D_FF,
        "model.max_seq_len": SEQ_LEN,
        "model.window_size": spec.w,
        "model.stride": spec.s,
        "model.d_phi": D_MODEL,
        "model.set_state_dim": D_MODEL,
        "resolved.d_phi": D_MODEL,
        "resolved.set_state_dim": D_MODEL,
        "resolved.output_residual_mode": "empty_only",
        "model.feature_mode": "geometry_only",
        "model.geometry.enabled": False,
        "model.token_mlp.enabled": False,
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if str(actual) != str(expected):
            raise ValueError(f"{spec.json_path} has {key}={actual!r}, expected {expected!r}")
    expected_allow = spec.w == 1 and spec.s == 1
    if str(meta.get("model.allow_token_token")) != str(expected_allow):
        raise ValueError(
            f"{spec.json_path} has allow_token_token={meta.get('model.allow_token_token')}, "
            f"expected {expected_allow}"
        )

    last = rows[-1]
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl"]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{spec.csv_path} missing finite final {key}")
    for key in [
        "ausa/candidate_count_mean",
        "ausa/candidate_count_max",
        "ausa/router_entropy_norm",
        "ausa/router_top1_weight",
    ]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{spec.csv_path} missing finite {key}")

    m_over_l = spec.m / SEQ_LEN
    empty_fraction = (spec.w - 1) / SEQ_LEN
    return {
        "phase": "A7",
        "family_slug": "set_dense_exact_empty_only",
        "family": "Set Dense empty_only",
        "backend": "exact",
        "seed": str(spec.seed),
        "lr": LR,
        "D": str(D_MODEL),
        "d_ff": str(D_FF),
        "L": str(SEQ_LEN),
        "w": str(spec.w),
        "s": str(spec.s),
        "M": str(spec.m),
        "M_over_L": f"{m_over_l:.8f}",
        "L_over_M": f"{SEQ_LEN / spec.m:.8f}",
        "empty_fraction": f"{empty_fraction:.8f}",
        "output_residual_mode": "empty_only",
        "config": "configs/paper_lr_norm/set_dense_exact.yaml",
        "csv_path": str(spec.csv_path.relative_to(ROOT)),
        "json_path": str(spec.json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_train_loss": last["train/loss"],
        "final_val_loss": last["val/loss"],
        "final_train_ppl": last["train/ppl"],
        "final_val_ppl": last["val/ppl"],
        "time_per_epoch_s": last.get("train/time_per_epoch_s", "NA"),
        "peak_vram_mib": last.get("train/peak_vram_mib", "NA"),
        "candidate_count_mean": last.get("ausa/candidate_count_mean", "NA"),
        "candidate_count_max": last.get("ausa/candidate_count_max", "NA"),
        "router_entropy_norm": last.get("ausa/router_entropy_norm", "NA"),
        "router_top1_weight": last.get("ausa/router_top1_weight", "NA"),
        "source_csv_sha256": sha256(spec.csv_path),
        "source_json_sha256": sha256(spec.json_path),
        "comparison_provenance": "new A7 empty_only set-side run",
    }


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["family_slug"], row["w"], row["s"], row["lr"])].append(row)

    baseline_vals = [
        float(row["final_val_ppl"])
        for row in rows
        if row["family_slug"] == "baseline_dense_exact"
    ]
    baseline_mean = mean(baseline_vals) if baseline_vals else float("nan")

    summary = []
    for (family_slug, w, s, lr), grp in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        vals = [float(row["final_val_ppl"]) for row in grp]
        train_vals = [float(row["final_train_ppl"]) for row in grp]
        times = [float_or_none(row["time_per_epoch_s"]) for row in grp]
        vrams = [float_or_none(row["peak_vram_mib"]) for row in grp]
        times_f = [v for v in times if v is not None]
        vrams_f = [v for v in vrams if v is not None]
        first = grp[0]
        mean_val = mean(vals)
        summary.append({
            "family_slug": family_slug,
            "family": first["family"],
            "backend": first["backend"],
            "lr": lr,
            "D": first["D"],
            "d_ff": first["d_ff"],
            "L": first["L"],
            "w": w,
            "s": s,
            "M": first["M"],
            "M_over_L": first["M_over_L"],
            "L_over_M": first["L_over_M"],
            "empty_fraction": first["empty_fraction"],
            "output_residual_mode": first["output_residual_mode"],
            "n": str(len(grp)),
            "seeds": ",".join(sorted((row["seed"] for row in grp), key=int)),
            "mean_final_val_ppl": f"{mean_val:.6f}",
            "std_final_val_ppl": f"{pstdev(vals):.6f}" if len(vals) > 1 else "0.000000",
            "min_final_val_ppl": f"{min(vals):.6f}",
            "max_final_val_ppl": f"{max(vals):.6f}",
            "mean_final_train_ppl": f"{mean(train_vals):.6f}",
            "mean_time_per_epoch_s": f"{mean(times_f):.6f}" if times_f else "NA",
            "mean_peak_vram_mib": f"{mean(vrams_f):.6f}" if vrams_f else "NA",
            "delta_vs_baseline_mean_ppl": (
                f"{mean_val - baseline_mean:.6f}"
                if family_slug != "baseline_dense_exact" and math.isfinite(baseline_mean)
                else "0.000000"
            ),
        })
    return summary


def main() -> None:
    failures: list[str] = []
    set_rows: list[dict[str, str]] = []
    baseline_rows: list[dict[str, str]] = []
    try:
        baseline_rows = load_baselines()
    except Exception as exc:
        failures.append(str(exc))
    for spec in expected_set_runs():
        try:
            set_rows.append(validate_set_run(spec))
        except Exception as exc:
            failures.append(str(exc))

    failures.extend(scan_logs())
    all_rows = baseline_rows + sorted(
        set_rows, key=lambda r: (int(r["w"]), int(r["s"]), int(r["seed"]))
    )

    fields = [
        "phase",
        "family_slug",
        "family",
        "backend",
        "seed",
        "lr",
        "D",
        "d_ff",
        "L",
        "w",
        "s",
        "M",
        "M_over_L",
        "L_over_M",
        "empty_fraction",
        "output_residual_mode",
        "config",
        "csv_path",
        "json_path",
        "rows",
        "final_train_loss",
        "final_val_loss",
        "final_train_ppl",
        "final_val_ppl",
        "time_per_epoch_s",
        "peak_vram_mib",
        "candidate_count_mean",
        "candidate_count_max",
        "router_entropy_norm",
        "router_top1_weight",
        "source_csv_sha256",
        "source_json_sha256",
        "comparison_provenance",
    ]
    summary_fields = [
        "family_slug",
        "family",
        "backend",
        "lr",
        "D",
        "d_ff",
        "L",
        "w",
        "s",
        "M",
        "M_over_L",
        "L_over_M",
        "empty_fraction",
        "output_residual_mode",
        "n",
        "seeds",
        "mean_final_val_ppl",
        "std_final_val_ppl",
        "min_final_val_ppl",
        "max_final_val_ppl",
        "mean_final_train_ppl",
        "mean_time_per_epoch_s",
        "mean_peak_vram_mib",
        "delta_vs_baseline_mean_ppl",
    ]

    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    all_path = TABLES / "a7_empty_only_calibration_all_runs.tsv"
    summary_path = TABLES / "a7_empty_only_calibration_summary.tsv"
    manifest_path = CHECKS / "a7_empty_only_calibration_manifest.json"
    audit_path = AUDIT / "A7_empty_only_calibration.md"

    write_tsv(all_path, all_rows, fields)
    summary_rows = summarize(all_rows) if all_rows else []
    write_tsv(summary_path, summary_rows, summary_fields)

    manifest = {
        "phase": "A7",
        "status": "pass" if not failures else "fail",
        "expected_new_set_runs": len(expected_set_runs()),
        "validated_new_set_runs": len(set_rows),
        "reused_baseline_runs": len(baseline_rows),
        "failures": failures,
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "dirty_status": run(["git", "status", "--short"]),
            "artifacts": {
            "all_runs": str(all_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
            "audit": str(audit_path.relative_to(ROOT)),
        },
        "extension_note": (
            "Includes the candidate-count-near-2 extension: w=2,s=1 seeds 0,1,2; "
            "w=3,s=1 seeds 0,1,2; w=2,s=2 seeds 0,1,2; and completes "
            "w=8,s=4 to seeds 0,1,2."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    audit_lines = [
        "# A7 Empty-Only Calibration",
        "",
        f"Status: {manifest['status'].upper()}",
        "",
        "## Baseline Provenance",
        "",
        "Dense token baseline rows are reused from A2, filtered by exact match: "
        "`Baseline token`, `backend=exact`, `D=384`, `d_ff=1536`, `L=512`, "
        "`epochs=10`, `lr=1e-4`, seeds `0,1,2`.",
        "",
        "## Artifacts",
        "",
        f"- All runs: `{all_path.relative_to(ROOT)}`",
        f"- Summary: `{summary_path.relative_to(ROOT)}`",
        f"- Manifest: `{manifest_path.relative_to(ROOT)}`",
        "",
        "## Validation",
        "",
        f"- New set-side runs: {len(set_rows)}/{len(expected_set_runs())}",
        f"- Reused baseline runs: {len(baseline_rows)}/3",
        f"- Log scan failures: {len(scan_logs())}",
        "",
        "## Candidate-Count Extension",
        "",
        "- `w=2,s=1`: completes the original one-seed point to three seeds.",
        "- `w=3,s=1`: tests a valid high-`M/L` topology with mean candidate count near three.",
        "- `w=2,s=2`: controls for the same window size under non-overlapping endpoint topology.",
        "- `w=8,s=4`: completes the remaining original one-seed topology to three seeds.",
    ]
    if failures:
        audit_lines += ["", "## Failures", ""]
        audit_lines += [f"- {failure}" for failure in failures]
    else:
        audit_lines += [
            "",
            "## Interpretation Guardrail",
            "",
            "A7 tests empirical convergence under the calibrated `empty_only` residual policy. "
            "It should not be described as exact Transformer equivalence; the set-side path "
            "still uses set pooling, set-stack processing, and routing projections.",
        ]
    audit_path.write_text("\n".join(audit_lines) + "\n")

    print(json.dumps(manifest, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
