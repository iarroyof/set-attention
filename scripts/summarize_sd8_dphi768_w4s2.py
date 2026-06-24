#!/usr/bin/env python3
"""Validate and summarize the SD-8.1 d_phi/set_state_dim=768 (4,2) follow-up."""

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


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"
AUDIT = ROOT / "audit"


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    raw: Path
    logs: Path
    config: str
    group: str
    table: Path
    comparison_table: Path
    manifest: Path
    audit: Path
    seq_len: int
    d_model: int
    d_ff: int
    layers: int
    heads: int
    d_phi: int
    set_state_dim: int
    epochs: int
    batch_size: int
    seeds: list[int]
    hash_bins: int


def mode_spec(mode: str) -> ModeSpec:
    if mode == "smoke":
        slug = "sd8_all_past_dense_dphi768_w4s2_smoke"
        return ModeSpec(
            mode=mode,
            raw=ROOT / "out" / "paper_mechanisms" / slug,
            logs=ROOT / "logs" / slug,
            config="configs/set_dictionary/sd8_all_past_dense_dphi768_w4s2_smoke.yaml",
            group=f"{slug}_D64_FF128_dphi128_setdim128",
            table=TABLES / f"{slug}_runs.tsv",
            comparison_table=TABLES / f"{slug}_comparison.tsv",
            manifest=CHECKS / f"{slug}_manifest.json",
            audit=AUDIT / "SD_8_1_dphi768_w4s2_smoke.md",
            seq_len=64,
            d_model=64,
            d_ff=128,
            layers=1,
            heads=4,
            d_phi=128,
            set_state_dim=128,
            epochs=1,
            batch_size=2,
            seeds=[0],
            hash_bins=32,
        )
    slug = "sd8_all_past_dense_dphi768_w4s2"
    return ModeSpec(
        mode=mode,
        raw=ROOT / "out" / "paper_mechanisms" / slug,
        logs=ROOT / "logs" / slug,
        config="configs/set_dictionary/sd8_all_past_dense_dphi768_w4s2.yaml",
        group=f"{slug}_D384_FF1536_dphi768_setdim768",
        table=TABLES / f"{slug}_runs.tsv",
        comparison_table=TABLES / f"{slug}_comparison.tsv",
        manifest=CHECKS / f"{slug}_manifest.json",
        audit=AUDIT / "SD_8_1_dphi768_w4s2.md",
        seq_len=512,
        d_model=384,
        d_ff=1536,
        layers=6,
        heads=8,
        d_phi=768,
        set_state_dim=768,
        epochs=10,
        batch_size=16,
        seeds=[0, 1, 2],
        hash_bins=128,
    )


@dataclass(frozen=True)
class ExpectedRun:
    spec: ModeSpec
    seed: int
    w: int = 4
    s: int = 2

    @property
    def m(self) -> int:
        return ((self.spec.seq_len - self.w) // self.s) + 1

    @property
    def name(self) -> str:
        return (
            f"{self.spec.group}_L{self.spec.seq_len}_w{self.w}_s{self.s}_"
            f"M{self.m}_lr1e-4_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return self.spec.raw / self.spec.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


def expected_runs(spec: ModeSpec) -> list[ExpectedRun]:
    return [ExpectedRun(spec, seed) for seed in spec.seeds]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
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


def meta_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return str(actual) == str(expected)
    if isinstance(expected, (int, float)):
        parsed = float_or_none(actual)
        return parsed is not None and math.isclose(parsed, float(expected), rel_tol=1e-9, abs_tol=1e-12)
    return str(actual) == str(expected)


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
                failures.append(f"{path.relative_to(ROOT)} contains standalone token {match.group()!r}")
    return failures


def validate_run(run_spec: ExpectedRun) -> dict[str, str]:
    spec = run_spec.spec
    if not run_spec.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {run_spec.csv_path}")
    if not run_spec.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {run_spec.json_path}")

    rows = read_csv(run_spec.csv_path)
    if len(rows) != spec.epochs:
        raise ValueError(f"{run_spec.csv_path} has {len(rows)} rows, expected {spec.epochs}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, spec.epochs + 1)):
        raise ValueError(f"{run_spec.csv_path} epochs are not 1..{spec.epochs}: {epochs}")
    for i, row in enumerate(rows, 1):
        for key, raw in row.items():
            if isinstance(raw, str) and raw.strip().lower() in {"nan", "inf", "-inf"}:
                raise ValueError(f"{run_spec.csv_path} row {i}: {key}={raw}")

    meta = json.loads(run_spec.json_path.read_text())
    checks = {
        "model.implementation": "set_only",
        "model.attention_family": "dense",
        "model.backend": "exact",
        "model.set_causality_mode": "strict_past",
        "model.output_residual_mode": "anchor_span",
        "model.anchor.enabled": False,
        "model.anchor.teacher.enabled": False,
        "model.token_mlp.enabled": False,
        "model.candidate_fiber": "all_past",
        "model.set_diversity.lambda_div": 0.0,
        "model.multivector_basis.enabled": False,
        "model.multivector_basis.r": 1,
        "model.router.score_mode": "dense",
        "model.router_topk": 8 if spec.mode == "smoke" else 16,
        "model.router_temperature": 1.0,
        "model.router_multihead": True,
        "model.pooling.mode": "soft_trimmed_boltzmann",
        "model.pooling.tau": 0.1,
        "model.pooling.q": 0.85,
        "model.d_model": spec.d_model,
        "model.dim_feedforward": spec.d_ff,
        "model.num_layers": spec.layers,
        "model.num_heads": spec.heads,
        "model.max_seq_len": spec.seq_len,
        "model.window_size": 4,
        "model.stride": 2,
        "model.d_phi": spec.d_phi,
        "model.set_state_dim": spec.set_state_dim,
        "model.feature_mode": "hashed_counts",
        "model.feature_params.num_bins": spec.hash_bins,
        "data.batch_size": spec.batch_size,
        "data.seq_len": spec.seq_len,
        "training.epochs": spec.epochs,
        "training.lr": 0.0001,
        "resolved.output_residual_mode": "anchor_span",
        "resolved.anchor_enabled": False,
        "resolved.anchor_pre_encoder_layers": 0,
        "resolved.anchor_teacher_enabled": False,
        "resolved.set_diversity_lambda_div": 0.0,
        "resolved.multivector_basis_enabled": False,
        "resolved.multivector_basis_r": 1,
        "resolved.candidate_fiber": "all_past",
        "resolved.router_score_mode": "dense",
        "resolved.d_phi": spec.d_phi,
        "resolved.set_state_dim": spec.set_state_dim,
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if not meta_matches(actual, expected):
            raise ValueError(f"{run_spec.json_path} has {key}={actual!r}, expected {expected!r}")

    last = rows[-1]
    required_finite = [
        "train/loss",
        "val/loss",
        "train/ppl",
        "val/ppl",
        "val/span_ablation_loss",
        "val/span_ablation_ppl",
        "val/span_ablation_delta_loss",
        "val/span_ablation_delta_ppl",
        "train/peak_vram_mib",
        "ausa/candidate_count_mean",
        "ausa/router_entropy_norm",
        "ausa/router_top1_weight",
        "ausa/pooling_neff_l2",
    ]
    if spec.mode == "full":
        required_finite.append("ausa/set_gram_spectral_entropy_norm")
    for key in required_finite:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{run_spec.csv_path} missing finite {key}")
    if last.get("train/anchor_loss") not in {"", "NA", None}:
        raise ValueError(f"{run_spec.csv_path} logged anchor loss for anchor-disabled run")

    return {
        "phase": "SD-8.1",
        "step": "all_past_doubled_dictionary_atom_width_w4s2",
        "mode": spec.mode,
        "family_slug": "sd8_all_past_dense_dphi768_w4s2",
        "family": "Set-Dictionary SD-8.1 all_past dphi768 setdim768",
        "implementation": "set_only",
        "backend_family": "dense",
        "backend": "exact",
        "seed": str(run_spec.seed),
        "lr": "1e-4",
        "D": str(spec.d_model),
        "d_ff": str(spec.d_ff),
        "layers": str(spec.layers),
        "heads": str(spec.heads),
        "L": str(spec.seq_len),
        "w": "4",
        "s": "2",
        "M": str(run_spec.m),
        "M_over_L": f"{run_spec.m / spec.seq_len:.8f}",
        "L_over_M": f"{spec.seq_len / run_spec.m:.8f}",
        "d_phi": str(spec.d_phi),
        "set_state_dim": str(spec.set_state_dim),
        "output_residual_mode": "anchor_span",
        "anchor_enabled": "False",
        "candidate_fiber": "all_past",
        "router_score_mode": "dense",
        "config": spec.config,
        "csv_path": str(run_spec.csv_path.relative_to(ROOT)),
        "json_path": str(run_spec.json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_train_loss": last["train/loss"],
        "final_val_loss": last["val/loss"],
        "final_train_ppl": last["train/ppl"],
        "final_val_ppl": last["val/ppl"],
        "span_ablation_ppl": last["val/span_ablation_ppl"],
        "span_ablation_delta_ppl": last["val/span_ablation_delta_ppl"],
        "span_ablation_delta_loss": last["val/span_ablation_delta_loss"],
        "time_per_epoch_s": last.get("train/time_per_epoch_s", "NA"),
        "peak_train_vram_mib": last["train/peak_vram_mib"],
        "candidate_count_mean": last["ausa/candidate_count_mean"],
        "router_entropy_norm": last["ausa/router_entropy_norm"],
        "router_top1_weight": last["ausa/router_top1_weight"],
        "pooling_neff_l2": last["ausa/pooling_neff_l2"],
        "set_gram_spectral_entropy_norm": last.get("ausa/set_gram_spectral_entropy_norm", "NA"),
        "source_csv_sha256": sha256(run_spec.csv_path),
        "source_json_sha256": sha256(run_spec.json_path),
    }


def summarize(rows: list[dict[str, str]]) -> dict[str, str]:
    ppls = [float(row["final_val_ppl"]) for row in rows]
    deltas = [float(row["span_ablation_delta_ppl"]) for row in rows]
    vrams = [float(row["peak_train_vram_mib"]) for row in rows]
    times = [float(row["time_per_epoch_s"]) for row in rows if float_or_none(row["time_per_epoch_s"]) is not None]
    return {
        "family_slug": "sd8_all_past_dense_dphi768_w4s2",
        "family": "SD-8.1 all_past d_phi=768 set_state_dim=768",
        "w": "4",
        "s": "2",
        "n": str(len(rows)),
        "seeds": ",".join(sorted(row["seed"] for row in rows)),
        "mean_final_val_ppl": f"{mean(ppls):.6f}",
        "std_final_val_ppl": f"{pstdev(ppls):.6f}" if len(ppls) > 1 else "0.000000",
        "mean_span_ablation_delta_ppl": f"{mean(deltas):.6f}",
        "std_span_ablation_delta_ppl": f"{pstdev(deltas):.6f}" if len(deltas) > 1 else "0.000000",
        "mean_peak_train_vram_mib": f"{mean(vrams):.6f}",
        "std_peak_train_vram_mib": f"{pstdev(vrams):.6f}" if len(vrams) > 1 else "0.000000",
        "mean_time_per_epoch_s": f"{mean(times):.6f}" if times else "NA",
    }


def comparison_rows(summary: dict[str, str], mode: str) -> list[dict[str, str]]:
    current_ppl = float(summary["mean_final_val_ppl"])
    current_vram = float(summary["mean_peak_train_vram_mib"])
    rows = [
        {
            "label": "SD-8.1 d_phi=768 setdim=768 all_past",
            "kind": "new",
            "topology": "(4,2)",
            "mean_val_ppl": f"{current_ppl:.6f}",
            "std_val_ppl": summary["std_final_val_ppl"],
            "mean_peak_vram_mib": f"{current_vram:.6f}",
            "delta_ppl_vs_new": "0.000000",
            "delta_vram_mib_vs_new": "0.000000",
            "source": "this run",
        }
    ]
    if mode != "full":
        return rows
    refs = [
        ("Dense token baseline", "baseline", 781.109436, 13407.220703, "a7_empty_only_calibration_summary.tsv"),
        ("Set Dense empty_only old ref", "baseline", 1273.6, 11807.3, "A8 favorable set conditions plan"),
        ("Best SD so far: SD-8 all_past d_phi=384 setdim=384", "sd_reference", 1288.603190, 11913.547852, "sd8_all_past_dense_routerdense_summary/runs"),
        ("Fixed S2 anchor_span d_phi=384 setdim=384", "sd_reference", 1288.973145, None, "SD_6_5_s2_anchoring_fixed"),
    ]
    for label, kind, ppl, vram, source in refs:
        rows.append({
            "label": label,
            "kind": kind,
            "topology": "(4,2)",
            "mean_val_ppl": f"{ppl:.6f}",
            "std_val_ppl": "NA",
            "mean_peak_vram_mib": f"{vram:.6f}" if vram is not None else "NA",
            "delta_ppl_vs_new": f"{ppl - current_ppl:.6f}",
            "delta_vram_mib_vs_new": f"{vram - current_vram:.6f}" if vram is not None else "NA",
            "source": source,
        })
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(spec: ModeSpec, rows: list[dict[str, str]], log_failures: list[str]) -> None:
    summary = summarize(rows)
    summary_path = spec.table.with_name(spec.table.stem.replace("_runs", "_summary") + ".tsv")
    comparisons = comparison_rows(summary, spec.mode)
    write_tsv(spec.table, rows)
    write_tsv(summary_path, [summary])
    write_tsv(spec.comparison_table, comparisons)

    manifest = {
        "status": "pass",
        "phase": "SD-8.1",
        "step": "all_past_doubled_dictionary_atom_width_w4s2",
        "mode": spec.mode,
        "validated_runs": len(rows),
        "expected_runs": len(expected_runs(spec)),
        "table": str(spec.table.relative_to(ROOT)),
        "summary_table": str(summary_path.relative_to(ROOT)),
        "comparison_table": str(spec.comparison_table.relative_to(ROOT)),
        "audit": str(spec.audit.relative_to(ROOT)),
        "log_failures": log_failures,
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
        "summary": summary,
        "comparisons": comparisons,
    }
    spec.manifest.parent.mkdir(parents=True, exist_ok=True)
    spec.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# SD-8.1 d_phi=768 set_state_dim=768 w4s2",
        "",
        "Status: PASS",
        "",
        f"Mode: `{spec.mode}`",
        f"Validated runs: {len(rows)} / {len(expected_runs(spec))}",
        "",
        "Contract:",
        "- `(w,s)=(4,2)`, `M=255` for full mode",
        "- `d_model=384`, `d_phi=768`, `set_state_dim=768` in full mode",
        "- `output_residual_mode=anchor_span`",
        "- `anchor.enabled=false` (CE only)",
        "- dense exact backend, `candidate_fiber=all_past`, `router.score_mode=dense`",
        "- deferred knobs disabled (`multivector_basis=false`, `r=1`, `set_diversity.lambda_div=0`)",
        "",
        "Summary:",
        "",
        "| n | seeds | mean val PPL | std | mean peak VRAM MiB | std VRAM | mean span-ablation delta PPL |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['n']} | {summary['seeds']} | {summary['mean_final_val_ppl']} | "
            f"{summary['std_final_val_ppl']} | {summary['mean_peak_train_vram_mib']} | "
            f"{summary['std_peak_train_vram_mib']} | {summary['mean_span_ablation_delta_ppl']} |"
        ),
        "",
        "Comparison:",
        "",
        "| label | mean val PPL | mean peak VRAM MiB | delta PPL vs new | delta VRAM MiB vs new |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['label']} | {row['mean_val_ppl']} | {row['mean_peak_vram_mib']} | "
            f"{row['delta_ppl_vs_new']} | {row['delta_vram_mib_vs_new']} |"
        )
    lines.extend([
        "",
        f"Runs table: `{spec.table.relative_to(ROOT)}`",
        f"Summary table: `{summary_path.relative_to(ROOT)}`",
        f"Comparison table: `{spec.comparison_table.relative_to(ROOT)}`",
        f"Manifest: `{spec.manifest.relative_to(ROOT)}`",
        "",
        "Notes:",
        "- Peak VRAM is raw `train/peak_vram_mib`, consistent with `audit/vram_overhead_audit.md`.",
        "- The Set Dense empty_only VRAM reference is the recorded old near-2 compressed point, not rerun here.",
    ])
    spec.audit.parent.mkdir(parents=True, exist_ok=True)
    spec.audit.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    args = parser.parse_args()
    spec = mode_spec(args.mode)
    rows = [validate_run(run_spec) for run_spec in expected_runs(spec)]
    log_failures = scan_logs(spec.logs)
    if log_failures:
        raise SystemExit("log scan failed:\n" + "\n".join(log_failures))
    write_outputs(spec, rows, log_failures)
    print(json.dumps({
        "status": "pass",
        "mode": spec.mode,
        "validated_runs": len(rows),
        "manifest": str(spec.manifest.relative_to(ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
