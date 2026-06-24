#!/usr/bin/env python3
"""Validate, summarize, and classify SD-6.5 fixed S2 anchoring rescue runs."""

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
S1_SUMMARY = TABLES / "sd5_s1_anchor_span_dense_summary.tsv"

TOKEN_BASELINE_PPL = 781.1
RECON_THRESHOLD = 0.5
VALID_RECON_GUARD = 1.2
FLAT_SLOPE_EPS = 0.01


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    raw: Path
    logs: Path
    config: str
    group: str
    table: Path
    epoch_table: Path
    trajectory_table: Path
    manifest: Path
    audit: Path
    seq_len: int
    d_model: int
    d_ff: int
    layers: int
    heads: int
    epochs: int
    batch_size: int
    seeds_by_topology: dict[tuple[int, int], list[int]]
    hash_bins: int


def mode_spec(mode: str) -> ModeSpec:
    if mode == "smoke":
        return ModeSpec(
            mode=mode,
            raw=ROOT / "out" / "paper_mechanisms" / "sd6_5_s2_anchoring_fixed_smoke",
            logs=ROOT / "logs" / "sd6_5_s2_anchoring_fixed_smoke",
            config="configs/set_dictionary/s2_anchor_span_dense_smoke.yaml",
            group="sd6_5_s2_anchoring_fixed_smoke_D64_FF128",
            table=TABLES / "sd6_5_s2_anchoring_fixed_smoke_runs.tsv",
            epoch_table=TABLES / "sd6_5_s2_anchoring_fixed_smoke_epoch_trajectory.tsv",
            trajectory_table=TABLES / "sd6_5_s2_anchoring_fixed_smoke_topology_trajectory.tsv",
            manifest=CHECKS / "sd6_5_s2_anchoring_fixed_smoke_manifest.json",
            audit=AUDIT / "SD_6_5_s2_anchoring_fixed_smoke.md",
            seq_len=64,
            d_model=64,
            d_ff=128,
            layers=1,
            heads=4,
            epochs=1,
            batch_size=2,
            seeds_by_topology={(4, 2): [0]},
            hash_bins=32,
        )
    return ModeSpec(
        mode=mode,
        raw=ROOT / "out" / "paper_mechanisms" / "sd6_5_s2_anchoring_fixed",
        logs=ROOT / "logs" / "sd6_5_s2_anchoring_fixed",
        config="configs/set_dictionary/s2_anchor_span_dense.yaml",
        group="sd6_5_s2_anchoring_fixed_D384_FF1536",
        table=TABLES / "sd6_5_s2_anchoring_fixed_runs.tsv",
        epoch_table=TABLES / "sd6_5_s2_anchoring_fixed_epoch_trajectory.tsv",
        trajectory_table=TABLES / "sd6_5_s2_anchoring_fixed_topology_trajectory.tsv",
        manifest=CHECKS / "sd6_5_s2_anchoring_fixed_manifest.json",
        audit=AUDIT / "SD_6_5_s2_anchoring_fixed.md",
        seq_len=512,
        d_model=384,
        d_ff=1536,
        layers=6,
        heads=8,
        epochs=10,
        batch_size=16,
        seeds_by_topology={(16, 8): [0, 1, 2], (4, 2): [0, 1, 2]},
        hash_bins=128,
    )


@dataclass(frozen=True)
class ExpectedRun:
    spec: ModeSpec
    w: int
    s: int
    seed: int

    @property
    def m(self) -> int:
        return ((self.spec.seq_len - self.w) // self.s) + 1

    @property
    def name(self) -> str:
        lr_tag = "1e-4".replace(".", "p")
        return (
            f"{self.spec.group}_L{self.spec.seq_len}_w{self.w}_s{self.s}_"
            f"M{self.m}_lr{lr_tag}_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return self.spec.raw / self.spec.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


def expected_runs(spec: ModeSpec) -> list[ExpectedRun]:
    return [
        ExpectedRun(spec, w, s, seed)
        for (w, s), seeds in spec.seeds_by_topology.items()
        for seed in seeds
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


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


def meta_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return str(actual) == str(expected)
    if isinstance(expected, (int, float)):
        parsed = float_or_none(actual)
        return parsed is not None and math.isclose(
            parsed, float(expected), rel_tol=1e-9, abs_tol=1e-12
        )
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
                failures.append(
                    f"{path.relative_to(ROOT)} contains standalone token {match.group()!r}"
                )
    return failures


def validate_run(spec: ExpectedRun) -> tuple[dict[str, str], list[dict[str, str]]]:
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {spec.csv_path}")
    if not spec.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {spec.json_path}")
    rows = read_csv(spec.csv_path)
    if len(rows) != spec.spec.epochs:
        raise ValueError(f"{spec.csv_path} has {len(rows)} rows, expected {spec.spec.epochs}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, spec.spec.epochs + 1)):
        raise ValueError(f"{spec.csv_path} epochs are not 1..{spec.spec.epochs}: {epochs}")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite CSV values in {spec.csv_path}: {bad[:5]}")

    meta = json.loads(spec.json_path.read_text())
    checks = {
        "model.implementation": "set_only",
        "model.attention_family": "dense",
        "model.backend": "exact",
        "model.set_causality_mode": "strict_past",
        "model.output_residual_mode": "anchor_span",
        "model.anchor.enabled": True,
        "model.anchor.target": "pre_encoder",
        "model.anchor.pre_encoder_layers": 2,
            "model.anchor.lambda_h": 0.1,
            "model.anchor.lambda_pre": 1.0,
            "model.anchor.pre_encoder_head": True,
            "model.anchor.detach_target": True,
        "model.anchor.norm": "layernorm",
        "model.anchor.teacher.enabled": False,
        "model.token_mlp.enabled": False,
        "model.candidate_fiber": "endpoint_window",
        "model.set_diversity.lambda_div": 0.0,
        "model.multivector_basis.enabled": False,
        "model.multivector_basis.r": 1,
        "model.router.score_mode": "candidate_gather",
        "model.router_topk": 8 if spec.spec.mode == "smoke" else 16,
        "model.router_temperature": 1.0,
        "model.router_multihead": True,
        "model.pooling.mode": "soft_trimmed_boltzmann",
        "model.pooling.tau": 0.1,
        "model.pooling.q": 0.85,
        "model.d_model": spec.spec.d_model,
        "model.dim_feedforward": spec.spec.d_ff,
        "model.num_layers": spec.spec.layers,
        "model.num_heads": spec.spec.heads,
        "model.max_seq_len": spec.spec.seq_len,
        "model.window_size": spec.w,
        "model.stride": spec.s,
        "model.d_phi": spec.spec.d_model,
        "model.set_state_dim": spec.spec.d_model,
        "model.feature_mode": "hashed_counts",
        "model.feature_params.num_bins": spec.spec.hash_bins,
        "data.batch_size": spec.spec.batch_size,
        "data.seq_len": spec.spec.seq_len,
        "training.epochs": spec.spec.epochs,
        "training.lr": 0.0001,
        "resolved.output_residual_mode": "anchor_span",
        "resolved.anchor_enabled": True,
        "resolved.anchor_target": "pre_encoder",
        "resolved.anchor_pre_encoder_layers": 2,
            "resolved.anchor_lambda_h": 0.1,
            "resolved.anchor_lambda_pre": 1.0,
            "resolved.anchor_pre_encoder_head": True,
            "resolved.anchor_detach_target": True,
        "resolved.anchor_norm": "layernorm",
        "resolved.anchor_teacher_enabled": False,
        "resolved.set_diversity_lambda_div": 0.0,
        "resolved.multivector_basis_enabled": False,
        "resolved.multivector_basis_r": 1,
        "resolved.candidate_fiber": "endpoint_window",
        "resolved.router_score_mode": "candidate_gather",
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if not meta_matches(actual, expected):
            raise ValueError(f"{spec.json_path} has {key}={actual!r}, expected {expected!r}")

    per_epoch: list[dict[str, str]] = []
    for row in rows:
        required_finite = [
            "train/loss",
            "val/loss",
                "train/ppl",
                "val/ppl",
                "train/anchor_loss",
                "train/anchor_pre_ce_loss",
                "train/anchor_pre_ce",
                "train/anchor_mse",
                "train/anchor/lambda_h",
                "train/anchor/lambda_pre",
                "train/anchor/pre_ce",
                "train/anchor/mse",
            "train/anchor/recon_error_norm",
            "val/span_ablation_loss",
            "val/span_ablation_ppl",
            "val/span_ablation_delta_loss",
            "val/span_ablation_delta_ppl",
            "train/grad_norm",
            "train/peak_vram_mib",
            "ausa/candidate_count_mean",
            "ausa/router_entropy_norm",
            "ausa/router_top1_weight",
            "ausa/pooling_neff_l2",
        ]
        if spec.spec.mode == "full":
            required_finite.append("ausa/set_gram_spectral_entropy_norm")
        for key in required_finite:
            if float_or_none(row.get(key)) is None:
                raise ValueError(f"{spec.csv_path} epoch {row.get('epoch')} missing finite {key}")
        per_epoch.append({
            "phase": "SD-6.5",
            "step": "S2-fixed",
            "mode": spec.spec.mode,
            "seed": str(spec.seed),
            "w": str(spec.w),
            "s": str(spec.s),
            "M": str(spec.m),
            "epoch": row["epoch"],
            "train_anchor_loss": row["train/anchor_loss"],
            "train_anchor_pre_ce_loss": row["train/anchor_pre_ce_loss"],
            "train_anchor_pre_ce": row["train/anchor_pre_ce"],
            "train_anchor_mse": row["train/anchor_mse"],
            "anchor_recon_error_norm": row["train/anchor/recon_error_norm"],
            "val_ppl": row["val/ppl"],
            "span_ablation_delta_ppl": row["val/span_ablation_delta_ppl"],
            "router_entropy_norm": row["ausa/router_entropy_norm"],
            "router_top1_weight": row["ausa/router_top1_weight"],
            "pooling_neff_l2": row["ausa/pooling_neff_l2"],
            "set_gram_spectral_entropy_norm": row.get("ausa/set_gram_spectral_entropy_norm", "NA"),
        })

    last = rows[-1]
    final = {
            "phase": "SD-6.5",
            "step": "S2-fixed",
            "mode": spec.spec.mode,
            "family_slug": "sd6_5_s2_anchoring_fixed",
            "family": "Set-Dictionary SD-6.5 S2 anchor_span + trained causal pre-encoder",
        "implementation": "set_only",
        "backend_family": "dense",
        "backend": "exact",
        "seed": str(spec.seed),
        "lr": "1e-4",
        "D": str(spec.spec.d_model),
        "d_ff": str(spec.spec.d_ff),
        "layers": str(spec.spec.layers),
        "heads": str(spec.spec.heads),
        "L": str(spec.spec.seq_len),
        "w": str(spec.w),
        "s": str(spec.s),
        "M": str(spec.m),
        "M_over_L": f"{spec.m / spec.spec.seq_len:.8f}",
        "L_over_M": f"{spec.spec.seq_len / spec.m:.8f}",
        "candidate_count_target": f"{spec.w / spec.s:.6f}",
        "output_residual_mode": "anchor_span",
            "anchor_enabled": "True",
            "anchor_lambda_h": "0.1",
            "anchor_lambda_pre": "1.0",
            "anchor_pre_encoder_head": "True",
            "candidate_fiber": "endpoint_window",
        "config": spec.spec.config,
        "csv_path": str(spec.csv_path.relative_to(ROOT)),
        "json_path": str(spec.json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_train_loss": last["train/loss"],
        "final_val_loss": last["val/loss"],
        "final_train_ppl": last["train/ppl"],
        "final_val_ppl": last["val/ppl"],
            "final_anchor_loss": last["train/anchor_loss"],
            "final_anchor_pre_ce_loss": last["train/anchor_pre_ce_loss"],
            "final_anchor_pre_ce": last["train/anchor_pre_ce"],
            "final_anchor_mse": last["train/anchor_mse"],
        "final_anchor_recon_error_norm": last["train/anchor/recon_error_norm"],
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
        "source_csv_sha256": sha256(spec.csv_path),
        "source_json_sha256": sha256(spec.json_path),
    }
    return final, per_epoch


def grouped_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["w"], row["s"]), []).append(row)
    out: list[dict[str, str]] = []
    for (w, s), group in sorted(groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        ppls = [float(row["final_val_ppl"]) for row in group]
        recon = [float(row["final_anchor_recon_error_norm"]) for row in group]
        deltas = [float(row["span_ablation_delta_ppl"]) for row in group]
        out.append({
            "w": w,
            "s": s,
            "n": str(len(group)),
            "seeds": ",".join(sorted(row["seed"] for row in group)),
            "mean_final_val_ppl": f"{mean(ppls):.6f}",
            "std_final_val_ppl": f"{pstdev(ppls):.6f}" if len(ppls) > 1 else "0.000000",
            "mean_final_anchor_recon_error_norm": f"{mean(recon):.6f}",
            "std_final_anchor_recon_error_norm": (
                f"{pstdev(recon):.6f}" if len(recon) > 1 else "0.000000"
            ),
            "mean_span_ablation_delta_ppl": f"{mean(deltas):.6f}",
            "std_span_ablation_delta_ppl": (
                f"{pstdev(deltas):.6f}" if len(deltas) > 1 else "0.000000"
            ),
        })
    return out


def topology_trajectory(epoch_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in epoch_rows:
        groups.setdefault((row["w"], row["s"], row["epoch"]), []).append(row)
    out: list[dict[str, str]] = []
    for (w, s, epoch), group in sorted(groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]), int(item[0][2]))):
        recon = [float(row["anchor_recon_error_norm"]) for row in group]
        ppls = [float(row["val_ppl"]) for row in group]
        deltas = [float(row["span_ablation_delta_ppl"]) for row in group]
        out.append({
            "w": w,
            "s": s,
            "epoch": epoch,
            "n": str(len(group)),
            "mean_anchor_recon_error_norm": f"{mean(recon):.6f}",
            "std_anchor_recon_error_norm": (
                f"{pstdev(recon):.6f}" if len(recon) > 1 else "0.000000"
            ),
            "mean_val_ppl": f"{mean(ppls):.6f}",
            "std_val_ppl": f"{pstdev(ppls):.6f}" if len(ppls) > 1 else "0.000000",
            "mean_span_ablation_delta_ppl": f"{mean(deltas):.6f}",
        })
    return out


def load_s1_summary() -> dict[tuple[str, str], dict[str, float]]:
    if not S1_SUMMARY.exists():
        raise FileNotFoundError(f"missing S1 summary: {S1_SUMMARY}")
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in read_tsv(S1_SUMMARY):
        out[(row["w"], row["s"])] = {
            "n": float(row["n"]),
            "mean_final_val_ppl": float(row["mean_final_val_ppl"]),
            "std_final_val_ppl": float(row["std_final_val_ppl"]),
        }
    return out


def classify(
    summary: list[dict[str, str]],
    trajectory: list[dict[str, str]],
    s1: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, str]]:
    traj_by_topology: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in trajectory:
        traj_by_topology.setdefault((row["w"], row["s"]), []).append(row)

    verdicts: list[dict[str, str]] = []
    for row in summary:
        key = (row["w"], row["s"])
        s1_row = s1.get(key)
        if s1_row is None:
            raise ValueError(f"missing S1 reference for topology {key}")
        series = sorted(traj_by_topology[key], key=lambda r: int(r["epoch"]))
        if len(series) >= 3:
            last3 = series[-3:]
            recon_slope = (
                float(last3[-1]["mean_anchor_recon_error_norm"])
                - float(last3[0]["mean_anchor_recon_error_norm"])
            ) / max(int(last3[-1]["epoch"]) - int(last3[0]["epoch"]), 1)
        else:
            recon_slope = 0.0
        s2_mean = float(row["mean_final_val_ppl"])
        s2_std = float(row["std_final_val_ppl"])
        s2_n = float(row["n"])
        s1_mean = s1_row["mean_final_val_ppl"]
        s1_std = s1_row["std_final_val_ppl"]
        s1_n = s1_row["n"]
        combined_ci = 1.96 * math.sqrt((s1_std**2 / s1_n) + (s2_std**2 / s2_n))
        ppl_delta_vs_s1 = s2_mean - s1_mean
        ppl_beats_s1 = s2_mean < (s1_mean - combined_ci)
        ppl_within_s1_ci = abs(ppl_delta_vs_s1) <= combined_ci
        final_recon = float(row["mean_final_anchor_recon_error_norm"])
        anchor_valid = final_recon < VALID_RECON_GUARD
        recon_signal = final_recon < RECON_THRESHOLD and recon_slope < 0.0
        recon_high = final_recon > RECON_THRESHOLD
        recon_flat = abs(recon_slope) <= FLAT_SLOPE_EPS

        if not anchor_valid:
            branch = "CONFOUNDED"
            verdict = "invalid-random-target-signature"
            recommendation = "fix pre-encoder training before any Branch A/B verdict"
        elif recon_signal and ppl_beats_s1:
            branch = "A"
            verdict = "signal-limited"
            recommendation = "SD-7: lambda_h=1.0, then lambda_div if still useful"
        elif recon_high and ppl_within_s1_ci:
            branch = "B"
            verdict = "capacity-limited"
            recommendation = "SD-8: D-fiber all_past CE-only first; r=2 only secondary"
        else:
            branch = "MIXED"
            verdict = "inconclusive"
            recommendation = "manual review before any downstream launch"

        verdicts.append({
            "w": row["w"],
            "s": row["s"],
            "branch": branch,
            "verdict": verdict,
            "recommendation": recommendation,
            "s1_mean_val_ppl": f"{s1_mean:.6f}",
            "s2_mean_val_ppl": f"{s2_mean:.6f}",
            "ppl_delta_vs_s1": f"{ppl_delta_vs_s1:.6f}",
            "combined_95ci": f"{combined_ci:.6f}",
            "ppl_beats_s1_beyond_ci": str(ppl_beats_s1),
            "ppl_within_s1_ci": str(ppl_within_s1_ci),
            "final_anchor_recon_error_norm": f"{final_recon:.6f}",
            "anchor_validity_guard_threshold": f"{VALID_RECON_GUARD:.6f}",
            "anchor_validity_guard_pass": str(anchor_valid),
            "last3_recon_slope_per_epoch": f"{recon_slope:.6f}",
            "recon_below_0p5_and_falling": str(recon_signal),
            "recon_above_0p5": str(recon_high),
            "recon_flat_abs_slope_le_0p01": str(recon_flat),
        })
    return verdicts


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    spec: ModeSpec,
    rows: list[dict[str, str]],
    epoch_rows: list[dict[str, str]],
    log_failures: list[str],
) -> str:
    summary = grouped_summary(rows)
    trajectory = topology_trajectory(epoch_rows)
    verdicts = classify(summary, trajectory, load_s1_summary()) if spec.mode == "full" else []
    confounded = any(row.get("branch") == "CONFOUNDED" for row in verdicts)

    write_tsv(spec.table, rows)
    write_tsv(spec.epoch_table, epoch_rows)
    write_tsv(spec.trajectory_table, trajectory)
    summary_path = spec.table.with_name(spec.table.stem.replace("_runs", "_summary") + ".tsv")
    verdict_path = spec.table.with_name(spec.table.stem.replace("_runs", "_verdict") + ".tsv")
    write_tsv(summary_path, summary)
    if verdicts:
        write_tsv(verdict_path, verdicts)

    status = "confounded" if confounded else "pass"
    manifest = {
        "status": status,
        "phase": "SD-6.5",
        "step": "S2-fixed",
        "mode": spec.mode,
        "validated_runs": len(rows),
        "expected_runs": len(expected_runs(spec)),
        "table": str(spec.table.relative_to(ROOT)),
        "summary_table": str(summary_path.relative_to(ROOT)),
        "epoch_table": str(spec.epoch_table.relative_to(ROOT)),
        "trajectory_table": str(spec.trajectory_table.relative_to(ROOT)),
        "verdict_table": str(verdict_path.relative_to(ROOT)) if verdicts else None,
        "audit": str(spec.audit.relative_to(ROOT)),
        "log_failures": log_failures,
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
        "reference_policy": {
            "s1_summary": str(S1_SUMMARY.relative_to(ROOT)),
            "token_dense_baseline_ppl": TOKEN_BASELINE_PPL,
            "refs_rerun": False,
            "valid_recon_guard": VALID_RECON_GUARD,
        },
        "summary": summary,
        "verdicts": verdicts,
    }
    spec.manifest.parent.mkdir(parents=True, exist_ok=True)
    spec.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# SD-6.5 Fixed S2 Anchoring Rescue",
        "",
        f"Status: {'STILL CONFOUNDED' if confounded else 'PASS'}",
        "",
        f"Mode: `{spec.mode}`",
        f"Validated runs: {len(rows)} / {len(expected_runs(spec))}",
        "",
        "Contract:",
        "- `output_residual_mode=anchor_span`",
        "- `anchor.enabled=true`, `anchor.target=pre_encoder`, `anchor.pre_encoder_layers=2`",
        "- `anchor.lambda_h=0.1`, `anchor.lambda_pre=1.0`, `anchor.pre_encoder_head=true`",
        "- `anchor.detach_target=true`, `anchor.norm=layernorm`",
        "- `anchor.teacher.enabled=false`",
        "- dense exact backend, `candidate_fiber=endpoint_window`",
        "- `token_mlp.enabled=false`, `multivector_basis=false`, `r=1`, `set_diversity.lambda_div=0`",
        "",
        "Topology summary:",
        "",
        "| w | s | n | seeds | mean val PPL | std | final recon error | mean span-ablation delta PPL |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['w']} | {row['s']} | {row['n']} | {row['seeds']} | "
            f"{row['mean_final_val_ppl']} | {row['std_final_val_ppl']} | "
            f"{row['mean_final_anchor_recon_error_norm']} | "
            f"{row['mean_span_ablation_delta_ppl']} |"
        )
    if verdicts:
        lines.extend(["", "Branch verdict:", ""])
        lines.append("| w | s | branch | PPL delta vs S1 | combined 95% CI | final recon | guard pass | last-3 recon slope | recommendation |")
        lines.append("| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |")
        for row in verdicts:
            lines.append(
                f"| {row['w']} | {row['s']} | {row['branch']} | "
                f"{row['ppl_delta_vs_s1']} | {row['combined_95ci']} | "
                f"{row['final_anchor_recon_error_norm']} | "
                f"{row['anchor_validity_guard_pass']} | "
                f"{row['last3_recon_slope_per_epoch']} | {row['recommendation']} |"
            )
    lines.extend([
        "",
        f"Runs table: `{spec.table.relative_to(ROOT)}`",
        f"Summary table: `{summary_path.relative_to(ROOT)}`",
        f"Epoch trajectory table: `{spec.epoch_table.relative_to(ROOT)}`",
        f"Topology trajectory table: `{spec.trajectory_table.relative_to(ROOT)}`",
    ])
    if verdicts:
        lines.append(f"Verdict table: `{verdict_path.relative_to(ROOT)}`")
    lines.extend([
        f"Manifest: `{spec.manifest.relative_to(ROOT)}`",
        "",
        "Notes:",
        "- `anchor/recon_error_norm` is emitted per epoch in the trajectory tables.",
        f"- Anchor validity guard: final topology mean `recon_error_norm` must be < {VALID_RECON_GUARD}.",
        "- Span-ablation metrics are evaluated during the trained run by zeroing `span_t` at validation.",
        "- No SD-7 or SD-8 follow-up was launched by this summarizer.",
    ])
    spec.audit.parent.mkdir(parents=True, exist_ok=True)
    spec.audit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    args = parser.parse_args()
    spec = mode_spec(args.mode)
    final_rows: list[dict[str, str]] = []
    epoch_rows: list[dict[str, str]] = []
    for run_spec in expected_runs(spec):
        final, per_epoch = validate_run(run_spec)
        final_rows.append(final)
        epoch_rows.extend(per_epoch)
    log_failures = scan_logs(spec.logs)
    if log_failures:
        raise SystemExit("log scan failed:\n" + "\n".join(log_failures))
    status = write_outputs(spec, final_rows, epoch_rows, log_failures)
    print(
        json.dumps(
            {
                    "status": status,
                "mode": spec.mode,
                "validated_runs": len(final_rows),
                "manifest": str(spec.manifest.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
