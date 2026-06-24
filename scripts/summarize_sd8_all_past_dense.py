#!/usr/bin/env python3
"""Validate and summarize SD-8 all_past dense-router runs."""

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
    expected_token_baseline_ppl: float | None


def mode_spec(mode: str) -> ModeSpec:
    if mode == "smoke":
        return ModeSpec(
            mode=mode,
            raw=ROOT / "out" / "paper_mechanisms" / "sd8_all_past_dense_routerdense_smoke",
            logs=ROOT / "logs" / "sd8_all_past_dense_routerdense_smoke",
            config="configs/set_dictionary/sd8_all_past_dense_smoke.yaml",
            group="sd8_all_past_dense_routerdense_smoke_D64_FF128",
            table=TABLES / "sd8_all_past_dense_routerdense_smoke_runs.tsv",
            manifest=CHECKS / "sd8_all_past_dense_routerdense_smoke_manifest.json",
            audit=AUDIT / "SD_8_all_past_dense_routerdense_smoke.md",
            seq_len=64,
            d_model=64,
            d_ff=128,
            layers=1,
            heads=4,
            epochs=1,
            batch_size=2,
            seeds_by_topology={(4, 2): [0]},
            hash_bins=32,
            expected_token_baseline_ppl=None,
        )
    return ModeSpec(
        mode=mode,
        raw=ROOT / "out" / "paper_mechanisms" / "sd8_all_past_dense_routerdense",
        logs=ROOT / "logs" / "sd8_all_past_dense_routerdense",
        config="configs/set_dictionary/sd8_all_past_dense.yaml",
        group="sd8_all_past_dense_routerdense_D384_FF1536",
        table=TABLES / "sd8_all_past_dense_routerdense_runs.tsv",
        manifest=CHECKS / "sd8_all_past_dense_routerdense_manifest.json",
        audit=AUDIT / "SD_8_all_past_dense_routerdense.md",
        seq_len=512,
        d_model=384,
        d_ff=1536,
        layers=6,
        heads=8,
        epochs=10,
        batch_size=16,
        seeds_by_topology={(16, 8): [0, 1, 2], (4, 2): [0, 1, 2]},
        hash_bins=128,
        expected_token_baseline_ppl=781.1,
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
                failures.append(
                    f"{path.relative_to(ROOT)} contains standalone token {match.group()!r}"
                )
    return failures


def validate_run(spec: ExpectedRun) -> dict[str, str]:
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
        "model.anchor.enabled": False,
        "model.anchor.teacher.enabled": False,
        "model.token_mlp.enabled": False,
        "model.candidate_fiber": "all_past",
        "model.set_diversity.lambda_div": 0.0,
        "model.multivector_basis.enabled": False,
        "model.multivector_basis.r": 1,
        "model.router.score_mode": "dense",
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
        "resolved.anchor_enabled": False,
        "resolved.anchor_pre_encoder_layers": 0,
        "resolved.anchor_teacher_enabled": False,
        "resolved.set_diversity_lambda_div": 0.0,
        "resolved.multivector_basis_enabled": False,
        "resolved.multivector_basis_r": 1,
        "resolved.candidate_fiber": "all_past",
        "resolved.router_score_mode": "dense",
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if not meta_matches(actual, expected):
            raise ValueError(f"{spec.json_path} has {key}={actual!r}, expected {expected!r}")

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
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{spec.csv_path} missing finite {key}")
    if last.get("train/anchor_loss") not in {"", "NA", None}:
        raise ValueError(f"{spec.csv_path} logged anchor loss for SD-8 anchor-disabled run")

    return {
        "phase": "SD-8",
        "step": "all_past_ce_only",
        "mode": spec.spec.mode,
        "family_slug": "sd8_all_past_dense_routerdense",
        "family": "Set-Dictionary SD-8 all_past dense-router",
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
        "candidate_count_target": f"{spec.m:.6f}",
        "output_residual_mode": "anchor_span",
        "anchor_enabled": "False",
        "candidate_fiber": "all_past",
        "router_score_mode": "dense",
        "config": spec.spec.config,
        "csv_path": str(spec.csv_path.relative_to(ROOT)),
        "json_path": str(spec.json_path.relative_to(ROOT)),
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
        "set_gram_spectral_entropy_norm": last["ausa/set_gram_spectral_entropy_norm"],
        "source_csv_sha256": sha256(spec.csv_path),
        "source_json_sha256": sha256(spec.json_path),
    }


def grouped_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["w"], row["s"]), []).append(row)
    out: list[dict[str, str]] = []
    for (w, s), group in sorted(groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        ppls = [float(row["final_val_ppl"]) for row in group]
        deltas = [float(row["span_ablation_delta_ppl"]) for row in group]
        out.append({
            "w": w,
            "s": s,
            "n": str(len(group)),
            "seeds": ",".join(sorted(row["seed"] for row in group)),
            "mean_final_val_ppl": f"{mean(ppls):.6f}",
            "std_final_val_ppl": f"{pstdev(ppls):.6f}" if len(ppls) > 1 else "0.000000",
            "mean_span_ablation_delta_ppl": f"{mean(deltas):.6f}",
            "std_span_ablation_delta_ppl": (
                f"{pstdev(deltas):.6f}" if len(deltas) > 1 else "0.000000"
            ),
        })
    return out


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(spec: ModeSpec, rows: list[dict[str, str]], log_failures: list[str]) -> None:
    summary = grouped_summary(rows)
    write_tsv(spec.table, rows)
    summary_path = spec.table.with_name(spec.table.stem.replace("_runs", "_summary") + ".tsv")
    write_tsv(summary_path, summary)
    manifest = {
        "status": "pass",
        "phase": "SD-8",
        "step": "all_past_ce_only",
        "mode": spec.mode,
        "validated_runs": len(rows),
        "expected_runs": len(expected_runs(spec)),
        "table": str(spec.table.relative_to(ROOT)),
        "summary_table": str(summary_path.relative_to(ROOT)),
        "audit": str(spec.audit.relative_to(ROOT)),
        "log_failures": log_failures,
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
        "reference_policy": {
            "token_dense_baseline_ppl": spec.expected_token_baseline_ppl,
            "refs_rerun": False,
            "full_refs": [
                {"topology": "16,8", "old_ska_dense_direct_ppl": 1422.8},
                {"topology": "4,2", "old_set_dense_empty_only_ppl": 1273.6},
            ]
            if spec.mode == "full"
            else [],
        },
        "summary": summary,
    }
    spec.manifest.parent.mkdir(parents=True, exist_ok=True)
    spec.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# SD-8 All-Past Dense-Router",
        "",
        "Status: PASS",
        "",
        f"Mode: `{spec.mode}`",
        f"Validated runs: {len(rows)} / {len(expected_runs(spec))}",
        "",
        "Contract:",
        "- `output_residual_mode=anchor_span`",
        "- `anchor.enabled=false` (CE only)",
        "- dense exact backend",
        "- `candidate_fiber=all_past`",
        "- `router.score_mode=dense` (same causal support, avoids all_past candidate-gather OOM)",
        "- `token_mlp.enabled=false`",
        "- deferred knobs disabled (`multivector_basis=false`, `r=1`, `set_diversity.lambda_div=0`)",
        "",
        "Topology summary:",
        "",
        "| w | s | n | seeds | mean val PPL | std | mean span-ablation delta PPL |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['w']} | {row['s']} | {row['n']} | {row['seeds']} | "
            f"{row['mean_final_val_ppl']} | {row['std_final_val_ppl']} | "
            f"{row['mean_span_ablation_delta_ppl']} |"
        )
    lines.extend([
        "",
        f"Runs table: `{spec.table.relative_to(ROOT)}`",
        f"Summary table: `{summary_path.relative_to(ROOT)}`",
        f"Manifest: `{spec.manifest.relative_to(ROOT)}`",
        "",
        "Notes:",
        "- Span-ablation metrics are evaluated during the trained run by zeroing `span_t` at validation.",
        "- SD-8 all_past CE-only has no anchor pre-encoder and should not log anchor auxiliary losses.",
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
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": spec.mode,
                "validated_runs": len(rows),
                "manifest": str(spec.manifest.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
