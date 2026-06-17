#!/usr/bin/env python3
"""Validate and summarize A9 candidate-gather router comparison."""

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
RAW = ROOT / "out" / "paper_mechanisms" / "a9_candidate_gather"
LOG_ROOT = ROOT / "logs" / "a9_candidate_gather"
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"
AUDIT = ROOT / "audit"
REFERENCE_SUMMARY = TABLES / "a7_backend_family_empty_only_augmented_summary.tsv"

SEQ_LEN = 512
D_MODEL = 384
D_FF = 1536
EPOCHS = 10
LR = "1e-4"
SEEDS = [0, 1, 2]
TOPOLOGIES = [(4, 2), (8, 4)]

FAMILIES = {
    "set_dense_exact": {
        "summary_slug": "set_dense_exact_empty_only",
        "family": "Set Dense empty_only + candidate_gather",
        "backend_family": "dense",
        "backend": "exact",
        "config": "configs/paper_lr_norm/set_dense_exact.yaml",
        "group": "a9_candidate_gather_set_dense_exact_D384_FF1536",
    },
    "set_sparse_local_band": {
        "summary_slug": "set_sparse_local_band_empty_only",
        "family": "Set Sparse empty_only + candidate_gather",
        "backend_family": "sparse",
        "backend": "local_band",
        "config": "configs/paper_lr_norm/set_sparse_local_band.yaml",
        "group": "a9_candidate_gather_set_sparse_local_band_D384_FF1536",
    },
    "set_linear_landmark": {
        "summary_slug": "set_linear_landmark_empty_only",
        "family": "Set Linear empty_only + candidate_gather",
        "backend_family": "linear",
        "backend": "landmark",
        "config": "configs/paper_lr_norm/set_linear_landmark.yaml",
        "group": "a9_candidate_gather_set_linear_landmark_D384_FF1536",
    },
}


@dataclass(frozen=True)
class ExpectedRun:
    slug: str
    w: int
    s: int
    seed: int

    @property
    def m(self) -> int:
        return ((SEQ_LEN - self.w) // self.s) + 1

    @property
    def spec(self) -> dict[str, str]:
        return FAMILIES[self.slug]

    @property
    def name(self) -> str:
        lr_tag = LR.replace(".", "p")
        return (
            f"a9_cg_{self.slug}_D384_FF1536_L{SEQ_LEN}_w{self.w}_s{self.s}_"
            f"M{self.m}_lr{lr_tag}_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return RAW / self.spec["group"] / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


def expected_runs() -> list[ExpectedRun]:
    return [
        ExpectedRun(slug, w, s, seed)
        for slug in FAMILIES
        for w, s in TOPOLOGIES
        for seed in SEEDS
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


def validate_run(spec: ExpectedRun) -> dict[str, str]:
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {spec.csv_path}")
    if not spec.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {spec.json_path}")
    rows = read_csv(spec.csv_path)
    if len(rows) != EPOCHS:
        raise ValueError(f"{spec.csv_path} has {len(rows)} rows, expected {EPOCHS}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, EPOCHS + 1)):
        raise ValueError(f"{spec.csv_path} epochs are not 1..{EPOCHS}: {epochs}")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite CSV values in {spec.csv_path}: {bad[:5]}")
    meta = json.loads(spec.json_path.read_text())
    family = spec.spec
    checks = {
        "model.implementation": "set_only",
        "model.backend": family["backend"],
        "model.set_causality_mode": "strict_past",
        "model.output_residual_mode": "empty_only",
        "model.router.score_mode": "candidate_gather",
        "resolved.router_score_mode": "candidate_gather",
        "model.d_model": D_MODEL,
        "model.dim_feedforward": D_FF,
        "model.max_seq_len": SEQ_LEN,
        "model.window_size": spec.w,
        "model.stride": spec.s,
        "model.d_phi": D_MODEL,
        "model.set_state_dim": D_MODEL,
        "resolved.d_phi": D_MODEL,
        "resolved.set_state_dim": D_MODEL,
        "model.feature_mode": "geometry_only",
        "model.geometry.enabled": False,
        "model.token_mlp.enabled": False,
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if str(actual) != str(expected):
            raise ValueError(f"{spec.json_path} has {key}={actual!r}, expected {expected!r}")
    if family["backend"] == "local_band" and str(meta.get("model.backend_params.radius")) != "4":
        raise ValueError(f"{spec.json_path} missing local_band radius=4")
    landmark_coverage = "NA"
    landmark_count = "NA"
    if family["backend"] == "landmark":
        expected_count = max(round(0.25 * spec.m), 2)
        if float_or_none(meta.get("resolved.landmark_coverage")) != 0.25:
            raise ValueError(f"{spec.json_path} missing resolved landmark_coverage=0.25")
        if str(meta.get("resolved.landmark_count")) != str(expected_count):
            raise ValueError(
                f"{spec.json_path} landmark_count={meta.get('resolved.landmark_count')}, "
                f"expected {expected_count}"
            )
        landmark_coverage = str(meta.get("resolved.landmark_coverage"))
        landmark_count = str(meta.get("resolved.landmark_count"))
    last = rows[-1]
    for key in [
        "train/loss",
        "val/loss",
        "train/ppl",
        "val/ppl",
        "train/peak_vram_mib",
        "ausa/candidate_count_mean",
        "ausa/router_entropy_norm",
        "ausa/router_top1_weight",
    ]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{spec.csv_path} missing finite {key}")
    return {
        "phase": "A9_candidate_gather",
        "family_slug": f"{spec.slug}_empty_only_candidate_gather",
        "reference_family_slug": family["summary_slug"],
        "family": family["family"],
        "implementation": "set_only",
        "backend_family": family["backend_family"],
        "backend": family["backend"],
        "seed": str(spec.seed),
        "lr": LR,
        "D": str(D_MODEL),
        "d_ff": str(D_FF),
        "L": str(SEQ_LEN),
        "w": str(spec.w),
        "s": str(spec.s),
        "M": str(spec.m),
        "M_over_L": f"{spec.m / SEQ_LEN:.8f}",
        "L_over_M": f"{SEQ_LEN / spec.m:.8f}",
        "output_residual_mode": "empty_only",
        "router_score_mode": "candidate_gather",
        "config": family["config"],
        "csv_path": str(spec.csv_path.relative_to(ROOT)),
        "json_path": str(spec.json_path.relative_to(ROOT)),
        "rows": str(len(rows)),
        "final_train_loss": last["train/loss"],
        "final_val_loss": last["val/loss"],
        "final_train_ppl": last["train/ppl"],
        "final_val_ppl": last["val/ppl"],
        "time_per_epoch_s": last.get("train/time_per_epoch_s", "NA"),
        "peak_vram_mib": last["train/peak_vram_mib"],
        "candidate_count_mean": last["ausa/candidate_count_mean"],
        "candidate_count_max": last.get("ausa/candidate_count_max", "NA"),
        "router_entropy_norm": last["ausa/router_entropy_norm"],
        "router_top1_weight": last["ausa/router_top1_weight"],
        "landmark_coverage": landmark_coverage,
        "landmark_count": landmark_count,
        "source_csv_sha256": sha256(spec.csv_path),
        "source_json_sha256": sha256(spec.json_path),
    }


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_references() -> dict[tuple[str, str, str], dict[str, str]]:
    refs: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_tsv(REFERENCE_SUMMARY):
        key = (row.get("family_slug", ""), row.get("w", ""), row.get("s", ""))
        if row.get("family_slug") in {v["summary_slug"] for v in FAMILIES.values()} and (
            row.get("w"),
            row.get("s"),
        ) in {("4", "2"), ("8", "4")}:
            refs[key] = row
    return refs


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    refs = load_references()
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family_slug"], row["reference_family_slug"], row["w"], row["s"])] .append(row)
    out = []
    for (slug, ref_slug, w, s), grp in sorted(groups.items()):
        vals = [float(row["final_val_ppl"]) for row in grp]
        train_vals = [float(row["final_train_ppl"]) for row in grp]
        vrams = [float(row["peak_vram_mib"]) for row in grp]
        times = [float_or_none(row["time_per_epoch_s"]) for row in grp]
        first = grp[0]
        ref = refs.get((ref_slug, w, s), {})
        ref_ppl = float_or_none(ref.get("mean_final_val_ppl"))
        ref_vram = float_or_none(ref.get("mean_peak_vram_mib"))
        mean_ppl = mean(vals)
        mean_vram = mean(vrams)
        out.append({
            "family_slug": slug,
            "reference_family_slug": ref_slug,
            "family": first["family"],
            "backend_family": first["backend_family"],
            "backend": first["backend"],
            "lr": first["lr"],
            "D": first["D"],
            "d_ff": first["d_ff"],
            "L": first["L"],
            "w": w,
            "s": s,
            "M": first["M"],
            "L_over_M": first["L_over_M"],
            "output_residual_mode": first["output_residual_mode"],
            "router_score_mode": first["router_score_mode"],
            "n": str(len(grp)),
            "seeds": ",".join(sorted((row["seed"] for row in grp), key=int)),
            "mean_final_val_ppl": f"{mean_ppl:.6f}",
            "std_final_val_ppl": f"{pstdev(vals):.6f}" if len(vals) > 1 else "0.000000",
            "mean_final_train_ppl": f"{mean(train_vals):.6f}",
            "mean_peak_vram_mib": f"{mean_vram:.6f}",
            "mean_time_per_epoch_s": f"{mean([v for v in times if v is not None]):.6f}" if any(v is not None for v in times) else "NA",
            "mean_candidate_count": f"{mean(float(row['candidate_count_mean']) for row in grp):.6f}",
            "reference_mean_final_val_ppl": ref.get("mean_final_val_ppl", "NA"),
            "reference_mean_peak_vram_mib": ref.get("mean_peak_vram_mib", "NA"),
            "delta_val_ppl_vs_dense_router": (
                f"{mean_ppl - ref_ppl:.6f}" if ref_ppl is not None else "NA"
            ),
            "delta_vram_mib_vs_dense_router": (
                f"{mean_vram - ref_vram:.6f}" if ref_vram is not None else "NA"
            ),
        })
    return out


def main() -> None:
    failures: list[str] = []
    rows: list[dict[str, str]] = []
    for spec in expected_runs():
        try:
            rows.append(validate_run(spec))
        except Exception as exc:  # noqa: BLE001
            failures.append(str(exc))
    failures.extend(scan_logs())

    all_path = TABLES / "a9_candidate_gather_all_runs.tsv"
    summary_path = TABLES / "a9_candidate_gather_summary.tsv"
    manifest_path = CHECKS / "a9_candidate_gather_manifest.json"
    audit_path = AUDIT / "A9_candidate_gather.md"

    fields = [
        "phase", "family_slug", "reference_family_slug", "family", "implementation",
        "backend_family", "backend", "seed", "lr", "D", "d_ff", "L", "w", "s", "M",
        "M_over_L", "L_over_M", "output_residual_mode", "router_score_mode", "config",
        "csv_path", "json_path", "rows", "final_train_loss", "final_val_loss",
        "final_train_ppl", "final_val_ppl", "time_per_epoch_s", "peak_vram_mib",
        "candidate_count_mean", "candidate_count_max", "router_entropy_norm",
        "router_top1_weight", "landmark_coverage", "landmark_count",
        "source_csv_sha256", "source_json_sha256",
    ]
    if rows:
        write_tsv(all_path, sorted(rows, key=lambda r: (r["family_slug"], int(r["w"]), int(r["s"]), int(r["seed"]))), fields)
        summary = summarize(rows)
        write_tsv(summary_path, summary, list(summary[0].keys()) if summary else [])
    else:
        summary = []

    manifest = {
        "phase": "A9_candidate_gather",
        "status": "pass" if not failures and len(rows) == len(expected_runs()) else "fail",
        "validated_runs": len(rows),
        "expected_runs": len(expected_runs()),
        "failures": failures,
        "checks": {
            "branch": run(["git", "branch", "--show-current"]),
            "head": run(["git", "rev-parse", "HEAD"]),
            "status_short": run(["git", "status", "--short"]),
        },
        "artifacts": {
            "all_runs": str(all_path.relative_to(ROOT)) if all_path.exists() else None,
            "summary": str(summary_path.relative_to(ROOT)) if summary_path.exists() else None,
            "audit": str(audit_path.relative_to(ROOT)),
        },
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (all_path, summary_path)
            if p.exists()
        },
    }
    CHECKS.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    AUDIT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# A9 Candidate-Gather Router Comparison",
        "",
        f"Status: {manifest['status'].upper()}",
        "",
        "Goal: remove dense token-to-set router scores/probabilities while preserving A7 empty_only semantics.",
        "",
        "Matrix: set dense/sparse/linear, topologies `(4,2)` and `(8,4)`, seeds `0,1,2`, `D=384`, `d_ff=1536`, `L=512`, LR `1e-4`, strict-past causal LM.",
        "",
        "Only intended change versus the A7 references: `model.router.score_mode=candidate_gather`.",
        "",
        "Artifacts:",
        f"- `{all_path.relative_to(ROOT)}`",
        f"- `{summary_path.relative_to(ROOT)}`",
        f"- `{manifest_path.relative_to(ROOT)}`",
        "",
        "Summary:",
    ]
    for row in summary:
        lines.append(
            "- {family} `(w,s)=({w},{s})`: PPL {ppl} "
            "(delta vs dense-router reference {dppl}), VRAM {vram} MiB "
            "(delta {dvram} MiB).".format(
                family=row["family"],
                w=row["w"],
                s=row["s"],
                ppl=row["mean_final_val_ppl"],
                dppl=row["delta_val_ppl_vs_dense_router"],
                vram=row["mean_peak_vram_mib"],
                dvram=row["delta_vram_mib_vs_dense_router"],
            )
        )
    if failures:
        lines.extend(["", "Failures:"])
        lines.extend(f"- {failure}" for failure in failures)
    audit_path.write_text("\n".join(lines) + "\n")

    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
