#!/usr/bin/env python3
"""Validate A7.5 seed-extension runs and build augmented A7 summaries."""

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
RAW = ROOT / "out" / "paper_mechanisms" / "a7_seed_extension"
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"
AUDIT = ROOT / "audit"
LOG_ROOT = ROOT / "logs" / "a7_seed_extension"
BASE_ALL = TABLES / "a7_backend_family_empty_only_all_runs.tsv"

SEQ_LEN = 512
D_MODEL = 384
D_FF = 1536
EPOCHS = 10
LR = "1e-4"
SEEDS = [3, 4]
SET_TOPOLOGIES = [(1, 1), (2, 1), (3, 1)]

BASELINE_FAMILIES = {
    "baseline_dense_exact": {
        "family": "Baseline Dense",
        "backend_family": "dense",
        "backend": "exact",
        "config": "configs/paper_lr_norm/baseline_dense_exact.yaml",
    },
    "baseline_sparse_local_band": {
        "family": "Token Sparse",
        "backend_family": "sparse",
        "backend": "local_band",
        "config": "configs/paper_lr_norm/baseline_sparse_local_band.yaml",
    },
    "baseline_linear_landmark": {
        "family": "Token Linear",
        "backend_family": "linear",
        "backend": "landmark",
        "config": "configs/paper_lr_norm/baseline_linear_landmark.yaml",
    },
}

SET_FAMILIES = {
    "set_dense_exact": {
        "slug": "set_dense_exact_empty_only",
        "family": "Set Dense empty_only",
        "backend_family": "dense",
        "backend": "exact",
        "config": "configs/paper_lr_norm/set_dense_exact.yaml",
    },
    "set_sparse_local_band": {
        "slug": "set_sparse_local_band_empty_only",
        "family": "Set Sparse empty_only",
        "backend_family": "sparse",
        "backend": "local_band",
        "config": "configs/paper_lr_norm/set_sparse_local_band.yaml",
    },
    "set_linear_landmark": {
        "slug": "set_linear_landmark_empty_only",
        "family": "Set Linear empty_only",
        "backend_family": "linear",
        "backend": "landmark",
        "config": "configs/paper_lr_norm/set_linear_landmark.yaml",
    },
}

FIELDS = [
    "phase", "family_slug", "family", "implementation", "backend_family", "backend",
    "seed", "lr", "D", "d_ff", "L", "w", "s", "M", "M_over_L", "L_over_M",
    "output_residual_mode", "config", "csv_path", "json_path", "rows",
    "final_train_loss", "final_val_loss", "final_train_ppl", "final_val_ppl",
    "time_per_epoch_s", "peak_vram_mib", "candidate_count_mean",
    "candidate_count_max", "router_entropy_norm", "router_top1_weight",
    "landmark_coverage", "landmark_count", "source_csv_sha256",
    "source_json_sha256", "comparison_provenance",
]

SUMMARY_FIELDS = [
    "family_slug", "family", "implementation", "backend_family", "backend", "lr",
    "D", "d_ff", "L", "w", "s", "M", "M_over_L", "L_over_M",
    "output_residual_mode", "n", "seeds", "mean_final_val_ppl",
    "std_final_val_ppl", "min_final_val_ppl", "max_final_val_ppl",
    "mean_final_train_ppl", "mean_time_per_epoch_s", "mean_peak_vram_mib",
    "mean_candidate_count", "landmark_coverage", "landmark_count",
]


@dataclass(frozen=True)
class ExpectedRun:
    kind: str
    family: str
    seed: int
    w: int | None = None
    s: int | None = None

    @property
    def m(self) -> int | None:
        if self.w is None or self.s is None:
            return None
        return ((SEQ_LEN - self.w) // self.s) + 1

    @property
    def group(self) -> str:
        return f"a7_seed_extension_{self.family}_D{D_MODEL}_FF{D_FF}"

    @property
    def name(self) -> str:
        lr_tag = LR.replace(".", "p")
        if self.kind == "baseline":
            return f"a7_seedext_{self.family}_D{D_MODEL}_FF{D_FF}_L{SEQ_LEN}_lr{lr_tag}_seed{self.seed}"
        return (
            f"a7_seedext_{self.family}_D{D_MODEL}_FF{D_FF}_L{SEQ_LEN}_"
            f"w{self.w}_s{self.s}_M{self.m}_lr{lr_tag}_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return RAW / self.group / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


def expected_runs() -> list[ExpectedRun]:
    runs = [
        ExpectedRun("baseline", family, seed)
        for family in BASELINE_FAMILIES
        for seed in SEEDS
    ]
    runs.extend(
        ExpectedRun("set", family, seed, w, s)
        for family in SET_FAMILIES
        for w, s in SET_TOPOLOGIES
        for seed in SEEDS
    )
    return runs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def base_row(
    *,
    phase: str,
    slug: str,
    family: str,
    implementation: str,
    backend_family: str,
    backend: str,
    seed: str,
    lr: str,
    d_model: str,
    d_ff: str,
    seq_len: str,
    w: str,
    s: str,
    m: str,
    config: str,
    csv_path: str,
    json_path: str,
    rows: str,
    last: dict[str, str],
    candidate_count_mean: str = "NA",
    candidate_count_max: str = "NA",
    router_entropy_norm: str = "NA",
    router_top1_weight: str = "NA",
    landmark_coverage: str = "NA",
    landmark_count: str = "NA",
    output_residual_mode: str = "NA",
    provenance: str = "",
) -> dict[str, str]:
    if m == "NA":
        m_over_l = "NA"
        l_over_m = "NA"
    else:
        m_int = int(m)
        m_over_l = f"{m_int / int(seq_len):.8f}"
        l_over_m = f"{int(seq_len) / m_int:.8f}"
    return {
        "phase": phase,
        "family_slug": slug,
        "family": family,
        "implementation": implementation,
        "backend_family": backend_family,
        "backend": backend,
        "seed": seed,
        "lr": lr,
        "D": d_model,
        "d_ff": d_ff,
        "L": seq_len,
        "w": w,
        "s": s,
        "M": m,
        "M_over_L": m_over_l,
        "L_over_M": l_over_m,
        "output_residual_mode": output_residual_mode,
        "config": config,
        "csv_path": csv_path,
        "json_path": json_path,
        "rows": rows,
        "final_train_loss": last.get("train/loss", "NA"),
        "final_val_loss": last.get("val/loss", "NA"),
        "final_train_ppl": last.get("train/ppl", "NA"),
        "final_val_ppl": last.get("val/ppl", "NA"),
        "time_per_epoch_s": last.get("train/time_per_epoch_s", "NA"),
        "peak_vram_mib": last.get("train/peak_vram_mib", "NA"),
        "candidate_count_mean": candidate_count_mean,
        "candidate_count_max": candidate_count_max,
        "router_entropy_norm": router_entropy_norm,
        "router_top1_weight": router_top1_weight,
        "landmark_coverage": landmark_coverage,
        "landmark_count": landmark_count,
        "source_csv_sha256": sha256(ROOT / csv_path),
        "source_json_sha256": sha256(ROOT / json_path),
        "comparison_provenance": provenance,
    }


def validate_baseline(spec: ExpectedRun) -> dict[str, str]:
    family = BASELINE_FAMILIES[spec.family]
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {spec.csv_path}")
    if not spec.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {spec.json_path}")
    rows = read_csv(spec.csv_path)
    if len(rows) != EPOCHS:
        raise ValueError(f"{spec.csv_path} has {len(rows)} rows, expected {EPOCHS}")
    if [int(r["epoch"]) for r in rows] != list(range(1, EPOCHS + 1)):
        raise ValueError(f"{spec.csv_path} epochs are not 1..{EPOCHS}")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite baseline CSV values in {spec.csv_path}: {bad[:5]}")
    meta = json.loads(spec.json_path.read_text())
    checks = {
        "model.implementation": "baseline_token",
        "model.backend": family["backend"],
        "model.d_model": D_MODEL,
        "model.dim_feedforward": D_FF,
        "model.max_seq_len": SEQ_LEN,
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
        coverage = float_or_none(meta.get("model.backend_params.landmark_coverage"))
        resolved = float_or_none(meta.get("resolved.landmark_coverage"))
        if coverage != 0.25 or resolved != 0.25:
            raise ValueError(f"{spec.json_path} missing landmark_coverage=0.25")
        landmark_coverage = str(meta.get("resolved.landmark_coverage", "NA"))
        landmark_count = str(meta.get("resolved.landmark_count", "NA"))
    return base_row(
        phase="A7.5_seed_extension",
        slug=spec.family,
        family=family["family"],
        implementation="baseline_token",
        backend_family=family["backend_family"],
        backend=family["backend"],
        seed=str(spec.seed),
        lr=LR,
        d_model=str(D_MODEL),
        d_ff=str(D_FF),
        seq_len=str(SEQ_LEN),
        w="NA",
        s="NA",
        m="NA",
        config=family["config"],
        csv_path=rel(spec.csv_path),
        json_path=rel(spec.json_path),
        rows=str(len(rows)),
        last=rows[-1],
        landmark_coverage=landmark_coverage,
        landmark_count=landmark_count,
        provenance="new A7.5 targeted baseline seed extension",
    )


def validate_set(spec: ExpectedRun) -> dict[str, str]:
    family = SET_FAMILIES[spec.family]
    assert spec.w is not None and spec.s is not None and spec.m is not None
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {spec.csv_path}")
    if not spec.json_path.exists():
        raise FileNotFoundError(f"missing JSON: {spec.json_path}")
    rows = read_csv(spec.csv_path)
    if len(rows) != EPOCHS:
        raise ValueError(f"{spec.csv_path} has {len(rows)} rows, expected {EPOCHS}")
    if [int(r["epoch"]) for r in rows] != list(range(1, EPOCHS + 1)):
        raise ValueError(f"{spec.csv_path} epochs are not 1..{EPOCHS}")
    ok, bad = finite_csv(rows)
    if not ok:
        raise ValueError(f"non-finite set CSV values in {spec.csv_path}: {bad[:5]}")
    meta = json.loads(spec.json_path.read_text())
    checks = {
        "model.implementation": "set_only",
        "model.backend": family["backend"],
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
        raise ValueError(f"{spec.json_path} allow_token_token mismatch")
    landmark_coverage = "NA"
    landmark_count = "NA"
    if family["backend"] == "local_band" and str(meta.get("model.backend_params.radius")) != "4":
        raise ValueError(f"{spec.json_path} missing local_band radius=4")
    if family["backend"] == "landmark":
        coverage = float_or_none(meta.get("model.backend_params.landmark_coverage"))
        resolved = float_or_none(meta.get("resolved.landmark_coverage"))
        expected_count = max(round(0.25 * spec.m), 2)
        if coverage != 0.25 or resolved != 0.25:
            raise ValueError(f"{spec.json_path} missing landmark_coverage=0.25")
        if str(meta.get("resolved.landmark_count")) != str(expected_count):
            raise ValueError(f"{spec.json_path} landmark_count mismatch")
        landmark_coverage = str(meta.get("resolved.landmark_coverage"))
        landmark_count = str(meta.get("resolved.landmark_count"))
    last = rows[-1]
    for key in [
        "train/loss",
        "val/loss",
        "train/ppl",
        "val/ppl",
        "ausa/candidate_count_mean",
        "ausa/candidate_count_max",
        "ausa/router_entropy_norm",
        "ausa/router_top1_weight",
    ]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{spec.csv_path} missing finite final {key}")
    return base_row(
        phase="A7.5_seed_extension",
        slug=family["slug"],
        family=family["family"],
        implementation="set_only",
        backend_family=family["backend_family"],
        backend=family["backend"],
        seed=str(spec.seed),
        lr=LR,
        d_model=str(D_MODEL),
        d_ff=str(D_FF),
        seq_len=str(SEQ_LEN),
        w=str(spec.w),
        s=str(spec.s),
        m=str(spec.m),
        output_residual_mode="empty_only",
        config=family["config"],
        csv_path=rel(spec.csv_path),
        json_path=rel(spec.json_path),
        rows=str(len(rows)),
        last=last,
        candidate_count_mean=last.get("ausa/candidate_count_mean", "NA"),
        candidate_count_max=last.get("ausa/candidate_count_max", "NA"),
        router_entropy_norm=last.get("ausa/router_entropy_norm", "NA"),
        router_top1_weight=last.get("ausa/router_top1_weight", "NA"),
        landmark_coverage=landmark_coverage,
        landmark_count=landmark_count,
        provenance="new A7.5 targeted set seed extension",
    )


def merge_rows(base_rows: list[dict[str, str]], extension_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for row in base_rows + extension_rows:
        key = (row["family_slug"], row["seed"], row["w"], row["s"], row["lr"], row["csv_path"])
        merged[key] = row
    return sorted(
        merged.values(),
        key=lambda r: (
            r["backend_family"],
            r["family_slug"],
            10**9 if r["w"] == "NA" else int(r["w"]),
            10**9 if r["s"] == "NA" else int(r["s"]),
            int(r["seed"]),
        ),
    )


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family_slug"], row["backend_family"], row["backend"], row["w"], row["s"], row["lr"])].append(row)
    out = []
    for (slug, backend_family, backend, w, s, lr), grp in sorted(groups.items()):
        vals = [float(row["final_val_ppl"]) for row in grp]
        train_vals = [float(row["final_train_ppl"]) for row in grp]
        times = [float_or_none(row["time_per_epoch_s"]) for row in grp]
        vrams = [float_or_none(row["peak_vram_mib"]) for row in grp]
        cands = [float_or_none(row["candidate_count_mean"]) for row in grp]
        first = grp[0]
        out.append({
            "family_slug": slug,
            "family": first["family"],
            "implementation": first["implementation"],
            "backend_family": backend_family,
            "backend": backend,
            "lr": lr,
            "D": first["D"],
            "d_ff": first["d_ff"],
            "L": first["L"],
            "w": w,
            "s": s,
            "M": first["M"],
            "M_over_L": first["M_over_L"],
            "L_over_M": first["L_over_M"],
            "output_residual_mode": first["output_residual_mode"],
            "n": str(len(grp)),
            "seeds": ",".join(sorted((row["seed"] for row in grp), key=int)),
            "mean_final_val_ppl": f"{mean(vals):.6f}",
            "std_final_val_ppl": f"{pstdev(vals):.6f}" if len(vals) > 1 else "0.000000",
            "min_final_val_ppl": f"{min(vals):.6f}",
            "max_final_val_ppl": f"{max(vals):.6f}",
            "mean_final_train_ppl": f"{mean(train_vals):.6f}",
            "mean_time_per_epoch_s": f"{mean([v for v in times if v is not None]):.6f}" if any(v is not None for v in times) else "NA",
            "mean_peak_vram_mib": f"{mean([v for v in vrams if v is not None]):.6f}" if any(v is not None for v in vrams) else "NA",
            "mean_candidate_count": f"{mean([v for v in cands if v is not None]):.6f}" if any(v is not None for v in cands) else "NA",
            "landmark_coverage": first["landmark_coverage"],
            "landmark_count": first["landmark_count"],
        })
    return out


def main() -> None:
    failures: list[str] = []
    extension_rows: list[dict[str, str]] = []
    for spec in expected_runs():
        try:
            extension_rows.append(validate_baseline(spec) if spec.kind == "baseline" else validate_set(spec))
        except Exception as exc:  # noqa: BLE001
            failures.append(str(exc))
    failures.extend(scan_logs())

    base_rows: list[dict[str, str]] = []
    try:
        base_rows = read_tsv(BASE_ALL)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"could not read base A7 table: {exc}")

    all_rows = merge_rows(base_rows, extension_rows) if base_rows else extension_rows
    summary_rows = summarize(all_rows) if all_rows else []

    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    all_path = TABLES / "a7_backend_family_empty_only_augmented_all_runs.tsv"
    summary_path = TABLES / "a7_backend_family_empty_only_augmented_summary.tsv"
    manifest_path = CHECKS / "a7_seed_extension_manifest.json"
    audit_path = AUDIT / "A7_seed_extension.md"

    write_tsv(all_path, all_rows, FIELDS)
    write_tsv(summary_path, summary_rows, SUMMARY_FIELDS)

    five_seed_rows = [
        row for row in summary_rows
        if row["n"] == "5" and (
            row["implementation"] == "baseline_token"
            or row["w"] in {"1", "2", "3"}
        )
    ]
    manifest = {
        "phase": "A7.5-seed-extension",
        "status": "pass" if not failures else "fail",
        "expected_new_runs": len(expected_runs()),
        "validated_new_runs": len(extension_rows),
        "base_rows_reused": len(base_rows),
        "augmented_rows": len(all_rows),
        "summary_rows": len(summary_rows),
        "five_seed_summary_rows": len(five_seed_rows),
        "failures": failures,
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "dirty_status": run(["git", "status", "--short"]),
        "artifacts": {
            "all_runs": str(all_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
            "audit": str(audit_path.relative_to(ROOT)),
        },
        "design": {
            "new_seeds": SEEDS,
            "baseline_families": sorted(BASELINE_FAMILIES),
            "set_families": sorted(SET_FAMILIES),
            "set_topologies": [{"w": w, "s": s} for w, s in SET_TOPOLOGIES],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    audit_lines = [
        "# A7.5 Targeted Seed Extension",
        "",
        f"Status: {manifest['status'].upper()}",
        "",
        "## Scope",
        "",
        "Seeds 3 and 4 were added for the convergence-critical A7 operating points: "
        "matched dense/sparse/linear token baselines and dense/sparse/linear "
        "set `empty_only` families at `(w,s)={(1,1),(2,1),(3,1)}`.",
        "",
        "The compressed A7 points outside this set remain at three seeds because "
        "their degradation is large and not the limiting uncertainty for the "
        "empirical convergence claim.",
        "",
        "## Validation",
        "",
        f"- New runs validated: {len(extension_rows)}/{len(expected_runs())}",
        f"- Reused base A7 rows: {len(base_rows)}",
        f"- Augmented all-run rows: {len(all_rows)}",
        f"- Augmented summary rows: {len(summary_rows)}",
        f"- Failures: {len(failures)}",
        "",
        "## Artifacts",
        "",
        f"- All runs: `{all_path.relative_to(ROOT)}`",
        f"- Summary: `{summary_path.relative_to(ROOT)}`",
        f"- Manifest: `{manifest_path.relative_to(ROOT)}`",
    ]
    if failures:
        audit_lines.extend(["", "## Failures", ""])
        audit_lines.extend(f"- {failure}" for failure in failures)
    audit_path.write_text("\n".join(audit_lines) + "\n")

    print(json.dumps(manifest, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
