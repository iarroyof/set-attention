#!/usr/bin/env python3
"""Validate and summarize A7 backend-family empty_only calibration evidence."""

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
RAW = ROOT / "out" / "paper_mechanisms" / "a7_backend_family_empty_only"
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"
AUDIT = ROOT / "audit"
LOG_ROOT = ROOT / "logs" / "a7_backend_family_empty_only"
DENSE_A7_ALL = TABLES / "a7_empty_only_calibration_all_runs.tsv"
TOKEN_CONTROLS = TABLES / "a2_baseline_controls_all_runs.tsv"

SEQ_LEN = 512
D_MODEL = 384
D_FF = 1536
EPOCHS = 10
LR = "1e-4"
SEEDS = [0, 1, 2]
TOPOLOGIES = [(1, 1), (2, 1), (3, 1), (2, 2), (4, 2), (8, 4), (16, 8), (32, 16)]

SET_FAMILIES = {
    "set_sparse_local_band": {
        "family": "Set Sparse empty_only",
        "backend_family": "sparse",
        "backend": "local_band",
        "config": "configs/paper_lr_norm/set_sparse_local_band.yaml",
        "group": "a7_backend_family_empty_only_set_sparse_local_band_D384_FF1536",
    },
    "set_linear_landmark": {
        "family": "Set Linear empty_only",
        "backend_family": "linear",
        "backend": "landmark",
        "config": "configs/paper_lr_norm/set_linear_landmark.yaml",
        "group": "a7_backend_family_empty_only_set_linear_landmark_D384_FF1536",
    },
}

TOKEN_BASELINES = {
    "baseline_sparse_local_band": {
        "family": "Token Sparse",
        "backend_family": "sparse",
        "backend": "local_band",
    },
    "baseline_linear_landmark": {
        "family": "Token Linear",
        "backend_family": "linear",
        "backend": "landmark",
    },
}


@dataclass(frozen=True)
class ExpectedSetRun:
    slug: str
    w: int
    s: int
    seed: int

    @property
    def spec(self) -> dict[str, str]:
        return SET_FAMILIES[self.slug]

    @property
    def m(self) -> int:
        return ((SEQ_LEN - self.w) // self.s) + 1

    @property
    def name(self) -> str:
        return (
            f"a7_empty_{self.slug}_D384_FF1536_L{SEQ_LEN}_w{self.w}_s{self.s}_"
            f"M{self.m}_lr{LR.replace('.', 'p')}_seed{self.seed}"
        )

    @property
    def csv_path(self) -> Path:
        return RAW / self.spec["group"] / f"{self.name}.csv"

    @property
    def json_path(self) -> Path:
        return self.csv_path.with_suffix(".json")


def expected_set_runs() -> list[ExpectedSetRun]:
    return [
        ExpectedSetRun(slug, w, s, seed)
        for slug in SET_FAMILIES
        for w, s in TOPOLOGIES
        for seed in SEEDS
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


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
        "final_train_loss": last.get("train/loss", last.get("final_train_loss", "NA")),
        "final_val_loss": last.get("val/loss", last.get("final_val_loss", "NA")),
        "final_train_ppl": last.get("train/ppl", last.get("final_train_ppl", "NA")),
        "final_val_ppl": last.get("val/ppl", last.get("final_val_ppl", "NA")),
        "time_per_epoch_s": last.get(
            "train/time_per_epoch_s", last.get("time_per_epoch_s", "NA")
        ),
        "peak_vram_mib": last.get("train/peak_vram_mib", last.get("peak_vram_mib", "NA")),
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


def load_dense_a7_rows() -> list[dict[str, str]]:
    if not DENSE_A7_ALL.exists():
        raise FileNotFoundError(f"missing dense A7 table: {DENSE_A7_ALL}")
    out = []
    for row in read_tsv(DENSE_A7_ALL):
        if row["family_slug"] == "baseline_dense_exact":
            out.append({
                **row,
                "implementation": "baseline_token",
                "backend_family": "dense",
                "landmark_coverage": "NA",
                "landmark_count": "NA",
                "comparison_provenance": "reused dense token baseline from A7/A2",
            })
        elif row["family_slug"] == "set_dense_exact_empty_only":
            out.append({
                **row,
                "implementation": "set_only",
                "backend_family": "dense",
                "landmark_coverage": "NA",
                "landmark_count": "NA",
                "comparison_provenance": "reused dense set empty_only A7 calibration",
            })
    return out


def load_token_baselines() -> list[dict[str, str]]:
    if not TOKEN_CONTROLS.exists():
        raise FileNotFoundError(f"missing token controls table: {TOKEN_CONTROLS}")
    out = []
    wanted = set(TOKEN_BASELINES)
    for row in read_tsv(TOKEN_CONTROLS):
        if row.get("family_slug") not in wanted:
            continue
        if row.get("lr") != LR or row.get("D") != str(D_MODEL) or row.get("d_ff") != str(D_FF):
            continue
        if row.get("L") != str(SEQ_LEN) or row.get("seed") not in {str(s) for s in SEEDS}:
            continue
        csv_path = ROOT / row["csv_path"]
        json_path = ROOT / row["json_path"]
        rows = read_csv(csv_path)
        if len(rows) != EPOCHS:
            raise ValueError(f"{csv_path} has {len(rows)} rows, expected {EPOCHS}")
        ok, bad = finite_csv(rows)
        if not ok:
            raise ValueError(f"non-finite token baseline CSV values in {csv_path}: {bad[:5]}")
        meta = json.loads(json_path.read_text())
        spec = TOKEN_BASELINES[row["family_slug"]]
        if meta.get("model.implementation") != "baseline_token":
            raise ValueError(f"{json_path} is not baseline_token")
        if meta.get("model.backend") != spec["backend"]:
            raise ValueError(f"{json_path} has backend {meta.get('model.backend')}")
        if spec["backend"] == "landmark":
            coverage = float_or_none(meta.get("model.backend_params.landmark_coverage"))
            resolved = float_or_none(meta.get("resolved.landmark_coverage"))
            if coverage != 0.25 or resolved != 0.25:
                raise ValueError(f"{json_path} missing landmark_coverage=0.25")
        out.append(base_row(
            phase="A2.4_reused_token_backend_baseline",
            slug=row["family_slug"],
            family=spec["family"],
            implementation="baseline_token",
            backend_family=spec["backend_family"],
            backend=spec["backend"],
            seed=row["seed"],
            lr=row["lr"],
            d_model=row["D"],
            d_ff=row["d_ff"],
            seq_len=row["L"],
            w="NA",
            s="NA",
            m="NA",
            config=row["config"],
            csv_path=row["csv_path"],
            json_path=row["json_path"],
            rows=str(len(rows)),
            last=rows[-1],
            landmark_coverage=str(meta.get("resolved.landmark_coverage", "NA")),
            landmark_count=str(meta.get("resolved.landmark_count", "NA")),
            provenance="reused v2.7 matched token-backend baseline; no set topology",
        ))
    counts = defaultdict(int)
    for row in out:
        counts[row["family_slug"]] += 1
    for slug in wanted:
        if counts[slug] != 3:
            raise ValueError(f"{slug} baseline count {counts[slug]}, expected 3")
    return sorted(out, key=lambda r: (r["family_slug"], int(r["seed"])))


def validate_set_run(spec: ExpectedSetRun) -> dict[str, str]:
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
    family = SET_FAMILIES[spec.slug]
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
    if family["backend"] == "local_band":
        if str(meta.get("model.backend_params.radius")) != "4":
            raise ValueError(f"{spec.json_path} missing local_band radius=4")
    if family["backend"] == "landmark":
        coverage = float_or_none(meta.get("model.backend_params.landmark_coverage"))
        resolved = float_or_none(meta.get("resolved.landmark_coverage"))
        expected_count = max(round(0.25 * spec.m), 2)
        if coverage != 0.25 or resolved != 0.25:
            raise ValueError(f"{spec.json_path} missing landmark_coverage=0.25")
        if str(meta.get("resolved.landmark_count")) != str(expected_count):
            raise ValueError(
                f"{spec.json_path} landmark_count={meta.get('resolved.landmark_count')}, "
                f"expected {expected_count}"
            )
        landmark_coverage = str(meta.get("resolved.landmark_coverage"))
        landmark_count = str(meta.get("resolved.landmark_count"))
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
    return base_row(
        phase="A7",
        slug=f"{spec.slug}_empty_only",
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
        csv_path=str(spec.csv_path.relative_to(ROOT)),
        json_path=str(spec.json_path.relative_to(ROOT)),
        rows=str(len(rows)),
        last=last,
        candidate_count_mean=last.get("ausa/candidate_count_mean", "NA"),
        candidate_count_max=last.get("ausa/candidate_count_max", "NA"),
        router_entropy_norm=last.get("ausa/router_entropy_norm", "NA"),
        router_top1_weight=last.get("ausa/router_top1_weight", "NA"),
        landmark_coverage=landmark_coverage,
        landmark_count=landmark_count,
        provenance="new A7 backend-family set empty_only run",
    )


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
    rows: list[dict[str, str]] = []
    dense_rows: list[dict[str, str]] = []
    token_rows: list[dict[str, str]] = []
    set_rows: list[dict[str, str]] = []
    try:
        dense_rows = load_dense_a7_rows()
    except Exception as exc:  # noqa: BLE001
        failures.append(str(exc))
    try:
        token_rows = load_token_baselines()
    except Exception as exc:  # noqa: BLE001
        failures.append(str(exc))
    for spec in expected_set_runs():
        try:
            set_rows.append(validate_set_run(spec))
        except Exception as exc:  # noqa: BLE001
            failures.append(str(exc))
    failures.extend(scan_logs())
    rows = dense_rows + token_rows + sorted(
        set_rows, key=lambda r: (r["family_slug"], int(r["w"]), int(r["s"]), int(r["seed"]))
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    all_path = TABLES / "a7_backend_family_empty_only_all_runs.tsv"
    summary_path = TABLES / "a7_backend_family_empty_only_summary.tsv"
    manifest_path = CHECKS / "a7_backend_family_empty_only_manifest.json"
    audit_path = AUDIT / "A7_backend_family_empty_only.md"

    fields = [
        "phase", "family_slug", "family", "implementation", "backend_family", "backend",
        "seed", "lr", "D", "d_ff", "L", "w", "s", "M", "M_over_L", "L_over_M",
        "output_residual_mode", "config", "csv_path", "json_path", "rows",
        "final_train_loss", "final_val_loss", "final_train_ppl", "final_val_ppl",
        "time_per_epoch_s", "peak_vram_mib", "candidate_count_mean",
        "candidate_count_max", "router_entropy_norm", "router_top1_weight",
        "landmark_coverage", "landmark_count", "source_csv_sha256",
        "source_json_sha256", "comparison_provenance",
    ]
    summary_fields = [
        "family_slug", "family", "implementation", "backend_family", "backend", "lr",
        "D", "d_ff", "L", "w", "s", "M", "M_over_L", "L_over_M",
        "output_residual_mode", "n", "seeds", "mean_final_val_ppl",
        "std_final_val_ppl", "min_final_val_ppl", "max_final_val_ppl",
        "mean_final_train_ppl", "mean_time_per_epoch_s", "mean_peak_vram_mib",
        "mean_candidate_count", "landmark_coverage", "landmark_count",
    ]
    write_tsv(all_path, rows, fields)
    summary_rows = summarize(rows) if rows else []
    write_tsv(summary_path, summary_rows, summary_fields)

    manifest = {
        "phase": "A7-backend-family",
        "status": "pass" if not failures else "fail",
        "expected_new_set_runs": len(expected_set_runs()),
        "validated_new_set_runs": len(set_rows),
        "reused_dense_a7_rows": len(dense_rows),
        "reused_token_backend_baseline_rows": len(token_rows),
        "failures": failures,
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "dirty_status": run(["git", "status", "--short"]),
        "artifacts": {
            "all_runs": str(all_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
            "audit": str(audit_path.relative_to(ROOT)),
        },
        "baseline_policy": (
            "Token baselines are reused as backend-specific horizontal references. "
            "They do not consume set topology, so rerunning them per (w,s) would "
            "only duplicate identical model definitions with nominal labels."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    audit_lines = [
        "# A7 Backend-Family Empty-Only Calibration",
        "",
        f"Status: {manifest['status'].upper()}",
        "",
        "## Baseline Policy",
        "",
        "Dense, sparse, and linear token baselines are used as matched backend "
        "references. Token baselines do not have a set candidate-fiber topology, "
        "so the same three-seed baseline is plotted as a horizontal reference "
        "for each backend family.",
        "",
        "## Validation",
        "",
        f"- New sparse/linear set-side runs: {len(set_rows)}/{len(expected_set_runs())}",
        f"- Reused dense A7 rows: {len(dense_rows)}",
        f"- Reused sparse/linear token baseline rows: {len(token_rows)}",
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
