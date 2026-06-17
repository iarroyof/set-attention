#!/usr/bin/env python3
"""Validate and summarize the A6.4 set-stack depth sweep.

Design:
  - 3 families x 2 capacity settings x 3 depths {6,8,10} x 3 seeds = 54 rows
  - Depth=6 rows are reused from the A6.3 TSV (a6_interface_bottleneck_all_runs.tsv)
    filtered to the two A6.4 capacity pairs: (384,384) and (768,512)
  - Depths 8 and 10 are new runs (36 new rows)

Hypothesis: num_layers (set-stack depth) is a bottleneck for SKA families,
especially those with restricted intra-set communication (sparse, linear backends).
Each SetAttentionBlock applies: Z^{l+1} = FFN(Attn_backend(Z^l) + Z^l).
Wider set_state_dim cannot substitute for depth when the backend restricts
which set states can communicate per layer.
"""

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
RAW = ROOT / "out" / "paper_mechanisms" / "a64_depth_sweep"
OUT = ROOT / "out" / "paper_integrated_evidence"
TABLES = OUT / "tables"
CHECKS = OUT / "checks"
AUDIT = ROOT / "audit"
LOG_ROOT = ROOT / "logs" / "a64_depth_sweep"

SEQ_LEN = 512
WINDOW = 16
STRIDE = 8
M = (SEQ_LEN - WINDOW) // STRIDE + 1
EPOCHS = 10
LR = "1e-4"
D_MODEL = 384
D_FF = 1536
# Capacity pairs: (set_state_dim, d_phi)
CAPACITY_PAIRS = [(384, 384), (768, 512)]
# Only NEW depths — depth=6 is reused from A6.3
NEW_DEPTHS = [8, 10]
REFERENCE_DEPTH = 6
SEEDS = [0, 1, 2]


@dataclass(frozen=True)
class Family:
    slug: str
    family: str
    backend: str
    config: str


FAMILIES = [
    Family("set_dense_exact", "Set Dense", "exact", "configs/paper_lr_norm/set_dense_exact.yaml"),
    Family("set_sparse_local_band", "Set Sparse", "local_band", "configs/paper_lr_norm/set_sparse_local_band.yaml"),
    Family("set_linear_landmark", "Set Linear", "landmark", "configs/paper_lr_norm/set_linear_landmark.yaml"),
]


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


def reference_rows() -> list[dict[str, str]]:
    """Read depth=6 rows from the A6.3 TSV for the two A6.4 capacity pairs."""
    path = TABLES / "a6_interface_bottleneck_all_runs.tsv"
    rows = read_delimited(path)
    wanted_slugs = {f.slug for f in FAMILIES}
    # The two A6.4 capacity pairs as (set_state_dim, d_phi) string tuples
    wanted_pairs = {(str(ssd), str(dphi)) for ssd, dphi in CAPACITY_PAIRS}
    refs: list[dict[str, str]] = []
    for row in rows:
        slug = row.get("family_slug", "")
        if slug not in wanted_slugs:
            continue
        pair = (row.get("set_state_dim", ""), row.get("d_phi", ""))
        if pair not in wanted_pairs:
            continue
        if row.get("seed") not in {"0", "1", "2"}:
            continue
        family = family_by_slug(slug)
        refs.append({
            "phase": "A6.4-reference",
            "source": "reused-A6.3",
            "family_slug": slug,
            "family": family.family,
            "backend": family.backend,
            "seed": row["seed"],
            "lr": LR,
            "D": str(D_MODEL),
            "d_ff": str(D_FF),
            "L": str(SEQ_LEN),
            "w": str(WINDOW),
            "s": str(STRIDE),
            "M": str(M),
            "d_phi": row["d_phi"],
            "set_state_dim": row["set_state_dim"],
            "num_layers": str(REFERENCE_DEPTH),
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
            "resolved_d_phi": row.get("resolved_d_phi", row["d_phi"]),
            "resolved_set_state_dim": row.get("resolved_set_state_dim", row["set_state_dim"]),
            "resolved_adapter_type": row.get("resolved_adapter_type", "NA"),
            "landmark_coverage": row.get("landmark_coverage", "NA"),
            "landmark_count": row.get("landmark_count", "NA"),
            "source_csv_sha256": row.get("source_csv_sha256", "NA"),
        })
    return refs


def new_run_path(family: Family, set_state_dim: int, d_phi: int, num_layers: int, seed: int) -> tuple[Path, Path]:
    lr_tag = LR.replace(".", "p")
    group = f"a64_depth_sweep_{family.slug}_D{D_MODEL}_FF{D_FF}"
    name = (
        f"a64_depth_{family.slug}_D{D_MODEL}_FF{D_FF}_setdim{set_state_dim}_"
        f"dphi{d_phi}_nl{num_layers}_w{WINDOW}_s{STRIDE}_lr{lr_tag}_seed{seed}"
    )
    csv_path = RAW / group / f"{name}.csv"
    return csv_path, csv_path.with_suffix(".json")


def landmark_count_str(family: Family) -> str:
    if family.backend != "landmark":
        return "NA"
    return str(max(round(0.25 * M), 2))


def validate_new_run(
    family: Family, set_state_dim: int, d_phi: int, num_layers: int, seed: int
) -> dict[str, str]:
    csv_path, json_path = new_run_path(family, set_state_dim, d_phi, num_layers, seed)
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
        "model.d_phi": d_phi,
        "model.set_state_dim": set_state_dim,
        "model.num_layers": num_layers,
        "resolved.d_phi": d_phi,
        "resolved.set_state_dim": set_state_dim,
    }
    for key, expected in checks.items():
        actual = meta.get(key)
        if str(actual) != str(expected):
            raise ValueError(f"{json_path} has {key}={actual!r}, expected {expected!r}")
    if family.backend == "landmark":
        if str(meta.get("resolved.landmark_coverage")) != "0.25":
            raise ValueError(f"{json_path} missing resolved.landmark_coverage=0.25")
        if str(meta.get("resolved.landmark_count")) != landmark_count_str(family):
            raise ValueError(
                f"{json_path} landmark_count={meta.get('resolved.landmark_count')}, "
                f"expected {landmark_count_str(family)}"
            )

    last = rows[-1]
    for key in ["train/loss", "val/loss", "train/ppl", "val/ppl"]:
        if float_or_none(last.get(key)) is None:
            raise ValueError(f"{csv_path} missing finite final {key}")

    return {
        "phase": "A6.4-new",
        "source": "new",
        "family_slug": family.slug,
        "family": family.family,
        "backend": family.backend,
        "seed": str(seed),
        "lr": LR,
        "D": str(D_MODEL),
        "d_ff": str(D_FF),
        "L": str(SEQ_LEN),
        "w": str(WINDOW),
        "s": str(STRIDE),
        "M": str(M),
        "d_phi": str(d_phi),
        "set_state_dim": str(set_state_dim),
        "num_layers": str(num_layers),
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
    for family in FAMILIES:
        for set_state_dim, d_phi in CAPACITY_PAIRS:
            for num_layers in NEW_DEPTHS:
                for seed in SEEDS:
                    try:
                        rows.append(validate_new_run(family, set_state_dim, d_phi, num_layers, seed))
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
    """Group by (family_slug, set_state_dim, d_phi, num_layers) and compute mean PPL.

    Delta is vs the depth=6 reference for the same (family, capacity) pair.
    """
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family_slug"], row["set_state_dim"], row["d_phi"], row["num_layers"])].append(row)

    # Compute depth=6 reference mean per (family_slug, set_state_dim, d_phi)
    ref_mean: dict[tuple[str, str, str], float] = {}
    for (slug, ssd, dphi, nl), group_rows in grouped.items():
        if nl == str(REFERENCE_DEPTH):
            ref_mean[(slug, ssd, dphi)] = mean(
                float(r["final_val_ppl"]) for r in group_rows
            )

    summary: list[dict[str, str]] = []
    for (slug, ssd, dphi, nl), group_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], int(item[0][1]), int(item[0][2]), int(item[0][3])),
    ):
        ppls = [float(r["final_val_ppl"]) for r in group_rows]
        train_ppls = [float(r["final_train_ppl"]) for r in group_rows]
        times = [float_or_none(r["time_per_epoch_s"]) for r in group_rows]
        vrams = [float_or_none(r["peak_vram_mib"]) for r in group_rows]
        times_f = [v for v in times if v is not None]
        vrams_f = [v for v in vrams if v is not None]
        mean_ppl = mean(ppls)
        ref = ref_mean.get((slug, ssd, dphi))
        delta = mean_ppl - ref if ref is not None else 0.0
        rel = (delta / ref * 100.0) if ref else 0.0
        family = family_by_slug(slug)
        summary.append({
            "phase": "A6.4",
            "family_slug": slug,
            "family": family.family,
            "backend": family.backend,
            "D": str(D_MODEL),
            "d_ff": str(D_FF),
            "set_state_dim": ssd,
            "d_phi": dphi,
            "num_layers": nl,
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
            "delta_val_ppl_vs_depth6": f"{delta:.6f}",
            "pct_delta_val_ppl_vs_depth6": f"{rel:.6f}",
        })
    return summary


def write_audit(manifest: dict[str, object], summary_rows: list[dict[str, str]]) -> None:
    lines = [
        "# A6.4 Set-Stack Depth Sweep",
        "",
        f"Status: {manifest['status'].upper()}",
        "",
        "## Hypothesis",
        "",
        "`num_layers` (set-processing stack depth) may be a bottleneck for SKA families, "
        "particularly those with restricted intra-set communication (sparse local-band, "
        "linear landmark backends). Each SetAttentionBlock applies "
        "Z^{l+1} = FFN(Attn_backend(Z^l, bias) + Z^l). With sparse/linear backends, "
        "only a subset of set states can communicate per layer, so information that "
        "requires multi-hop propagation through set space requires more layers. "
        "Wider set_state_dim (capacity) cannot substitute for depth (reach).",
        "",
        "## Implementation Math",
        "",
        "SetOnlyLM builds `num_layers` SetAttentionBlocks: "
        "`self.blocks = nn.ModuleList([SetAttentionBlock(...) for _ in range(num_layers)])`. "
        "Each block: Z^{l+1} = FFN(Attn_backend(Z^l) + Z^l) where Z^l ∈ R^{M × set_state_dim}. "
        "The router reads routed context (still set_state_dim-wide) and projects back to d_model. "
        "This sweep fixes (set_state_dim, d_phi) at the two A6.4 capacity pairs "
        "and varies depth in {6,8,10} to test whether more layers recover the PPL gap.",
        "",
        "## Summary",
        "",
        "| family | backend | set_state_dim | d_phi | num_layers | n | mean val PPL | delta vs depth=6 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['family']} | {row['backend']} | {row['set_state_dim']} | "
            f"{row['d_phi']} | {row['num_layers']} | {row['n']} | "
            f"{row['mean_final_val_ppl']} | {row['delta_val_ppl_vs_depth6']} |"
        )
    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        "The depth bottleneck hypothesis is supported when increasing `num_layers` "
        "lowers validation PPL, especially for sparse/linear backends relative to dense. "
        "If depth gain is larger for sparse/linear, this indicates restricted per-layer "
        "communication (not just set-state width) limits representation quality.",
        "",
        "## Artifacts",
        "",
        "- All runs TSV: `out/paper_integrated_evidence/tables/a64_depth_sweep_all_runs.tsv`",
        "- Summary TSV: `out/paper_integrated_evidence/tables/a64_depth_sweep_summary.tsv`",
        "- Manifest: `out/paper_integrated_evidence/checks/a64_depth_sweep_manifest.json`",
        "",
        "## Validation",
        "",
        f"- Total expected rows: {manifest['expected_runs']}",
        f"- Total validated rows: {manifest['validated_runs']}",
        f"- New expected runs: {manifest['expected_new_runs']}",
        f"- New validated runs: {manifest['validated_new_runs']}",
        f"- Reused rows (depth=6 from A6.3): {manifest['reused_rows']}",
        f"- Log failures: {len(manifest.get('log_failures', []))}",
        f"- Failures: {len(manifest.get('failures', []))}",
    ])
    if manifest.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in manifest["failures"])
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "A6_4_depth_sweep.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    refs = reference_rows()
    new, new_failures = validate_new_rows()
    failures.extend(new_failures)
    log_failures = scan_logs()
    failures.extend(log_failures)

    expected_reused = len(FAMILIES) * len(CAPACITY_PAIRS) * len(SEEDS)   # 3×2×3 = 18
    expected_new = len(FAMILIES) * len(CAPACITY_PAIRS) * len(NEW_DEPTHS) * len(SEEDS)  # 3×2×2×3 = 36
    expected_total = expected_reused + expected_new  # 54

    if len(refs) != expected_reused:
        failures.append(f"reused row count {len(refs)} != expected {expected_reused}")

    rows = refs + new

    all_runs_path = TABLES / "a64_depth_sweep_all_runs.tsv"
    summary_path = TABLES / "a64_depth_sweep_summary.tsv"
    manifest_path = CHECKS / "a64_depth_sweep_manifest.json"

    all_fields = [
        "phase", "source", "family_slug", "family", "backend", "seed", "lr",
        "D", "d_ff", "L", "w", "s", "M", "d_phi", "set_state_dim", "num_layers",
        "config", "csv_path", "json_path", "rows", "final_train_loss", "final_val_loss",
        "final_train_ppl", "final_val_ppl", "time_per_epoch_s", "peak_vram_mib",
        "resolved_d_phi", "resolved_set_state_dim", "resolved_adapter_type",
        "landmark_coverage", "landmark_count", "source_csv_sha256",
    ]
    if rows:
        write_tsv(all_runs_path, rows, all_fields)
        summary_rows = summarize(rows)
        write_tsv(
            summary_path,
            summary_rows,
            [
                "phase", "family_slug", "family", "backend", "D", "d_ff",
                "set_state_dim", "d_phi", "num_layers", "lr", "n",
                "mean_final_val_ppl", "std_final_val_ppl",
                "mean_final_train_ppl", "std_final_train_ppl",
                "mean_peak_vram_mib", "mean_time_per_epoch_s",
                "delta_val_ppl_vs_depth6", "pct_delta_val_ppl_vs_depth6",
            ],
        )
    else:
        summary_rows = []

    manifest = {
        "status": "pass" if not failures and len(rows) == expected_total else "fail",
        "phase": "A6.4",
        "expected_runs": expected_total,
        "validated_runs": len(rows),
        "expected_new_runs": expected_new,
        "validated_new_runs": len(new),
        "reused_rows": len(refs),
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
        "reused_rows": manifest["reused_rows"],
        "failures": failures[:10],
        "summary": str(summary_path.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
    }, indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
