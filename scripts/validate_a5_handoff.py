#!/usr/bin/env python3
"""Build the final Phase A reproducibility handoff.

This script validates the completed A1-A4 evidence manifests, checks source
CSV/JSON metadata, and writes the artifact index plus audit note.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
AUDIT = ROOT / "audit"

TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])(?:nan|NaN|-inf|inf)(?![A-Za-z0-9_])")


@dataclass(frozen=True)
class ManifestSpec:
    phase: str
    path: str
    intended_use: str


MANIFESTS = [
    ManifestSpec("A2", "out/paper_integrated_evidence/checks/a2_grid_manifest.json", "LR-normalized headline/family grids for main tables"),
    ManifestSpec("A2.4", "out/paper_integrated_evidence/checks/a2_baseline_controls_manifest.json", "v2.7 matched sparse/linear token controls for LR-normalized comparisons"),
    ManifestSpec("A3.1", "out/paper_integrated_evidence/checks/a3_window_sweep_manifest.json", "fixed-stride set-family window sweep for mechanism figure"),
    ManifestSpec("A3.1-control", "out/paper_integrated_evidence/checks/a3_window_baseline_controls_manifest.json", "matched token-control window overlays"),
    ManifestSpec("A3.2", "out/paper_integrated_evidence/checks/a3_pooltau_sweep_manifest.json", "pooling-temperature sweep with error bars"),
    ManifestSpec("A3.3", "out/paper_integrated_evidence/checks/a3_stride_sweep_manifest.json", "demoted stride-sweep complement"),
    ManifestSpec("A4.1", "out/paper_integrated_evidence/checks/a41_smoke_manifest.json", "long-context smoke gate"),
    ManifestSpec("A4.2", "out/paper_integrated_evidence/checks/a42_slice_manifest.json", "long-context family slice"),
    ManifestSpec("A4.2-control", "out/paper_integrated_evidence/checks/a4_long_context_baseline_controls_manifest.json", "matched long-context sparse/linear token controls"),
    ManifestSpec("A4.3", "out/paper_integrated_evidence/checks/a4_convergence_manifest.json", "30-epoch convergence panel"),
    ManifestSpec("A6.1", "out/paper_integrated_evidence/checks/a6_dphi_capacity_manifest.json", "d_phi set-token interface capacity ablation"),
    ManifestSpec("A6.2", "out/paper_integrated_evidence/checks/a6_set_state_width_manifest.json", "explicit set-state dimensionality capacity ablation"),
    ManifestSpec("A6.3", "out/paper_integrated_evidence/checks/a6_interface_bottleneck_manifest.json", "set-token interface bottleneck ablation"),
    ManifestSpec("A6.4", "out/paper_integrated_evidence/checks/a64_depth_sweep_manifest.json", "set-processing stack depth ablation"),
]

AUDITS = [
    "audit/A1_9_gate.json",
    "audit/A2_grid_handoff.md",
    "audit/A2_4_baseline_controls.md",
    "audit/A3_1_window_sweep.md",
    "audit/A3_1_baseline_controls.md",
    "audit/A3_2_pooltau_sweep.md",
    "audit/A3_3_stride_sweep.md",
    "audit/A4_1_smoke.md",
    "audit/A4_2_slice.md",
    "audit/A4_2_baseline_controls.md",
    "audit/A4_3_convergence.md",
    "audit/A6_1_dphi_capacity.md",
    "audit/A6_2_set_state_width.md",
    "audit/A6_3_interface_bottleneck.md",
    "audit/A6_4_depth_sweep.md",
]

OPTIONAL_EXTRA_MANIFESTS = [
    "out/paper_integrated_evidence/checks/integrable_evidence_manifest.json",
    "out/paper_integrated_evidence/checks/lrnorm_headline_manifest.json",
]


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


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def has_nonfinite_token(path: Path) -> list[str]:
    issues: list[str] = []
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for i, row in enumerate(reader, 1):
                for key, raw in row.items():
                    if isinstance(raw, str) and raw.strip().lower() in {"nan", "inf", "-inf"}:
                        issues.append(f"{path}: row {i} {key}={raw}")
    else:
        text = path.read_text(errors="replace")
        match = TOKEN_RE.search(text)
        if match:
            issues.append(f"{path}: standalone {match.group()!r}")
    return issues


def row_count(path: Path) -> int:
    if path.suffix.lower() == ".tsv":
        return len(read_csv(path, delimiter="\t"))
    if path.suffix.lower() == ".csv":
        return len(read_csv(path))
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data)
    return len(path.read_text(errors="replace").splitlines())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def manifest_generated_paths(manifest: dict) -> list[str]:
    out: list[str] = []
    for item in manifest.get("generated", []):
        if isinstance(item, dict) and item.get("path"):
            out.append(str(item["path"]))
    for key in ["all_runs_tsv", "summary_tsv"]:
        if manifest.get(key):
            out.append(str(manifest[key]))
    return out


def manifest_source_csvs(manifest: dict) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in manifest.get("source_csvs", []):
        if isinstance(item, dict) and item.get("path"):
            out.append({"path": str(item["path"]), "sha256": str(item.get("sha256", ""))})
    return out


def all_runs_source_csvs(manifest: dict) -> list[dict[str, str]]:
    all_runs = manifest.get("all_runs_tsv")
    if not all_runs:
        return []
    path = ROOT / str(all_runs)
    if not path.exists():
        return []
    rows = read_csv(path, delimiter="\t")
    out: list[dict[str, str]] = []
    for row in rows:
        csv_path = row.get("csv_path")
        if csv_path:
            out.append({
                "path": csv_path,
                "sha256": row.get("source_csv_sha256", ""),
            })
    return out


def is_na(value: object) -> bool:
    return value in {None, "", "NA", "None"}


def infer_family(meta: dict) -> str:
    impl = str(meta.get("model.implementation", "NA"))
    attn = str(meta.get("model.attention_family", "NA"))
    backend = str(meta.get("model.backend", "NA"))
    if impl == "baseline_token":
        if attn == "dense" and backend == "exact":
            return "baseline_dense_exact"
        if backend == "local_band":
            return "baseline_sparse_local_band"
        if backend == "landmark":
            return "baseline_linear_landmark"
    if impl == "set_only":
        if attn == "dense" and backend == "exact":
            return "set_dense_exact"
        if backend == "local_band":
            return "set_sparse_local_band"
        if backend == "landmark":
            return "set_linear_landmark"
    return f"{impl}_{attn}_{backend}"


def expected_m(meta: dict) -> int | None:
    if meta.get("model.implementation") != "set_only":
        return None
    seq_len = int(meta.get("data.seq_len", meta.get("model.max_seq_len")))
    w = int(meta["model.window_size"])
    s = int(meta["model.stride"])
    return ((seq_len - w) // s) + 1


def validate_metadata(json_path: Path) -> list[str]:
    issues: list[str] = []
    meta = json.loads(json_path.read_text())
    required = [
        "training.seed",
        "training.lr",
        "model.d_model",
        "model.dim_feedforward",
        "model.attention_family",
        "model.backend",
        "model.implementation",
    ]
    for key in required:
        if is_na(meta.get(key)):
            issues.append(f"{json_path}: missing {key}")
    impl = meta.get("model.implementation")
    backend = meta.get("model.backend")
    if impl == "set_only":
        set_required = [
            "model.window_size",
            "model.stride",
            "model.set_causality_mode",
            "resolved.d_phi",
            "resolved.adapter_type",
            "resolved.pooling_alpha",
            "resolved.hash_seed",
            "resolved.hash_normalize",
            "resolved.router_min_temp",
        ]
        for key in set_required:
            if is_na(meta.get(key)):
                issues.append(f"{json_path}: missing {key}")
        if meta.get("model.set_causality_mode") != "strict_past":
            issues.append(f"{json_path}: set_causality_mode={meta.get('model.set_causality_mode')}")
        m_val = expected_m(meta)
        if m_val is not None:
            seq_len = int(meta.get("data.seq_len", meta.get("model.max_seq_len")))
            if seq_len == 512 and int(meta.get("model.window_size")) == 16 and int(meta.get("model.stride")) == 8 and m_val != 63:
                issues.append(f"{json_path}: expected M=63, got {m_val}")
            if seq_len == 2048 and int(meta.get("model.window_size")) == 16 and int(meta.get("model.stride")) == 8 and m_val != 255:
                issues.append(f"{json_path}: expected M=255, got {m_val}")
    if backend == "landmark":
        for key in ["model.backend_params.landmark_coverage", "resolved.landmark_coverage", "resolved.landmark_count"]:
            if is_na(meta.get(key)):
                issues.append(f"{json_path}: missing {key}")
    return issues


def artifact_row(phase: str, path: Path, status: str, intended_use: str) -> dict[str, str]:
    return {
        "phase": phase,
        "artifact_path": rel(path),
        "rows": str(row_count(path)),
        "sha256": sha256(path),
        "status": status,
        "intended_paper_use": intended_use,
    }


def write_tsv(path: Path, rows: Iterable[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for path in [CHECKS, TABLES, AUDIT]:
        path.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    artifact_rows: list[dict[str, str]] = []
    checked_manifests: list[dict[str, str]] = []
    checked_audits: list[dict[str, str]] = []
    checked_source_csvs = 0
    checked_jsons = 0
    unique_sources: dict[str, str] = {}
    unique_generated: dict[str, tuple[str, str]] = {}

    for spec in MANIFESTS:
        path = ROOT / spec.path
        if not path.exists():
            failures.append(f"missing manifest: {spec.path}")
            continue
        manifest = json.loads(path.read_text())
        status = str(manifest.get("status", "missing"))
        if status.lower() != "pass":
            failures.append(f"{spec.path}: status={status}")
        if manifest.get("failures"):
            failures.append(f"{spec.path}: failures={manifest.get('failures')}")
        checked_manifests.append({
            "phase": spec.phase,
            "path": spec.path,
            "status": status,
            "validated_runs": str(manifest.get("validated_runs", "NA")),
            "expected_runs": str(manifest.get("expected_runs", "NA")),
        })
        artifact_rows.append(artifact_row(spec.phase, path, status, spec.intended_use))

        for gen in manifest_generated_paths(manifest):
            unique_generated[gen] = (spec.phase, spec.intended_use)
        for item in manifest_source_csvs(manifest):
            unique_sources[item["path"]] = item["sha256"]
        for item in all_runs_source_csvs(manifest):
            unique_sources[item["path"]] = item["sha256"]

    for extra in OPTIONAL_EXTRA_MANIFESTS:
        path = ROOT / extra
        if path.exists():
            manifest = json.loads(path.read_text())
            artifact_rows.append(
                artifact_row("extra", path, str(manifest.get("status", "present")), "supplemental consolidated manifest")
            )

    for path_str, (phase, use) in sorted(unique_generated.items()):
        path = ROOT / path_str
        if not path.exists():
            failures.append(f"missing generated artifact: {path_str}")
            continue
        artifact_rows.append(artifact_row(phase, path, "present", use))
        failures.extend(has_nonfinite_token(path))

    for csv_path_str, recorded_hash in sorted(unique_sources.items()):
        csv_path = ROOT / csv_path_str
        if not csv_path.exists():
            failures.append(f"missing source CSV: {csv_path_str}")
            continue
        actual = sha256(csv_path)
        if recorded_hash and actual != recorded_hash:
            failures.append(f"{csv_path_str}: sha256 mismatch {actual} != {recorded_hash}")
        failures.extend(has_nonfinite_token(csv_path))
        checked_source_csvs += 1
        json_path = csv_path.with_suffix(".json")
        if not json_path.exists():
            failures.append(f"missing source JSON: {rel(json_path)}")
        else:
            checked_jsons += 1
            failures.extend(validate_metadata(json_path))

    for audit in AUDITS:
        path = ROOT / audit
        if not path.exists():
            failures.append(f"missing audit: {audit}")
            continue
        checked_audits.append({"path": audit, "sha256": sha256(path)})
        artifact_rows.append(artifact_row("audit", path, "present", "phase audit/provenance"))

    artifact_rows = sorted(artifact_rows, key=lambda row: (row["phase"], row["artifact_path"]))
    index_path = CHECKS / "final_artifact_index.tsv"
    write_tsv(
        index_path,
        artifact_rows,
        ["phase", "artifact_path", "rows", "sha256", "status", "intended_paper_use"],
    )

    final_manifest = {
        "status": "pass" if not failures else "fail",
        "checked_manifests": checked_manifests,
        "checked_audits": checked_audits,
        "checked_source_csvs": checked_source_csvs,
        "checked_source_jsons": checked_jsons,
        "artifact_index": rel(index_path),
        "artifact_index_sha256": sha256(index_path),
        "failures": failures,
        "git": {
            "branch": run(["git", "branch", "--show-current"]),
            "head": run(["git", "rev-parse", "HEAD"]),
            "status_short_count": len(run(["git", "status", "--short"])["stdout"]),
        },
    }
    final_manifest_path = CHECKS / "final_reproducibility_manifest.json"
    final_manifest_path.write_text(json.dumps(final_manifest, indent=2) + "\n")

    lines = [
        "# A5.4 Final Reproducibility Handoff",
        "",
        "Status: PASS" if not failures else "Status: FAIL",
        "",
        "## Scope",
        "",
        "This handoff consolidates completed Phase A evidence after A1-A4, the v2.7 matched token-backend controls, A4.3 convergence, and A6 capacity/bottleneck ablations.",
        "",
        "## Validation Summary",
        "",
        f"- Required manifests checked: {len(checked_manifests)} / {len(MANIFESTS)}",
        f"- Source CSVs checked: {checked_source_csvs}",
        f"- Source JSON metadata files checked: {checked_jsons}",
        f"- Indexed artifacts: {len(artifact_rows)}",
        f"- Final artifact index: `{rel(index_path)}`",
        f"- Final reproducibility manifest: `{rel(final_manifest_path)}`",
        "",
        "## Manifest Checks",
        "",
        "| phase | manifest | status | validated | expected |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for item in checked_manifests:
        lines.append(
            f"| {item['phase']} | `{item['path']}` | {item['status']} | {item['validated_runs']} | {item['expected_runs']} |"
        )
    lines.extend(["", "## TSV / Artifact Index", ""])
    lines.append("| artifact | rows | sha256 | intended use |")
    lines.append("| --- | ---: | --- | --- |")
    for row in artifact_rows:
        if row["artifact_path"].endswith(".tsv"):
            lines.append(
                f"| `{row['artifact_path']}` | {row['rows']} | `{row['sha256']}` | {row['intended_paper_use']} |"
            )
    lines.extend(["", "## Audits Checked", ""])
    for item in checked_audits:
        lines.append(f"- `{item['path']}` sha256={item['sha256']}")
    lines.extend([
        "",
        "## Matched-Controls Coverage Statement",
        "",
        "- A2.4 supplies matched `baseline_sparse_local_band` and `baseline_linear_landmark` controls for LR-normalized headline/family comparisons.",
        "- A3.1-control supplies matched token-backend overlays for the fixed-stride window sweep.",
        "- A4.2-control supplies matched sparse/linear token controls at long context (`L=2048`).",
        "- A4.3 supplies a 30-epoch panel containing dense/sparse/linear token baselines and dense/sparse/linear SKA variants.",
        "- Historical A2/A3/A4 set-family artifacts that predate v2.7 are still valid, but any backend-family interpretation must use or cite the matched-control artifacts above.",
        "",
        "## A6 Capacity Ablation Statement",
        "",
        "- A6.1 shows that increasing `d_phi` helps some set families but does not produce a broad monotonic gain.",
        "- A6.2 shows explicit `set_state_dim` helps SetSparse at 512 but does not broadly improve SKA.",
        "- A6.3 shows moderate `d_phi` increases partially relieve interface bottlenecks, especially for SetLinear, but matched `d_phi=set_state_dim` often worsens PPL.",
        "- A6.4 rejects the set-stack-depth bottleneck under this budget: depth 8/10 worsens validation PPL versus depth 6 across all tested families and capacity pairs.",
        "",
        "## Writing-Agent Caveats",
        "",
        "- A4.3 convergence favors token baselines over SKA at the 30-epoch LR-normalized reference; do not overclaim convergence wins for SKA.",
        "- SKA memory advantage appears in the long-context slice: compare `audit/A4_2_slice.md` and `audit/A4_2_baseline_controls.md` before writing the long-context claim.",
        "- Tables/figures must distinguish dense-baseline-only historical artifacts from v2.7 matched-control artifacts.",
        "- A6 capacity ablations should be written as diagnostics, not as evidence of a simple missing-capacity fix.",
        "- App D.2 should be dropped or rebuilt from the canonical LR-normalized baseline per `audit/A1_3_reconciliation.md`.",
        "",
        "## Failures",
        "",
    ])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- None.")
    lines.append("")
    handoff_path = AUDIT / "A5_4_handoff.md"
    handoff_path.write_text("\n".join(lines))

    print(json.dumps({
        "status": final_manifest["status"],
        "manifests": len(checked_manifests),
        "source_csvs": checked_source_csvs,
        "source_jsons": checked_jsons,
        "indexed_artifacts": len(artifact_rows),
        "artifact_index": rel(index_path),
        "handoff": rel(handoff_path),
        "failures": failures[:20],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
