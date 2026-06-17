#!/usr/bin/env python3
"""Build a consolidated report for post-A1 causal LM experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "out" / "paper_integrated_evidence" / "tables"
CHECKS = ROOT / "out" / "paper_integrated_evidence" / "checks"
AUDIT = ROOT / "audit"


@dataclass(frozen=True)
class ArtifactSpec:
    phase: str
    manifest: str
    table: str
    intended_use: str
    caveat: str


SPECS = [
    ArtifactSpec(
        "A2",
        "out/paper_integrated_evidence/checks/a2_grid_manifest.json",
        "out/paper_integrated_evidence/tables/a2_grid_all_runs.tsv",
        "LR-normalized dense baseline and set-family provenance; dense-baseline-only historical context.",
        "Use matched v2.7 controls for reviewer-facing backend-family claims.",
    ),
    ArtifactSpec(
        "A2.4",
        "out/paper_integrated_evidence/checks/a2_baseline_controls_manifest.json",
        "out/paper_integrated_evidence/tables/a2_baseline_controls_summary.tsv",
        "Matched sparse and linear token-backend controls at the LR-normalized headline point.",
        "Controls are at D=384,d_ff=1536,L=512,w=16,s=8 only.",
    ),
    ArtifactSpec(
        "A3.1",
        "out/paper_integrated_evidence/checks/a3_window_sweep_manifest.json",
        "out/paper_integrated_evidence/tables/a3_window_sweep_summary.tsv",
        "Fixed-stride window-size mechanism sweep for candidate-count effects.",
        "Set-family mechanism evidence; not a headline token-baseline comparison.",
    ),
    ArtifactSpec(
        "A3.1-control",
        "out/paper_integrated_evidence/checks/a3_window_baseline_controls_manifest.json",
        "out/paper_integrated_evidence/tables/a3_window_baseline_controls_summary.tsv",
        "Token sparse/linear overlays for the A3.1 window sweep.",
        "Use to distinguish set mechanism from backend attribution.",
    ),
    ArtifactSpec(
        "A3.2",
        "out/paper_integrated_evidence/checks/a3_pooltau_sweep_manifest.json",
        "out/paper_integrated_evidence/tables/a3_pooltau_sweep_summary.tsv",
        "Pooling-temperature sweep with error bars across set families.",
        "Mechanism evidence for pooling support and transport.",
    ),
    ArtifactSpec(
        "A3.3",
        "out/paper_integrated_evidence/checks/a3_stride_sweep_manifest.json",
        "out/paper_integrated_evidence/tables/a3_stride_sweep_summary.tsv",
        "Stride/candidate-count complement sweep.",
        "Demoted complement; useful for topology diagnostics.",
    ),
    ArtifactSpec(
        "A4.1",
        "out/paper_integrated_evidence/checks/a41_smoke_manifest.json",
        "out/paper_integrated_evidence/tables/a41_smoke_all_runs.tsv",
        "Long-context smoke/proof of feasible batch policy.",
        "Not a final comparison table.",
    ),
    ArtifactSpec(
        "A4.2",
        "out/paper_integrated_evidence/checks/a42_slice_manifest.json",
        "out/paper_integrated_evidence/tables/a42_slice_all_runs.tsv",
        "Long-context set-family quality/memory slice.",
        "Pair with A4.2 controls for matched backend claims.",
    ),
    ArtifactSpec(
        "A4.2-control",
        "out/paper_integrated_evidence/checks/a4_long_context_baseline_controls_manifest.json",
        "out/paper_integrated_evidence/tables/a4_long_context_baseline_controls_summary.tsv",
        "Long-context matched sparse/linear token controls.",
        "Dense token baseline remains memory-heavy but strongest in PPL.",
    ),
    ArtifactSpec(
        "A4.3",
        "out/paper_integrated_evidence/checks/a4_convergence_manifest.json",
        "out/paper_integrated_evidence/tables/a4_convergence_summary.tsv",
        "Thirty-epoch convergence panel.",
        "Convergence favors token baselines under tested settings.",
    ),
    ArtifactSpec(
        "A6.1",
        "out/paper_integrated_evidence/checks/a6_dphi_capacity_manifest.json",
        "out/paper_integrated_evidence/tables/a6_dphi_capacity_summary.tsv",
        "d_phi set-token interface capacity ablation.",
        "Moderate d_phi gains are family-specific, not a complete bottleneck fix.",
    ),
    ArtifactSpec(
        "A6.2",
        "out/paper_integrated_evidence/checks/a6_set_state_width_manifest.json",
        "out/paper_integrated_evidence/tables/a6_set_state_width_summary.tsv",
        "Explicit set-state dimensionality sweep at fixed token width.",
        "Wider set state does not reliably close the PPL gap.",
    ),
    ArtifactSpec(
        "A6.3",
        "out/paper_integrated_evidence/checks/a6_interface_bottleneck_manifest.json",
        "out/paper_integrated_evidence/tables/a6_interface_bottleneck_summary.tsv",
        "Joint set-state and d_phi interface bottleneck sweep.",
        "Matched d_phi=set_state_dim often worsens validation PPL.",
    ),
    ArtifactSpec(
        "A6.4",
        "out/paper_integrated_evidence/checks/a64_depth_sweep_manifest.json",
        "out/paper_integrated_evidence/tables/a64_depth_sweep_summary.tsv",
        "Set-stack depth bottleneck test.",
        "Depth 8/10 worsens validation PPL versus depth 6.",
    ),
    ArtifactSpec(
        "A7",
        "out/paper_integrated_evidence/checks/a7_empty_only_calibration_manifest.json",
        "out/paper_integrated_evidence/tables/a7_empty_only_calibration_summary.tsv",
        "Calibrated empty_only token-limit and compression path.",
        "Empirical convergence as M/L->1, not exact Transformer equivalence.",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as fh:
        return max(sum(1 for _ in csv.reader(fh, delimiter="\t")) - 1, 0)


def manifest_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "invalid-json"
    value = data.get("status")
    if value is None:
        value = "pass" if data.get("overall_pass") is True else "unknown"
    return str(value)


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in SPECS:
        manifest = ROOT / spec.manifest
        table = ROOT / spec.table
        rows.append({
            "phase": spec.phase,
            "manifest": spec.manifest,
            "manifest_status": manifest_status(manifest),
            "manifest_sha256": sha256(manifest) if manifest.exists() else "MISSING",
            "table": spec.table,
            "table_rows": str(row_count(table)),
            "table_sha256": sha256(table) if table.exists() else "MISSING",
            "intended_paper_use": spec.intended_use,
            "caveat": spec.caveat,
        })
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "phase",
        "manifest",
        "manifest_status",
        "manifest_sha256",
        "table",
        "table_rows",
        "table_sha256",
        "intended_paper_use",
        "caveat",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    missing = [r for r in rows if r["manifest_status"] not in {"pass", "true", "True"} or r["table_sha256"] == "MISSING"]
    lines = [
        "# Stabilized Post-A1 Causal LM Experiment Report",
        "",
        f"Overall status: {'PASS' if not missing else 'CHECK'}",
        "",
        "This report consolidates the post-A1 causal LM experiment record so the work is not lost even if the current SKA implementation is not presented as a perplexity improvement.",
        "",
        "## Reporting Principles",
        "",
        "- Use strict-past, T1 dropped trailing windows, and explicit output residual policy labels.",
        "- Distinguish dense-baseline-only historical artifacts from v2.7 matched token-backend controls.",
        "- Treat A6 capacity sweeps as diagnostics, not as evidence of a simple missing-capacity fix.",
        "- Treat A7 as empirical convergence under the calibrated set pipeline, not exact Transformer equivalence.",
        "- Preserve source CSV/JSON hashes and manifest status for every reported table or figure.",
        "- Exclude pre-A1, noncausal, and causality-unverified artifacts from reviewer-facing causal LM claims unless they are rebuilt and revalidated under the post-A1 causal LM protocol.",
        "",
        "## Artifact Index",
        "",
        "| Phase | Status | Rows | Intended paper use | Caveat |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['manifest_status']} | {row['table_rows']} | "
            f"{row['intended_paper_use']} | {row['caveat']} |"
        )
    lines += [
        "",
        "## Writing Caveats",
        "",
        "- The current causal SKA implementation is strongest as a diagnostic framework for compression, routing support, pooling support, and memory tradeoffs.",
        "- Matched token baselines remain stronger in perplexity at the tested operating points.",
        "- The A7 singleton limit shows SetDense `empty_only` approaching the dense token baseline but still trailing it.",
        "- Long-context results support a memory advantage, not a quality advantage.",
        "- Historical unreconciled appendix slices should be dropped or rebuilt from canonical LR-normalized artifacts.",
    ]
    if missing:
        lines += ["", "## Items Requiring Attention", ""]
        for row in missing:
            lines.append(
                f"- {row['phase']}: manifest_status={row['manifest_status']}, table={row['table']}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = build_rows()
    index_path = CHECKS / "causal_experiment_report_index.tsv"
    report_path = AUDIT / "causal_experiment_report.md"
    write_tsv(index_path, rows)
    write_markdown(report_path, rows)
    print(index_path)
    print(report_path)


if __name__ == "__main__":
    main()
