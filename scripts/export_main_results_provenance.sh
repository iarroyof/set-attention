#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/set-attention}"
OUTDIR="${2:-$ROOT/out/provenance_exports}"
OUTFILE="$OUTDIR/main_results_provenance.txt"

mkdir -p "$OUTDIR"

python3 - <<'PY' "$ROOT" "$OUTFILE"
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
OUTFILE = Path(sys.argv[2])

def read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))

def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def read_json(path):
    with open(path) as fh:
        return json.load(fh)

def get_cfg(obj, key):
    if key in obj:
        return obj[key]
    cfg = obj.get("config", {})
    return cfg.get(key)

def last_row(csv_path):
    rows = read_csv(csv_path)
    return rows[-1] if rows else {}

def section(title):
    return f"\n{'='*100}\n{title}\n{'='*100}\n"

lines = []

# --------------------------------------------------------------------------------------
# A. What we currently consider the most important main-paper tables and figures
# --------------------------------------------------------------------------------------
lines.append(section("A. CURRENT MOST IMPORTANT MAIN-PAPER TABLES AND FIGURES"))

lines.append(
"""MAIN TABLES TO KEEP IN / AROUND THE MAIN PAPER
1. tab:matched-baseline-vs-setonly
   Purpose: strict matched quality / efficiency comparison
   Source family:
   - out/metrics/action1A_LR1e-4_baseline_seed0.{csv,json}
   - out/metrics/action1A_LR1e-4_set_dense_M64_seed0.{csv,json}
   - out/metrics/action1A_LR1e-4_set_sparse_M64_seed0.{csv,json}
   - out/metrics/action1A_LR1e-4_set_linear_M64_seed0.{csv,json}

2. Tier C topology/transition result
   Preferred main figure(s):
   - out/paper_bundle/fig_phase_candidates_vs_ppl.png
   - out/paper_bundle/fig_phase_candidates_vs_entropy.png
   - out/paper_bundle/fig_phase_candidates_vs_top1.png
   Primary summary TSV:
   - out/metrics/paper_action0_topology_temp.tsv
   - out/metrics/paper_action1_boundary_stride.tsv

3. Tier C pooling / transport result
   Preferred main figure(s):
   - out/paper_bundle/fig_pooltau_vs_ppl.png
   - out/paper_bundle/fig_pooltau_vs_neff.png
   - out/paper_bundle/fig_pooltau_vs_rho_pa.png
   Primary summary TSV:
   - out/metrics/paper_action1_pooltau_sweep.tsv

4. Time-quality comparison figures
   - out/paper_bundle/fig_time_vs_ppl.png
   - out/paper_bundle/actionD3_curves.png
   - out/paper_bundle/actionD4_ff_sensitivity.png
   Primary summary TSV:
   - out/metrics/actionD_epoch10_summary.tsv

SUPPORT / APPENDIX TABLES
- out/metrics/paper_action0_anchor_seed_sweep.tsv
- out/metrics/paper_action1A_parity_LR1e-4_epoch10.tsv
- out/paper_support_sparse_linear_diag/paper_sparse_linear_with_diagnostics.tsv
- out/paper_support_sparse_linear_diag/paper_sparse_linear_with_diagnostics_compact.tsv
"""
)

# --------------------------------------------------------------------------------------
# B. Strict matched table provenance
# --------------------------------------------------------------------------------------
lines.append(section("B. STRICT MATCHED TABLE PROVENANCE (tab:matched-baseline-vs-setonly)"))

strict_runs = [
    "action1A_LR1e-4_baseline_seed0",
    "action1A_LR1e-4_set_dense_M64_seed0",
    "action1A_LR1e-4_set_sparse_M64_seed0",
    "action1A_LR1e-4_set_linear_M64_seed0",
]

strict_keys = [
    "task",
    "data.dataset",
    "data.seq_len",
    "data.batch_size",
    "training.seed",
    "training.lr",
    "training.epochs",
    "training.weight_decay",
    "training.warmup_steps",
    "training.precision",
    "training.grad_accum_steps",
    "training.clip_grad_norm",
    "model.implementation",
    "model.attention_family",
    "model.backend",
    "model.feature_mode",
    "model.router_type",
    "model.router_topk",
    "model.pooling.mode",
    "model.pooling.tau",
    "model.router_multihead",
    "model.pooling_multihead",
    "model.d_model",
    "model.num_layers",
    "model.num_heads",
    "model.dim_feedforward",
    "model.dropout",
    "model.attn_dropout",
    "model.resid_dropout",
    "model.ffn_dropout",
    "model.window_size",
    "model.stride",
    "model.vocab_size",
    "model.architecture",
    "model.causal",
    "model.backend_params.radius",
    "model.backend_params.k_s",
    "model.backend_params.landmark_coverage",
    "model.backend_params.k",
]

strict_metric_keys = [
    "epoch",
    "val/loss",
    "val/ppl",
    "train/loss",
    "train/ppl",
    "train/time_per_epoch_s",
    "train/peak_vram_mib",
    "ausa/routing_entropy_norm",
    "ausa/router_top1_weight",
    "ausa/pooling_neff_ratio",
    "ausa/grad_ratio_pool_rho_p",
    "ausa/grad_ratio_set_stack_rho_a",
    "ausa/grad_ratio_total_rho_pa",
]

for base in strict_runs:
    csv_path = ROOT / "out" / "metrics" / f"{base}.csv"
    json_path = ROOT / "out" / "metrics" / f"{base}.json"
    obj = read_json(json_path)
    last = last_row(csv_path)

    lines.append(f"\nRUN: {base}")
    lines.append(f"  source_csv  = {csv_path.relative_to(ROOT)}")
    lines.append(f"  source_json = {json_path.relative_to(ROOT)}")
    lines.append("  CONFIG:")
    for k in strict_keys:
        lines.append(f"    {k} = {get_cfg(obj, k)}")
    lines.append("  FINAL METRICS:")
    for k in strict_metric_keys:
        lines.append(f"    {k} = {last.get(k, '')}")

# --------------------------------------------------------------------------------------
# C. Summary TSV provenance for Tier C tables
# --------------------------------------------------------------------------------------
lines.append(section("C. TIER C SUMMARY TSV PROVENANCE"))

summary_files = [
    "paper_action0_anchor_seed_sweep.tsv",
    "paper_action0_topology_temp.tsv",
    "paper_action1_boundary_stride.tsv",
    "paper_action1_pooltau_sweep.tsv",
    "paper_action1A_parity_LR1e-4_epoch10.tsv",
    "actionD_epoch10_summary.tsv",
]

for name in summary_files:
    path = ROOT / "out" / "metrics" / name
    rows = read_tsv(path)
    lines.append(f"\nFILE: {path.relative_to(ROOT)}")
    lines.append(f"  rows = {len(rows)}")
    lines.append(f"  columns = {list(rows[0].keys()) if rows else []}")
    if rows:
        lines.append("  first_row =")
        for k, v in rows[0].items():
            lines.append(f"    {k} = {v}")
        if len(rows) > 1:
            lines.append("  last_row =")
            for k, v in rows[-1].items():
                lines.append(f"    {k} = {v}")

# --------------------------------------------------------------------------------------
# D. Sparse/linear support provenance with diagnostics
# --------------------------------------------------------------------------------------
lines.append(section("D. SPARSE / LINEAR SUPPORT PROVENANCE WITH DIAGNOSTICS"))

support_diag = ROOT / "out" / "paper_support_sparse_linear_diag" / "paper_sparse_linear_with_diagnostics.tsv"
support_compact = ROOT / "out" / "paper_support_sparse_linear_diag" / "paper_sparse_linear_with_diagnostics_compact.tsv"
support_manifest = ROOT / "out" / "paper_support_sparse_linear_diag" / "paper_sparse_linear_with_diagnostics_manifest.json"

if support_diag.exists():
    rows = read_tsv(support_diag)
    lines.append(f"FILE: {support_diag.relative_to(ROOT)}")
    lines.append(f"  rows = {len(rows)}")
    lines.append(f"  columns = {list(rows[0].keys()) if rows else []}")
    for r in rows:
        lines.append(f"\n  run_name_short = {r.get('run_name_short','')}")
        for k in [
            "group_tag",
            "source_csv",
            "source_json",
            "training.lr",
            "model.attention_family",
            "model.backend",
            "val/ppl",
            "train/time_per_epoch_s",
            "train/peak_vram_mib",
            "ausa/routing_entropy_norm",
            "ausa/router_top1_weight",
            "ausa/pooling_neff_ratio",
            "ausa/grad_ratio_pool_rho_p",
            "ausa/grad_ratio_set_stack_rho_a",
            "ausa/grad_ratio_total_rho_pa",
            "diagnostics_present",
            "evidence_note",
        ]:
            lines.append(f"    {k} = {r.get(k,'')}")
else:
    lines.append(f"Missing: {support_diag.relative_to(ROOT)}")

if support_compact.exists():
    rows = read_tsv(support_compact)
    lines.append(f"\nFILE: {support_compact.relative_to(ROOT)}")
    lines.append(f"  rows = {len(rows)}")
    lines.append(f"  columns = {list(rows[0].keys()) if rows else []}")

if support_manifest.exists():
    lines.append(f"\nFILE: {support_manifest.relative_to(ROOT)}")
    lines.append(support_manifest.read_text())

# --------------------------------------------------------------------------------------
# E. Figure-to-source mapping for the current most important figures
# --------------------------------------------------------------------------------------
lines.append(section("E. FIGURE-TO-SOURCE MAPPING"))

figure_map = [
    ("out/paper_bundle/fig_time_vs_ppl.png", "out/metrics/actionD_epoch10_summary.tsv"),
    ("out/paper_bundle/actionD3_curves.png", "out/metrics/actionD_epoch10_summary.tsv"),
    ("out/paper_bundle/actionD4_ff_sensitivity.png", "out/metrics/actionD_epoch10_summary.tsv"),
    ("out/paper_bundle/fig_phase_candidates_vs_ppl.png", "out/metrics/paper_action0_topology_temp.tsv and/or out/metrics/paper_action1_boundary_stride.tsv"),
    ("out/paper_bundle/fig_phase_candidates_vs_entropy.png", "out/metrics/paper_action0_topology_temp.tsv and/or out/metrics/paper_action1_boundary_stride.tsv"),
    ("out/paper_bundle/fig_phase_candidates_vs_top1.png", "out/metrics/paper_action0_topology_temp.tsv and/or out/metrics/paper_action1_boundary_stride.tsv"),
    ("out/paper_bundle/fig_pooltau_vs_ppl.png", "out/metrics/paper_action1_pooltau_sweep.tsv"),
    ("out/paper_bundle/fig_pooltau_vs_neff.png", "out/metrics/paper_action1_pooltau_sweep.tsv"),
    ("out/paper_bundle/fig_pooltau_vs_rho_pa.png", "out/metrics/paper_action1_pooltau_sweep.tsv"),
]

for fig, src in figure_map:
    fig_path = ROOT / fig
    lines.append(f"\nFIGURE: {fig}")
    lines.append(f"  exists = {fig_path.exists()}")
    lines.append(f"  primary_source = {src}")

# --------------------------------------------------------------------------------------
# F. Minimal paper-ready statements
# --------------------------------------------------------------------------------------
lines.append(section("F. MINIMAL PAPER-READY PROVENANCE STATEMENTS"))

lines.append(
"""STRICT MATCHED TABLE (deterministically verified from current chat)
All rows use:
- task=lm
- dataset=wikitext2
- seq_len=512
- batch_size=16
- seed=0
- lr=1e-4
- epochs=10
- warmup_steps=1000
- d_model=384
- num_layers=6
- num_heads=8
- dim_feedforward=1536
- dropout=attn_dropout=resid_dropout=ffn_dropout=0.1
- vocab_size=76618
- architecture=transformer_lm
- causal=True

Set-only rows additionally use:
- feature_mode=hashed_counts
- router_type=learned
- router_topk=16
- pooling.mode=soft_trimmed_boltzmann
- pooling.tau=0.1
- router_multihead=True
- pooling_multihead=False
- window_size=16
- stride=8

Backend-specific parameters:
- sparse/local_band: radius=4
- linear/landmark: landmark_coverage=0.25
- dense/exact: no backend-specific radius/landmark parameter
"""
)

OUTFILE.write_text("\n".join(lines))
print(f"Wrote {OUTFILE}")
PY

echo
echo "Done."
echo "Output file: $OUTFILE"
