#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/set-attention}"
OUTDIR="${2:-$ROOT/out/final_paper_bundle}"
TMPDIR="$(mktemp -d)"

echo "ROOT=$ROOT"
echo "OUTDIR=$OUTDIR"

mkdir -p "$OUTDIR"/{tables/summary,tables/raw,plots/main,plots/appendix,metadata,instructions,checks,overleaf_ready}

# ----------------------------
# 1. Copy final summary tables
# ----------------------------
cp "$ROOT/out/metrics/paper_action0_anchor_seed_sweep.tsv"    "$OUTDIR/tables/summary/"
cp "$ROOT/out/metrics/paper_action0_topology_temp.tsv"        "$OUTDIR/tables/summary/"
cp "$ROOT/out/metrics/paper_action1_boundary_stride.tsv"      "$OUTDIR/tables/summary/"
cp "$ROOT/out/metrics/paper_action1_pooltau_sweep.tsv"        "$OUTDIR/tables/summary/"
cp "$ROOT/out/metrics/actionD_epoch10_summary.tsv"            "$OUTDIR/tables/summary/"
cp "$ROOT/out/metrics/paper_action1A_parity_LR1e-4_epoch10.tsv" "$OUTDIR/tables/summary/"

# ---------------------------------
# 2. Copy March raw provenance files
# ---------------------------------
for f in \
  action0_anchor_s4_w16_T1_lr1e4_seed0 \
  action0_anchor_s4_w16_T1_lr1e4_seed1 \
  action0_anchor_s4_w16_T1_lr1e4_seed2 \
  action0_tempSweep_s4_w16_T0p5_lr1e4_seed0 \
  action0_tempSweep_s4_w16_T1_lr1e4_seed0 \
  action0_tempSweep_s4_w16_T2_lr1e4_seed0 \
  action0_tempSweep_s4_w16_T4_lr1e4_seed0 \
  action0_learned_s6_w16_temp0.5_LR1e-4_seed0 \
  action0_learned_s6_w16_temp1.0_LR1e-4_seed0 \
  action0_learned_s6_w16_temp2.0_LR1e-4_seed0 \
  action0_learned_s6_w16_temp4.0_LR1e-4_seed0 \
  action1_boundary_s3_w16_T1_lr1e4_seed0 \
  action1_boundary_s5_w16_T1_lr1e4_seed0 \
  action1_pooltau_0.05_s4_w16_T1_lr1e4_seed0 \
  action1_pooltau_0.1_s4_w16_T1_lr1e4_seed0 \
  action1_pooltau_0.2_s4_w16_T1_lr1e4_seed0
do
  for ext in csv json; do
    src="$ROOT/out/metrics/${f}.${ext}"
    if [[ -f "$src" ]]; then
      cp "$src" "$OUTDIR/tables/raw/"
    fi
  done
done

# ------------------------------------------
# 3. Extract plots from the existing bundle
# ------------------------------------------
if [[ -f "$ROOT/out/paper_bundle.tar.gz" ]]; then
  tar -xzf "$ROOT/out/paper_bundle.tar.gz" -C "$TMPDIR"
else
  echo "ERROR: $ROOT/out/paper_bundle.tar.gz not found"
  exit 1
fi

PB="$TMPDIR/out/paper_bundle"
if [[ ! -d "$PB" ]]; then
  echo "ERROR: extracted paper_bundle directory not found"
  exit 1
fi

# Main-text plots
for f in \
  fig_time_vs_ppl.png \
  fig_entropy_vs_ppl.png \
  fig_top1_vs_ppl.png \
  fig_phase_candidates_vs_ppl.png \
  fig_phase_candidates_vs_entropy.png \
  fig_phase_candidates_vs_top1.png \
  fig_pooltau_vs_ppl.png \
  fig_pooltau_vs_neff.png \
  fig_pooltau_vs_rho_p.png \
  fig_pooltau_vs_rho_pa.png \
  fig_candidates_vs_rho_pa.png \
  fig_rho_pa_vs_ppl.png \
  actionD3_curves.png \
  actionD4_ff_sensitivity.png
do
  [[ -f "$PB/$f" ]] && cp "$PB/$f" "$OUTDIR/plots/main/"
done

# Appendix/support plots
for f in \
  fig_s4_temp_ppl.png \
  fig_s4_temp_entropy.png \
  fig_s4_temp_top1.png \
  fig_s4_temp_rho_p.png \
  fig_s4_temp_rho_pa.png \
  fig_s6_temp_entropy.png \
  fig_s6_temp_top1.png \
  fig_s16_temp_entropy.png \
  fig_s16_temp_top1.png \
  fig_temp_sweep_entropy.png \
  fig_temp_sweep_top1.png \
  fig_temp_compare_entropy_s6_vs_s16.png \
  fig_temp_compare_top1_s6_vs_s16.png
do
  [[ -f "$PB/$f" ]] && cp "$PB/$f" "$OUTDIR/plots/appendix/"
done

# ------------------------------------------
# 4. Machine-readable manifest
# ------------------------------------------
cat > "$OUTDIR/metadata/manifest.json" <<'JSON'
{
  "tier_c_main_summary_tables": [
    "tables/summary/paper_action0_anchor_seed_sweep.tsv",
    "tables/summary/paper_action0_topology_temp.tsv",
    "tables/summary/paper_action1_boundary_stride.tsv",
    "tables/summary/paper_action1_pooltau_sweep.tsv",
    "tables/summary/actionD_epoch10_summary.tsv"
  ],
  "tier_c_appendix_summary_tables": [
    "tables/summary/paper_action1A_parity_LR1e-4_epoch10.tsv"
  ],
  "main_text_plots": [
    "plots/main/fig_time_vs_ppl.png",
    "plots/main/fig_entropy_vs_ppl.png",
    "plots/main/fig_top1_vs_ppl.png",
    "plots/main/fig_phase_candidates_vs_ppl.png",
    "plots/main/fig_phase_candidates_vs_entropy.png",
    "plots/main/fig_phase_candidates_vs_top1.png",
    "plots/main/fig_pooltau_vs_ppl.png",
    "plots/main/fig_pooltau_vs_neff.png",
    "plots/main/fig_pooltau_vs_rho_p.png",
    "plots/main/fig_pooltau_vs_rho_pa.png",
    "plots/main/fig_candidates_vs_rho_pa.png",
    "plots/main/fig_rho_pa_vs_ppl.png",
    "plots/main/actionD3_curves.png",
    "plots/main/actionD4_ff_sensitivity.png"
  ],
  "appendix_plots": [
    "plots/appendix/fig_s4_temp_ppl.png",
    "plots/appendix/fig_s4_temp_entropy.png",
    "plots/appendix/fig_s4_temp_top1.png",
    "plots/appendix/fig_s4_temp_rho_p.png",
    "plots/appendix/fig_s4_temp_rho_pa.png",
    "plots/appendix/fig_s6_temp_entropy.png",
    "plots/appendix/fig_s6_temp_top1.png",
    "plots/appendix/fig_s16_temp_entropy.png",
    "plots/appendix/fig_s16_temp_top1.png",
    "plots/appendix/fig_temp_sweep_entropy.png",
    "plots/appendix/fig_temp_sweep_top1.png",
    "plots/appendix/fig_temp_compare_entropy_s6_vs_s16.png",
    "plots/appendix/fig_temp_compare_top1_s6_vs_s16.png"
  ]
}
JSON

# ------------------------------------------
# 5. Human-readable plan / instructions
# ------------------------------------------
cat > "$OUTDIR/instructions/00_README_FIRST.md" <<'MD'
# Final Paper Bundle

This bundle is organized for an OpenAI agent to write the results sections in progressive, manageable steps with reduced hallucination risk.

## Scope
Use only the post-parity March summary layer for final paper construction:

- `paper_action0_anchor_seed_sweep.tsv`
- `paper_action0_topology_temp.tsv`
- `paper_action1_boundary_stride.tsv`
- `paper_action1_pooltau_sweep.tsv`
- `actionD_epoch10_summary.tsv`

Use `paper_action1A_parity_LR1e-4_epoch10.tsv` only as appendix / parity justification.

## Tier C subsection order
1. Anchor stability at the reference operating point
2. Feasible operating region over topology and temperature
3. Boundary transition under stride / candidate-count change
4. Pooling concentration and gradient transport
5. Matched quality / efficiency comparison against baseline

## Folder layout
- `tables/summary/`: paper-facing summary TSVs
- `tables/raw/`: March provenance CSV/JSON files
- `plots/main/`: figures intended for main text
- `plots/appendix/`: figures intended for appendix
- `instructions/`: prompts, table plans, anti-hallucination rules
- `checks/`: scripts to validate produced LaTeX tables
- `overleaf_ready/`: target location for generated sections/macros
MD

cat > "$OUTDIR/instructions/01_SECTION_PLAN.md" <<'MD'
# Section-by-section plan for the agent

## Section A — Tier C setup
State that only the post-parity March summary layer is used for final Tier C reporting.

## Section B — Anchor stability
Source table:
- `tables/summary/paper_action0_anchor_seed_sweep.tsv`

Primary fields:
- `ausa/router_candidate_count_struct_mean`
- `ausa/routing_entropy_norm`
- `ausa/router_top1_weight`
- `ausa/pooling_neff_ratio`
- `ausa/grad_ratio_pool_rho_p`
- `ausa/grad_ratio_set_stack_rho_a`
- `ausa/grad_ratio_total_rho_pa`
- `val/ppl`
- `train/time_per_epoch_s`
- `train/peak_vram_mib`

## Section C — Feasible operating region
Source table:
- `tables/summary/paper_action0_topology_temp.tsv`

Primary fields:
- `model.stride`
- `model.window_size`
- `model.router_temperature`
- `ausa/router_candidate_count_struct_mean`
- `ausa/routing_entropy_norm`
- `ausa/router_top1_weight`
- `ausa/grad_ratio_pool_rho_p`
- `ausa/grad_ratio_set_stack_rho_a`
- `ausa/grad_ratio_total_rho_pa`
- `val/ppl`

Preferred figures:
- `plots/main/fig_phase_candidates_vs_ppl.png`
- `plots/main/fig_phase_candidates_vs_entropy.png`
- `plots/main/fig_phase_candidates_vs_top1.png`

## Section D — Boundary stride transition
Source table:
- `tables/summary/paper_action1_boundary_stride.tsv`

Primary fields:
- `model.stride`
- `model.window_size`
- `model.router_temperature`
- `ausa/router_candidate_count_struct_mean`
- `ausa/routing_entropy_norm`
- `ausa/router_top1_weight`
- `ausa/pooling_neff_ratio`
- `ausa/grad_ratio_pool_rho_p`
- `ausa/grad_ratio_set_stack_rho_a`
- `ausa/grad_ratio_total_rho_pa`
- `val/ppl`
- `train/time_per_epoch_s`
- `train/peak_vram_mib`

## Section E — Pooling concentration and trainability
Source table:
- `tables/summary/paper_action1_pooltau_sweep.tsv`

Primary fields:
- `model.pooling.tau`
- `ausa/pooling_neff_ratio`
- `ausa/pooling_neff_l2`
- `ausa/pooling_effective_support`
- `ausa/routing_entropy_norm`
- `ausa/router_top1_weight`
- `ausa/grad_ratio_pool_rho_p`
- `ausa/grad_ratio_set_stack_rho_a`
- `ausa/grad_ratio_total_rho_pa`
- `val/ppl`
- `train/time_per_epoch_s`
- `train/peak_vram_mib`

Preferred figures:
- `plots/main/fig_pooltau_vs_ppl.png`
- `plots/main/fig_pooltau_vs_neff.png`
- `plots/main/fig_pooltau_vs_rho_pa.png`

## Section F — Matched quality / efficiency against baseline
Source table:
- `tables/summary/actionD_epoch10_summary.tsv`

Primary fields:
- `impl`
- `model.d_model`
- `model.dim_feedforward`
- `val/loss`
- `val/ppl`
- `train/loss`
- `train/ppl`
- `train/time_per_epoch_s`
- `train/peak_vram_mib`
- `ausa/grad_ratio_pool_rho_p`
- `ausa/grad_ratio_set_stack_rho_a`
- `ausa/grad_ratio_total_rho_pa`
- `ausa/routing_entropy_norm`
- `ausa/router_top1_weight`
- `ausa/router_candidate_count_struct_mean`

Preferred figures:
- `plots/main/fig_time_vs_ppl.png`
- `plots/main/actionD3_curves.png`
- `plots/main/actionD4_ff_sensitivity.png`
MD

cat > "$OUTDIR/instructions/02_ANTI_HALLUCINATION_RULES.md" <<'MD'
# Non-negotiable table/figure rules for the agent

1. Never invent a numeric value.
2. Every LaTeX table row must be backed by an explicit row in a TSV file.
3. If a value is rounded in the LaTeX table, preserve the unrounded source value in an adjacent machine-readable audit file.
4. Do not rename columns semantically unless the original TSV column is cited in comments or audit output.
5. Do not infer missing metrics from similarly named metrics.
6. Use only:
   - `tables/summary/*.tsv` for paper-facing numbers
   - `tables/raw/*` only for provenance checks
7. Use `paper_action1A_parity_LR1e-4_epoch10.tsv` only as appendix / parity note support.
8. Prefer manageable writing steps:
   - one subsection at a time
   - then one table at a time
   - then one figure block at a time
9. Before finalizing any subsection, create an audit file that lists:
   - source TSV
   - selected rows
   - selected columns
   - exact values used
10. If a metric is not present in the source TSV, explicitly say it is unavailable in this bundle.
MD

cat > "$OUTDIR/instructions/03_TABLE_BLUEPRINTS.md" <<'MD'
# Table blueprints

## Table T1 — Anchor stability
Source: `tables/summary/paper_action0_anchor_seed_sweep.tsv`
Rows: one per seed
Columns:
- training.seed
- val/ppl
- ausa/router_candidate_count_struct_mean
- ausa/routing_entropy_norm
- ausa/router_top1_weight
- ausa/pooling_neff_ratio
- ausa/grad_ratio_pool_rho_p
- ausa/grad_ratio_set_stack_rho_a
- ausa/grad_ratio_total_rho_pa
- train/time_per_epoch_s
- train/peak_vram_mib

## Table T2 — Topology-temperature feasible region
Source: `tables/summary/paper_action0_topology_temp.tsv`
Rows: one per topology-temperature setting
Columns:
- model.window_size
- model.stride
- model.router_temperature
- ausa/router_candidate_count_struct_mean
- ausa/routing_entropy_norm
- ausa/router_top1_weight
- ausa/grad_ratio_pool_rho_p
- ausa/grad_ratio_set_stack_rho_a
- ausa/grad_ratio_total_rho_pa
- val/ppl

## Table T3 — Boundary stride summary
Source: `tables/summary/paper_action1_boundary_stride.tsv`
Rows: one per stride
Columns:
- model.window_size
- model.stride
- model.router_temperature
- ausa/router_candidate_count_struct_mean
- ausa/routing_entropy_norm
- ausa/router_top1_weight
- ausa/pooling_neff_ratio
- ausa/grad_ratio_pool_rho_p
- ausa/grad_ratio_set_stack_rho_a
- ausa/grad_ratio_total_rho_pa
- val/ppl
- train/time_per_epoch_s
- train/peak_vram_mib

## Table T4 — Pooling sweep summary
Source: `tables/summary/paper_action1_pooltau_sweep.tsv`
Rows: one per tau
Columns:
- model.pooling.tau
- ausa/pooling_neff_ratio
- ausa/pooling_neff_l2
- ausa/pooling_effective_support
- ausa/routing_entropy_norm
- ausa/router_top1_weight
- ausa/grad_ratio_pool_rho_p
- ausa/grad_ratio_set_stack_rho_a
- ausa/grad_ratio_total_rho_pa
- val/ppl
- train/time_per_epoch_s
- train/peak_vram_mib

## Table T5 — Matched baseline vs set-only comparison
Source: `tables/summary/actionD_epoch10_summary.tsv`
Rows: one per paired configuration and implementation
Columns:
- impl
- model.d_model
- model.dim_feedforward
- val/loss
- val/ppl
- train/loss
- train/ppl
- train/time_per_epoch_s
- train/peak_vram_mib
- ausa/routing_entropy_norm
- ausa/router_top1_weight
- ausa/router_candidate_count_struct_mean
- ausa/grad_ratio_pool_rho_p
- ausa/grad_ratio_set_stack_rho_a
- ausa/grad_ratio_total_rho_pa
MD

cat > "$OUTDIR/instructions/04_AGENT_PROGRESSIVE_WORKFLOW.md" <<'MD'
# Recommended progressive workflow for an OpenAI agent

## Step 1
Read:
- `instructions/00_README_FIRST.md`
- `instructions/01_SECTION_PLAN.md`
- `instructions/02_ANTI_HALLUCINATION_RULES.md`
- `instructions/03_TABLE_BLUEPRINTS.md`

## Step 2
Build only Section B (Anchor stability), plus:
- one audit file in `checks/audit_section_B.json`
- one LaTeX snippet in `overleaf_ready/section_B_anchor_stability.tex`

## Step 3
Build only Section C (Feasible operating region), plus:
- one audit file
- one LaTeX snippet

## Step 4
Build only Section D

## Step 5
Build only Section E

## Step 6
Build only Section F

## Step 7
Assemble:
- `overleaf_ready/results_section_combined.tex`
- `overleaf_ready/results_figures_only.tex`
- `overleaf_ready/results_tables_only.tex`

At every step, the agent must stop and verify that each table cell came from the stated TSV source.
MD

# ------------------------------------------
# 6. LaTeX verification helper
# ------------------------------------------
cat > "$OUTDIR/checks/verify_table_values.py" <<'PY'
#!/usr/bin/env python3
import csv
import json
import re
import sys
from pathlib import Path

"""
Usage:
  python3 verify_table_values.py \
      --audit checks/audit_section_B.json

Expected audit JSON schema:
{
  "table_id": "T1",
  "source_tsv": "tables/summary/paper_action0_anchor_seed_sweep.tsv",
  "columns": ["training.seed", "val/ppl"],
  "rows": [
    {
      "match": {"training.seed": 0},
      "used_values": {"training.seed": 0, "val/ppl": 73.1234}
    }
  ]
}
"""

def load_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))

def normalize(v):
    return str(v).strip()

def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--audit":
        print("Usage: python3 verify_table_values.py --audit <audit.json>")
        sys.exit(2)

    audit_path = Path(sys.argv[2])
    audit = json.loads(audit_path.read_text())
    rows = load_tsv(audit["source_tsv"])

    ok = True
    for i, spec in enumerate(audit["rows"], start=1):
        matches = []
        for r in rows:
            good = True
            for k, v in spec["match"].items():
                if normalize(r.get(k, "")) != normalize(v):
                    good = False
                    break
            if good:
                matches.append(r)

        if len(matches) != 1:
            print(f"[FAIL] row {i}: expected exactly 1 source match, got {len(matches)}")
            ok = False
            continue

        src = matches[0]
        for k, v in spec["used_values"].items():
            if normalize(src.get(k, "")) != normalize(v):
                print(f"[FAIL] row {i}: column {k}: expected {v}, source has {src.get(k)}")
                ok = False

    if ok:
        print("[OK] audit matches source TSV exactly")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
PY
chmod +x "$OUTDIR/checks/verify_table_values.py"

# ------------------------------------------
# 7. Overleaf helper include file
# ------------------------------------------
cat > "$OUTDIR/overleaf_ready/results_include_stub.tex" <<'TEX'
% Add these progressively as the agent writes them:
% \input{section_B_anchor_stability.tex}
% \input{section_C_feasible_region.tex}
% \input{section_D_boundary_stride.tex}
% \input{section_E_pooling_concentration.tex}
% \input{section_F_matched_comparison.tex}
TEX

# ------------------------------------------
# 8. Tarball for upload
# ------------------------------------------
tar -czf "$ROOT/out/final_paper_bundle.tar.gz" -C "$OUTDIR" .

echo
echo "Done."
echo "Bundle directory: $OUTDIR"
echo "Tarball: $ROOT/out/final_paper_bundle.tar.gz"
echo "Next: upload final_paper_bundle.tar.gz to the agent / Overleaf workflow."
