# MRP-7A: Legacy Manuscript Content Deprecation And Current-Model Rewrite

Status: ACTIVE; current-evidence figure/table/prose pass compiled

Owner: current paper-update worker until reassigned

Updated: 2026-07-15 after full-matrix/short-B3 Results audit and MRP-3 null summary.

## Mission

Move active manuscript narrative away from legacy SKA dense/sparse/landmark and
single-resolution `empty_only` framing, while preserving that material as
deprecated provenance. Replace it with exact-dense multiresolution
set-dictionary sections that use the final MRP-1 matrix and diagnostics.

This task is an MRP-7 subtask that is allowed before final MRP-7 because it
does not need new experiments. It only reorganizes and rewrites paper content
around already validated MRP-1/MRP-6D evidence.

## Required Retrieval Context

1. `../set_dictionary_research_main_plan.md`
2. `../../audit/phase_sd_status.md`
3. `../../audit/SD_dense_paper5_final_20260708.md`
4. `../../docs/sd_dense_paper5_matrix.md`
5. `../../memory/set-attention-research-direction.md`
6. `../../out/final_paper_bundle/overleaf_ready/example_paper.tex`

Do not read legacy manuscript sections as launch authority or current
empirical claims.

## Write Scope

- Current manuscript narrative in
  `out/final_paper_bundle/overleaf_ready/example_paper.tex`
- Deprecated-paper-content index under
  `docs/archive/deprecated_paper_content/`
- Optional extracted legacy snippets under
  `docs/archive/deprecated_paper_content/`
- Paper-synthesis audit notes under `audit/`

Do not alter theorem statements without MRP-6D review. Do not launch
experiments.

## Deprecation Rule

Legacy sections may be removed from the active PDF when they are one of:

- matched SKA dense/sparse/landmark backend controls;
- A7 singleton/near-token `empty_only` calibration used as a current claim;
- A-series LR, window, pooling, sparse, landmark, or linear figures presented
  as active evidence;
- old single-stream/direct-residual paper claims;
- any text implying landmark/linear efficiency for the current branch.

Preserve the material as historical provenance by recording its old labels,
source figures/tables, and reason for deprecation in
`docs/archive/deprecated_paper_content/README.md`. Large TeX extracts may be
stored in separate files there if needed, but active claims should point to the
new exact-dense MRP-1 audit.

## Current-Model Replacement Sections

Rewrite the active Results around these subsections:

1. **Exact-Dense Matched Protocol**: state the fixed control tuple and the
   matched exact-dense token baseline; no sparse/landmark family comparison.
2. **Matched Token Control And Multiresolution Frontier**: use the full MRP-1
   PPL/VRAM matrix by `(L,batch)` island.
3. **All-Fine Control Is Not The Optimum**: use b0 as the high-resolution set
   endpoint and show that b25 often beats it; do not claim b0 equals token.
4. **Blur Sweep And Capacity Boundary**: describe b0->b100 as the allocation
   path, and treat `L4096/B4` token/b0/b25 as censored feasibility.
5. **Fine/Coarse Mechanism Diagnostics**: use existing ablation, effective
   range, routing entropy/top1, and rarity/position diagnostics.
6. **Claim Boundary And Next Mechanism Tests**: state that MRP-2 is still
   needed for natural AR-hit attribution and that MRP-3 completed as a
   null/inconclusive MQAR probe because the registered task did not reach the
   support accuracy regime. No new experiment is implied by the paper rewrite
   itself.

The prose style should be exploratory and reviewer-facing: explain what the
patterns suggest, what they do not prove, and why the next mechanism tests are
natural.

## Current-Model Figure Program

The current paper figures must be analogous in role to the legacy diagnostic
figures, not simple one-metric bar charts.  Main figures should answer a
research question and expose a comparative pattern:

1. **Blur-allocation frontier path**
   (`fig_sd_blur_path_frontier.pdf`): plot PPL against peak GiB for
   b0$\to$b25$\to$b50$\to$b75$\to$b100 across the registered `(L,B)` matrix,
   including the short-B3 bridge and censored `L4096/B4` boundary, with the
   matched token point shown when available.  Purpose: show that the useful
   operating point is usually interior, not all-fine/token-like and not
   all-coarse.
2. **Operating-regime map**
   (`fig_sd_operating_regime_map.pdf`): heatmap over all final matrix islands
   and blur settings, with set-minus-token PPL and GiB saved in each cell;
   censored L4096/B4 rows must be labeled explicitly.  Purpose: show where the
   multiresolution advantage appears as context/batch pressure grows.
3. **Fine/coarse routing diagnostics**
   (`fig_sd_group_routing_diagnostics.pdf`): use line plots with confidence
   intervals over coarse-head fraction inside fixed `(L,B)` islands.  Purpose:
   show that fine and coarse labels correspond to learned routing behavior,
   with fine paths local/concentrated and coarse paths broader/higher-entropy.
4. **Mechanism/allocation sweep**
   (`fig_sd_mechanism_allocation.pdf`): align PPL--memory frontier views with
   route-removal responsibility along coarse-head fraction for fixed-batch B3
   and B4 paths.  Purpose: show that b25 is an interior optimum, later blur
   saves memory but becomes quality destructive, and the memory frontier is not
   obtained by making the coarse stream the dominant predictive carrier.

Auxiliary span and bucket diagnostic point plots may be generated for
inspection, but they are not sufficient main-paper figures unless promoted
into one of the comparative views above. Do not use histogram/bar-chart
encodings for scalar diagnostics that are not counts.

## Definition Of Done

- Deprecated legacy content is no longer active main-paper evidence.
- The active Results section is centered on the current exact-dense
  multiresolution set-dictionary matrix.
- The full MRP-1 matrix and interpretive sections compile in the manuscript.
- Deprecated content is indexed under
  `docs/archive/deprecated_paper_content/`.
- The main plan delegates this paper-rewrite subtask explicitly.
- The PDF compiles without new missing files.

## Durable Handoff

Status: ACTIVE; current-evidence rewrite in progress.

Last completed action: subplan created, deprecated-content index added, main
plan delegation updated, active Results rewritten around the final MRP-1
exact-dense matrix plus short-B3 bridge, and the current-evidence
figure/table/prose layer added.  The main figures now follow the legacy-paper
diagnostic role with current-model evidence: a blur-allocation frontier path,
an operating-regime map, fine/coarse routing line diagnostics, and an
allocation/mechanism sweep.  Earlier literal routing/span/bucket bar charts
were demoted or converted to scalar point diagnostics rather than main
histogram-style evidence visuals.
The abstract, introduction, and active model overview now use the exact-dense
multiresolution set-dictionary framing instead of dense/sparse/landmark
A-series framing.  Legacy Results subsections from `Matched Backend Controls`
through `Capacity Diagnostics` are hidden from the active PDF under `\iffalse`
and indexed as deprecated provenance.

Files changed:

- `docs/agent_plans/mrp7a_legacy_paper_content_deprecation.md`
- `docs/archive/deprecated_paper_content/README.md`
- `docs/set_dictionary_research_main_plan.md`
- `out/final_paper_bundle/overleaf_ready/example_paper.tex`
- `out/final_paper_bundle/overleaf_ready/tables/sd_grid_compact_frontier.tex`
- `out/final_paper_bundle/plots/main/fig_sd_exact_dense_frontier.tex`
- `out/final_paper_bundle/plots/main/fig_sd_exact_dense_frontier.pdf`
- `out/final_paper_bundle/plots/main/fig_sd_blur_path_frontier.tex`
- `out/final_paper_bundle/plots/main/fig_sd_blur_path_frontier.pdf`
- `out/final_paper_bundle/plots/main/fig_sd_operating_regime_map.tex`
- `out/final_paper_bundle/plots/main/fig_sd_operating_regime_map.pdf`
- `out/final_paper_bundle/plots/main/fig_sd_mechanism_allocation.tex`
- `out/final_paper_bundle/plots/main/fig_sd_mechanism_allocation.pdf`
- `out/final_paper_bundle/plots/main/fig_sd_group_routing_diagnostics.tex`
- `out/final_paper_bundle/plots/main/fig_sd_group_routing_diagnostics.pdf`
- `out/final_paper_bundle/plots/main/fig_sd_span_ablation_diagnostics.tex`
- `out/final_paper_bundle/plots/main/fig_sd_span_ablation_diagnostics.pdf`
- `out/final_paper_bundle/plots/main/fig_sd_bucket_diagnostics.tex`
- `out/final_paper_bundle/plots/main/fig_sd_bucket_diagnostics.pdf`
- `scripts/build_mrp1_paper_assets.py`

Commands/tests and outcomes:

- `pdflatex -interaction=nonstopmode -halt-on-error example_paper.tex` parsed
  the source but could not overwrite `example_paper.pdf`, likely because the
  PDF was open/locked.
- `pdflatex -interaction=nonstopmode -halt-on-error -jobname=example_paper_mrp7a_check example_paper.tex`
  succeeded and produced `example_paper_mrp7a_check.pdf`.
- `python scripts/build_mrp1_paper_assets.py` generated the compact frontier
  table and TikZ figure source from
  `out/paper_integrated_evidence/checks/sd_grid_seeded_v1_final_20260708/cells.tsv`.
- `pdflatex -interaction=nonstopmode -halt-on-error fig_sd_exact_dense_frontier.tex`
  succeeded and produced `fig_sd_exact_dense_frontier.pdf`.
- `pdflatex -interaction=nonstopmode -halt-on-error -jobname=example_paper_mrp7a_current_check example_paper.tex`
  succeeded twice and produced `example_paper_mrp7a_current_check.pdf`.
- `pdflatex -interaction=nonstopmode -halt-on-error fig_sd_group_routing_diagnostics.tex`
  succeeded and produced `fig_sd_group_routing_diagnostics.pdf`.
- `pdflatex -interaction=nonstopmode -halt-on-error fig_sd_span_ablation_diagnostics.tex`
  succeeded and produced `fig_sd_span_ablation_diagnostics.pdf`.
- `pdflatex -interaction=nonstopmode -halt-on-error fig_sd_bucket_diagnostics.tex`
  succeeded and produced `fig_sd_bucket_diagnostics.pdf`.
- `pdflatex -interaction=nonstopmode -halt-on-error fig_sd_blur_path_frontier.tex`
  succeeded and produced `fig_sd_blur_path_frontier.pdf`.
- `pdflatex -interaction=nonstopmode -halt-on-error fig_sd_operating_regime_map.tex`
  succeeded and produced `fig_sd_operating_regime_map.pdf`.
- `pdflatex -interaction=nonstopmode -halt-on-error -jobname=fig_sd_mechanism_allocation_new fig_sd_mechanism_allocation.tex`
  succeeded; the generated PDF was copied to
  `fig_sd_mechanism_allocation.pdf` to replace stale output.
- `pdflatex -interaction=nonstopmode -halt-on-error -jobname=example_paper_mrp7a_diag_check2 example_paper.tex`
  succeeded twice and produced `example_paper_mrp7a_diag_check2.pdf`.

Artifacts and digests: final MRP-1 audit is
`audit/SD_dense_paper5_final_20260708.md`.

Host/PID/log/ETA: no active host work.

Decision or gate result: no clarification blocks the rewrite. Deprecated
content remains recoverable, but not active evidence. Final MRP-7 remains
blocked on MRP-2/5; MRP-3 can only be reported as a null/inconclusive MQAR
probe, and MRP-7A may continue current-evidence cleanup.

Known incident or limitation: final MRP-7 remains blocked on MRP-2/5, so this
subtask must not write final AR/PG-19 claims or positive MQAR specialization
claims.  The compile still emits pre-existing missing-bibliography warnings
and appendix overfull boxes; these are not introduced by the current MRP-1
figure/table/diagnostics pass.
The main `example_paper.pdf` may be locked by a viewer; use the temporary
compile product for validation or close the locked PDF before regenerating the
canonical file.

Next atomic action: continue rewriting Future Work, Conclusion, and appendices
to remove remaining legacy A-series claims from active prose, keeping only
provenance references and replacing active claims with current MRP-1/MRP-6D
wording.  After MRP-2 completes, add the registered AR-hit result; include
MRP-3 only as a completed null/inconclusive MQAR probe unless a future approved
protocol replaces it.

Inputs required: none beyond current MRP-1 artifacts.
