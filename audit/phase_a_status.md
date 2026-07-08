# Phase A Status Tracker

> **Historical original-campaign tracker.** Do not use this file to select current set-dictionary
> work. Current state is `audit/phase_sd_status.md`; current cells are
> `docs/sd_dense_paper5_matrix.md`.

Last updated: 2026-06-16 by Codex after validating A9 candidate-gather router comparison.

## v2.7 Scope Amendment

Matched token-backend baseline controls are required before final B5/A5 handoff. Completed A2/A3/A4 set-family artifacts remain valid, but dense-baseline-only comparisons are incomplete for reviewer-facing backend attribution.

New required controls:

- `baseline_sparse_local_band`: token Transformer with local-band backend, no set pooling/routing.
- `baseline_linear_landmark`: token Transformer with landmark backend, no set pooling/routing.

These are additive tasks. Do not discard completed A2/A3/A4 artifacts. Continue from the current phase sequence, then fill the control gaps before final paper-table handoff.

## Summary

| Phase | Task | Status | Runs | Audit / Notes |
| --- | --- | --- | --- | --- |
| A0 | Preflight (env fingerprint, smoke train, baseline tests) | ✅ DONE | — | Confirmed in Context doc; container: Python 3.11.0rc1, PyTorch 2.5.1+cu124, 2×RTX4090 |
| A1.1 | Option-1 bank + residual (T1 tail drop, strict-past, R1 residual) | ✅ DONE | — | `src/models/set_only/banks.py`, `set_only_lm.py`, `router.py` modified |
| A1.2 | Causality tests | ✅ DONE | — | `tests/test_causality.py` (untracked, exists on blue-demon) |
| A1.3 | Reconcile App D.2 vs Table 8 | ✅ DONE | — | `audit/A1_3_reconciliation.md` |
| A1.4 | Hyperparameter exposure (config propagation contract) | ✅ DONE | — | `normalize.py`, `schema.py`, `compatibility.py` modified |
| A1.5 | Cache fingerprint / W&B step checks | ✅ DONE | — | `audit/A1_5_cache_wandb.md` |
| A1.6 | Landmark remediation (linspace-rounded, coverage param) | ✅ DONE | — | `src/set_attention/backends/landmark.py` modified |
| A1.7 | Diagnostics rewire (Option-1 candidate fiber) | ✅ DONE | — | `src/models/set_only/diagnostics.py` modified |
| A1.8 | Documentation scan/update | ✅ DONE | — | `audit/A1_8_doc_changes.md` |
| A1.9 | Audit and smoke gate | ✅ DONE | — | Go/no-go passed; A2 launched |
| A1.10 | Nyström deprecation | ✅ DONE | — | `src/set_attention/backends/nystrom.py` hard-fails; `configs/_deprecated/` created |
| A2.0 | Smoke gate (10 epochs, dense baseline + dense SKA) | ✅ DONE | — | Pre-condition for A2.1+ |
| A2.1 | Stability table (anchor topology reference) | ✅ DONE | 153 total | `audit/A2_grid_handoff.md` — Status: PASS |
| A2.2 | LR-normalized matched grid (LR-norm headline reference) | ✅ DONE | 153 total | Included in A2 handoff; multiple D/d_ff sizes |
| A2.3 | Family slice (baseline-token, SetDense, SetSparse, SetLinear) | ✅ DONE | 153 total | Included in A2 handoff |
| A2.4 | Matched token-backend baseline controls (`baseline_sparse_local_band`, `baseline_linear_landmark`) | ✅ DONE | 30 | `audit/A2_4_baseline_controls.md` — Status: PASS (validated_runs=30/30); launched 2026-05-13 17:03 CST, completed 18:34 CST |
| A3.1 | Window-size sweep (fixed s=4, w ∈ {6,8,12,16,20,24}) | ✅ DONE | 24 | `audit/A3_1_window_sweep.md` — Status: PASS |
| A3.1-control | Window-sweep token-backend control overlays (`baseline_sparse_local_band`, `baseline_linear_landmark`) | ✅ DONE | 24 | `audit/A3_1_baseline_controls.md` — Status: PASS (validated_runs=24/24); launched 2026-05-13 21:02 CST, completed 22:15 CST |
| A3.2 | Pooling-temperature sweep (fixed s=4, w=16, τ ∈ {0.05,0.1,0.2}) | ✅ DONE | 27 | `audit/A3_2_pooltau_sweep.md` — Status: PASS; local and remote manifests confirmed pass on 2026-05-13 |
| A3.3 | Stride sweep (fixed w=16, s ∈ {4,8,12,16}) — demoted complement | ✅ DONE | 24 | `audit/A3_3_stride_sweep.md` — Status: PASS (validated_runs=24/24, synced 2026-05-11) |
| A4.1 | Long-context smoke (L=2048) | ✅ DONE | 2 | `audit/A4_1_smoke.md` — Status: PASS (validated_runs=2/2, batch_size=4, synced 2026-05-11). Note: B=16 OOMs at L=2048 fp32; B=4 fits ~6 GiB. |
| A4.2 | Long-context slice (LR-norm headline reference, L=2048) | ✅ DONE | 12 | `audit/A4_2_slice.md` — Status: PASS (validated_runs=12/12, 4 families × 3 seeds, synced 2026-05-12). mean_val_ppl: baseline=928.2, set_dense=1511.5, set_sparse=1429.4, set_linear=1434.7. |
| A4.2-control | Long-context matched token-backend controls (`baseline_sparse_local_band`, `baseline_linear_landmark`) | ✅ DONE | 6 | `audit/A4_2_baseline_controls.md` — Status: PASS (validated_runs=6/6); launched 2026-05-13 22:17 CST, completed 22:48 CST |
| A4.3 | Convergence run (≥30 epochs, best config) | ✅ DONE | 6 | `audit/A4_3_convergence.md` — Status: PASS (validated_runs=6/6, 30 epochs); launched 2026-05-14 08:06 CST, completed 09:07 CST |
| A4.4 | Optional MQAR / associative recall | ⏳ PENDING | — | Optional; skip if budget constrained |
| A5 | Reproducibility handoff (TSVs + metadata per locked contract) | ✅ DONE | — | `audit/A5_4_handoff.md` — Status: PASS; final index `out/paper_integrated_evidence/checks/final_artifact_index.tsv`; B5 cleared |
| A6.1 | d_phi capacity sweep (`d_phi` ∈ {384,512,768}, SKA families only) | ✅ DONE | 27 | `audit/A6_1_dphi_capacity.md` — Status: PASS (validated_runs=27/27); best mean val PPL: Set Dense d_phi=384, Set Sparse d_phi=512, Set Linear d_phi=512 |
| A6.2 | Explicit set-state dimensionality sweep (`set_state_dim` ∈ {384,512,768}) | ✅ DONE | 36 | `audit/A6_2_set_state_width.md` — Status: PASS (validated_runs=36/36; 18 reused + 18 new); token width fixed at D=384 and d_phi fixed at 384 |
| A6.3 | Set-token interface bottleneck sweep (`set_state_dim,d_phi` matched/partial) | ✅ DONE | 54 | `audit/A6_3_interface_bottleneck.md` — Status: PASS (validated_runs=54/54; 27 reused + 27 new). Raising `d_phi` partially helps some wider set-state runs but matched `d_phi=set_state_dim` does not consistently recover degradation. |
| A6.4 | Set-stack depth sweep (`num_layers` ∈ {6,8,10}, 2 capacity pairs) | ✅ DONE | 54 | `audit/A6_4_depth_sweep.md` — Status: PASS (validated_runs=54/54; 18 reused + 36 new). Depth bottleneck hypothesis REJECTED: more layers consistently worsens val PPL across all families (depth=6 is best). Overfitting signature: train PPL flat ~350-400 while val PPL rises with depth. |
| A7.0 | Empty-only calibration feature and smoke/test gate | ✅ FEATURE IMPLEMENTED | — | `model.output_residual_mode` added with `direct`, `empty_only`, `none`; tests pass on blue-demon Docker. Full calibration experiments not launched. Plan: `audit/A7_empty_only_calibration_plan.md`. |
| A7.1-A7.3 | Empty-only token-limit and compression calibration experiments | ✅ DONE | 24 new + 3 reused baseline | `audit/A7_empty_only_calibration.md` — Status: PASS. New SetDense `empty_only` runs validated 24/24; matched A2 dense baseline rows reused 3/3 by exact provenance. Token-limit `w=1,s=1,M=512` mean val PPL 800.8 vs baseline 781.1; degradation increases as `M/L` falls. |
| A7.4 | Backend-family empty-only calibration (`SetSparse`, `SetLinear` on dense A7 topologies) | ✅ DONE | 48 new + 33 reused references | `audit/A7_backend_family_empty_only.md` — Status: PASS (validated_new_set_runs=48/48; combined summary rows=27). Matched token sparse/linear baselines are reused as horizontal references because token baselines do not consume set candidate-fiber topology. Figure: `out/final_paper_bundle/plots/main/fig_a7_backend_family_compression.png`. |
| A7.5 | Targeted seed extension for convergence-critical A7 points | ✅ DONE | 24 new | `audit/A7_seed_extension.md` — Status: PASS (validated_new_runs=24/24). New seeds `{3,4}` for token baselines `baseline_dense_exact`, `baseline_sparse_local_band`, `baseline_linear_landmark`, and set families `set_dense_exact`, `set_sparse_local_band`, `set_linear_landmark` at `(w,s)={(1,1),(2,1),(3,1)}`. Augmented summary: `out/paper_integrated_evidence/tables/a7_backend_family_empty_only_augmented_summary.tsv`; 12 convergence-critical rows now have five seeds. Quality-efficiency frontier figure: `out/final_paper_bundle/plots/main/fig_a7_seed_extension_fine_grained.png`; compact operating-point comparison table: `tab:a7-operating-point-comparison`; VRAM overhead audit: `audit/vram_overhead_audit.md` (no subtraction justified). |
| A8 | Favorable set-attention conditions plan | ✅ FOLLOW-UP DONE | 6 smoke + 10 follow-up | `audit/A8_favorable_set_conditions_plan.md`. Direct `L=8192` seed-0 smoke passed; see `audit/A8_3_largeL_smoke.md`. Selective 5-seed `L=8192` follow-up passed for `baseline_linear_landmark` vs `set_linear_landmark` at `(w,s)=(8,4)`; see `audit/A8_3_l8192_linear_followup.md`. Result: SetLinear uses ~50.35% of matched linear baseline VRAM but mean val PPL is worse by +1132.91, so the smoke quality signal did not hold. Next recommended work is A8.0 candidate-gather routing or memory/retrieval-favorable tasks, not broadening current-implementation L=8192 grids. |
| A8-hybrid | Hybrid token/set sparse progressive topology sweep | ✅ DONE | 9 | `audit/A8_hybrid_sparse_progressive.md` — Status: PASS (validated_runs=9/9). Initial run exposed a hybrid diagnostics bug caused by sharing one `SetDiagnostics` object across set layers with different `M`; fixed by per-set-layer diagnostics aggregation. Relaunched on blue-demon at 2026-06-14 12:05 CST. Matrix: sparse/local-band hybrids `TTSSSS`, `TSTSTS`, `TTTTSS`, seeds `{0,1,2}`, `D=384,d_ff=1536,L=512,lr=1e-4`, progressive topologies `(4,2)` then `(8,4)`. Configs under `configs/a8_hybrid/` are now the source of truth for model/data/training hyperparameters; launcher overrides only seed/output/log identity for future resumes/reruns. Result: best pattern `TTTTSS` mean val PPL 2393.36, still poor; seeds 3,4 not recommended unless a confirmatory negative result is requested. |
| A9 | Candidate-gather learned router implementation and matched comparison | ✅ DONE | 18 | `audit/A9_candidate_gather.md` — Status: PASS (validated_runs=18/18). Redundancy-1 fix implemented as `model.router.score_mode=candidate_gather`, with `dense` retained as historical/debug mode. Focused Docker tests passed. Matrix: set dense/sparse/linear, `(w,s)={(4,2),(8,4)}`, seeds `{0,1,2}`, `empty_only`, `D=384,d_ff=1536,L=512,lr=1e-4`. Result: semantics/config validation passed, but L=512 peak VRAM did not improve versus dense-router A7 references; candidate-gather rows were +27.7 MiB at `(4,2)` and +118.8 MiB at `(8,4)` on average. |
| B5 | Results integration from A5/A6 handoff | ✅ DONE | — | `audit/B5_results_integration.md` — Status: PASS. Current NeurIPS bundle source is `out/final_paper_bundle/overleaf_ready/example_paper.tex`; latest local bundle build is `out/final_paper_bundle/checks/compile_logs/run_future_work_20260613/example_paper.pdf` (46 pages, no fatal LaTeX errors or unresolved references/citations). |

## Blocking Dependencies

```
B5 results/writeup integration was cleared by A5 handoff; A6.1-A6.4 add-on capacity ablations are available for B5.
A9 candidate-gather validation/summarization is complete. The next memory-focused step should inspect why the L=512 training peak is dominated by non-router terms or allocator/diagnostic overhead before claiming a practical VRAM win from candidate-gather routing.
```

## Incidents

| File | Phase | Summary |
| --- | --- | --- |
| `audit/incident_A1_9_diagnostics_nan_20260509.md` | A1.9 | False-positive nan in diagnostics scan |
| `audit/incident_A2_0_baseline_diagnostics_nan_20260509.md` | A2.0 | Same false-positive nan in baseline scan |

Root cause of both incidents: `scan_logs()` matched `"nan"` as a plain substring inside normal English words (e.g. "planning", "channel"). Fixed in `summarize_a3_pooltau_sweep.py` and `summarize_a3_stride_sweep.py` by replacing with word-boundary regex: `(?<![A-Za-z0-9_])(?:nan|NaN|-inf|inf)(?![A-Za-z0-9_])`.

## Key Config Constants (post-A1.1 locked)

| Quantity | Value |
| --- | --- |
| LR-norm headline reference | D=384, d_ff=1536, w=16, s=8, M=63 (L=512) |
| Anchor topology reference | D=384, d_ff=1536, w=16, s=4, M=125 (L=512) |
| Causality mode | `strict_past` |
| Tail policy | T1 (drop partial trailing windows) |
| Landmark coverage | `landmark_coverage=0.25` |
| Pooling alpha | `10.0` |
| Hash seed | `13` |
| Router min_temp | `0.5` |
| Git branch | `paper/final-results-bundle` |
| Git HEAD (at A3.2 run) | `1174947643b001647c4fb92cc48e88c75f044be4` |

## Current Next Step

The v2.7 matched token-backend controls are complete:

1. A2.4 manifest: `out/paper_integrated_evidence/checks/a2_baseline_controls_manifest.json` — pass, 30/30.
2. A3.1-control manifest: `out/paper_integrated_evidence/checks/a3_window_baseline_controls_manifest.json` — pass, 24/24.
3. A4.2-control manifest: `out/paper_integrated_evidence/checks/a4_long_context_baseline_controls_manifest.json` — pass, 6/6.

A5 reproducibility handoff passed. A6.1 d_phi capacity sweep passed. A6.2 explicit `model.set_state_dim` sweep passed. A6.3 interface bottleneck sweep passed after Docker restart and detached resume. The bottleneck hypothesis is partially supported for moderate `d_phi` increases, but not as a complete explanation because matched `d_phi=set_state_dim` often worsens validation PPL. Do not edit LaTeX until explicitly requested.

A6.4 set-stack depth sweep complete. Depth bottleneck hypothesis rejected: increasing num_layers 6→8→10 worsens validation PPL monotonically for all three families at both capacity settings. Train PPL is flat (~350–400) while val PPL rises — classic overfitting / generalization gap. Depth=6 is the optimal set-stack depth under the current training budget (10 epochs, LR=1e-4).

B5 Results integration and the first B1-B4/B6 consistency pass are complete in the current NeurIPS final bundle. The paper now uses conservative matched-control claims, strict-past endpoint bank definitions, T1 dropped trailing windows, configurable output residual routing, candidate-fiber diagnostics, pre-shifted loss normalization, and provenance-only treatment of unreconciled historical appendix slices. A7 adds `model.output_residual_mode=empty_only` for calibrated token-limit and compression experiments. SetDense token-limit/compression calibration and backend-family empty-only calibration both passed; summaries are `out/paper_integrated_evidence/tables/a7_empty_only_calibration_summary.tsv`, `out/paper_integrated_evidence/tables/a7_backend_family_empty_only_summary.tsv`, and the A7.5 five-seed augmented summary `out/paper_integrated_evidence/tables/a7_backend_family_empty_only_augmented_summary.tsv`. The latest local PDF after adding the expanded Future Work priority list and A8 plan is `out/final_paper_bundle/overleaf_ready/example_paper.pdf` (46 pages, no fatal LaTeX errors or unresolved citations/references in `out/final_paper_bundle/checks/compile_logs/run_future_work_20260613/example_paper.log`). VRAM values remain raw `train/peak_vram_mib`; `audit/vram_overhead_audit.md` found no removable non-architectural overhead requiring subtraction. A8 is planned but not launched.
