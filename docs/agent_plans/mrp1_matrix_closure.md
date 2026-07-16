# MRP-1: Five-Seed Exact-Dense Matrix Closure

Status: PASS

Owner: current experiment operations role

Updated: 2026-07-14 after post-close short-B3 bridge audit.

## Mission

Complete, validate, summarize, and freeze the exact-dense paper matrix with
applied seeds `0..4`, without changing its registered cells.

## Required Retrieval Context

1. `../set_dictionary_research_main_plan.md`
2. `../../audit/phase_sd_status.md`
3. `../sd_dense_paper5_matrix.md`
4. `../../audit/SD_9_7_handoff.md`
5. `../sd_dense_matched_comparison_plan.md`
6. `../../scripts/sd_grid_status.py`

## Active State

The final 2026-07-08 queue snapshot is:

- blue-demon: 120/120 endpoint-valid complete; driver exited normally; both
  GPUs idle; final package pulled and validated;
- lizmark corrected/replacement queues: exited normally; 135/135 corrected
  rows endpoint-valid after replacing the 15 mixed `L3584,B4` diagnostic rows.

The final Lizmark status check found no active grid driver or worker and both
GPUs idle. Pause/release Lizmark unless a new approved stage explicitly needs
it. Future Lizmark launches still require exclusive admission after the
incident in `../../audit/incident_lizmark_gpu_contention_20260702.md`.

Post-close bridge note: the later `short_b3` extension completed the previously
missing `L512/B3` and `L1024/B3` descriptive islands with 60/60 endpoint-valid
rows. These rows complete the full paper visualization over B3/B4/B16 short
operating points and reinforce the b25 interior pattern. They are not part of
the original closed 255-row paper5 bundle, do not change the frozen
`b*=b25`, and do not authorize another MRP-1 sweep.

## Locked Matrix

- rows: token plus `{b0,b25,b50,b75,b100}`;
- full WikiText-2, no `data.limit`;
- 10 epochs;
- exact backend;
- registered shape and optimizer settings;
- repeated legacy `L4096/B4` token/b0/b25 OOM cells remain 3/3 and are not
  retried by this matrix.

No b62, B2, landmark, sparse, fixed-k, or architecture row may be added.

## Completion Procedure

Completed procedure:

1. Check each host once for the grid driver, workers, GPU state, and final queue
   log.
2. Run the canonical status scanner at target epoch 10.
3. Pull only new exact-dense paper5 CSV, JSON, logs, and closed-OOM evidence.
4. Validate every successful artifact:
   - correct model family, row, `L`, batch, and applied seed;
   - `training.seed_applied=true`, requested/applied/torch seed equality,
     deterministic mode requested, and benchmark mode off;
   - experiment contract `sd_grid_seeded_v1` and diagnostics contract
     `current_matrix_v1`;
   - exact backend;
   - empty/absent exact-token `backend_params`;
   - full data and 10 registered epochs;
   - `anchor_span`, token MLP off, anchor off, CE-only, endpoint-window for set;
   - peak train VRAM present independently of smoke rows.
   - fine/coarse group diagnostics present for every group instantiated by the
     row; baseline attention diagnostics present for token rows.
5. Scan logs for word-boundary NaN/Inf, traceback, runtime failure, unexpected
   OOM, and W&B step corruption.
6. Reconcile expected successful rows, terminal exclusive OOMs, duplicates,
   and missing rows. Stop and write an incident on any mismatch.
7. Join admission telemetry by cell. Reject every non-exclusive or
   occupancy-unknown attempt from all analysis and require a retry; accept only
   exclusive OOMs as censored feasibility.
8. Update `docs/sd_dense_paper5_matrix.md` and
   `audit/phase_sd_status.md` from RUNNING to DONE only after validation.

Exact CUDA replay is not part of the achieved MRP-1 contract because CuBLAS
ran without a fail-closed workspace configuration. Preserve that qualification
from `audit/incident_mrp0_prelaunch_gap_20260706.md`.

## Analysis Contract

For each `(L,batch)` island, using only corrected seeded rows:

- report all seed-level PPL and peak-VRAM values;
- report mean, sample SD, and 95% Student-t CI;
- compute the feasible PPL/mean-peak-VRAM Pareto frontier;
- compare each mixed row with b0 and b100;
- compare set rows with token only within the identical island;
- keep B3, B4, and B16 separate;
- report OOM as censored feasibility.

Legacy rows remain a separate unpaired-replicate sensitivity analysis. Never
pool them with corrected rows or use them to fill a missing corrected seed.

Freeze `b*` before MRP-2/3/5 outcomes are inspected:

1. use the `L=2048,B=4` island;
2. consider interior rows b25, b50, and b75;
3. select the row with minimum mean validation PPL;
4. record its mean PPL, CI, VRAM, and selection timestamp in
   `audit/SD_dense_paper5_results.md`;
5. do not change `b*` during later datasets or tasks.

This freeze is complete: `b*=b25`, recorded in
`audit/SD_dense_paper5_final_20260708.md`. MRP-1 is closed; later launches
still require their own explicit approvals.

## Deliverables

- `audit/SD_dense_paper5_results.md`
- machine-readable per-run and per-cell summaries under
  `out/paper_integrated_evidence/checks/`
- final blur/Pareto tables and plots
- updated matrix, tracker, main-plan status, and Phase-B current plan
- explicit frozen `b*`

## Definition Of Done

Every expected corrected cell is either validated successful or registered
terminal exclusive OOM; all logs and applied-seed metadata pass; legacy sensitivity is
separate; `b*` is frozen; both hosts are recorded idle; and no later experiment
has been launched.

## Durable Handoff

Status: PASS.

Last completed action: Blue ended normally at 120/120 strict endpoint-valid
cells. Lizmark ended normally after the replacement wave; artifacts/logs were
pulled locally on 2026-07-08. The strict scanner accepted 255 endpoint-valid
full-data rows under `SD_GRID_REQUIRE_CONTRACT=sd_grid_seeded_v1`.

Files changed: final-grid summary artifacts, final audit, manuscript table,
and canonical planning/status documentation.

Commands/tests and outcomes: strict `sd_grid_status.py` scan passed. The final
Lizmark status check showed no active grid process, both GPUs idle, queue log
complete, and header-only corrected OOM registries.

Artifacts and digests: `out/paper_integrated_evidence/checks/sd_grid_seeded_v1_final_20260708/`
and `audit/SD_dense_paper5_final_20260708.md`.

Host/PID/log/ETA: no active PIDs. Log:
`logs/sd_grid_lizmark_paper5_seeded_v1_replacements_20260707.log`.

Decision or gate result: MRP-1 is closed. `b*=b25` remains frozen. MRP-2 and
MRP-3 already have recorded approval for their registered work; continue only
their registered evaluation/completion steps.

Known incidents or limitations: legacy labels were not applied. Grid
identity/diagnostics remediation is in
`audit/incident_sd_grid_identity_diagnostics_20260701.md`; Lizmark admission
and artifact disposition are in
`audit/incident_lizmark_gpu_contention_20260702.md`; the epoch-probe recovery
is in `audit/incident_sd_grid_epoch_gradient_probe_cadence_20260704.md`.
MRP-0 platform validation passed, but new stages still require registered
preflights and checkpoint/data provenance.

Next atomic action: no further MRP-1 experiment action. Downstream work should
continue with the already registered MRP-2 AR-hit evaluation and MRP-3 MQAR
completion. Do not run more Lizmark work without a new approved stage.

Inputs required: host access, current queue logs/artifacts, canonical scanner,
and the registered matrix.
