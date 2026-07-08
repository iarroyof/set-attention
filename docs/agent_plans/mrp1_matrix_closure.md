# MRP-1: Five-Seed Exact-Dense Matrix Closure

Status: REPLACEMENT RUNNING

Owner: current experiment operations role

Updated: 2026-07-07 after Lizmark replacement launch.

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

The frozen 2026-07-06 queue snapshot is:

- blue-demon: 120/120 endpoint-valid complete; driver exited normally; both
  GPUs idle; final package pulled and validated;
- lizmark corrected PID `2879441`: exited normally; 135/135 first-pass rows
  completed, 120 endpoint-valid, and 15 mixed `L3584,B4` rows required
  replacement for endpoint gradient diagnostics.
- lizmark replacement driver PID `3940226`: running the 15 registered
  `L3584,B4` mixed replacement rows; workers `3940654` and `3940655`.

The Lizmark external-workload watcher PID `3049751` was stopped earlier, but
the 2026-07-07 check found `cancer_rl_agent__deferred_until_sd_grid_release`
running again with no current GPU allocation. The external workload must be
kept or restored to a non-contending state before replacement rows launch.

The one permitted post-launch health check passed on each host. Lizmark
requires exclusive admission after the incident in
`../../audit/incident_lizmark_gpu_contention_20260702.md`. Do not poll either
host again until the user explicitly asks for status or reports completion.

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

On the first explicit completion/status request:

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
`audit/SD_dense_paper5_results.md`. It does not satisfy MRP-1's remaining
matrix-closure or MRP-0 dependencies.

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

Status: REPLACEMENT RUNNING.

Last completed action: Blue ended normally at 120/120 strict endpoint-valid
cells. Lizmark ended normally and its artifacts/log were pulled locally on
2026-07-07. The merged first-pass snapshot contains 255 full-data rows:
240 strict endpoint-valid rows plus 15 mixed `L3584,B4` rows whose endpoint
gradient diagnostics remain `NA`. Those 15 invalid records and markers were
archived outside the corrected root; the replacement dry run planned exactly
15 cells and skipped 120; the replacement driver was launched.

Files changed: epoch-scoped gradient-probe scheduling, its regression test,
partial-grid summarizer/artifacts, incident audit, and canonical
planning/status documentation. The probe fix is deployed on both hosts;
existing active containers retain their imported pre-fix code.

Commands/tests and outcomes: final launcher/handoff tests passed 6/6. The
strict dry run reported 133 plans and two accepted skips, and the post-resume
health check showed one exclusive set-attention process per GPU and header-only
OOM registries.

Artifacts and digests: pending final sync.

Host/PID/log/ETA: replacement driver PID `3940226`, workers `3940654` and
`3940655`, log
`logs/sd_grid_lizmark_paper5_seeded_v1_replacements_20260707.log`. Initial
active cells were b25 seeds 0 and 1, with both GPUs at about 41.9 GiB.

Decision or gate result: MRP-1 is not closed. Replacement wave is running
after external workload was stopped again.

Known incidents or limitations: legacy labels were not applied. Grid
identity/diagnostics remediation is in
`audit/incident_sd_grid_identity_diagnostics_20260701.md`; Lizmark admission
and artifact disposition are in
`audit/incident_lizmark_gpu_contention_20260702.md`; the epoch-probe recovery
is in `audit/incident_sd_grid_epoch_gradient_probe_cadence_20260704.md`.
Full MRP-0 checkpoint/data/loader work remains open.

Next atomic action: stop polling until user requests status or reports
completion. Then pull replacement artifacts, inspect admission telemetry and
OOM registries, and rerun the strict scanner before accepting final MRP-1
results.

Inputs required: host access, current queue logs/artifacts, canonical scanner,
and the registered matrix.
