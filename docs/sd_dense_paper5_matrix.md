# Exact-Dense Five-Seed Paper Matrix

Status: FIRST PASS COMPLETE; 15 L3584/B4 MIXED REPLACEMENTS RUNNING.

Updated: 2026-07-07 after Lizmark replacement launch.

Reproducibility note: the completed legacy rows logged labels `seed=0..4` but
did not apply them to RNG state. Preserve those rows as unpaired stochastic
replicates. They do not satisfy the five-seed confirmation requirement. See
`audit/incident_training_seed_not_applied_20260630.md`.

The confirmation matrix is being rerun in
`out/paper_mechanisms/sd_grid_seeded_v1` with actual seeds `0..4`,
deterministic mode, applied-seed assertions, and the
`current_matrix_v1` diagnostics contract.

Qualification: seeds are applied, but exact CUDA replay is not certified.
All 120 pulled Blue logs warn that CuBLAS operations remained nondeterministic
because `CUBLAS_WORKSPACE_CONFIG` was not set and the runner used
`warn_only=True`. Checkpoint and data/tokenizer digest provenance are also
absent. See `audit/incident_mrp0_prelaunch_gap_20260706.md`.

## Objective

Confirm the exact-dense multiresolution Pareto findings with five seeds at every trainable condition
reported in the main paper. Every supported island uses the same blur set
`{b0,b25,b50,b75,b100}` plus matched exact token attention. No legacy
replicate is reused as an applied seed.

All cells use full WikiText-2, 10 epochs, exact dense attention, `D=384`, `d_ff=1536`, 6 layers,
8 heads, LR `1e-4`, and warmup metadata `1000`.

Set guards are `output_residual_mode=anchor_span`, token MLP off, anchor off, CE-only,
endpoint-window, and no re-read/all-past/multivector.

## Matrix

| Host | Island | Five-seed paper rows | Corrected seeded state |
|---|---|---|---|
| blue-demon | L512/B16 | token + b0/b25/b50/b75/b100 | 30/30 valid done |
| blue-demon | L512/B4 | token + b0/b25/b50/b75/b100 | 30/30 valid done |
| blue-demon | L1024/B4 | token + b0/b25/b50/b75/b100 | 30/30 valid done |
| blue-demon | L2048/B3 | token + b0/b25/b50/b75/b100 | 30/30 valid done |
| lizmark | L2048/B4 | token + b0/b25/b50/b75/b100 | 30/30 valid done |
| lizmark | L3584/B4 | token + b0/b25/b50/b75/b100 | 15 endpoint-valid done; 15 mixed replacement rows running |
| lizmark | L3584/B3 | token + b0/b25/b50/b75/b100 | 30/30 valid done |
| lizmark | L4096/B3 | token + b0/b25/b50/b75/b100 | 30/30 valid done |
| lizmark | L4096/B4 | b50/b75/b100 | 15/15 valid done |

Corrected total: 255 runs, with 120 on blue-demon and 135 on lizmark. No
legacy row is reused to satisfy an applied seed.

The L4096/B4 token, b0, and b25 rows remain repeated 3/3 legacy OOM outcomes.
They predate the 2026-07-02 Lizmark contention incident and are not rerun in
the corrected namespace. Because their legacy launchers did not archive
external-process telemetry, describe them as observed legacy feasibility
outcomes, not retrospectively certified exclusive-capacity measurements.

The b62 row is excluded from this five-seed matrix. Existing b62 artifacts remain valid exploratory
evidence but are not enqueued or pooled into the regular five-row blur comparison.

## Targeted comparisons

1. At each matched island, compute the feasible PPL/peak-VRAM Pareto frontier over all five blur rows.
2. Compare every blur row against exact token only within identical `(L,batch)` islands.
3. Use L2048 and L3584 B4-versus-B3 pairs to measure native-batch sensitivity; do not pool batches.
4. Use L4096/B3 for a supported quality comparison and L4096/B4 for the registered memory-feasibility
   result.
5. Treat b62 as an exploratory supplementary row outside the regular five-seed matrix.

## Scheduler

Use only:

```bash
GRID_PROFILE=paper5 SEEDS="0 1 2 3 4" HOST_TAG=<blue|lizmark> \
  GRID_NAMESPACE=sd_grid_seeded_v1 RUN_TAG=seeded_v1 \
  REQUIRE_APPLIED_SEED=1 TRAINING_DETERMINISTIC=true \
  GPU0=0 GPU1=1 bash scripts/run_sd_grid.sh
```

The metadata scanner skips completed seeds. One queue is allowed per host.
Lizmark additionally requires `REQUIRE_EXCLUSIVE_GPU=1` and
`ALLOW_GPU_CORESIDENCY=0`. Any CUDA process or failed occupancy query defers a
cell before container creation, with a second occupancy check immediately
before `docker run`. The known competing container is stopped and has a
one-shot post-grid restart handoff. After launch, perform one health check and
stop polling until the user requests status.

## Launch state

The initial reduced-row-selection queues launched at 08:44 CST were stopped before any new run completed.
They were superseded without deleting any prior valid artifact.

The 2026-06-30 queues are classified as legacy unseeded-replicate queues and
have ended. The corrected Lizmark queue launched after cleanup and testing.
Untracked GPU co-residency was then detected; affected corrected artifacts were
quarantined before the guarded resume.

| Host | Driver PID | Corrected cells | Queue log | Snapshot active row | ETA / action |
|---|---:|---:|---|---|---|
| blue-demon | exited normally | 120/120 endpoint-valid done | `logs/sd_grid_blue_paper5_seeded_v1.log` | none; both GPUs idle | final package pulled and validated |
| lizmark replacement | 3940226 | 15 mixed L3584/B4 replacements planned; first two active at launch check | `logs/sd_grid_lizmark_paper5_seeded_v1_replacements_20260707.log` | b25 seed0/seed1 active | wait for completion, then pull and strict-validate |

Blue closed with 120 strict endpoint-valid rows, 120 done markers, and 120
exit-0 registry records. The final result/log package is local. Word-boundary
NaN/Inf, traceback, runtime, unexpected OOM, and W&B-step scanning found zero
affected logs.

The earlier two four-epoch corrected rows and one fail-fast configuration
attempt are archived outside aggregation roots. The Lizmark contention-exposed
seed 2 and partial attempts are also outside the live tree. They cannot satisfy
or duplicate a matrix cell. Cleanup and incident details:
`audit/SD_grid_duplicate_cleanup_20260701.md` and
`audit/incident_sd_grid_identity_diagnostics_20260701.md`, and
`audit/incident_lizmark_gpu_contention_20260702.md`.

The 2026-07-07 status check found the driver finished and both GPUs idle.
The first pass produced PPL/VRAM for all 255 corrected rows, but the strict
scanner still rejects the 15 mixed `L3584,B4` rows because their endpoint
gradient diagnostics remain `NA`. Those invalid first-pass rows were archived
outside the corrected root. The unchanged `paper5` manifest dry-run reported
exactly 15 replacement plans and 120 skips before launch.

The final strict-exclusive resume passed 6 launcher/handoff contract tests, a
133-plan/2-accepted-skip dry run, and one post-launch health check. Admission
recorded no prior CUDA process on either GPU; the only two GPU processes were
the set-attention workers. Both OOM registries were header-only.
The competing container is stopped and renamed
`cancer_rl_agent__deferred_until_sd_grid_release` until
the corrected matrix, including replacement rows, is validated. Restart watcher
PID `3049751` was stopped on 2026-07-04 because the current driver would
otherwise release the external workload before the 15 rejected mixed
`L3584/B4` cells can be rerun. Do not launch a second driver, manually start
that container, or poll either queue without an explicit status request.

## Partial Result Snapshot

The frozen 2026-07-07 PPL/VRAM matrices, paired comparisons, and provisional
frontiers are in `audit/SD_dense_paper5_partial_20260707.md` and
`out/paper_integrated_evidence/checks/sd_grid_seeded_v1_post_lizmark_20260707/`.

Current complete-island mean frontiers:

| Island | Nondominated rows |
|---|---|
| L512/B16 | token, b50, b75, b100 |
| L512/B4 | b25, b50, b75, b100 |
| L1024/B4 | token, b50, b75, b100 |
| L2048/B3 | b25, b50, b75, b100 |
| L2048/B4 | b25, b50, b75, b100 |
| L3584/B4 | b25, b50, b75, b100 (provisional; mixed rows require replacement) |
| L3584/B3 | b25, b50, b75, b100 |
| L4096/B3 | b25, b50, b75, b100 |
| L4096/B4 | b50, b75, b100 |

The strongest complete set-vs-token mean result is `L2048,B4` b25:
`916.4 +/- 33.7` PPL and `18116.7` MiB versus token
`942.8 +/- 19.4` PPL and `18633.3` MiB. Its paired PPL delta
`-26.5 +/- 48.1` still crosses zero, so this is a mean Pareto win rather than
a confirmed quality separation.

The registered `L2048,B4` selection island is complete. The minimum interior
mean PPL selects and freezes `b*=b25` before any downstream empirical outcome;
see `audit/SD_dense_paper5_results.md`. This freeze does not close MRP-1 or
authorize MRP-2/3/5.

The strict 2026-07-07 scan confirmed the epoch-level gradient-probe cadence
defect still affects all 15 completed mixed `L3584,B4` rows. The remote source
contains the instrumentation-only fix, and the replacement wave is now running.
Launch audit: `audit/SD_dense_paper5_replacements_20260707.md`. Incident and recovery procedure:
`audit/incident_sd_grid_epoch_gradient_probe_cadence_20260704.md`.
