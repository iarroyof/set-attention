# Dense Matched Set-vs-Token Plan

> Historical execution plan. Current matrix and GPU-admission policy are in
> `docs/sd_dense_paper5_matrix.md` and
> `audit/incident_lizmark_gpu_contention_20260702.md`. The repeated legacy
> `L4096/B4` OOMs predate that incident but lack archived external-process
> telemetry; do not present them as certified exclusive-capacity measurements.

Status: PRIMARY AND FRONTIER MATRICES COMPLETE; FIVE-SEED PAPER TOP-UP RUNNING. Last updated 2026-06-30.

This document supersedes landmark-era SD-9.5--SD-9.7 launch plans for new experimental work. Landmark,
sparse, and fixed-k families are disabled until the user explicitly reopens one.

Program-level continuation after this matrix is governed by
`docs/set_dictionary_research_main_plan.md` and its task plans under
`docs/agent_plans/`. This document remains the scientific contract for MRP-1
islands only.

## Question

Under a fully matched exact-dense protocol, does mixed-resolution set-dictionary attention:

1. improve the PPL/peak-VRAM frontier over uniform all-fine and all-coarse set banks; and
2. match or beat dense token attention as context length grows?

PPL comparisons are valid only within an island of constant
`(backend,batch,L,architecture,lr,warmup,data,epochs,seeds)`.

## Fixed protocol

- Backend: exact dense for both set and token.
- Model: `D=384`, `d_ff=1536`, 6 layers, 8 heads.
- Training: WikiText-2 full data, 10 epochs, LR `1e-4`, warmup metadata `1000`. Primary grid used
  seeds `{0,1,2}`; current paper cells use `{0,1,2,3,4}`.
- Those configured seed labels were not applied to RNG state by the current
  unified runner. Analyze completed cells as unpaired stochastic replicates;
  do not use paired-seed statistics.
- Current five-seed blur rows:
  - b0: 8 fine / 0 coarse
  - b25: 6 fine / 2 coarse
  - b50: 4 fine / 4 coarse
  - b75: 2 fine / 6 coarse
  - b100: 0 fine / 8 coarse
- b62 (3 fine / 5 coarse) exists in the earlier primary/frontier grids but is exploratory and is not
  part of the regular five-seed paper matrix.
- Fine group `(w,s)=(2,1)`; coarse group `(4,2)`.
- Set guards: `anchor_span`, token MLP off, anchor off, CE-only, endpoint-window fiber, no re-read,
  all-past, or multivector.
- Record validation PPL and independent peak train VRAM for every completed run.

## Matched islands

| Island | Host | Purpose |
|---|---|---|
| exact, `L=512,B=16` | blue-demon | GOLD operating point and continuity with SD-9 short |
| exact, `L=512,B=4` | blue-demon | measured bridge from B16 to the cross-length batch |
| exact, `L=1024,B=4` | blue-demon | short-to-medium scaling |
| exact, `L=2048,B=4` | lizmark | medium scaling |
| exact, `L=4096,B=4` | lizmark | dense feasibility frontier |

Do not average B16 and B4. Do not compare their absolute PPL as an architecture effect.

## Current progress

Dense set sweep:

| Island | Result |
|---|---|
| L512/B16 | all six blur rows complete, 3/3 seeds |
| L512/B4 | all six blur rows complete, 3/3 seeds |
| L1024/B4 | all six blur rows complete, 3/3 seeds |
| L2048/B4 | all six blur rows complete, 3/3 seeds |
| L4096/B4 | b50/b62/b75/b100 complete; b0/b25 OOM 3/3 |

Set-only means:

| Island | Lowest-PPL feasible row | Mean PPL | Mean peak VRAM MiB |
|---|---|---:|---:|
| L512/B16 | b25 | 860.629 | 13790.5 |
| L512/B4 | b25 | 961.859 | 4074.8 |
| L1024/B4 | b25 | 931.052 | 8037.4 |
| L2048/B4 | b25 | 920.105 | 18097.3 |
| L4096/B4 | b50 | 900.563 | 41369.5 |

Interpretation before token controls:

- b25 Pareto-dominates all-fine on PPL and VRAM through `L=2048`.
- At `L=4096`, b0 and b25 exceed the 49 GiB limit. b50 is the best-PPL feasible row, not a proven
  unconstrained optimum.
- These conclusions are set-vs-set only.

## Exact-token incident and retry

The first new token attempts failed before training because `run_sd_grid.sh` loaded the landmark token
YAML and attempted to clear inherited `backend_params` with `{}`. Exact-backend validation rejected it.

The failure did not affect set rows. It does mean no new set-vs-token conclusion exists yet.

Resolution:

- exact token uses `configs/paper_lr_norm/baseline_dense_exact.yaml`;
- `backend_params` must be absent/empty;
- the grid scanner selects registered epoch 10 from any protocol-matched longer reusable run;
- regression/config checks and token-only dry runs passed on both hosts.

Current retry:

- blue-demon PID `3489171` ended normally: 8/8 completed;
- lizmark PID `2153424` ended normally: L2048 3/3 completed, L4096 3/3 OOM.

Both hosts are idle. The matched result is `audit/SD_dense_matched_results.md`.

## Completion and analysis contract

The completed primary analysis followed this contract; reuse it for any approved top-up:

1. Validate exact backend, no backend parameters, no data limit, 10 epochs, correct seeds/B/L, and peak VRAM.
2. Scan logs for word-boundary NaN/Inf, traceback, and OOM.
3. Report per-seed values and mean/CI for PPL and peak VRAM.
4. Compare the full set blur curve and feasible Pareto frontier within each island.
5. Compare the best feasible mixed row with the matched exact token row.
6. Report the L512 B16-to-B4 offset separately.
7. Treat a token OOM as a feasibility result, not as a token PPL value.
8. Historical note: the original plan proposed winner-only seeds `{3,4}`. That recommendation is
   superseded by `docs/sd_dense_paper5_matrix.md`.

## Deferred work

- No landmark or sparse follow-up is active.
- `docs/archive/deferred/sd_linear_matrix_plan.md` is a deferred historical design note, not an
  approved queue.
- SD-10a and SD-11 remain held until this dense comparison is reviewed.

## Dense-frontier extension

Motivation: L4096/B4 b50 has unexpectedly strong PPL while token, b0, and b25 OOM. An isolated smaller
batch at L4096 would create an uninterpretable new island. The extension therefore uses an overlap
rectangle that separates batch and length:

| Point | Host | Role |
|---|---|---|
| L2048/B3 | blue-demon | B4-to-B3 batch bridge at an already solved length |
| L3584/B4 | lizmark | same-B4 near-frontier point between L2048 and L4096 |
| L3584/B3 | lizmark | second B4-to-B3 bridge |
| L4096/B3 | lizmark | closest smaller-batch completion attempt |

Each point contains all six blur rows plus exact token. Stage 1 runs seed 0 only, but it is a full-data,
10-epoch run, not a smoke/probe. Its PPL is a valid replicate; it is not a multi-seed conclusion.

Comparisons enabled:

- B4 length behavior: L2048 -> L3584 -> feasible L4096 rows.
- B3 length behavior: L2048 -> L3584 -> L4096.
- Batch effect at fixed L: B4 versus B3 at L2048 and L3584.
- Within-island blur/PPL/VRAM curves at every completed point.

Use `GRID_PROFILE=frontier SEEDS=0 scripts/run_sd_grid.sh`. The profile is opt-in and emits no primary
cells. In-flight duplicate identity includes batch.

If B3 still OOMs at L4096, the prepared `GRID_PROFILE=b2` fallback repeats the same structure with B2.
Do not launch B2 until the seed-0 B3 rectangle has been analyzed. Extend seeds 1--2 only for supported,
scientifically useful cells after that review.

Launch state:

- blue-demon PID `3657887`, `logs/sd_grid_blue_frontier_seed0.log`: DONE 7/7 L2048/B3 cells;
- lizmark PID `2185668`, `logs/sd_grid_lizmark_frontier_seed0.log`: DONE 21/21 rectangle cells;
- all 28 cells completed full-data training for 10/10 epochs and passed guard/metric/log validation;
- both hosts were idle at this frontier transition and B2 was not launched;
  the later paper5 queues in `audit/phase_sd_status.md` are now active.

Blue L2048/B3 result: b25 is the best set row (`900.7514` PPL, `13797.1` MiB); exact token is
`885.7863` PPL at `14189.7` MiB. The seed-0 ordering relative to B4 flips, confirming that native batch
changes optimization behavior and must remain an explicit comparison island. Details:
`audit/SD_dense_frontier_extension.md`.

Completed frontier findings:

- L3584/B4: b25 `846.937 / 38552.9 MiB` versus token `890.875 / 41076.2 MiB`; b25 is Pareto-better
  for seed 0.
- L3584/B3: b62 `915.146 / 24467.7 MiB` versus token `869.081 / 31035.0 MiB`; this is a
  quality/memory tradeoff and the preferred set blur shifts.
- L4096/B3: b25 `840.564 / 35350.1 MiB` versus token `969.223 / 37955.4 MiB`; b25 is Pareto-better
  for seed 0.
- B3 materially changes optimization and cannot be treated as a memory-only replacement for B4.
- These are exploratory full runs, not paper-grade population estimates. Details and the complete
  tables are in `audit/SD_dense_frontier_extension.md`.

## Five-seed paper confirmation

The corrected main-paper matrix uses the regular blur set `{b0,b25,b50,b75,b100}` plus exact token at
every supported island. It includes both B3 bridges at L2048/L3584. L4096/B4 includes supported
b50/b75/b100 while token/b0/b25 remain repeated 3/3 legacy OOM cells. Existing b62 artifacts are retained as
exploratory evidence but are not part of the five-seed queue.

Launched 2026-06-30:

- blue-demon driver PID `526557`: 58 missing runs;
- lizmark driver PID `2389855`: 90 missing runs.

Both queues passed matching host dry runs and one post-launch health check. Full matrix, logs, ETA, and
comparison contract: `docs/sd_dense_paper5_matrix.md`.

The superseded reduced-row-selection queues were stopped before any new cell completed; no prior valid run
was removed.

Do not poll until explicitly requested. Do not launch B2 or the full frontier cross-product. Gradient
accumulation is not required for the registered L4096/B4 feasibility result; add an effective-B4
control only if a matched L4096/B4 quality claim becomes necessary.
