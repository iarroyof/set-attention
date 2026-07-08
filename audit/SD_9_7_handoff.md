# Dense Matched Multiresolution Handoff

Status: corrected exact-dense five-seed paper matrix RUNNING.

## Read first

1. `docs/set_dictionary_research_main_plan.md`
2. `audit/phase_sd_status.md`
3. `docs/agent_plans/mrp1_matrix_closure.md`
4. `docs/sd_dense_paper5_matrix.md`
5. `audit/incident_sd_dense_token_exact_backend_params.md`
6. `audit/incident_training_seed_not_applied_20260630.md`
7. `audit/incident_lizmark_gpu_contention_20260702.md`

The archived `audit/SD_9_7_handoff_landmark_legacy_20260625.md` is provenance only.

## Scientific target

Test whether multi-resolution set-dictionary attention gives a reproducible PPL/peak-VRAM advantage over:

1. uniform all-fine and all-coarse set banks; and
2. matched dense token attention,

within exact-backend islands that hold batch, length, architecture, LR, warmup metadata, data, epochs,
and seeds fixed.

Do not compare absolute PPL across B16 and B4. Use `L=512` only as a measured batch bridge.

## Active matrix

- Regular set rows: `{b0,b25,b50,b75,b100}` plus exact token, five seeds.
- Blue-demon: L512/B16, L512/B4, L1024/B4, L2048/B3.
- Lizmark: L2048/B4, L3584/B4, L3584/B3, L4096/B3, L4096/B4.
- At L4096/B4, token/b0/b25 remain repeated 3/3 legacy OOM outcomes;
  b50/b75/b100 are five-seeded.
- Existing b62 artifacts remain exploratory and are not enqueued.

Resolved primary set frontier:

- all set rows through `L=2048` completed 3/3;
- at `L=4096`, b50/b62/b75/b100 completed 3/3;
- at `L=4096`, b0 and b25 OOMed 3/3 on 49 GiB and are closed frontier cells.

## Token retry resolution

The original exact-token attempts produced zero training epochs due to inherited landmark
`backend_params`. They are invalid and excluded.

The fixed scheduler selects `baseline_dense_exact.yaml` for exact token rows. Final outcome:

- blue PID `3489171`: 8/8 cells completed;
- lizmark PID `2153424`: L2048 3/3 completed; L4096 OOM 3/3.

At completion of that token-retry transition both hosts were idle. Results and
validation are in `audit/SD_dense_matched_results.md`; the later paper5 queues
listed below now own both hosts.

## Standing check procedure

On an explicit user status request:

1. Check `pgrep -fa 'run_sd_grid.sh|run_experiment.py'` and `nvidia-smi` once on both hosts.
2. Run `SD_GRID_TARGET_EPOCHS=10 python3 scripts/sd_grid_status.py out/paper_mechanisms`.
3. Pull only new exact-dense paper5 set/token CSV/JSON/log files; do not pull or aggregate landmark rows.
4. Validate token metadata:
   - `model.implementation=baseline_token`
   - `model.backend=exact`
   - `model.backend_params` absent/empty
   - `data.limit` absent/null
   - correct B/L/seed
   - 10 epoch rows
   - `train/peak_vram_mib` present
5. Scan completed logs for word-boundary NaN/Inf, traceback, and OOM.
6. Join Lizmark rows to `gpu_admission_lizmark.tsv`; reject every
   non-exclusive or occupancy-unknown attempt from all analysis.
7. Update `audit/phase_sd_status.md` and the dense plan before launching anything else.

Configured seed labels in the legacy queue were not applied to RNG state.
Preserve every legacy artifact, but summarize those rows as unpaired
stochastic replicates and do not compute paired-seed statistics. Corrected
`sd_grid_seeded_v1` rows apply and assert seeds `0..4`.

## Analysis contract

For each matched island report:

- per-seed and mean/CI validation PPL;
- mean peak train VRAM;
- full set blur curve and feasible Pareto frontier;
- best mixed versus all-fine/all-coarse;
- best feasible mixed versus exact token;
- whether the set advantage changes with L;
- B16 versus B4 offset at `L=512`, reported separately from architecture effects.

At `L=4096`, the token OOM is a repeated legacy observed-feasibility outcome,
not a token-quality comparison or retrospectively certified exclusive-capacity
measurement. New terminal OOM claims require admission-certified exclusivity.

## Guards

- No landmark, sparse, fixed-k, re-read, all-past, multivector, or anchor launches.
- Set rows remain `anchor_span`, token MLP off, anchor off, CE-only, endpoint-window.
- Never aggregate `data.limit` rows.
- Use `scripts/run_sd_grid.sh` only; legacy SD-9.6/9.7 launchers are retired for new work.
- One grid driver per host; dry-run first; one post-launch health check; then stop polling.
- No commit unless the user approves.

## Frontier profile

`GRID_PROFILE=frontier` contains only:

- blue: L2048/B3, six set blurs plus token;
- lizmark: L3584/B4, L3584/B3, and L4096/B3, six set blurs plus token.

The `SEEDS=0` full 10-epoch/full-data contract is complete. It is a valid stochastic replicate, not a conclusive
multi-seed result. `GRID_PROFILE=b2` is unnecessary because every B3 row fit and must remain unlaunched.

Completed launch:

- blue PID `3657887`, log `logs/sd_grid_blue_frontier_seed0.log`: DONE 7/7 and locally validated;
- lizmark PID `2185668`, log `logs/sd_grid_lizmark_frontier_seed0.log`: DONE 21/21 and locally
  validated.

Both hosts are idle. The 28 cells have full data, 10 epochs, independent peak VRAM, correct exact-dense
metadata, and clean logs. Complete result audit: `audit/SD_dense_frontier_extension.md`.

Selected seed-0 outcomes:

- L3584/B4 b25 Pareto-dominates token (`846.937/38552.9 MiB` versus
  `890.875/41076.2 MiB`);
- L3584/B3 favors token quality, with b62 as the set memory tradeoff;
- L4096/B3 b25 Pareto-dominates token (`840.564/35350.1 MiB` versus
  `969.223/37955.4 MiB`).

Native B3 and B4 are different optimization islands. Do not promote these single-seed comparisons to
paper claims.

## Active paper5 queue

The user approved the selected five-seed main-paper matrix in
`docs/sd_dense_paper5_matrix.md`. Current corrected state is:

- blue-demon complete: 120/120 endpoint-valid rows, driver exited,
  `logs/sd_grid_blue_paper5_seeded_v1.log`;
- lizmark first pass complete: 135/135 core rows, 120 endpoint-valid, 15
  mixed `L3584,B4` rows rejected for endpoint gradient diagnostics;
- lizmark replacement running: driver PID `3940226`, workers `3940654` and
  `3940655`, log
  `logs/sd_grid_lizmark_paper5_seeded_v1_replacements_20260707.log`.

The corrected matrix uses `{b0,b25,b50,b75,b100}` plus token at every supported island and excludes
b62. Contention-exposed Lizmark artifacts are quarantined outside the live
tree. The external container was stopped again before the replacement launch.
The replacement dry run planned exactly 15 cells and skipped 120 valid lizmark
cells. Stop polling until an explicit user request. Do not start another queue,
`cancer_rl_agent`, or any deferred family while this replacement driver is
active.
