# Canonical Documentation Audit

Status: SUPERSEDED by the multiresolution program-plan addendum below; remote mirror propagation pending.

Date: 2026-06-30.

## Current Direction

The active branch is `set-dictionary/anchor-span`. Current evidence is exact-dense multiresolution set
dictionary versus matched exact-dense token attention. The five-seed paper blur rows are
`{b0,b25,b50,b75,b100}`. B3 and B4 remain separate comparison islands.

Coverage-scaled landmark, sparse, fixed-k, Nyström, SD-10a, SD-11, re-read, all-past, and multivector
work are inactive unless explicitly reopened by the user.

## Canonical Chain

1. `docs/set_dictionary_research_main_plan.md`
2. `audit/phase_sd_status.md`
3. the assigned task under `docs/agent_plans/`
4. `docs/sd_dense_paper5_matrix.md` while MRP-1 remains active
5. `docs/set_dictionary_dev_agent_prompt.md`
6. `docs/set_dictionary_model_provenance_for_math_agent.md`
7. `docs/revision_source_of_truth_definitions.md`

## Program-plan addendum

The later same-day audit found and corrected additional restart-safety gaps:

- added a canonical MRP-0--MRP-7 research program and independent agent plans;
- marked v3.0 as architecture/history rather than current execution authority;
- corrected the active branch name in onboarding;
- removed already-satisfied branch-creation prerequisites from the dev prompt;
- corrected the MRP-1 handoff's stale host-idle/token-only pull wording;
- updated Phase-B state to reflect the running five-replicate matrix;
- recorded the open configured-seed-not-applied incident and prohibited paired
  statistics on current artifacts;
- made the current multiresolution theory/proof replacement an explicit
  four-agent workflow.

The earlier `PASS` described only the exact-dense versus legacy-backend
boundary. It must not be read as proof that all restart-safety issues were
already resolved.

## Corrections

- Reordered onboarding and branch prompts around the current matrix/tracker.
- Replaced the root revision context with a current exact-dense environment overlay; archived the old
  Phase-A/v2.7 body.
- Replaced shared research memory with a current exact-dense synthesis; archived the landmark-era
  synthesis.
- Archived all top-level `codex_plan*.md`, `Untitled-*.md`, and `improvements_matrix.md` brainstorms.
- Archived the unapproved fixed-k plan under `docs/archive/deferred/`.
- Added `docs/README.md`, `docs/archive/README.md`, and `audit/README.md`.
- Marked the v2.7 lock, Phase-A tracker, landmark SD-9.5/9.6 files, and old ICML TeX files as
  historical.
- Corrected host authority and credential guidance.
- Corrected the false claim that `training.grad_accum_steps` is implemented.
- Added the L512/B16 seed-0 reuse audit: AdamW uses a constant LR, no scheduler is constructed or
  stepped, warmup is metadata-only, and epoch count only bounds the outer loop.
- Added an empirical-hold comment to the canonical manuscript source so legacy landmark/sparse prose
  cannot be mistaken for current evidence.

## Residual Historical References

Landmark and Nyström still appear where technically or historically necessary:

- code/config capability documentation;
- the explicitly historical sections of the v3.0 design record;
- original audit/provenance files;
- the archived brainstorm/deferred/context directories;
- manuscript text awaiting replacement after the five-seed matrix completes.

Those references are labeled historical, inactive, deprecated, or capability-only and do not override
the canonical chain.

## Checks

- No brainstorm/fixed-k plan remains at the top level of `docs/`.
- No active document references the old brainstorm or fixed-k paths.
- Current matrix/tracker/handoff agree on exact backend, five seeds, regular blur rows, hosts, and PIDs.
- `git diff --check` passes.
- No experiment queue was polled or modified during this documentation audit.

Remote note: propagation of the archive layout and final canonical edits to blue-demon/lizmark was
attempted but blocked by the execution-approval service limit. Do not assume the remote documentation
mirrors are current until an explicit sync succeeds. This does not affect the already running
experiment processes.

## 2026-07-01 Seed-Incident Addendum

The prior PID/seed agreement check is superseded by
`audit/incident_training_seed_not_applied_20260630.md`. Legacy labels `0..4`
were not applied RNG seeds and are now documented only as unpaired stochastic
replicates.

The canonical main plan, tracker, MRP-1 subplan, paper5 matrix, onboarding,
dev-agent prompt, root context, config contract, README, and immediate memory
now agree on:

- isolated corrected namespace `sd_grid_seeded_v1`;
- applied deterministic seeds `0..4`;
- 120 corrected blue cells running under PID `1858067`;
- 135 corrected lizmark cells pending its legacy PID `2389855` exit;
- no remote executable-source sync while a host has an active training/grid
  process;
- one post-launch health check followed by stop-polling.

The later 2026-07-01 grid-identity audit further requires
`current_matrix_v1` diagnostics, native-batch-aware identity, full-data
rejection for every non-null limit, and duplicate-ID failure. See
`audit/incident_sd_grid_identity_diagnostics_20260701.md` and
`audit/SD_grid_duplicate_cleanup_20260701.md`. Current blue PID is `1914084`;
corrected lizmark remains unlaunched while legacy PID `2389855` is active.

## 2026-07-02 Lizmark Admission Addendum

The final paragraph above is a dated transition, not current state. The
corrected Lizmark queue is now running under PID `2879441`; current state is
only in `audit/phase_sd_status.md`.

After detecting an unrelated dual-GPU process, the live matrix, main plan,
MRP-1/MRP-6B/MRP-7 subplans, onboarding, dev prompt, environment context,
model provenance, comparison-control memory, matched-results audit, and
immediate research-direction memory were aligned on one policy:

- co-resident success is valid for PPL and allocated peak VRAM only;
- co-resident wall-time and throughput are excluded;
- co-resident OOM is non-terminal and remains queued;
- only an admission-certified exclusive OOM is terminal capacity evidence;
- legacy `L4096/B4` OOMs predate this incident but lack retrospective
  external-process telemetry.

Artifact disposition, launcher guards, tests, PIDs, and archive paths are in
`audit/incident_lizmark_gpu_contention_20260702.md`.

Final policy supersedes the co-residency bullets above: Lizmark now requires
exclusive occupancy for every corrected cell. The competing container is
stopped under the deferred name, and watcher PID `3049751` owns restoration
and restart after the set-attention grid and all CUDA processes release both
devices.
