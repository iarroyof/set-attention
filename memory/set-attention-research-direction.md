---
name: set-attention-research-direction
description: Current exact-dense multiresolution set-dictionary direction
metadata:
  type: project
  updated: 2026-07-08
---

# Current Research Direction

The active branch is `set-dictionary/anchor-span`. Current experiments compare multiresolution
set-dictionary attention with matched token attention using the **exact dense backend only**.

## Active Evidence

- Full WikiText-2, 10 epochs, LR `1e-4`, `D=384`, `d_ff=1536`, 6 layers, 8 heads.
- Set rows use `anchor_span`, token MLP off, anchor off, CE-only, endpoint-window.
- The paper matrix uses blur rows `{b0,b25,b50,b75,b100}` plus exact token at every supported
  `(L,batch)` island.
- Legacy labels `seed=0..4` were not applied to RNG state. Those artifacts are
  unpaired stochastic replicates only. The confirmation matrix reruns every
  supported cell with actual seeds `0..4` in `sd_grid_seeded_v1`; no legacy
  artifact satisfies a corrected seed.
- The corrected matrix is closed. Blue has 120/120 strict endpoint-valid rows;
  Lizmark has 135/135 after the 15 mixed `L3584,B4` replacement rows
  (`b25/b50/b75`, seeds 0--4). The final strict scan accepted 255 CSVs under
  `SD_GRID_REQUIRE_CONTRACT=sd_grid_seeded_v1`.
- Corrected aggregation is fail-closed on experiment/diagnostics contracts,
  native batch, applied seed, full-data status, and duplicate IDs. Fine/coarse
  training diagnostics are separate; archived partial/fail-fast attempts are
  outside `out/paper_mechanisms`.
- MRP-1 is PASS. Pause/release Lizmark until a new approved stage explicitly
  needs it.
- MRP-0 passed Blue container validation on 2026-07-07. The validation covered
  focused tests, duplicate strict token/b25 smokes, checkpoint replay, and
  eval-only immutability. Run one-step full-shape preflights before any
  selected retraining launch.
- MRP-6A is PASS, MRP-6B is analytic PASS, MRP-6C is analytic PASS, and
  MRP-6D canonical TeX integration passed clean build. Empirical
  specialization wording still waits for MRP-3.
- MRP-3 generator/trainer infrastructure is ready and passed Blue container
  tests, dry-run, launch-guard, and a tiny CPU smoke. MRP-2 and MRP-3 are now
  gated by explicit launch approval rather than by MRP-1 completion. MRP-5
  still requires MRP-2/MRP-3 review and explicit approval.
- B3 and B4 are separate optimization islands; never pool their PPL values.
- L4096/B4 token, b0, and b25 are repeated 3/3 legacy OOM outcomes that predate
  the 2026-07-02 contention incident. Their old launchers lack external-process
  telemetry, so they are observed legacy feasibility outcomes rather than
  retrospectively certified exclusive-capacity measurements.
- The primary defensible result is a mixed-resolution quality/memory frontier and an exact-dense
  memory-feasibility extension, not universal PPL superiority over token attention.
- Final MRP-1 analysis: `audit/SD_dense_paper5_final_20260708.md`.

Live matrix and status:

- `docs/set_dictionary_research_main_plan.md`
- `docs/agent_plans/`
- `docs/sd_dense_paper5_matrix.md`
- `audit/phase_sd_status.md`
- `audit/SD_9_7_handoff.md`
- `audit/SD_dense_matched_results.md`
- `audit/SD_dense_frontier_extension.md`

## Inactive Or Historical

- Coverage-scaled landmark (`landmark_coverage=0.25`) is quadratic up to a constant factor because
  landmark count scales with `M`. It is historical quality/reference evidence only.
- Nyström, fixed-k landmark, sparse, SD-10a, SD-11, re-read, all-past, and multivector work are not
  active and require explicit user approval.
- Do not execute archived brainstorms, legacy status snapshots, or landmark-era handoffs.
- The pre-2026-06-30 research synthesis is preserved at
  `memory/archive/set-attention-research-direction_legacy_through_20260625.md`.

## Paper Framing

Use “set-mediated token-level causal prediction” and “causal dictionary atoms.” The intended headline
is that moderate blur can improve the exact-dense set frontier and extend the feasible context boundary
while retaining quality. Do not claim that the coverage landmark implementation is linear or
sub-quadratic.

The next registered program adds only natural AR-hit slicing, standalone
synthetic MQAR, and one tokenizer-matched PG-19 transfer study. Formal work
replaces the legacy single-stream/direct-residual appendix with a
multiresolution `anchor_span` theory. See the canonical main plan for gates.
