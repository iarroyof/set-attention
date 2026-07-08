# MRP-7: Final Paper Synthesis And Reproducibility Bundle

Status: BLOCKED

Owner: UNASSIGNED

Updated: 2026-06-30.

## Mission

Integrate the validated empirical program and audited theory into one concise
NeurIPS paper, with the full grid and negative results in the appendix.

## Hard Dependencies

- MRP-1 complete;
- MRP-2 and MRP-3 complete;
- MRP-4 either complete or `NOT_TRIGGERED`;
- MRP-5 complete;
- MRP-6D complete.

## Required Retrieval Context

Read the main plan, every final MRP audit, the MRP-6D proof ledger, the
canonical manuscript, and Phase-B progress logs. Do not use intermediate chat
summaries.

## Write Scope

- non-theory narrative/tables/figures in the canonical TeX
- final plots and tables
- `out/final_paper_bundle/checks/current_plan.md`
- reproducibility manifests and final bundle
- `audit/MRP_7_paper_synthesis.md`

Do not alter theorem statements without returning the change to MRP-6D audit.

## Main-Paper Evidence Budget

Keep:

1. one exact-dense WT2 blur/PPL/VRAM frontier figure;
2. one compact matched token/set table including the L4096 feasibility
   boundary;
3. one MQAR mechanism figure with lag and group ablation;
4. one tokenizer-matched WT2/PG-19 transfer table;
5. at most four main-text formal results.

Move full per-replicate grids, AR frequency/lag slices, sensitivity results,
parameter counts, fit residuals, and null outcomes to the appendix.

## Claim Rules

- Say constant-factor exact-dense memory reduction, never subquadratic.
- Say fixed-batch observed feasibility when legacy token rows OOM; reserve a
  standalone capacity claim for admission-certified exclusive OOM evidence.
- Say interior blur is supported only on datasets where the registered gate
  passes.
- Say MQAR measures discrete associative recall by lag, not signal frequency.
- Describe natural AR-hit evidence as reduced-scale unless Pile-scale
  pretraining was actually performed.
- State that b0 is a set endpoint, not token attention.
- Distinguish unpaired MRP-1 replicates from applied-seed MRP-2/3/5 runs.
- Preserve negative/null results and confidence intervals.

## Reproducibility Bundle

Include:

- every active config and resolved fingerprint;
- data/tokenizer/checkpoint digests;
- exact run manifests and applied seeds;
- per-run PPL/VRAM and OOM evidence;
- per-run GPU admission state, with every non-exclusive corrected Lizmark
  attempt excluded from evidence;
- summarizer/log-scan outputs;
- theorem validation tests;
- clean LaTeX compile log and PDF digest.

## Definition Of Done

Every abstract/conclusion claim maps to evidence, all legacy backend language
is removed or historical, the main text remains compact, the appendix and
bundle are complete, and the final PDF compiles cleanly.

## Durable Handoff

Status: BLOCKED.

Last completed action: evidence budget, claim rules, and reproducibility bundle
registered.

Files changed: this subplan only during registration.

Commands/tests and outcomes: none.

Artifacts and digests: none.

Host/PID/log/ETA: none.

Decision or gate result: final synthesis cannot start.

Known incident or limitation: current manuscript still contains legacy
single-stream and inactive-backend claims.

Next atomic action: wait for every hard dependency and MRP-6D clean build.

Inputs required: final MRP audits, proof ledger, plots/tables, manifests, and
canonical TeX.
