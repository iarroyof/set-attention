# Multiresolution Research Program Plan Audit

Status: PASS for planning and ownership; empirical launches remain gated.

Date: 2026-06-30.

## Purpose

Verify that `docs/set_dictionary_research_main_plan.md` can be resumed by
independent agents without chat context, conflicting file ownership, silent
architecture changes, or ambiguous experiment selection.

## Decomposition Rationale

| Workstream | Why it is an independent end-to-end task |
|---|---|
| MRP-0 platform | Seed application, checkpoints, masked metrics, and provenance are shared correctness infrastructure; implementing them separately prevents three task agents from creating incompatible formats. |
| MRP-1 matrix closure | It owns already-running remote processes, artifact sync, validation, statistics, and frozen `b*`; mixing it with new code would violate the active queue policy. |
| MRP-2 natural AR hits | It is an evaluation definition and token-level accounting problem with a fixed natural-LM matrix, distinct from synthetic generation and corpus transfer. |
| MRP-3 MQAR | It owns a standalone generator, masked objective, lag metrics, ablations, calibration, and registered interaction gate. |
| MRP-4 sensitivity | It is conditional and changes one topology dimension; isolating it prevents an exploratory architecture from contaminating the primary model claim. |
| MRP-5 PG-19 | It owns tokenizer/corpus provenance and a fixed cross-domain matrix; no natural-dataset tuning feeds back into `b*`. |
| MRP-6A formal foundations | It defines the executed model and proves causality/context factorization before approximation assumptions are introduced. |
| MRP-6B memory theory | It combines exact tensor/parameter counts with a censored empirical fit and has no need to edit approximation proofs. |
| MRP-6C approximation/allocation | It owns conditional coupling and allocation results and explicitly consumes, rather than invents, MQAR evidence. |
| MRP-6D proof integration | Independent re-proof and sole TeX ownership prevents proof authors from self-approving manuscript statements and avoids concurrent TeX conflicts. |
| MRP-7 synthesis | Final narrative, tables, figures, and bundle require all claims to be frozen; it must not run concurrently with formal TeX integration. |

## Concurrency Waves

### Wave A: current

- MRP-1 continues remotely with no polling or source changes.
- MRP-0 may be implemented and tested locally.
- MRP-6A and the analytic portion of MRP-6B may proceed in separate memo files.

### Wave B: after MRP-0/MRP-1 and formal-foundation gates

- MRP-2 and MRP-3 may run concurrently because their write sets are disjoint.
- MRP-6B completes its empirical fit.
- MRP-6C starts only after MRP-6A and analytic MRP-6B pass, then waits for the
  MRP-3 interpretation gate before final empirical interpretation.

### Wave C

- MRP-4 runs only if its deterministic trigger fires.
- MRP-5 runs after the frozen `b*`, MRP-2 completion, and MRP-3 review.
- MRP-6D starts only after all three proof memos pass.

### Wave D

- MRP-7 performs final paper and reproducibility integration.

## Shared-File Serialization

| Shared surface | Sole owner before final integration |
|---|---|
| runner/checkpoints/logger/config task semantics | MRP-0 |
| generic ordered WikiText-2 interface | MRP-0 |
| natural AR definitions/evaluator | MRP-2 |
| synthetic MQAR generator/trainer | MRP-3 |
| PG-19 adapter and fixed tokenizer | MRP-5 |
| canonical formal TeX | MRP-6D |
| non-theory final TeX and bundle | MRP-7 |
| live process tracker and main-plan status | current program-integration/experiment-operations role only |

MRP-2 and MRP-5 do not both edit `src/data/wikitext2.py`; both consume the
MRP-0 ordered-text contract. MRP-6A/B/C do not edit the canonical TeX.
Concurrent task agents record state in their subplans/audits; the coordinator
serializes those transitions into the shared tracker.

## Determinism Audit

Every conditional has a fixed resolution:

- MRP-1 freezes `b*` by minimum mean PPL at `L2048/B4` before later outcomes.
- MRP-2 uses a fixed support threshold and does not enlarge its dataset.
- MRP-3 selects batch by descending supported preflight and uses a registered
  interaction statistic and CI.
- MRP-4 runs one fixed `(16,8)` sensitivity row only when primary MQAR has
  adequate support/accuracy but a null interaction; otherwise it becomes
  `NOT_TRIGGERED`. No topology search follows.
- MRP-5 uses one tokenizer, one token budget, four fixed rows, and no PG-19 blur
  tuning.
- MRP-6 statements are assigned to exact, conditional, or empirical tiers.

There is no “choose whichever looks best” instruction.

Every future training/prefetch matrix also requires explicit user approval
recorded by the shared-tracker owner. A scientific trigger never authorizes a
launch by itself.

## Critical Risks And Controls

1. **Configured seed not applied.**
   MRP-0 blocks new runs; MRP-1 uses unpaired replicate statistics.
2. **No checkpoints.**
   MRP-0 defines the contract; MRP-2 performs a fixed targeted retrain if host
   inventory confirms none.
3. **Unified runner hardcodes WikiText-2.**
   MRP-3 remains standalone; MRP-5 owns the PG-19 adapter.
4. **Masked MQAR denominator bug.**
   MRP-0 and MRP-3 require hand-computed `-100` tests.
5. **Legacy theory mismatches current model.**
   MRP-6D begins from a defect ledger and independently re-proves every result.
6. **Distance/frequency conflation.**
   Main plan and MRP-3/6C explicitly prohibit it.
7. **OOM overclaim.**
   All plans state fixed-batch/hardware feasibility and censored treatment.
8. **Remote-document drift.**
   Local workspace remains canonical; remote propagation is a separate,
   explicit action and never inferred.

## Independent Review Corrections

An independent post-draft review found seven blocking issues. They were
resolved before this audit retained `PASS`:

1. removed the false row-wise-top-k term from the routing-matrix rank bound;
2. replaced per-stratum underidentified VRAM fits with one identifiable
   lizmark-B4 constrained fit and held-out other strata;
3. added explicit approval as a hard dependency for every future launch and
   prefetch;
4. blocked MRP-6C until MRP-6A and analytic MRP-6B pass;
5. assigned the shared tracker/main-plan write surface to one coordinator;
6. filled the complete durable handoff schema in every subplan and corrected
   MRP-0's stale next action;
7. removed legacy v3 conditional permissions from the current dev prompt.

## Residual Governance Limit

The new files are durable in the local workspace but are not protected by
version history until the user approves a commit. No commit or remote sync was
performed during this planning task.

## Verdict

The program is decomposed into disjoint end-to-end tasks with deterministic
gates, explicit write ownership, immediate/retrieval/archive memory, and a
durable handoff schema. MRP-0, MRP-6A, and analytic MRP-6B can proceed without
interfering with the running MRP-1 queues.
