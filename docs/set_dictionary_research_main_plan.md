# Multiresolution Set-Dictionary Research Program

Status: ACTIVE; MRP-1, MRP-2, MRP-3, and MRP-lca-cmp are complete.  The
current paper integrates the local-language-model, associative-recall, and
global-aggregation evidence; the global-recipe natural-AR bridge is complete,
while MRP-5 transfer remains a separate follow-up.

Updated: 2026-08-12 after the natural-AR global-recipe bridge was verified.

This is the canonical program-level plan for the current
`set-dictionary/anchor-span` research direction. It supersedes future-work
instructions in the legacy SD ladder and old manuscript prose. It does not
authorize an experiment launch by itself. Live processes remain governed by
`audit/phase_sd_status.md`.

The decomposition and write-ownership audit is
`audit/set_dictionary_research_program_plan_audit_20260630.md`.

## 1. Research Objective

Establish whether exact-dense multiresolution set-dictionary attention provides
a reproducible quality/memory frontier, determine what its fine and coarse
streams actually contribute, test whether the result transfers beyond
WikiText-2, and replace the legacy single-stream theory with a code-faithful
formal account.

The intended contribution is:

1. an exact-dense, constant-factor memory allocation principle across set
   resolutions;
2. a causal set-mediated prediction path in which all historical context
   reaches the logits through the routed span;
3. a measured interior fine/coarse allocation frontier;
4. controlled evidence about natural and synthetic associative recall; and
5. a theorem/proof suite whose assumptions and conclusions match the executed
   model.

The work does **not** claim subquadratic complexity, universal superiority over
token attention, or token-attention equivalence of the all-fine `b0` row.

## 2. Canonical Precedence

When two documents disagree, use this order:

1. `docs/set_dictionary_research_main_plan.md` -- program scope, dependencies,
   and research gates.
2. `audit/phase_sd_status.md` -- live jobs, incidents, PIDs, logs, and current
   operational transition.
3. The active subplan under `docs/agent_plans/` -- exact task protocol and
   deliverables.
4. `docs/sd_dense_paper5_matrix.md` -- MRP-1 cells while that queue is active.
5. `docs/set_dictionary_model_provenance_for_math_agent.md` -- implemented
   forward path and code-grounded mathematical objects.
6. `docs/revision_source_of_truth_definitions.md` -- code-backed definitions.
7. `out/final_paper_bundle/overleaf_ready/example_paper.tex` -- current
   manuscript source, never launch authority.

`docs/ska_set_dictionary_revision_plan_v3_0.md`, the SD-9.x landmark audits,
Phase-A plans, and every file under `docs/archive/` are retrieval/provenance
material. They cannot override this plan.

## 3. Memory Model

### Immediate memory

Every continuing agent reads only the following before deciding what to do:

1. this main plan;
2. `audit/phase_sd_status.md`;
3. the one active subplan assigned to that agent;
4. `memory/set-attention-research-direction.md`;
5. the subplan's latest handoff or incident, if one exists.

This tier answers: what is active, what is blocked, what is the next atomic
action, and what must not be launched.

### Retrieval memory

Read only when required by the active subplan:

- environment and hosts: `set_attention_agent_onboarding.md` and
  `Context For Revision Agent after NeurIPS2026 LLM feedback.md`;
- experiment comparison rules:
  `memory/experiment-comparison-control-tuple.md`;
- active architecture:
  `configs/set_dictionary/sd9_multiresolution.yaml`,
  `docs/set_dictionary_model_provenance_for_math_agent.md`, and the cited code;
- current matrix history: `docs/sd_dense_matched_comparison_plan.md`,
  `audit/SD_dense_matched_results.md`, and
  `audit/SD_dense_frontier_extension.md`;
- theory and paper: the canonical manuscript and the MRP-6 proof memos;
- external protocols: the primary Zoology/MQAR and PG-19 sources recorded in
  the relevant subplans.

### Archive memory

Files under `docs/archive/`, `memory/archive/`, and audit files explicitly named
`legacy` are read-only provenance. No command, launch matrix, or current claim
may be recovered from them.

### Durable state rule

Chat history is never project state. Before stopping, every agent updates:

1. its subplan status block;
2. a task audit or incident file;
3. the exact next atomic action and required inputs;
4. a tracker-update handoff to the current shared-tracker owner when
   operational or program state changed.

No agent commits, pushes, or syncs remote source trees without explicit user
approval. Until such approval, the local workspace is canonical but lacks
version-history protection; record that limitation in the handoff.

## 4. Locked Architecture And Evidence Boundary

The primary model remains:

- exact dense backend for set and token comparisons;
- `D=384`, `d_ff=1536`, 6 layers, 8 heads;
- fine bank `(w,s)=(2,1)`;
- coarse bank `(w,s)=(4,2)`;
- blur rows `{b0,b25,b50,b75,b100}`;
- `output_residual_mode=anchor_span`;
- token MLP disabled, trained anchor disabled, CE only;
- `candidate_fiber=endpoint_window`;
- no re-read, `all_past`, multivector, landmark, sparse, or Nyström path.

Each resolution owns a separate bank, pool, set stack, and router. The routed
group outputs are concatenated and projected before the LM head. Both streams
have global causal receptive fields through their exact set-attention stacks.
Window and stride change atomization, pooling distortion, atom count, stream
width, and memory; they do not impose a fine-stream distance cutoff.

Consequences for claims:

- lag is not signal frequency;
- a long-lag MQAR result does not prove a slowly varying coarse language signal;
- `b0` is still a set model because `(w,s)=(2,1)`;
- the leading score-memory reduction remains inside `O(L^2)`;
- a token OOM is a fixed `(L,batch,hardware,implementation)` feasibility result,
  not proof that the context length is intrinsically unsupported.

## 5. Program Graph

| ID | End-to-end owner | Current state | Hard dependency | Canonical subplan |
|---|---|---|---|---|
| MRP-0 | Shared reproducibility platform | PASS | Blue container validation passed on 2026-07-07; run full-shape preflights before any selected retraining launch | `agent_plans/mrp0_reproducibility_platform.md` |
| MRP-1 | Five-seed matrix closure | PASS | Blue 120/120 and Lizmark 135/135 endpoint-valid; strict scan accepted 255 CSVs; later short-B3 bridge added 60/60 endpoint-valid descriptive rows; `b*=b25` remains frozen | `agent_plans/mrp1_matrix_closure.md` |
| MRP-2 | Natural AR-hit evaluation | COMPLETE; PROTOCOL PASS; SCIENTIFIC NULL | 12/12 registered local rows plus 12/12 global-recipe bridge rows; no set-specific repeated-bigram AR advantage under either recipe | `agent_plans/mrp2_natural_ar_hits.md` |
| MRP-3 | Synthetic MQAR mechanism study | COMPLETE; NULL/INCONCLUSIVE | all 18 registered rows completed and summarized; frozen `b25` accuracy mean `0.0002474`, below the `0.90` support threshold | `agent_plans/mrp3_mqar_mechanism.md` |
| MRP-4 | Scale-separation sensitivity | NOT_TRIGGERED | MRP-3 failed support condition 3, so scale-separation rows must not launch from this result | `agent_plans/mrp4_scale_separation.md` |
| MRP-lca-cmp | Long-context aggregation / compressed-memory comparison | COMPLETE | dropout-free b75 reaches L4096 endpoint parity at 13.7% lower peak VRAM; global routing is task-specific and degrades WikiText-2 LM in the reverse bridge | `agent_plans/mrp_lca_cmp.md` |
| MRP-5 | Tokenizer-matched WT2/PG-19 transfer | UNBLOCKED; LAUNCH APPROVAL REQUIRED | MRP-2 protocol is complete and null; transfer remains a separate experiment and is not authorized by this plan alone | `agent_plans/mrp5_pg19_transfer.md` |
| MRP-6A | Formal architecture, causality, context path | PASS | code-faithful memo and focused tests passed in Blue container | `agent_plans/mrp6a_formal_architecture.md` |
| MRP-6B | Exact-dense memory/frontier theory | PASS | analytic memo now counts exact set scores and global dense-router scores separately; paper comparisons use measured, not fitted, peak VRAM | `agent_plans/mrp6b_memory_frontier_theory.md` |
| MRP-6C | Approximation and allocation theory | analytic PASS; empirical specialization not established | MRP-3 completed but did not reach the support regime; keep the approximation/allocation theory conditional | `agent_plans/mrp6c_multiresolution_approximation.md` |
| MRP-6D | Independent proof audit and TeX integration | PASS; 2026-08-10 AMENDMENT | theory is nested under Model Overview and separately counts the two executed score tensors; empirical specialization remains task-scoped | `agent_plans/mrp6d_theory_integration.md` |
| MRP-7A | Legacy paper-content deprecation and current-model rewrite | ACTIVE; no new experiments | current Results now include MRP-1, MQAR, both natural-AR recipes, LCA, and the WikiText-2 reverse bridge; legacy sections remain excluded | `agent_plans/mrp7a_legacy_paper_content_deprecation.md` |
| MRP-7 | Final paper synthesis and reproducibility bundle | ACTIVE; FINAL CLOSEOUT PENDING | global-recipe AR evidence is integrated; final closeout still tracks the separately approval-gated MRP-5 transfer | `agent_plans/mrp7_paper_synthesis.md` |

Agents own disjoint task files until integration. MRP-6A/B/C write proof memos and
tests, not the canonical TeX. Only MRP-6D edits the formal appendix. MRP-7A may
edit current Results narrative/tables to remove deprecated legacy content and
replace it with MRP-1 exact-dense evidence. Full MRP-7 edits the final abstract,
conclusion, and reproducibility bundle from completed evidence.  Final closeout
must preserve the completed MRP-2/MRP-3 null interpretations, keep the
global-recipe natural-AR bridge separate from registered MRP-2 rows, and
identify MRP-5 transfer as a follow-up rather than completed evidence.

`audit/phase_sd_status.md` and this main plan have one write owner: the
program-integration/status worker named in the tracker. Task agents update only
their subplan and task audit, then provide a handoff for that owner to fold
into the shared tracker.

## 6. Deterministic Execution Sequence

1. MRP-1 is closed. Blue contributed 120 endpoint-valid rows and Lizmark
   contributed 135 endpoint-valid rows after the `L3584,B4` replacement wave.
   The strict scanner accepted 255 corrected CSVs on 2026-07-08. The later
   short-B3 bridge completed 60/60 additional endpoint-valid descriptive rows
   for `L512/B3` and `L1024/B3`; it reinforces the b25 interior pattern but
   does not reopen MRP-1, change `b*`, or authorize additional WT2 sweeps.
   Pause/release Lizmark until a new registered stage explicitly needs it.
2. MRP-0 passed Blue container validation on 2026-07-07. Run one-step
   full-shape preflights before any selected retraining launch. See
   `audit/MRP_0_reproducibility_platform.md`.
3. MRP-6A/B/C proof memos passed focused Blue-container tests, and MRP-6D
   independent proof audit plus canonical TeX integration passed clean build.
   MRP-3 did not establish the empirical specialization premises, so keep the
   theory conditional.
4. The registered `L=2048,B=4` rule froze `b*=b25` before any AR, MQAR, or
   PG-19 outcome. Use the final MRP-1 audit
   `audit/SD_dense_paper5_final_20260708.md` for paper matrix values.
5. MRP-2 checkpoint retraining and registered AR-hit evaluation are complete.
   Its literal sequence-block bootstrap gate is null under the local routing
   recipe. The separately labeled global-recipe bridge is also complete and
   finds no set-specific repeated-bigram advantage. Do not pool bridge rows
   with the registered matrix or interpret this proxy as proof that no internal
   retrieval computation exists.
6. MRP-3 completed all 18 registered primary rows but failed the support gate:
   frozen `b25` accuracy is near chance and far below `0.90`. Mark MRP-4
   `NOT_TRIGGERED` because the primary task is inadequate; do not launch
   scale-separation rows from this result.
7. MRP-lca-cmp is complete. Its LCA result uses the separately declared global
   routing recipe and must not be pooled with the WikiText-2 matrix; the reverse
   bridge establishes that the global recipe is not a universal replacement.
8. MRP-5 is now dependency-unblocked but still requires explicit transfer
   approval. If launched, reuse its AR-hit evaluator on both new
   tokenizer-matched corpora with no additional training rows.
9. Preserve the passed MRP-6D analytic theory integration. MRP-3 permits only a
   null/inconclusive mechanism-probe statement, not a positive specialization
   interpretation.
10. MRP-7A may proceed with the current Results rewrite around MRP-1, MQAR,
   both natural-AR recipes, LCA, and the WikiText-2 reverse bridge. It must not
   add a PG-19 result or turn the recipe-robust AR-proxy null into a claim that
   the architecture cannot perform retrieval.
11. MRP-7 performs the final claim/evidence audit and paper build.

No later step may be promoted because an earlier result looks promising.

## 7. Shared Statistical Contract

- Legacy MRP-1 rows are called **replicates**, not seeds. Their configured
  labels were not applied and they cannot support exact reruns or paired-seed
  inference.
- Corrected `sd_grid_seeded_v1` rows apply seeds `0..4` before stochastic
  construction, log the applied state, and satisfy the registered config and
  diagnostics contracts. Only these rows satisfy the five-seed closure
  requirement.
- MRP-1 summaries use corrected per-cell means, sample standard deviations,
  and 95% Student-t confidence intervals. Legacy summaries remain separately
  labeled sensitivity evidence and are never pooled with corrected rows.
- New MRP-2/3/5 runs apply and log the actual RNG seed, dataset fingerprint,
  tokenizer digest, checkpoint digest, and resolved config fingerprint.
- Every comparison holds the control tuple fixed:
  dataset/tokenizer, objective, architecture width/depth, backend, `L`, native
  batch, effective batch, optimizer, LR, warmup, training-token budget, seed,
  hardware class, and metric implementation.
- PPL comparisons never cross batch islands or tokenizers.
- PG-19 and tokenizer-matched WT2 are compared within dataset. Report
  bits-per-byte in addition to PPL.
- OOM is a censored feasibility observation. It is not assigned an artificial
  PPL or VRAM value.
- Every corrected Lizmark row requires no other CUDA workload at cell start
  and end. Any co-resident or occupancy-unknown attempt is quarantined and
  retried, and cannot support a paper metric or terminal OOM claim.
- Every summarizer uses word-boundary NaN/Inf detection and rejects limited,
  smoke, malformed-seed, metadata-incomplete, or wrong-backend rows.

## 8. Pre-registered Research Decisions

### Natural associative recall

MRP-2 is a reduced-scale adaptation of the Zoology AR-hit analysis, not a
reproduction of its 10B-token Pile experiment. It evaluates repeated
in-context bigram completions and reports the full distribution over training
bigram frequency and lag. No Pile pretraining claim is made.

### Synthetic MQAR

MRP-3 trains directly on generated MQAR sequences. Pile pretraining is not
required. The primary mechanism claim is a group-by-lag ablation interaction,
not a frequency decomposition of language.

### Cross-corpus transfer

MRP-5 uses one fixed public byte-level subword tokenizer and exactly four rows:
token, b0, frozen `b*`, and b100. Blur is not tuned on PG-19. The training
budget is the number of next-token targets consumed by ten tokenizer-matched
WikiText-2 passes.

### Theory

Formal claims are separated into:

- Tier A: exact structural identities and causality results;
- Tier B: conditional approximation/stability results with explicit
  assumptions;
- Tier C: empirical propositions and conjectures.

No rank, entropy, or topology theorem may be described as a perplexity or
generalization theorem.

## 9. Program-Level Definition Of Done

The program is complete only when:

1. MRP-1 has a validated applied-five-seed exact-dense audit;
2. the RNG/checkpoint provenance defect is fixed for every new run;
3. AR-hit and MQAR results are either conclusive under their registered support
   criteria or explicitly recorded as null/inconclusive;
4. PG-19 transfer is complete under the frozen protocol;
5. every main-text claim maps to a validated table, figure, theorem, or
   limitation;
6. the appendix defines the executed multiresolution `anchor_span` model and
   contains complete, independently audited proofs;
7. legacy landmark/sparse/direct-residual claims are removed or clearly
   historical;
8. the paper compiles without undefined references and the reproducibility
   manifest records config, data, tokenizer, checkpoint, and artifact digests.

## 10. Restart Handoff Contract

Every subplan update ends with:

```text
Status:
Last completed action:
Files changed:
Commands/tests and outcomes:
Artifacts and digests:
Host/PID/log/ETA, if applicable:
Decision or gate result:
Known incident or limitation:
Next atomic action:
Inputs required for that action:
```

An agent that cannot fill these fields has not produced a durable handoff.
