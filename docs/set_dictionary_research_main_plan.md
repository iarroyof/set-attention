# Multiresolution Set-Dictionary Research Program

Status: ACTIVE; MRP-1 corrected exact-dense five-seed matrix is REPLACEMENT RUNNING.

Updated: 2026-07-07.

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
| MRP-1 | Five-seed matrix closure | REPLACEMENT RUNNING | Blue 120/120 complete; Lizmark first pass complete; 15 L3584/B4 mixed diagnostic replacements launched; `b*=b25` frozen | `agent_plans/mrp1_matrix_closure.md` |
| MRP-2 | Natural AR-hit evaluation | BLOCKED | `b*=b25` is frozen; MRP-1 completion and explicit launch approval remain | `agent_plans/mrp2_natural_ar_hits.md` |
| MRP-3 | Synthetic MQAR mechanism study | generator/trainer infrastructure READY; launches BLOCKED | MRP-1 completion and explicit approval for launches | `agent_plans/mrp3_mqar_mechanism.md` |
| MRP-4 | Scale-separation sensitivity | HOLD, deterministic trigger only | MRP-3 trigger and separate explicit launch approval | `agent_plans/mrp4_scale_separation.md` |
| MRP-5 | Tokenizer-matched WT2/PG-19 transfer | BLOCKED | MRP-1 frozen `b*`, MRP-2 complete, MRP-3 reviewed, explicit launch approval recorded | `agent_plans/mrp5_pg19_transfer.md` |
| MRP-6A | Formal architecture, causality, context path | PASS | code-faithful memo and focused tests passed in Blue container | `agent_plans/mrp6a_formal_architecture.md` |
| MRP-6B | Exact-dense memory/frontier theory | analytic PASS; final fit awaits MRP-1 | analytic memo and focused tests passed in Blue container; empirical matrix artifacts pending | `agent_plans/mrp6b_memory_frontier_theory.md` |
| MRP-6C | Approximation and allocation theory | analytic PASS; empirical interpretation BLOCKED | empirical interpretation awaits MRP-3 | `agent_plans/mrp6c_multiresolution_approximation.md` |
| MRP-6D | Independent proof audit and TeX integration | PASS | canonical TeX integration and clean build passed; empirical specialization text remains conditional until MRP-3 | `agent_plans/mrp6d_theory_integration.md` |
| MRP-7 | Final paper synthesis and reproducibility bundle | BLOCKED | MRP-1/2/3/5 and MRP-6D resolved; MRP-4 pass or not-triggered | `agent_plans/mrp7_paper_synthesis.md` |

Agents own disjoint task files until integration. MRP-6A/B/C write proof memos and
tests, not the canonical TeX. Only MRP-6D edits the formal appendix. MRP-7 edits
the remaining narrative, tables, figures, and final bundle after MRP-6D.

`audit/phase_sd_status.md` and this main plan have one write owner: the MRP-1
experiment-operations role while paper5 runs. Task agents update only their
subplan and task audit, then provide a handoff for that owner to fold into the
shared tracker. After MRP-1 closes, a replacement program-integration owner
must be explicitly named in the tracker before another agent edits either
shared file.

## 6. Deterministic Execution Sequence

1. Let the Lizmark `L3584,B4` replacement wave run under exclusive-GPU policy.
   Blue is complete and idle; Lizmark first pass is complete; the 15
   endpoint-diagnostic replacements were launched on 2026-07-07. Do not
   duplicate the driver or allow a competing CUDA workload during replacement
   rows.
2. MRP-0 passed Blue container validation on 2026-07-07. Run one-step
   full-shape preflights before any selected retraining launch. See
   `audit/MRP_0_reproducibility_platform.md`.
3. MRP-6A/B/C proof memos passed focused Blue-container tests, and MRP-6D
   independent proof audit plus canonical TeX integration passed clean build.
   Final empirical specialization wording still waits for MRP-3.
4. After the 15 registered `L3584,B4` mixed diagnostic-retry cells finish,
   perform one status/pull/validation pass. Close the matrix only after the
   strict scanner accepts the replacements. The registered `L=2048,B=4` rule
   has already frozen `b*=b25`, before any AR, MQAR, or PG-19 outcome.
5. Run MRP-2 and MRP-3 only after MRP-1 closes and the user approves their
   registered matrices.
6. Evaluate the MRP-3 gate. If support/accuracy prerequisites pass but the
   interaction CI is not positive, execute exactly MRP-4. If the interaction
   passes, or if support/accuracy is inadequate, mark MRP-4
   `NOT_TRIGGERED` with the registered reason. No other topology is introduced.
7. Run MRP-5 after MRP-2, reusing its AR-hit evaluator on both new
   tokenizer-matched corpora with no additional training rows.
8. Preserve the passed MRP-6D analytic theory integration. After MRP-3, add
   only the registered empirical-specialization interpretation allowed by its
   outcome.
9. MRP-7 performs the final claim/evidence audit and paper build.

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
