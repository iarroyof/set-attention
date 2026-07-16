# MRP-6D: Independent Proof Audit And Manuscript Integration

Status: PASS; empirical specialization not established by null MRP-3

Owner: MRP-6D theory integration worker

Updated: 2026-07-15 after MRP-3 completed as null/inconclusive.

## Mission

Independently verify the new proof memos, remove or repair invalid legacy
results, and integrate one coherent current-model theory into the NeurIPS
manuscript and appendix.

This is the only theory workstream allowed to edit the canonical TeX.

## Required Retrieval Context

Read completely:

1. `../set_dictionary_research_main_plan.md`
2. all MRP-6A/B/C subplans and proof memos
3. all MRP-6A/B/C audits and tests
4. `../set_dictionary_model_provenance_for_math_agent.md`
5. `../../out/final_paper_bundle/overleaf_ready/example_paper.tex`
6. exact implementation files cited by every proposed theorem
7. MRP-1/2/3/5 audits available at integration time

## Write Scope

- `out/final_paper_bundle/overleaf_ready/example_paper.tex`
- theory-specific bibliography entries
- `audit/MRP_6D_theory_integration.md`
- compile/proof lint outputs under
  `out/final_paper_bundle/checks/`

Do not rewrite empirical tables owned by MRP-7 except where a theorem
cross-reference requires a placeholder.

## Mandatory Audit Before Editing

Re-prove or reject every proposed result without relying on the memo author's
conclusion. Produce a ledger with:

- theorem label and tier;
- code object represented;
- assumptions;
- proof status;
- empirical dependency;
- retained/repaired/removed legacy label;
- reviewer-facing claim allowed.

At minimum audit these known defects:

1. legacy detailed model is single-stream/direct-residual;
2. routing distributions from different group atom spaces were aggregated;
3. finite-temperature softmax was claimed to realize Dirac boundaries;
4. Boltzmann maximum-entropy theorem omitted moment feasibility and
   nonconstant-energy cases;
5. pooling gradients omitted overlapping sets, anchor, router query, and
   descriptor paths;
6. Jacobian sandwich assumed pooling was the only computation path;
7. mechanistic proof used Lipschitz assumptions on the wrong operator family;
8. pooling collapse corollary compared unweighted errors;
9. empty-fiber proof used the first endpoint incorrectly;
10. span ablation was called a unigram regime;
11. candidate-gather was described as future work;
12. b0/singleton limits were conflated with token attention.

## Legacy Label Disposition Ledger

Audit every label below even if the new appendix removes it:

| Legacy labels | Primary review owner | Required disposition |
|---|---|---|
| `prop:empty-strict-past-fibers`, `prop:interior-candidate-count` | MRP-6A | restate per group and repair proof |
| `thm:per-head-entropy`, `cor:normalized-head-entropy` | MRP-6A | retain per group with finite-temperature qualification |
| `thm:aggregated-vs-head-entropy`, `rem:entropy-gap-interpretation` | MRP-6A | remove cross-group form; retain only group-local version |
| `prop:topk-support-entropy` | MRP-6A | restate per group with `min(C_t^g,k)` |
| `thm:degenerate-topology`, `cor:stride-equals-window` | MRP-6A | restate per group and separate empty fibers |
| `prop:single-head-common-mixture`, `thm:strict-routing-inclusion`, `thm:routing-dimension-growth`, `thm:routing-rank-gain`, `cor:topology-limited-routing-capacity` | MRP-6A | replace with tagged direct-sum assignment/capacity results; distinguish admissible laws from parameter realizability |
| `thm:boltzmann-maxent`, `rem:soft-trimmed-boltzmann` | MRP-6D | repair moment-feasibility, boundary, and constant-energy cases; state relation to trimming exactly |
| `thm:gradient-decomposition-pooling`, `cor:direct-coupling-gradients` | MRP-6C/6D | retain only as explicitly named partial-branch derivatives or remove |
| `thm:raw-gradient-ratio-multiplicative`, `prop:logged-gradient-ratios`, `cor:bottleneck-localization` | MRP-6D | retain algebraic identity only where variables match logged graph objects; remove unsupported bottleneck causality |
| `thm:necessary-feasible-topology`, `cor:stride-less-than-window` | MRP-6A | restate as group-local nondegenerate-routing conditions, not quality conditions |
| `thm:mechanistic-decomposition`, `rem:mechanistic-bound-meaning` | MRP-6C | replace with multigroup coupling bound and assumptions on both operator families |
| `prop:effective-support-bounds`, `prop:pooling-concentration-regimes` | MRP-6D | retain exact probability identities; remove direct quality implications |
| `prop:jacobian-sandwich` | MRP-6D | remove unless rewritten for branch-isolated VJPs that include every active graph path |
| `prop:entropy-feasible-routing`, `prop:capacity-stability` | MRP-6A/6D | retain only structural/rank statements; remove empirical quality thresholds |
| `cor:loss-sensitivity-pooling` | MRP-6C/6D | remove or replace by correctly weighted conditional statement |
| `cor:topology-collapse-boundary` | MRP-6A | retain only per-group routing collapse conclusion |
| `def:feasible-operating-region`, `cor:joint-experimental-design`, `rem:what-is-proven` | MRP-6D | rewrite around the exact-dense multiresolution program and current nonclaims |

The four compact main-text legacy results
`thm:main-routing-entropy`, `thm:main-multihead-capacity`,
`prop:main-pooling-transport`, and `thm:main-feasible-region` are replaced only
after their appendix dependencies pass this ledger.

## Required Appendix Structure

1. Notation and dimensions.
2. Multiresolution bank construction.
3. Group-local pooling, features, set stacks, and routing.
4. Concatenated `anchor_span` readout.
5. Tier A exact results:
   - bank/candidate lemmas;
   - causality;
   - contextual-path factorization;
   - per-group entropy/top-k;
   - direct-sum routing capacity;
   - exact dense score-memory law.
6. Tier B conditional results:
   - routed approximation/coupling bound;
   - loss stability bound;
   - pooling collision/recovery boundary;
   - discrete interior allocation.
7. Tier C empirical propositions and explicitly labeled conjectures.
8. Proofs in full, not sketches.
9. Code/diagnostic correspondence table.
10. Limitations and nonclaims.

The theory section must be at least as complete as the legacy appendix, while
removing invalid volume rather than preserving theorem count.

## Main-Text Formal Results

Select at most four main-text statements:

1. causal contextual-path factorization;
2. direct-sum multigroup routing capacity;
3. exact-dense memory coefficient;
4. conditional interior-allocation/approximation result.

Each main statement points to a complete appendix proof. Do not place
empirical PPL or OOM conclusions inside theorem statements.

## Validation

- all MRP-6 tests pass;
- no unresolved `SD-audit` TODO remains;
- no undefined references/citations;
- theorem environments and labels are unique;
- dimensions match the current config and code;
- `pdflatex`/bibliography build succeeds from a clean compile directory;
- PDF text inspection confirms equations and proofs are present;
- every use of “linear,” “subquadratic,” “token equivalent,” “identifiable,”
  and “receptive field” passes a context audit.

## Definition Of Done

The proof ledger is complete, every retained theorem is independently
verified, invalid legacy results are removed or corrected, the appendix
formalizes the executed model, and the clean manuscript build passes.

## Durable Handoff

Status: PASS; empirical specialization not established by null MRP-3.

Last completed action: independently audited the MRP-6A/B/C proof packages,
replaced the canonical main theory and detailed proof appendix with the
current multiresolution `anchor_span` theory, and wrote the MRP-6D ledger.

Files changed: `out/final_paper_bundle/overleaf_ready/example_paper.tex`,
`audit/MRP_6D_theory_integration.md`, and this subplan handoff only.

Commands/tests and outcomes: see `audit/MRP_6D_theory_integration.md` for the
local test and TeX validation record.

Artifacts and digests: compile/proof lint outputs are under
`out/final_paper_bundle/checks/`.

Host/PID/log/ETA: local CPU/LaTeX validation only; no experiments launched.

Decision or gate result: MRP-6D canonical TeX integration is complete for the
analytic theory package. The completed MRP-3 result did not reach the support
regime, so mechanism specialization remains unestablished and must not be
stated as a positive empirical result.

Known incident or limitation: the older model-object appendix remains as
provenance for the broader Set Attention manuscript, but its legacy proposition
labels were renamed and the current theory appendix explicitly supersedes it
for set-dictionary claims.

Next atomic action: MRP-7 should perform final paper synthesis after the
remaining empirical gates close, preserving the MRP-6D nonclaims.

Inputs required: none for the analytic theory package. A future empirical
specialization statement would require a new approved mechanism result that
reaches its support regime.
