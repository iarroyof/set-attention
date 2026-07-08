# MRP-6A: Formal Architecture, Causality, And Context Path

Status: PASS

Owner: MRP-6A theory worker

Updated: 2026-07-07.

## Mission

Replace the legacy one-bank/direct-residual formal model with a complete
mathematical definition of the executed multiresolution `anchor_span` model,
then prove its causal and contextual-path properties.

This agent writes a proof memo and validation tests. It does not edit the
canonical manuscript; MRP-6D owns TeX integration.

## Required Retrieval Context

Read all of:

1. `../set_dictionary_research_main_plan.md`
2. `../set_dictionary_model_provenance_for_math_agent.md`
3. `../revision_source_of_truth_definitions.md`
4. `../../configs/set_dictionary/sd9_multiresolution.yaml`
5. `../../src/models/set_only/banks.py`
6. `../../src/models/set_only/ska_block.py`
7. `../../src/models/set_only/router.py`
8. multiresolution construction and forward path in
   `../../src/models/set_only/set_only_lm.py`
9. current formal appendix in
   `../../out/final_paper_bundle/overleaf_ready/example_paper.tex`
10. causality tests under `../../tests/`

## Write Scope

- `docs/theory/multiresolution_formal_model.md` (new)
- focused formal/numerical tests under `tests/theory/`
- `audit/MRP_6A_formal_architecture.md`

## Formal Object Contract

Define:

1. resolution groups `g in G` and a disjoint head partition
   `U_g`, with `sum_g H_g=H`;
2. stream widths `D_g=D H_g/H` and the divisibility conditions enforced by
   code;
3. bank sizes
   `M_g=floor((L-w_g)/s_g)+1`, starts, endpoints, and sets;
4. group candidate fibers
   `C_t^g={m:t-w_g<e_m^g<=t}`;
5. pooling laws, pooled atoms, endpoint-local geometry/content descriptors;
6. independent causal set stacks `Phi_g`;
7. headwise routing laws and routed group vectors `r_t^g`;
8. concatenation `r_t=Concat_g r_t^g`, output projection `W_o`, and
   `span_t=W_o r_t`;
9. thin anchor `h_t^0=e(x_t)+p_t`;
10. logits
    `ell_t=W_lm(h_t^0+span_t)` and prediction of `x_{t+1}`.

State empty-fiber behavior and distinguish router heads from set-attention
heads. Define entropy and top-1 diagnostics group-locally; do not aggregate
probability vectors over incompatible atom spaces without a tagged
disjoint-union definition.

## Required Results

### A. Per-group bank lemmas

Correctly prove:

- empty-fiber characterization;
- exact/interior candidate counts;
- endpoint locality.

Repair the legacy proof error that incorrectly keeps the first endpoint in all
later fibers. Use the most recent stride-grid endpoint and the assumption
`s_g<=w_g`.

### B. Causal closure theorem

Prove by induction over each set-stack layer:

1. pooled atom `m` in group `g` depends only on tokens through endpoint
   `e_m^g`;
2. masked exact set attention preserves this endpoint measurability;
3. endpoint-window routing at token `t` consumes only atoms with
   `e_m^g<=t`;
4. concatenation, `W_o`, anchor addition, and the LM head preserve dependence
   on `x_<=t`.

Explicitly include hashed-count and geometry features in the measurability
argument. The conclusion is next-token causality, not strict exclusion of the
current input token.

### C. Contextual-path factorization theorem

For two histories sharing current token and position, prove:

```text
ell_t(x)-ell_t(x')
  = W_lm W_o (r_t(x)-r_t(x')).
```

Therefore every dependence on `x_<t` factors through the routed span. State
the observability limitation: softmax cannot observe span differences mapped
by `W_lm W_o` to a constant-logit vector. Do not use the unqualified term
“identifiable.”

### D. Multigroup routing-capacity theorem

Define the block-supported routing object over the tagged disjoint union of
group atom indices. Prove:

```text
rank(Pi_hat_t)
  = sum_g rank(Pi_t^g)
  <= sum_g min(H_g, C_t^g).
```

State product-simplex dimension as an admissible-assignment result. Do not
claim that finite-temperature dot-product parameters realize every boundary
Dirac law. Top-k limits each routing row's support and entropy, but row-wise
top-k sparsity does not by itself bound matrix rank by `k`.

### E. Span-ablation interpretation

Prove only that setting the span to zero removes all historical-context paths.
The remaining anchor still depends on the current token and position, so do
not call it a pure unigram model.

## Legacy Theorem Corrections Owned Here

- per-group entropy/top-k restatement;
- multigroup rank/dimension restatement;
- finite-temperature realization qualification;
- empty-fiber proof repair;
- removal of direct/empty-only residual assumptions from current-model
  statements.

Do not modify pooling maximum-entropy or gradient theorems; MRP-6D audits
those separately.

## Validation

1. Exhaustively enumerate small `(L,w,s)` banks and compare candidate formulas
   with `banks.py`.
2. Run existing and new future-token perturbation causality tests for b0, b*,
   and b100.
3. Verify group widths, bank sizes, and candidate counts against resolved
   runtime metadata.
4. Numerically verify the context-path logit identity on fixed checkpoints or
   initialized models with dropout disabled.
5. Verify block-rank bounds on generated routing matrices.

## Definition Of Done

The proof memo contains complete definitions, statements, assumptions, and
proofs; every code-correspondence claim has a source reference; all validation
tests pass; and the audit lists exactly which legacy theorem labels should be
replaced or retained.

## Durable Handoff

Status: PASS.

Last completed action: wrote the code-grounded multiresolution formal model
memo, focused CPU validation tests, and MRP-6A audit. Runtime validation passed
in the Blue `set-attention-dev:cu124` container.

Files changed: `docs/theory/multiresolution_formal_model.md`,
`tests/theory/test_mrp6a_formal_model.py`,
`audit/MRP_6A_formal_architecture.md`, and this subplan status/handoff only.

Commands/tests and outcomes: local `python -m py_compile
tests/theory/test_mrp6a_formal_model.py` passed. Local runtime pytest was
unavailable, so the focused theory suite was run in the Blue container:
`python -m pytest -q tests/theory/test_mrp6a_formal_model.py
tests/theory/test_mrp6b_memory_frontier.py` reported `11 passed, 2 warnings`.

Artifacts and digests: proof memo and audit listed above; no experiment
artifacts.

Host/PID/log/ETA: Blue-demon container validation completed; no training or
experiment launch.

Decision or gate result: MRP-6A proof memo package and focused validation
pass; ready for MRP-6D independent proof audit after MRP-6C is available.

Known incident or limitation: local Python validation environment is incomplete;
line-number citations were intentionally avoided because the workspace contains
unrelated dirty edits.

Next atomic action: start MRP-6C analytic approximation/allocation work or
promote this memo to MRP-6D once MRP-6C is also complete.

Inputs required: MRP-6C memo for final independent integration.
