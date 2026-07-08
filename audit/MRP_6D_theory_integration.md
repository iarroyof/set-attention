# MRP-6D Theory Integration Audit

Status: PASS with empirical specialization still conditional on MRP-3.

Updated: 2026-07-07.

## Scope

This audit covers the independent proof review and canonical TeX integration
for the current exact-dense multiresolution set-dictionary branch. No
experiments were launched. Empirical tables were not rewritten.

Canonical objects audited:

- strict-past endpoint banks and candidate fibers;
- independent fine/coarse banks, poolers, set stacks, features, and routers;
- candidate-gather routing with empty-fiber zero routes;
- `anchor_span` readout with token MLP and trained anchor disabled;
- exact dense set-attention score tensor materialization;
- conditional routed approximation and discrete allocation results.

## Proof Ledger

| Label or result | Tier | Code object | Assumptions | Status | Empirical dependency | Reviewer-facing claim |
|---|---|---|---|---|---|---|
| `prop:empty-strict-past-fibers`, `prop:interior-candidate-count` | A | `banks.py` strict-past endpoint bank | integer window/stride, group-local bank | Repaired | none | candidate counts are per group and boundary-aware |
| `thm:mrp6-causal-closure` | A | bank, feature, set stack, router, readout | deterministic forward path or fixed training randomness | Retained new | none | logits at `t` depend only on `x_<=t` |
| `thm:contextual-path-factorization` | A | `anchor_span` readout | same current token and position; token MLP/anchor disabled | Retained new | none | historical context factors through routed span |
| `thm:per-head-entropy`, `cor:normalized-head-entropy` | A | per-group router softmax | nonempty group fiber; fixed support for temperature monotonicity | Repaired | none | entropy/top-k bounds are group-local |
| `thm:aggregated-vs-head-entropy`, `rem:entropy-gap-interpretation` | A | group-local routing probabilities | common finite atom space inside one group | Repaired | none | aggregation valid only within group or tagged union |
| `prop:topk-support-entropy` | A | router top-k mask | top-k applied before softmax | Repaired | none | support bounded by `min(c_t^g,k)` |
| `thm:degenerate-topology`, `cor:stride-equals-window` | A | endpoint-window fibers | group-local singleton fiber | Repaired | none | singleton group fibers force deterministic routing |
| common-mixture / strict-inclusion / dimension / rank legacy labels | A | multigroup routing matrix | tagged disjoint union of group atoms | Replaced | none | direct-sum rank ceiling is `sum_g min(H_g,c_t^g)` |
| `thm:exact-dense-score-memory` | A | `DenseExactBackend.forward` | exact dense backend, full square scores | Retained new | none | blur changes score-memory coefficient inside quadratic family |
| `thm:boltzmann-maxent`, `rem:soft-trimmed-boltzmann` | A/C | pooling rule in `banks.py` | pure Gibbs only under feasible moment/fixed-energy constraint | Repaired | none | implemented pooling is soft-trimmed Gibbs reweighting |
| pooling-gradient labels | B | active full autograd graph | all branch paths would need to be named | Removed | none | no full-graph pooling-gradient theorem retained |
| raw/logged gradient ratio labels | C | diagnostic ratio definitions | variables match logged graph objects | Partially retained | diagnostics only | algebraic stabilizer identity only, no bottleneck causality |
| necessary feasible topology / stride-less-than-window | C | candidate count and routing capacity | group-local nondegeneracy | Repaired | empirical quality not implied | structural routing condition only |
| `thm:mechanistic-decomposition`, bound meaning | B | multigroup route and span | reference atoms, coupling, TV transport, norm bounds | Repaired | premises require diagnostics | conditional routed approximation and span bound |
| effective support / pooling concentration | C | pooling weights | finite probability law | Retained | quality implications empirical | exact identities only |
| `prop:jacobian-sandwich` | B | branch-isolated VJP, not active graph | all active graph paths named | Removed | none | not stated for active model |
| entropy feasible routing / capacity stability | C | routing diagnostics and ranks | structural thresholds only | Repaired | thresholds empirical | no quality threshold theorem |
| loss sensitivity to pooling | B | readout loss | local Lipschitz readout and reference assumptions | Repaired | constants empirical | conditional stability bound |
| topology collapse boundary | C | group-local singleton fibers | per-group condition | Repaired | none | collapse is per group |
| feasible region / joint design / what is proven | C | fixed control tuple | diagnostics and conditional constants | Rewritten | MRP-3 for specialization | diagnostic program definition and nonclaims |
| main-text four legacy statements | A/B/C | current model | see appendix proofs | Replaced | MRP-3 for allocation specialization | four allowed formal statements only |

## Required Defect Audit

- Single-stream/direct-residual model: removed from current formal claims;
  `anchor_span` equations now drive the main theory and appendix.
- Cross-group routing aggregation: rejected unless routed over tagged atoms.
- Finite-temperature Dirac realization: stated only as a limit.
- Boltzmann max-entropy: repaired as pure Gibbs/feasible-moment scope; trimming
  is a reweighting.
- Pooling gradients and Jacobian sandwich: removed for the active full graph.
- Mechanistic proof: replaced by multigroup coupling with explicit reference
  and loss-stability assumptions.
- Pooling collapse quality corollary: downgraded to exact probability
  identities plus conditional diagnostics.
- Empty-fiber proof: repaired with moving endpoint grid.
- Span ablation: described as removing historical-context paths, not unigram.
- Candidate gather: described as the current implemented router path.
- b0/singleton/token attention: explicit nonclaim retained.

## TeX Integration

Changed canonical TeX:

- replaced the main theory section with four formal statements:
  contextual-path factorization, direct-sum multigroup routing capacity,
  exact-dense memory coefficient, and conditional interior allocation;
- replaced the detailed theory appendix with the current multiresolution
  definitions, Tier A/B/C results, full proofs, code correspondence,
  limitations, and legacy-label disposition;
- renamed legacy model-object labels to avoid duplicate references while
  preserving the appendix provenance text;
- added `longtable` for the disposition table.

## Validation Record

Commands run locally:

```text
python tests/theory/test_mrp6b_memory_frontier.py
python tests/theory/test_mrp6c_approximation_allocation.py
python tests/theory/test_mrp6a_formal_model.py
python -m py_compile tests/theory/test_mrp6a_formal_model.py tests/theory/test_mrp6b_memory_frontier.py tests/theory/test_mrp6c_approximation_allocation.py
pdflatex -interaction=nonstopmode -halt-on-error -jobname=mrp6d_build -output-directory=../checks/mrp6d_tex example_paper.tex
pdftotext out/final_paper_bundle/checks/mrp6d_tex/mrp6d_build.pdf out/final_paper_bundle/checks/mrp6d_tex/mrp6d_build.txt
```

MRP-6A runtime test still depends on local `torch`; Blue-container pytest had
already passed for MRP-6A/B/C before this integration. The local MRP-6A runtime
attempt failed with `ImportError: cannot import name 'nn' from 'torch'`, matching
the earlier local-environment limitation.

Canonical TeX build status: PASS. The final clean-job PDF is
`out/final_paper_bundle/checks/mrp6d_tex/mrp6d_build.pdf`; the final log has
no undefined references and no unresolved citation warnings.

Forbidden/mis-scoped language audit:

- no unresolved `SD-audit` TODO remains;
- no unqualified `identifiable`, `receptive field`, or `token equivalent`
  language remains in the TeX;
- `subquadratic` appears only in explicit nonclaim sentences;
- `linear` appears as the existing landmark-backend family and linear-algebra
  terminology, not as a current subquadratic claim.

Remaining LaTeX warnings:

- one float specifier changed from `h` to `ht`;
- overfull boxes in the legacy-disposition longtable from long theorem-label
  strings;
- existing underfull layout warnings elsewhere in the manuscript.

## Known Limitations

The earlier model-object appendix still contains provenance for the older
single-resolution architecture. The current theory section explicitly
supersedes it for set-dictionary claims, and duplicate theorem labels were
renamed so reviewer-facing references resolve to the current proofs.

MRP-3-dependent specialization remains conditional. No empirical specialization,
PPL superiority, or OOM theorem is claimed.
