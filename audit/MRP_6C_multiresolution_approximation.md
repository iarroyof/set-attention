# MRP-6C Multiresolution Approximation Audit

Status: ANALYTIC PASS; empirical interpretation BLOCKED on MRP-3.

Updated: 2026-07-07.

## Scope

This audit covers the analytic MRP-6C memo package only. It does not edit the
canonical TeX, shared trackers, or experiment launch files, and it does not
claim empirical blur selection.

## Deliverables

- Proof memo:
  `docs/theory/multiresolution_approximation_allocation.md`
- Focused CPU-local checks:
  `tests/theory/test_mrp6c_approximation_allocation.py`
- Subplan status/handoff update:
  `docs/agent_plans/mrp6c_multiresolution_approximation.md`

## Result Map

| Requirement | Status | Evidence |
|---|---|---|
| Per-group routed approximation lemma | PASS | Memo Section 3 proves `||Delta r_t^{u,g}|| <= epsilon_g + 2 B_g xi_g` with `xi_g` as total variation and constant `2` from `||mu-rho||_1=2TV`. |
| Concatenated span/loss bound | PASS | Memo Section 4 proves the direct-sum bound and states two-family Lipschitz assumptions for implemented and reference readouts. |
| Pooling collision/token-recovery boundary | PASS | Memo Section 5 separates identical-dictionary non-recovery, continuous dimension compression, discrete-domain caveat, and token-equivalence caveat. |
| Registered b25 dimension check | PASS | Memo Section 5 and tests verify `288(L-1)+96(L/2-1)=336L-384 < 384L` for registered even lengths. |
| Discrete interior-allocation theorem | PASS | Memo Section 6 proves strict memory monotonicity, interior minimizer existence under diminishing returns plus sign change, and Pareto conditions. |
| Rank-ceiling limitation | PASS | Memo Section 6 shows b25/b50/b75 share ceiling `4`, so rank cannot select among them. |
| MQAR interpretation limits | PASS | Memo Section 7 limits MRP-3 to distant discrete-association preservation and lag-ablation tests, not frequency decomposition. |
| Legacy theorem disposition | PASS | Memo Section 8 classifies legacy statements as retain, repair, or remove. |

## Validation

Local lightweight command:

```text
python tests/theory/test_mrp6c_approximation_allocation.py
```

Expected coverage:

- total-variation constant in the routed approximation bound;
- direct-sum span bound arithmetic;
- exact score allocation monotonicity with coarse heads;
- registered b25 pre-stack dimension compression;
- interior minimizer existence under discrete diminishing returns;
- all-fine Pareto condition;
- rank-ceiling limitation for b25/b50/b75.

Blue container validation:

```text
python -m pytest -q tests/theory/test_mrp6c_approximation_allocation.py

7 passed
```

## Claim Boundaries

- The memo is conditional theory. It does not assert that natural language has
  a smooth/low-frequency coarse component.
- It does not infer b25 selection from rank, topology, or memory.
- It does not treat OOM as a theorem about intrinsic maximum context length.
- It does not claim `M=L` or `(w,s)=(1,1)` is token-attention equivalence.
- Empirical specialization remains blocked until MRP-3 evidence is available.

## Handoff

Last completed action: implemented the analytic MRP-6C proof memo, focused
CPU-local tests, and this audit.

Files changed: `docs/theory/multiresolution_approximation_allocation.md`,
`tests/theory/test_mrp6c_approximation_allocation.py`,
`audit/MRP_6C_multiresolution_approximation.md`, and the MRP-6C subplan
status/handoff only.

Commands/tests and outcomes: `python
tests/theory/test_mrp6c_approximation_allocation.py` passed 7 focused checks;
`python -m py_compile tests/theory/test_mrp6c_approximation_allocation.py`
passed; Blue container pytest for
`tests/theory/test_mrp6c_approximation_allocation.py` passed.

Artifacts and digests: no experiment artifacts.

Host/PID/log/ETA: none; no training or experiment launch.

Decision or gate result: analytic MRP-6C proof package passes local focused
validation and is ready for MRP-6D independent proof audit. Empirical
interpretation waits for MRP-3.

Known incident or limitation: local workspace contains unrelated dirty edits
owned by other workers; this package intentionally avoids shared trackers and
canonical TeX.

Next atomic action: MRP-6D should independently audit MRP-6A/B/C and integrate
accepted formal results after respecting its own write scope.

Inputs required: MRP-3 validated mechanism audit for empirical specialization.
