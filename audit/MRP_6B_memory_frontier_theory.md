# MRP-6B Memory Frontier Theory Audit

Status: PASS; amended for the completed matrix and global dense-router LCA
recipe.  The optional empirical fit is deferred and is not used in the paper.

Updated: 2026-08-10.

## Scope Completed

Implemented the analytic MRP-6B deliverable in
`docs/theory/exact_dense_memory_frontier.md`.

The memo states:

- exact strict-past set counts for fine `(2,1)` and coarse `(4,2)` banks;
- full-square dense score tensor count
  `A_score = B K sum_g H_g M_g^2`;
- leading blur coefficients for b0, b25, b50, b75, and b100;
- finite monotonic score-count reduction when replacing a fine head by a
  coarse head;
- activation and router score scaling under endpoint-window candidate gather;
- exact implemented parameter formulas, including stream-width splits,
  hashed-count feature builders, linear content-bias adapters, learned routers,
  shared `d_ff`, and length-dependent position embeddings;
- concrete runtime/training parameter counts for registered lengths;
- peak-VRAM interpretation limits;
- MRP-1-compatible Pareto/frontier definitions;
- measured-VRAM interpretation and explicit separation of set-score and
  dense-router-score tensor counts.

## Validation

Focused CPU-local formula tests were added under `tests/theory/`:

- strict-past set counts;
- leading score coefficients;
- finite score-count monotonicity in coarse heads;
- registered parameter count table at `L=2048`;
- quadratic class preservation;
- fit-row rejection for cross-host/batch pooling and uncertified OOMs.

Blue container validation completed:

```text
python -m pytest -q \
  tests/theory/test_mrp6a_formal_model.py \
  tests/theory/test_mrp6b_memory_frontier.py

11 passed, 2 warnings in 0.72s
```

## Empirical Fit Status

The MRP-1 replacement rows are complete.  No constrained VRAM fit was run;
the paper uses measured within-island peaks and the analytic count only for
directional explanation.  The fit is therefore deferred rather than blocked.
Legacy OOM rows remain fixed-admission feasibility observations unless they
carry exclusive telemetry.

## Write-Scope Compliance

Edited only MRP-6B-owned files:

- `docs/theory/exact_dense_memory_frontier.md`;
- `tests/theory/test_mrp6b_memory_frontier.py`;
- `audit/MRP_6B_memory_frontier_theory.md`;
- `docs/agent_plans/mrp6b_memory_frontier_theory.md` status/handoff.

No canonical TeX, shared tracker, launcher, config, or experiment artifact was
edited. No experiments were launched.

## Handoff

Last completed action: analytic exact-dense memory/frontier memo and focused
formula tests added.

Files changed:
`docs/theory/exact_dense_memory_frontier.md`;
`tests/theory/test_mrp6b_memory_frontier.py`;
`audit/MRP_6B_memory_frontier_theory.md`;
`docs/agent_plans/mrp6b_memory_frontier_theory.md`.

Commands/tests and outcomes: local `python
tests/theory/test_mrp6b_memory_frontier.py` passed. Local pytest is not
installed, so the focused MRP-6A/6B pytest suite was run in the Blue
`set-attention-dev:cu124` container and reported `11 passed, 2 warnings`.

Artifacts and digests: no generated empirical artifacts.

Host/PID/log/ETA: no launched jobs.

Decision or gate result: MRP-6B passes for the current paper; no fitted VRAM
value is used as evidence.

Known incident or limitation: default local Python lacks project runtime
dependencies and pytest. Empirical fit is intentionally absent.

Next atomic action: none required.  Revisit the constrained fit only as an
optional model-checking appendix.

Inputs required: validated final MRP-1 lizmark `B=4` regular blur rows and
admission-certified OOM metadata, if any.
