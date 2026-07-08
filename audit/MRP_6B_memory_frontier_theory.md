# MRP-6B Memory Frontier Theory Audit

Status: ANALYTIC PASS; empirical fit BLOCKED on MRP-1 replacement closure.

Updated: 2026-07-07.

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
- pending empirical fit and final-table replacement gates.

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

Not run and not finalized. The 15 `L3584,B4` mixed replacement rows are still
the registered blocker. Legacy `L4096,B4` token/b0/b25 OOM rows remain
observed feasibility outcomes only because their launchers did not archive
external-process telemetry.

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

Decision or gate result: analytic MRP-6B passes; final empirical fit remains
blocked on strict MRP-1 closure.

Known incident or limitation: default local Python lacks project runtime
dependencies and pytest. Empirical fit is intentionally absent.

Next atomic action: after MRP-1 closes, implement the constrained lizmark-B4
VRAM fit and replace pending sections with final fit tables.

Inputs required: validated final MRP-1 lizmark `B=4` regular blur rows and
admission-certified OOM metadata, if any.
