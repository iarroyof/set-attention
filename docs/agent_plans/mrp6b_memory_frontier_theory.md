# MRP-6B: Exact-Dense Memory And Frontier Theory

Status: PASS for current MRP-1 empirical interpretation

Owner: MRP-6B theory worker

Updated: 2026-07-08.

## Mission

Prove the exact leading score-memory law for multiresolution dense set
attention, characterize its constant-factor blur dependence, and test its
predictions against measured VRAM and censored OOM boundaries.

This agent writes a proof memo, fit code, and audit. MRP-6D owns manuscript
integration.

## Required Retrieval Context

1. `../set_dictionary_research_main_plan.md`
2. `../set_dictionary_model_provenance_for_math_agent.md`
3. `../../src/set_attention/backends/dense_exact.py`
4. multiresolution stream construction in
   `../../src/models/set_only/set_only_lm.py`
5. `../../configs/set_dictionary/sd9_multiresolution.yaml`
6. `../../audit/SD_dense_matched_results.md`
7. `../../audit/SD_dense_frontier_extension.md`
8. final MRP-1 audit: `../../audit/SD_dense_paper5_final_20260708.md`

## Write Scope

- `docs/theory/exact_dense_memory_frontier.md` (new)
- `scripts/analyze_sd_memory_law.py` (new)
- focused formula/fit tests under `tests/theory/`
- machine-readable fit outputs under
  `out/paper_integrated_evidence/checks/`
- `audit/MRP_6B_memory_frontier_theory.md`

## Required Structural Results

### A. Exact bank and score counts

For group `g`, prove:

```text
M_g = floor((L-w_g)/s_g)+1
A_score = B K sum_g H_g M_g^2.
```

`A_score` counts elements in the materialized dense score tensors across
batch and layers. The implementation materializes full `M_g x M_g` matrices
before applying the causal mask; do not replace this by a triangular-storage
count.

For fine `(2,1)` and coarse `(4,2)`:

```text
M_f=L-1
M_c=floor((L-4)/2)+1
```

and for even `L`, `M_c=L/2-1`.

### B. Blur coefficient

With coarse-head fraction `p=H_c/H`, prove:

```text
A_score/(B K H L^2)
  = (1-p)/s_f^2 + p/s_c^2 + O(1/L)
  = 1 - 3p/4 + O(1/L).
```

List the leading coefficients for b0, b25, b50, b75, and b100. Prove that
replacing one fine head by one coarse head strictly decreases score count for
the registered lengths while preserving `Theta(L^2)`.

### C. Non-score terms and parameter accounting

Derive the exact blur dependence of:

- Q/K/V/output projection parameters `sum_g O(D_g^2)`;
- FFN parameters under the implemented shared `d_ff`;
- pooled/set-state activations `sum_g O(B M_g D_g)`;
- group feature, geometry, router, and output-projection terms.

Mixed streams have different `sum_g D_g^2` from uniform endpoints. Report
runtime inference/training parameter counts for every blur row. Do not describe
the sweep as parameter-identical if it is not.

### D. Feasible-length prediction

Separate theorem from empirical model. For a fixed hardware/native-batch
stratum use:

```text
V
  = alpha
    + beta_linear B K sum_g M_g D_g
    + beta_score B K sum_g H_g M_g^2
    + beta_param P(p)
    + error.
```

The theorem determines the quadratic score term only. Fit all coefficients
from successful **lizmark B4** regular-blur cells at
`L in {2048,3584,4096}`. `P(p)` is the exact runtime parameter count derived
above. Use B3, B16, and blue-demon cells only as held-out/descriptive checks;
do not fit separate underidentified coefficients to one-length strata.

From the fitted model, compute a predicted maximum feasible `L` under the
49-GiB lizmark budget with uncertainty. Test whether it correctly orders the
`L4096/B4` b0/b25 legacy OOM and b50/b75/b100 success boundary. Do not claim a
hard bracket or point-exact OOM prediction unless exclusive admission
telemetry exists for the OOM row.

Fit by constrained least squares with nonnegative coefficients. Estimate
uncertainty with 10,000 cell-level bootstrap resamples. Report percentile
intervals and leave-one-length-out error within lizmark B4, then report
B3/B16/blue residuals without refitting. A secondary sensitivity fit may use
the repeated legacy OOM rows as lower-bound inequalities, but it must state
that their launchers did not archive external-process telemetry. Only a
corrected exclusive OOM can constrain the primary fit.

## Empirical Analysis Rules

- use peak train VRAM, never smoke VRAM;
- keep B3/B4/B16 and host classes separate;
- include all successful regular blur rows;
- use only admission-certified exclusive OOMs as primary right-censored
  capacity observations; keep legacy OOMs in a labeled sensitivity analysis;
- report fitted residuals, leave-one-length-out prediction, and held-out-stratum residuals;
- do not fit landmark data;
- do not infer asymptotic speed or wall-clock improvement from VRAM alone.

## Tests

1. Formula counts match runtime tensor shapes for small groups.
2. Coefficients match exact finite-`L` ratios.
3. Score count is monotone in coarse heads.
4. Runtime parameter counts match analytic counts.
5. Fit code rejects cross-batch and cross-host pooling.
6. Only exclusive OOM cells enter the primary fit, and only as inequalities.
7. Synthetic known-coefficient data recovers the expected fit.

## Definition Of Done

The memo contains complete proofs and limitations, every regular blur row has
parameter/score counts, the calibrated model and censored checks are
reproducible, and the audit distinguishes exact complexity statements from
empirical VRAM predictions.

## Durable Handoff

Status: analytic work PASS; empirical fit BLOCKED on MRP-1.

Last completed action: finite-count exact-dense memory/frontier memo and
focused formula tests added for the active multiresolution branch.

Files changed: `docs/theory/exact_dense_memory_frontier.md`;
`tests/theory/test_mrp6b_memory_frontier.py`;
`audit/MRP_6B_memory_frontier_theory.md`; this subplan status/handoff.

Commands/tests and outcomes: local `python
tests/theory/test_mrp6b_memory_frontier.py` passed. Local pytest is not
installed, so the focused MRP-6A/6B pytest suite was run in the Blue
`set-attention-dev:cu124` container:
`python -m pytest -q tests/theory/test_mrp6a_formal_model.py
tests/theory/test_mrp6b_memory_frontier.py` reported `11 passed, 2 warnings`.

Artifacts and digests: no generated empirical artifacts.

Host/PID/log/ETA: none; no experiments launched.

Decision or gate result: analytic MRP-6B deliverable passes; empirical fit may
not start until strict MRP-1 closure.

Known incident or limitation: score count predicts a leading term, not total
VRAM or exact OOM; one-length strata are held out rather than independently
fit. Local default Python lacks project runtime dependencies and pytest.

Next atomic action: after MRP-1 closes, implement and run the constrained
lizmark-B4 VRAM fit and replace pending empirical sections.

Inputs required: validated lizmark-B4 MRP-1 table and any admission-certified
OOM metadata.
