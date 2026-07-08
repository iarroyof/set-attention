# MRP-6C: Multiresolution Approximation And Interior Allocation

Status: ANALYTIC PASS; empirical interpretation BLOCKED on MRP-3

Owner: MRP-6C theory worker

Updated: 2026-07-07.

## Mission

Develop a conditional approximation theory that explains when mixed
resolutions can outperform uniform endpoints and a discrete allocation theorem
that explains why an interior head split can exist.

Do not assume that long-lag signals are temporally smooth or low frequency.
Do not claim that topology/rank alone predicts PPL or selects b25.

## Required Retrieval Context

1. `../set_dictionary_research_main_plan.md`
2. `mrp6a_formal_architecture.md`
3. `mrp6b_memory_frontier_theory.md`
4. `mrp3_mqar_mechanism.md`
5. `../set_dictionary_model_provenance_for_math_agent.md`
6. current mechanistic bound and assumptions in the canonical manuscript

## Write Scope

- `docs/theory/multiresolution_approximation_allocation.md` (new)
- numerical allocation, dimension, and bound checks under `tests/theory/`
- `audit/MRP_6C_multiresolution_approximation.md`

Do not edit the canonical TeX.

## Required Definitions

For each group define:

- a reference contextual target or reference atom family;
- an explicit atom correspondence/coupling between the reference and group
  bank;
- atom approximation error `epsilon_g`;
- routing-law transport error `xi_g`;
- atom norm bound `B_g`;
- downstream group projection and concatenation operators;
- pooling distortion as a data-dependent within-set diameter or coupling
  error, not a temporal-smoothness assumption.

State which quantities are observable diagnostics and which are theoretical
proxies.

## Required Results

### A. Per-group routed approximation lemma

Under the registered coupling and bounded-atom assumptions, bound each routed
group error by atom error plus routing transport, for example:

```text
||Delta r_t^g|| <= epsilon_g + 2 B_g xi_g.
```

Specify the probability metric used for `xi_g` and prove the constant.

### B. Concatenated span bound

Using the direct-sum head geometry, prove a bound of the form:

```text
||Delta span_t||
 <= ||W_o|| [
      sum_g H_g (epsilon_g + 2 B_g xi_g)^2
    ]^(1/2).
```

Then derive a conditional loss bound under an explicit local Lipschitz
constant for the LM readout. Include assumptions on both model and reference
operator families; repair the legacy proof's one-sided Lipschitz use.

### C. Pooling collision and token-recovery boundary

Prove:

1. if two histories with the same current token produce identical
   multigroup dictionaries, every deterministic downstream stack/router gives
   the same span and cannot separate the histories;
2. dimension compression of the continuous pre-stack map rules out a global
   continuous left inverse on an open set when output dimension is lower;
3. this does not prove collisions on the discrete vocabulary domain;
4. `M=L` or `(w,s)=(1,1)` alone does not imply token-attention equivalence.

For the registered 6-fine/2-coarse SD9 split, verify the continuous pre-stack
dimension calculation and its assumptions.

### D. Discrete interior-allocation theorem

Let `n` be coarse heads and define:

```text
A(n)=(H-n)M_f^2+nM_c^2
E(n)=E_f(H-n)+E_c(n)+E_int(n).
```

Prove:

- `A(n)` is strictly decreasing;
- under discrete diminishing returns and a sign change in marginal quality
  gains, an interior minimizer of `E(n)` exists;
- a mixed allocation is Pareto-better than all-fine exactly when it has no
  greater error and strictly lower memory, or strictly lower error and no
  greater memory;
- rank ceilings can distinguish mixed from uniform allocations but cannot
  select among b25/b50/b75 when their registered ceiling is equal.

State blur-optimum movement with `L` only as a conjecture unless dense
multi-length evidence establishes it.

### E. MQAR interpretation

Use MRP-3 only to test whether coarse contextualized atoms preserve distant
discrete associations and whether ablation effects vary by lag. Do not call
this validation of a high-frequency/slow-signal language decomposition.

If MRP-3 is null, retain the conditional theorems and state that their
specialization premises were not empirically established.

## Legacy Theory Audit Owned Here

Explicitly assess:

- the current mechanistic decomposition theorem;
- pooling-gradient formulas that omit overlapping-set and anchor/router paths;
- the Jacobian sandwich that assumes pooling is the only path;
- the unsupported inference from low effective support to large approximation
  error;
- token-limit/convergence statements.

Classify each as retain, repair with a named partial derivative, or remove.

## Definition Of Done

Every theorem has explicit domains, assumptions, and complete proof; no
distance/frequency conflation remains; allocation claims stop at existence and
Pareto conditions; and the audit maps each result to required diagnostics and
legacy theorem replacements.

## Durable Handoff

Status: ANALYTIC PASS; empirical interpretation BLOCKED on MRP-3.

Last completed action: completed the conditional approximation/allocation
proof memo, focused CPU-local test file, MRP-6C audit, and local validation.

Files changed: `docs/theory/multiresolution_approximation_allocation.md`,
`tests/theory/test_mrp6c_approximation_allocation.py`,
`audit/MRP_6C_multiresolution_approximation.md`, and this subplan
status/handoff only.

Commands/tests and outcomes: `python
tests/theory/test_mrp6c_approximation_allocation.py` passed 7 focused checks;
`python -m py_compile tests/theory/test_mrp6c_approximation_allocation.py`
passed; Blue container pytest for
`tests/theory/test_mrp6c_approximation_allocation.py` passed.

Artifacts and digests: no experiment artifacts.

Host/PID/log/ETA: none.

Decision or gate result: analytic MRP-6C proof package passes and is ready for
MRP-6D independent proof audit; empirical interpretation remains blocked on
MRP-3.

Known incident or limitation: MRP-3 lag evidence, when available, cannot be
interpreted as signal frequency or smoothness; the analytic memo remains
conditional.

Next atomic action: MRP-6D should independently audit MRP-6A/B/C and integrate
accepted formal results after respecting its own write scope.

Inputs required: validated MRP-3 audit for empirical specialization; no
experiment input required for the analytic package.
