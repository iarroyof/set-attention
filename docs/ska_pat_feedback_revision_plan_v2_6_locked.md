# SKA Paper -- PAT Feedback Revision Plan (v2.7 LOCKED)

Status: locked for implementation and writing agents.

This plan supersedes the pasted v2.5 plan and the v2.6 locked text. It folds in the code-review fixes, resolves the two blocking architectural decisions, restores the reviewer-critical matched token-backend baseline controls, and names the context files that a new chat must consult before editing code or LaTeX.

## v2.7 Baseline-Control Amendment

The paper must include matched token-attention controls for the sparse and linear backend families. This is required for a complete reviewer-facing comparison.

Previously completed A2/A3/A4 set-family experiments remain valid and should not be discarded. They compare `baseline_token` dense attention against `SetDense`, `SetSparse`, and `SetLinear`. However, they do not isolate whether a difference comes from the set mechanism or from the backend choice. The revision therefore adds a required baseline-control extension:

- `baseline_dense_exact`: existing dense token Transformer baseline.
- `baseline_sparse_local_band`: token Transformer using the same `local_band` backend/radius as `SetSparse`, without set pooling/routing.
- `baseline_linear_landmark`: token Transformer using the same `landmark` backend/coverage as `SetLinear`, without set pooling/routing.
- `set_dense_exact`: SKA with exact set backend.
- `set_sparse_local_band`: SKA with local-band set backend.
- `set_linear_landmark`: SKA with landmark set backend.

This extension is additive. Continue/resume the already-running phases in order, but every final table/figure/handoff that compares backend families must include the two missing token controls or explicitly label itself as "SKA variants vs dense token baseline only".

Implementation note: `src/models/baseline_token/attention.py` already has code paths for `local_band` and `landmark`. Before launching control grids, verify these paths are active, causal, strict enough for AR LM, and use `landmark_coverage=0.25` for the landmark control.

New required extension tasks:

- **A2.4 Baseline-backend control grid.** Run `baseline_sparse_local_band` and `baseline_linear_landmark` for the LR-norm headline reference (`D=384,d_ff=1536,L=512,w=16,s=8,M=63`) across the same LR grid and seeds used for A2.3 family slice. Produce summary TSVs and manifest.
- **A3-control overlays.** For mechanism figures where backend attribution matters, run token sparse/linear controls for the matched sweep axis if feasible. Minimum required overlays: A3.1 fixed-stride window sweep for `baseline_sparse_local_band` and `baseline_linear_landmark`; optional for A3.2/A3.3 if compute permits.
- **A4-control long-context slice.** Run `baseline_sparse_local_band` and `baseline_linear_landmark` at `L=2048` with the same seed count and batch policy used for the completed long-context family slice. If either control OOMs or is unsupported, record the failure as an explicit result.
- **A4 convergence.** The convergence run should compare the selected best SKA variant against the corresponding matched token-backend control and the dense token baseline when budget permits. Do not mark locked-plan convergence complete until a `>=30` epoch artifact exists.
- **A5 final handoff.** The final handoff must state which results include matched backend controls and which historical tables are dense-baseline-only.

## Required Context Files

Open these first in any new chat/session:

- `docs/ska_pat_feedback_revision_plan_v2_6_locked.md` -- this locked plan, now with the v2.7 baseline-control amendment.
- `docs/revision_source_of_truth_definitions.md` -- code-backed source of truth for current definitions and values.
- `Context For Revision Agent after NeurIPS2026 LLM feedback.md` -- execution environment, blue-demon workflow, repo-sync rules, and causality finding.
- `docs/example_paper_working_agent.tex` -- current LaTeX working draft.
- `configs/hyperparameters.md` -- public config contract.

Do not use older pasted plans as authoritative after this file exists. Older plans are historical context only. If a status tracker says A4.2 is a "long-context slice", interpret that as the completed long-context slice; convergence remains pending unless a `>=30` epoch artifact exists.

## Locked Architectural Decisions

### D1: Token Residual Path Under Option 1

Decision: **R1 -- direct embedding residual**.

Implementation rule:

```text
r_t = routed set-stack output for token t
final_t = h_t^(0) + r_t
logits_t = LMHead(final_t)
```

For tokens with no candidates (`C_t = 0`), use `r_t = 0`, so `final_t = h_t^(0)`. Do not special-case warmup by bypassing the same final-state formula.

Rationale:

- Fixes the current-token access gap created by endpoint-only strict-past routing.
- Preserves causality.
- Keeps the inference/sealing story cleaner than per-query causal pooling.
- Requires only a small theory edit: token state depends on the prefix through two causal paths, direct `h_t^(0)` and routed set-stack context.

Writing consequence:

The dependence-set lemma must prove:

```text
final_t = h_t^(0) + routed_context_t depends only on tokens <= t.
```

Do not claim the routed path alone always contains token `t`; it generally does not.

### D2: Tail Boundary Policy

Decision: **T1 -- drop partial trailing windows**.

Implementation rule:

```python
starts = range(0, seq_len - window_size + 1, stride)
M = floor((L - w) / s) + 1
S_m = {a_m, ..., a_m + w - 1}
```

No clipped partial trailing windows in the autoregressive LM paper configuration.

For the LR-norm headline reference:

```text
L = 512, w = 16, s = 8
M = floor((512 - 16) / 8) + 1 = 63
```

For the long-context reference:

```text
L = 2048, w = 16, s = 8
M = floor((2048 - 16) / 8) + 1 = 255
```

Source-of-truth note:

`docs/revision_source_of_truth_definitions.md` describes the current pre-A1.1 implementation where `M = ceil(L/s)`. After A1.1 lands, update that file to state the post-fix autoregressive LM paper topology above.

## Option 1 Candidate Family

For token position `t` and set endpoint `e_m = max(S_m)`:

```math
\mathcal C_t = \{m : t-w < e_m \le t\}.
```

With T1, every active set has exactly `w` tokens and the candidate-count bound is clean:

```math
C_t \in \{\lfloor w/s \rfloor, \lceil w/s \rceil\}
```

for interior tokens after warmup and before the final uncovered suffix. Tokens not covered by any sealed endpoint use the residual path only (`r_t = 0`, `final_t = h_t^(0)`).

For `L=512, w=16, s=8, M=63`:

- endpoints are `15, 23, ..., 511`;
- after warmup, candidate count is bounded by `2`;
- the old clipped-tail violation disappears.

## Config / Hyperparameter Propagation Contract

The repo does **not** use pydantic for the main experiment configs. The active path is:

```text
configs/*.yaml
  -> src/config/normalize.py::normalize_config
  -> src/config/schema.py::validate_config
  -> src/config/compatibility.py::validate_compatibility
  -> scripts/run_experiment.py
  -> model constructors
  -> submodule constructors
  -> src/train/experiment_logger.py
```

Every new diagnostic, input config, or hyperparameter must follow this path:

1. Add exactly one canonical YAML key.
2. Default it in `src/config/normalize.py`.
3. Allow it in `src/config/schema.py`.
4. Range-check or semantically validate it in `src/config/compatibility.py`.
5. Thread it through `scripts/run_experiment.py` and model constructors.
6. Thread it into the actual submodule constructor.
7. Log the resolved runtime value in run metadata.
8. Update `configs/hyperparameters.md` and `docs/revision_source_of_truth_definitions.md`.
9. Add a focused default-preservation or analytical test.

Canonical keys to expose:

| Group | Key | Default | Notes |
| --- | --- | --- | --- |
| `model.pooling` | `alpha` | `10.0` | trim sharpness |
| `model.pooling` | `learnable_alpha` | `false` | verify existing default |
| `model.feature_params` | `hash_seed` | `13` | hashed-count seed |
| `model.feature_params` | `normalize` | `true` | hashed-count normalization |
| `model.feature_params` | `num_bins` | `128` | hash bins |
| `model.router` | `min_temp` | `0.5` | learned-router temperature floor |
| `model` | `d_phi` | `null` | `null -> d_model` in normalization/resolution |
| `model` | `adapter_type` | `"auto"` | log resolved adapter type |
| `model.backend_params` | `landmark_coverage` | `0.25` | landmark backend only |
| `model.backend_params` | `radius` | existing | local-band backend |

Replace any stale wording that says defaults live in `schema.py`. Defaults live in `normalize.py`; `schema.py` is the allow-list/shape gate.

## Diagnostics Propagation Contract

For any new diagnostic:

1. Compute it in the relevant diagnostics module, primarily `src/models/set_only/diagnostics.py`.
2. Use the current Option-1 candidate fiber/mask, not legacy `t in S_m` membership.
3. Add the metric key to `src/train/metrics_schema.py`.
4. Confirm `src/train/loop.py` collects it through the existing `model.diagnostics` path.
5. Log raw and normalized variants when applicable.
6. Add an analytical test:
   - uniform logits -> normalized entropy `1.0`;
   - single-mode logits -> top-1 `1.0`;
   - candidate counts match `build_window_bank` exactly.

## Landmark Backend Remediation

The landmark backend is the only linear-family backend used in this revision cycle.

Current defect:

```python
stride = max(m // num_landmarks, 1)
idx = torch.arange(0, m, stride)[:num_landmarks]
```

This creates stride-dependent coverage and can leave a late-bank tail unselected.

Locked replacement:

```python
def _select_landmarks(self, m, device):
    if self.num_landmarks >= m:
        return torch.arange(m, device=device)
    K = self.num_landmarks
    idx = torch.tensor(
        [round(i * (m - 1) / (K - 1)) for i in range(K)],
        device=device,
        dtype=torch.long,
    )
    return idx
```

Coverage parameter:

```text
rho = model.backend_params.landmark_coverage
K = max(round(rho * M), 2)
```

Canonical examples:

- `M=63, rho=0.25 -> K=16`, indices from `round(i*62/15)`: `[0, 4, 8, 12, 17, 21, 25, 29, 33, 37, 41, 45, 50, 54, 58, 62]`.
- Historical v2.5 example before T1: `M=64, rho=0.25 -> K=16`, indices `[0, 4, 8, 13, 17, 21, 25, 29, 34, 38, 42, 46, 50, 55, 59, 63]`.
- Do **not** write `{0,4,8,...,60}` anywhere.

Do not globally delete `num_landmarks` from all code paths until the deprecated Nyström handling is decided and implemented. For the active landmark backend, migrate configs to `landmark_coverage`.

## Nyström Backend Policy

Decision: deprecated and excluded from this revision cycle.

Implementation:

- `src/config/compatibility.py` rejects `backend: nystrom` for active experiment configs with a clear error.
- `src/set_attention/backends/nystrom.py` should hard-fail construction with `RuntimeError("NystromBackend is deprecated for this revision cycle; use landmark backend.")`.
- Move active YAMLs using `backend: nystrom` to `configs/_deprecated/`.
- Do not delete the backend file in this revision.

Do not describe Nyström as a tested backend in the paper.

## Reference Recipe Terminology

Use these labels exactly:

- **LR-norm headline reference**: `D=384, d_ff=1536, w=16, s=8`. After T1, `L=512 -> M=63`; `L=2048 -> M=255`.
- **Anchor topology reference**: `D=384, d_ff=1536, w=16, s=4`. After T1, `L=512 -> M=125`.

Do not write bare "reference recipe" in implementation tasks or paper tables. Name the family.

## Phase A -- Code / Empirical Agent

### A0 Preflight

Run on blue-demon before edits:

- environment and container fingerprint;
- baseline tests;
- smoke training;
- resolved-config snapshot;
- deterministic eval-mode forward fingerprint;
- import sanity check.

Use blue-demon Docker Compose for anything importing `torch`, project model modules, datasets, CUDA, or training utilities.

### A1 Architectural Correctness

Required order:

1. **A1.1 Option-1 bank and residual implementation**
   - Drop partial trailing windows.
   - Build candidate fiber from endpoints `t-w < e_m <= t`.
   - Add direct residual `final_t = h_t^(0) + r_t`.
   - Tests in `tests/test_banks_option1.py`.
2. **A1.2 Causality tests**
   - Perturb future inputs and verify `final_t` is invariant for all `t`.
   - Run dense, sparse, landmark.
3. **A1.3 Reconcile App D.2 vs Table 8**
   - Diff configs; reproduce or drop App D.2.
4. **A1.4 Hyperparameter exposure**
   - Follow the config propagation contract above.
   - Default-preservation check is deterministic eval-mode forward fingerprint identity, not training bitwise identity.
5. **A1.5 Cache fingerprint / W&B step checks**
6. **A1.6 Landmark remediation**
   - Coverage parameter and linspace-rounded anchored indices.
7. **A1.7 Diagnostics rewire**
   - Use Option-1 candidate fiber.
8. **A1.8 Documentation scan/update**
   - Update `configs/hyperparameters.md`, `docs/revision_source_of_truth_definitions.md`, YAMLs, README/docs, tests.
9. **A1.10 Nyström deprecation**
   - Hard reject active use.
10. **A1.9 Audit and smoke gate**
   - Runs after A1.1-A1.8 and A1.10.

Go/no-go: do not launch A2 until A1.9 passes.

### A2 Multi-Seed Reruns

- A2.0 pre-launch smoke gate at full 10 epochs for dense baseline + dense SKA.
- A2.1 stability table at the anchor topology reference.
- A2.2 LR-normalized matched grid at LR-norm headline reference.
- A2.3 family slice for baseline-token, SetDense, SetSparse, SetLinear.
- A2.4 matched token-backend baseline-control grid for `baseline_sparse_local_band` and `baseline_linear_landmark`, using the same LR/seed/reference setup as the A2.3 family slice.

Linear-family runs use landmark backend with `landmark_coverage=0.25`.

### A3 Mechanism Sweeps

- Window-size sweep at fixed stride.
- Pooling-temperature sweep.
- Stride sweep demoted to complement and captioned as confounded by `M`.
- Baseline-control overlays for sparse/linear token baselines where attribution matters. Minimum required: A3.1 window sweep overlays for `baseline_sparse_local_band` and `baseline_linear_landmark`.

### A4 Scale-Up Evidence

- Long-context smoke at `L=2048`.
- Long-context slice at LR-norm headline reference.
- Convergence run to at least 30 epochs.
- Long-context matched token-backend controls for `baseline_sparse_local_band` and `baseline_linear_landmark`.
- Optional MQAR / associative recall.

### A5 Reproducibility Handoff

Produce TSVs and metadata with:

```text
seed, LR, D, d_ff, family, backend, w, s, M,
resolved_d_phi, resolved_adapter_type, pooling_alpha,
hash_seed, hash_normalize, router_min_temp,
landmark_coverage, landmark_count, source_csv_sha256
```

Final audit file: `audit/A5_4_handoff.md`.

The A5 handoff must include an explicit "matched controls coverage" section listing which final tables/figures include `baseline_sparse_local_band` and `baseline_linear_landmark`, and which artifacts predate the v2.7 amendment.

## Phase B -- Writing / LaTeX Agent

Use labels/section context, not raw line/equation numbers.

### B1 Theory Corrections

- Fix Theorem B.18 / Cor B.19 vector-calculus layout with explicit outer product.
- Disambiguate path-specific vs full gradient.
- Update candidate-count proposition for Option 1 with T1.
- Add general-position assumption to Theorem B.11.
- Broaden B.30 Lipschitz assumptions to model and reference operators.
- Loss normalization: align paper to code. The dataloader uses pre-shifted chunks (`x=chunk[:-1]`, `y=chunk[1:]`); the training loss averages over all `L` label positions. Do not mention BOS unless the implementation actually adds one.
- Clean notation: `A^(h)` vs `\widetilde A^(h)`, `X` vs `\mathbf X`, `local_band` vs `local-band`.

### B2 Definitions

- Hashed-count features with `G_hash=128`, `hash_seed=13`, normalized counts.
- Geometry bias `G` with `gamma=1.0`, `beta=0.0`.
- Content bias from linear adapter:
  ```math
  B^{(h)}_{ij} = <A^{(h)} phi_i, B_param^{(h)} phi_j>
  ```
- Numerical values: `H_r=8`, `d_phi=384`, `alpha=10.0`, `tau_min=0.5`, hash settings.
- Landmark rule: coverage fraction plus linspace-rounded anchored indices. For post-T1 LR-norm headline at `L=512`, use `M=63`, `K=16`, indices `[0, 4, 8, 12, 17, 21, 25, 29, 33, 37, 41, 45, 50, 54, 58, 62]`.
- Router top-1 metric is head-averaged probability concentration, not modal head agreement.
- Rename hash-bin count to `G_hash`; reserve `G` for geometry bias.

### B3 Causality Reframe

Define Option-1 candidate family and residual path in Section 3.

Lemma:

```text
final_t = h_t^(0) + routed_context_t depends only on tokens <= t.
```

Proof:

- `h_t^(0)` depends only on token `t`;
- every selected set has endpoint `<= t`;
- inter-set causal mask only allows dependencies from earlier/equal set indices;
- therefore routed context is prefix-causal;
- sum of causal direct and routed paths is causal.

### B4 References / Checklist / Figures

Populate missing references, checklist, limitations, and figure legibility.

### B5 Results

Replace tables and figures only after A2-A4 artifacts pass A5.4.

### B6 Hierarchical Causality Appendix

Frame inference cache claims as **proposed incremental decoding design**, not current implementation.

Allowed wording:

```text
The architecture admits an incremental-decoding implementation in which sealed set-stack states are cached per set; the present code recomputes the full bank and set stack during training/evaluation.
```

Do not write that current code already maintains a KV cache or avoids recomputation.

Section 1 / Section 3 paragraph must not claim the routed path alone spans the entire prefix including current token. Say:

```text
The final representation combines a direct current-token path with a routed prefix-context path over sealed set states.
```

### B7 Rebuttal

Use:

- fixed;
- deferred-to-future-work;
- reviewer-misconception.

Self-flagged improvements:

- strict-past routing plus residual causal lemma;
- landmark coverage remediation;
- hyperparameter exposure and resolved metadata logging.

## Error Handling

On any failed DoD:

1. Stop.
2. Write `audit/incident_<phase>_<task>_<timestamp>.md`.
3. Categorize as surface bug, logic/spec bug, cross-task interaction, environment/infrastructure, remote-execution failure, or genuine ambiguity.
4. Fix once if safe; rerun the entire affected DoD set.
5. Escalate if the same incident repeats, if the spec is contradicted, or if projected A2 runtime exceeds budget.

## Locked Summary

Locked choices:

- Causality fix: Option 1 endpoint-based strict-past routing.
- Token access fix: R1 direct embedding residual.
- Tail policy: T1 drop partial trailing windows.
- Linear backend: landmark only.
- Landmark parameter: `landmark_coverage=0.25`.
- Matched controls: dense/sparse/linear token-attention baselines are required for final backend-family comparisons.
- Config stack: `normalize.py -> schema.py -> compatibility.py`, no pydantic assumption.
- Inference cache: proposed design only unless implemented later.
