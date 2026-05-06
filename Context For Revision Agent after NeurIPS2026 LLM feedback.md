**Context For Revision Agent**

Repository/source of truth:
- Primary experiment server: `blue-demon:~/set-attention`
- Local paper/workspace mirror may exist at:
  `/mnt/d/UserFolders/Documents/GitHub/set-attention`
- Treat blue-demon as the source of truth for completed experiments and generated artifacts.
- Preserve the current implementation as an optional configurable mode. Do not delete it; it may remain useful for non-causal / bidirectional appendix diagnostics.

**Critical Finding**

The current set-only LM implementation has an autoregressive causality leak.

Verified files:
- `src/models/set_only/banks.py`
- `src/models/set_only/set_only_lm.py`
- `src/models/set_only/router.py`
- `src/train/loop.py`
- `src/data/wikitext2.py`

Current behavior:
- `build_window_bank()` builds forward windows `[start, start + window_size)`.
- `Bank.pool()` pools all valid tokens in each set with only padding masks.
- `token_to_sets` includes every set containing token `t`.
- `SetOnlyLM.causal=True` only applies an inter-set causal mask after pooling.
- `LearnedRouter` and `UniformRouter` route over `token_to_sets` without enforcing `max S_m <= t`.
- WikiText-2 labels are next-token shifted, but this does not prevent leakage because `z_t` can already contain future input tokens through pooled set states.

Therefore:
- Current AR LM perplexity claims are not causally valid.
- Current non-causal set-only results may be retained only as non-causal / bidirectional / diagnostic appendix evidence.
- The revision should not defend the old AR headline as-is.

---

## Branches To Implement Or Support

Implement the fork cleanly so we can select the strongest path later.

### Branch 0: Current Non-Causal Architecture

Purpose:
- Preserve current implementation for diagnostics, bidirectional tasks, and appendix comparisons.

Suggested config:
```yaml
model:
  set_causality_mode: noncausal
```

Semantics:
- Current window bank.
- Current pooling.
- Current routing over all containing sets.
- Must be explicitly labeled non-causal when used in AR contexts.

### Branch 1b: End-Aligned Causal Bank

This is the likely strongest defensible path if we want to preserve efficiency.

Semantics:
- Each set is a past-facing or end-aligned window.
- Token `t` may route only to sets whose maximum token index is `<= t`.
- Add fallback singleton/current-token sets so early positions do not have empty candidate families.

Suggested config:
```yaml
model:
  set_causality_mode: end_aligned
  causal_fallback: singleton_current
```

Required implementation:
- Add bank metadata:
  - `set_starts`
  - `set_ends`
  - `set_indices`
  - `token_to_sets_causal`
  - candidate count per token
- Routing must use `token_to_sets_causal` when `set_causality_mode=end_aligned`.
- Pooling can remain one pooled state per set.
- Set-attention stack can remain set-level with causal set-to-set mask.
- Add explicit tests that no token representation depends on future tokens.

Paper math:
\[
\mathcal C_t^-=\{m:\max S_m\le t\},
\qquad
z_t^{(u)}=\sum_{m\in\mathcal C_t^-}\pi_{t,m}^{(u)}\tilde s_m^{(u)}.
\]

### Branch 1a: Causal Prefix Pooling

This is the cleanest semantic fix but likely weakens the efficiency story.

Semantics:
\[
s_{m,t}^{(0)}
=
\sum_{u\in S_m,\ u\le t}
\omega_{u,m,t}h_u^{(0)}.
\]

Implementation implications:
- Pooling becomes token-conditioned.
- Routing consumes per-token causal set summaries.
- More expensive and more invasive.
- Implement only if selected explicitly or after 1b underperforms badly.

Suggested config:
```yaml
model:
  set_causality_mode: prefix_pooling
```

### Branch 2: Bidirectional / Encoder Pivot

Purpose:
- Keep current architecture as valid by changing task framing.
- Suitable tasks: MLM, classification, retrieval, LRA-style tasks, bidirectional sequence diagnostics.

Suggested config:
```yaml
model:
  set_causality_mode: bidirectional
task:
  objective: mlm
```

### Branch 3: Theory + Diagnostic Study

Purpose:
- Remove competitive AR LM claim.
- Keep Tier C mechanisms as the contribution:
  - candidate-count collapse
  - pooling concentration
  - gradient transport
  - cross-family dense/sparse/linear behavior

No architecture fix strictly required, but all AR wording must be removed or caveated.

---

## Required Code Workstreams

### A. Modeling Code

Files to inspect/edit:
- `src/models/set_only/banks.py`
- `src/models/set_only/set_only_lm.py`
- `src/models/set_only/router.py`
- `src/config/schema.py`
- `scripts/run_experiment.py`

Add:
- `set_causality_mode`
- `causal_fallback`
- clear validation errors for incompatible settings
- metadata-rich `Bank` object

Suggested enum:
```python
SetCausalityMode = Literal[
    "noncausal_current",
    "end_aligned",
    "prefix_pooling",
    "bidirectional",
]
```

Keep old behavior exactly reachable through `noncausal_current`.

### B. Dataset / Objective

Files:
- `src/data/wikitext2.py`
- `src/train/loop.py`

Tasks:
- Confirm AR shift remains correct for causal branches.
- Add optional leak-test dataset or utility where future tokens can be perturbed while past tokens are fixed.
- For bidirectional branch, add MLM-style masking only if Branch 2 is selected.

### C. Diagnostics

Add causality diagnostics:
1. Perturbation test:
   - Compute `logits_t` for a sequence.
   - Change tokens `x_{>t}` only.
   - Confirm `logits_t` is unchanged under causal modes.

2. Gradient test:
   - Compute loss at position `t`.
   - Confirm gradients w.r.t. embeddings/tokens at positions `>t` are zero or numerically negligible.

3. Candidate-family audit:
   - For every `t`, assert all routed sets satisfy `max S_m <= t` in end-aligned mode.

Output diagnostics to CSV/JSON:
```text
causality_violation_max_abs
future_grad_norm
num_invalid_future_candidates
mean_causal_candidate_count
min_causal_candidate_count
```

### D. Interface

Expose through:
- YAML configs
- CLI overrides
- experiment manifests
- result CSVs

Every run must record:
```text
set_causality_mode
causal_fallback
window_size
stride
mean_candidate_count
min_candidate_count
max_candidate_count
pooling_mode
router_type
router_topk
router_multihead
backend
backend_params
seed
git_commit
config_sha256
source_config_path
```

---

## Latest Main-Paper Grid To Preserve As Historical / Appendix Evidence

Current reference operating recipe used in the latest main-paper experiments:

```text
Dataset: WikiText-2 LM
Sequence length: 512
Batch size: 16
Epochs: 10
Warmup: 1000
Model width: D=384
Layers: 6
Heads: 8
FFN width: 1536
Dropout family: 0.1
Seed: historically seed 0, latest normalized runs may include more
Vocabulary size: 76618
Architecture: causal transformer_lm for baseline
Set features: hashed_counts
Router: learned
router_topk: 16
router_multihead: true
pooling: soft_trimmed_boltzmann
tau_pool: 0.1
pooling_multihead: false
window size: 16
stride: 8
Dense set backend: dense_exact
Sparse set backend: local_band, radius=4
Linear set backend: landmark, num_landmarks=24
```

Important:
- These current set-only AR LM results were produced under the leaking set-only architecture.
- They may be kept only as historical/non-causal appendix evidence unless rerun under a causal branch.
- Do not use them as headline AR evidence.

---

## Strong Defensible Experiment Plan

Given dedicated dual RTX 4090 workstation, do not use relaxed evidence. Use strong seeding.

### Minimum Causal AR Grid

Use Branch 1b first.

Seeds:
```text
0, 1, 2, 3, 4
```

Learning rates:
```text
5e-5, 1e-4, 2e-4, 5e-4
```

Families:
```text
baseline_dense_exact
set_dense_exact_end_aligned
set_sparse_local_band_end_aligned
set_linear_landmark_end_aligned
```

Reference size:
```text
L=512
B=16
D=384
layers=6
heads=8
d_ff=1536
epochs=10 initially, then extend best settings to convergence
```

Topology sweeps:
```text
window_size: 8, 16, 32, 64
stride: 4, 8, 16, 32
```

Pooling temperature:
```text
tau_pool: 0.03, 0.05, 0.1, 0.2, 0.5, 1.0
```

Router temperature:
```text
tau_router: 0.5, 1.0, 2.0
```

Longer context extension:
```text
L: 1024, 2048, 4096
```

For long context:
- Use best few operating points only.
- Include baseline where feasible.
- If dense baseline becomes infeasible, report memory/time limits explicitly.

### Required Comparisons

At minimum:
- Dense causal Transformer baseline.
- Causal Set Dense.
- Causal Set Sparse.
- Causal Set Linear.

If time permits, add one published efficient-attention baseline:
- Performer
- Longformer
- Routing Transformer

Do not claim broad efficient-attention superiority without at least one published efficient-attention comparator.

---

## Provenance Requirements

Reuse existing provenance style, but tighten it.

For every run, write:
```text
out/runs/<run_id>/
  config.yaml
  resolved_config.yaml
  metrics.csv
  diagnostics.csv
  causality_audit.json
  manifest.json
  stdout.log
  stderr.log
```

`manifest.json` must contain:
```json
{
  "run_id": "...",
  "git_commit": "...",
  "git_status_short": "...",
  "host": "blue-demon",
  "gpu_name": "...",
  "cuda_version": "...",
  "python_version": "...",
  "torch_version": "...",
  "config_sha256": "...",
  "source_config_path": "...",
  "seed": 0,
  "set_causality_mode": "end_aligned",
  "started_at": "...",
  "ended_at": "...",
  "metrics_csv": "...",
  "causality_audit": "..."
}
```

Summary scripts must never reconstruct numbers from memory. They must read only:
- metrics CSVs
- diagnostics CSVs
- manifest JSONs
- causality audit JSONs

---

## Paper Revision Guidance

Until causal reruns are complete:
- Remove headline AR LM claims based on old set-only runs.
- Move old non-causal set-only AR results to appendix as diagnostic/non-causal evidence.
- Keep Tier C theory and diagnostics if framed architecture-independently.
- State that causal variants are under evaluation only after results exist.

If Branch 1b succeeds:
- Main paper can return to AR LM framing.
- Rebuild Table 1 and headline figure only from causal runs.
- Add causality audit table in main or appendix.

If Branch 1b fails:
- Do not force the narrative.
- Pivot to Branch 2 or Branch 3.

---

## Immediate First Tasks For New Codex Chat

1. Inspect current code and confirm the leak with a minimal failing test.
2. Add `set_causality_mode=noncausal_current` preserving exact old behavior.
3. Implement `end_aligned` bank/routing mode.
4. Add causality audit tests and diagnostics.
5. Generate reference configs for:
   - baseline dense exact
   - set dense exact end-aligned
   - set sparse local-band end-aligned
   - set linear landmark end-aligned
6. Add run scripts for dual RTX 4090 scheduling.
7. Add provenance manifests for every run.
8. Only after tests pass, launch the seed/LR grid on blue-demon.