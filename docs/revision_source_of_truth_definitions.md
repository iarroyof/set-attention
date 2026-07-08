# Revision Source of Truth: Set-Attention Definitions

This document records the code-backed definitions and values that should be treated as source-of-truth for revision writing. It is intentionally narrow: definitions here come from the current repo implementation and current paper configs, not from prose assumptions.

Current experiment-scope note (2026-06-30): the implementation still exposes `exact`, `local_band`, and
`landmark`, but the active set-dictionary comparison uses **exact only**. Backend availability in this
document is a code/schema statement, not authorization to launch that backend. Coverage-scaled landmark
results are historical reference evidence and must not be described as linear/sub-quadratic scaling.
Current cells are defined by `docs/sd_dense_paper5_matrix.md`.

Revision-plan pointer:

- Current program authority is `docs/set_dictionary_research_main_plan.md`,
  followed by `audit/phase_sd_status.md` and the assigned task under
  `docs/agent_plans/`.
- `docs/sd_dense_paper5_matrix.md` remains the MRP-1 cell authority while its
  queues are active.
- The v2.7 PAT plan and `audit/phase_a_status.md` are historical original-campaign records. Their
  sparse/landmark tasks are not active branch requirements.
- Phase B (paper writing) action tracking uses `out/final_paper_bundle/checks/current_plan.md` and `out/final_paper_bundle/checks/progress_log.md`.
- A1.1 implements the locked autoregressive LM bank topology: partial trailing windows are dropped in `set_causality_mode=strict_past`, so the LR-norm headline reference has `M = floor((512 - 16) / 8) + 1 = 63`.
- The historical clipped-window topology remains available as `set_causality_mode=noncausal` for noncausal/bidirectional use.

Active evidence boundary:

- Reviewer-facing experimental claims for this revision use only post-A1 causal LM artifacts: `set_causality_mode=strict_past`, T1 dropped trailing windows, explicit residual policy, validated per-run JSON metadata, and passing manifests under `out/paper_integrated_evidence/checks/`.
- Pre-A1, noncausal, or causality-unverified artifacts are legacy/internal audit material only and must not be mixed into active tables, figures, claims, or summaries unless rerun and revalidated under the post-A1 causal LM protocol.
- Legacy-only locations include `out/paper_bundle/`, `out/paper_complements_bundle/`, `out/paper_complements/`, older `out/metrics/paper_action*` artifacts, and old dense-only mechanism plots generated before the A1 strict-past correction.
- Use the terms "post-A1 causal LM", "causal LM", or "strict-past causal LM"; do not use the old typo for causal status.

## Current Exact-Dense Set-Dictionary Configuration

The active paper matrix fixes:

| Quantity | Value |
|---|---|
| Backend | exact dense for set and token |
| Shape | `D=384`, `d_ff=1536`, 6 layers, 8 heads |
| Fine group | `(w,s)=(2,1)` |
| Coarse group | `(w,s)=(4,2)` |
| Blur rows | `{b0,b25,b50,b75,b100}` |
| Output policy | `anchor_span` |
| Token MLP / anchor | disabled / disabled |
| Objective / fiber | CE-only / `endpoint_window` |
| Training | full WikiText-2, 10 epochs, LR `1e-4`, five seeds |

Exact island ownership and batch/length values live in `docs/sd_dense_paper5_matrix.md`.

## Historical LR-Normalized Reference Configuration

The following table describes the older post-parity LR-normalized reference runs. It is retained for
provenance and is not the active set-dictionary launch configuration:

| Quantity | Value | Source |
| --- | --- | --- |
| Implementation | `set_only` | `configs/paper_lr_norm/set_dense_exact.yaml:2` |
| Attention family / backend | `dense` / `exact` | `configs/paper_lr_norm/set_dense_exact.yaml:3-4` |
| Model width | `d_model = 384` | `configs/paper_lr_norm/set_dense_exact.yaml:7` |
| Attention heads | `H = 8` | `configs/paper_lr_norm/set_dense_exact.yaml:8` |
| Head width | `d_h = d_model / H = 48` | derived from config and `src/models/set_only/set_only_lm.py:304-305` |
| Layers | `6` | `configs/paper_lr_norm/set_dense_exact.yaml:9` |
| FFN width | `1536` | `configs/paper_lr_norm/set_dense_exact.yaml:10` |
| Sequence length | `512` | `configs/paper_lr_norm/set_dense_exact.yaml:15`, `configs/paper_lr_norm/set_dense_exact.yaml:34` |
| Window size | `w = 16` | `configs/paper_lr_norm/set_dense_exact.yaml:17` |
| Stride | `s = 8` | `configs/paper_lr_norm/set_dense_exact.yaml:18` |
| Number of sets | `M = floor((512 - 16) / 8) + 1 = 63` in `set_causality_mode=strict_past` | `src/models/set_only/banks.py:105-156` |
| Set feature mode | `hashed_counts` | `configs/paper_lr_norm/set_dense_exact.yaml:24` |
| Router | learned, multihead, `router_topk = 16` | `configs/paper_lr_norm/set_dense_exact.yaml:25-27` |
| Router implementation | `model.router.score_mode = candidate_gather` computes learned-router scores only over the current candidate fiber; `dense` remains available as the historical dense masked implementation | `src/config/normalize.py`, `src/models/set_only/router.py` |
| Router temperature | `router_temperature = 1.0`; learned-router floor `model.router.min_temp = 0.5` | `src/config/normalize.py`, `src/models/set_only/set_only_lm.py`, `src/models/set_only/router.py` |
| Pooling | `soft_trimmed_boltzmann`, `tau = 0.1`, `q = 0.85` | `configs/paper_lr_norm/set_dense_exact.yaml:19-22` |
| Pooling sharpness | `model.pooling.alpha = 10.0`, `model.pooling.learnable_alpha = false` | `src/config/normalize.py`, `src/models/set_only/banks.py` |
| Pooling multihead | `false` | `configs/paper_lr_norm/set_dense_exact.yaml:23` |
| Hashed-count settings | `num_bins = 128`, `hash_seed = 13`, `normalize = true` | `src/config/normalize.py`, `src/set_attention/features/hashed_counts.py` |
| Feature dimension / adapter | `model.d_phi = null` resolves to `d_model = 384`; `model.adapter_type = auto` resolves to `linear` for the LR-normalized dense set config | `src/config/normalize.py`, `src/models/set_only/set_only_lm.py`, `src/set_attention/adapter_factory.py` |
| Set-state width | `model.set_state_dim = null` resolves to `d_model = 384`; pooled set states, set-attention blocks, backend value states, and routed set context use this width before projection back to token width | `src/config/normalize.py`, `src/models/set_only/set_only_lm.py`, `src/models/set_only/router.py` |
| Landmark coverage / count | For linear-landmark runs, `model.backend_params.landmark_coverage = 0.25`; at `L=512,w=16,s=8`, `M=63` and `K=16` | `src/config/normalize.py`, `src/models/set_only/set_only_lm.py`, `src/set_attention/backends/landmark.py` |
| Output residual mode | `model.output_residual_mode = direct` preserves the A1 direct residual; calibration runs may use `empty_only` or `none` explicitly | `src/config/normalize.py`, `src/models/set_only/set_only_lm.py` |
| Token MLP | disabled | `configs/paper_lr_norm/set_dense_exact.yaml:28-29` |
| Causal set stack mask | enabled | `configs/paper_lr_norm/set_dense_exact.yaml:16`, `src/models/set_only/set_only_lm.py:436-441` |

The Action 1 topocap config differs on width and stride: `d_model = 512`, `d_h = 64`, `stride = 4`, `M = floor((512 - 16) / 4) + 1 = 125` under `set_causality_mode=strict_past`, and it explicitly sets `router_temperature = 1.0`; see `configs/paper/action1_topocap_D512_FF1024_s4_w16_T1_lr1e4_seed0.yaml:8-36`. I did not find this `topocap` config family as a source for a currently reported draft number. The generated artifact using this shape is named `out/metrics/pending_topocap_D512_FF1024_s4.{csv,json}` and should be treated as a side/pending experiment unless a later draft explicitly cites it.

## Hashed-Count Features

For `feature_mode: hashed_counts`, the feature builder is `HashedCountFeatureBuilder` in `src/set_attention/features/hashed_counts.py`.

Definition:

- Each set `S_j` receives a count vector `c_j in R^G`, where `G = num_bins`.
- Current default is `G = 128`.
- Token ids are hashed by `bin(x) = (x * 1315423911 + hash_seed) mod G`.
- Current default `hash_seed = 13`.
- Counts are normalized by set size by default: `c_j <- c_j / |S_j|`.
- Geometry and hashed counts are projected separately, concatenated, then fused to form `phi_attn`.
- For learned routing in the current set-only LM path, `desc_router` is the pooled set state `Z_j`, not the hashed-count router projection, because `set_states` is passed into the feature builder.

Code provenance:

- Constructor defaults: `num_bins=128`, `normalize=True`, `hash_seed=13`, `fusion="mlp"` in `src/set_attention/features/hashed_counts.py:10-22`.
- Normalized config defaults: `model.feature_params.num_bins=128`, `model.feature_params.hash_seed=13`, and `model.feature_params.normalize=true` in `src/config/normalize.py`.
- Config validation and compatibility checks: `src/config/schema.py` and `src/config/compatibility.py`.
- Set-only LM passes all three values into `HashedCountFeatureBuilder`.
- Hash function: `src/set_attention/features/hashed_counts.py:48-49`.
- Count construction and normalization: `src/set_attention/features/hashed_counts.py:55-68`.
- Geometry/count fusion into `phi_attn`: `src/set_attention/features/hashed_counts.py:70-84`.
- `desc_router = set_states` in the current LM path: `src/set_attention/features/hashed_counts.py:86-91` and `src/models/set_only/set_only_lm.py:369-379`.

## Geometry Term `G`

The geometry term is the set-position bias matrix:

`G_ij = geom_bias_from_delta(delta_indices(set_positions))`.

In hashed-count mode, `exp(G)` is also projected and fused into `phi_attn` when geometry-in-feature is enabled. Separately, `G` is passed as an attention bias unless `geometry.apply_as_bias` is disabled.

Current defaults:

- `gamma = 1.0`, `beta = 0.0`.
- Geometry enabled by default.
- Geometry is applied as bias and included in `phi_attn` by default.

Code provenance:

- `gamma`, `beta` constructor defaults: `src/models/set_only/set_only_lm.py:61-62`.
- Geometry flags and defaults: `src/models/set_only/set_only_lm.py:164-173`.
- Hashed-count geometry bias and `exp(G)` row features: `src/set_attention/features/hashed_counts.py:70-75`.
- Attention-bias gating: `src/models/set_only/set_only_lm.py:414-416`.

Interface exposure note: if the manuscript uses `G` for the hash-count dimension instead of geometry, use `G = 128` only for the hash bin count and avoid overloading it with the geometry matrix.

## Content Bias `B^(h)`

The set-attention content bias is produced by the adapter from `phi_attn`:

`B = adapter(phi_attn)`.

For a linear adapter, per-head parameters are:

- `A^(h) in R^{d_h x d_phi}`
- `B_param^(h) in R^{d_h x d_phi}`

and the returned content-bias tensor is:

`bias^(h)_{ij} = <A^(h) phi_i, B_param^(h) phi_j>`.

Code provenance:

- Content bias creation and use: `src/models/set_only/set_only_lm.py:417-419`, `src/models/set_only/set_only_lm.py:443-444`.
- Linear adapter parameters and einsums: `src/set_attention/adapters.py:7-18`.
- Nonlinear and hybrid adapter alternatives: `src/set_attention/adapters.py:21-62`.
- Auto adapter selection: `src/set_attention/adapter_factory.py:14-20`.

Current paper config implication:

- `d_phi` defaults to `d_model`, so LR-normalized dense has `d_phi = 384`.
- `set_state_dim` defaults to `d_model`, so LR-normalized dense has `set_state_dim = 384`.
- `d_h = set_state_dim / H = 48`, so `select_adapter_type(d_phi=384, d_h=48)` selects `linear`.
- There are `H = 8` per-head content-bias slices.

Interface exposure note: `adapter_type` is now normalized as `auto` when absent. The runtime resolved adapter type and resolved `d_phi` are logged as `resolved.adapter_type` and `resolved.d_phi`.

## `set_state_dim`

Definition:

- `set_state_dim` is the width of pooled set states after token-window pooling, the set-attention stack hidden states, backend value projections, and routed set context before it is projected back to token width.
- If `model.set_state_dim` is `null` or absent before normalization, runtime resolution sets `set_state_dim = d_model`.
- In the default strict-past LM mode, the routed set context is projected to `d_model` and added to the direct token residual before the LM head.

Current values:

- LR-normalized dense set config: `set_state_dim = 384`.
- A6.2 explicit set-state capacity sweep holds token width at `d_model = 384` and `d_phi = 384` while sweeping `set_state_dim`.

Code provenance:

- Normalized config default `model.set_state_dim = null`: `src/config/normalize.py`.
- Constructor argument and runtime fallback: `src/models/set_only/set_only_lm.py`.
- Pooled set-state input/output projections: `src/models/set_only/set_only_lm.py`.
- Set stack/backends run at `set_state_dim`: `src/models/set_only/set_only_lm.py` and `src/models/set_only/ska_block.py`.
- Learned router accepts token width, set-state width, and descriptor width separately: `src/models/set_only/router.py`.
- Logged runtime value: `resolved.set_state_dim` in `scripts/run_experiment.py` and `src/train/experiment_logger.py`.

## `d_phi`

Definition:

- `d_phi` is the set feature dimension used for `phi_attn` and the adapter.
- If `model.d_phi` is `null` or absent before normalization, runtime resolution sets `d_phi = d_model`.

Current values:

- LR-normalized dense set config: `d_phi = 384`.
- Action 1 topocap side/pending config: `d_phi = 512`.

Code provenance:

- Normalized config default `model.d_phi = null`: `src/config/normalize.py`.
- Constructor argument and runtime fallback: `src/models/set_only/set_only_lm.py`.
- Fallback to `d_model`: `src/models/set_only/set_only_lm.py:154-157`.
- Passed to hashed-count feature builder: `src/models/set_only/set_only_lm.py:179-189`.
- Passed to learned multihead router and adapter: `src/models/set_only/set_only_lm.py:289-314`.
- Logged runtime value: `resolved.d_phi` in `scripts/run_experiment.py` and `src/train/experiment_logger.py`.

## Pooling `alpha`, `tau`, and `tau_min`

For `soft_trimmed_boltzmann`, each set pools token embeddings by:

- computing squared distance from the set mean,
- keeping a soft quantile mask using `sigmoid(alpha * (threshold - d2))`,
- applying logits `(-d2 / tau) + log(mask_soft + 1e-8)`,
- softmaxing over the window tokens.

Current values:

- Paper configs set `tau = 0.1` and `q = 0.85`.
- `alpha` is defaulted by normalization as `model.pooling.alpha = 10.0`.
- `learnable_alpha` is defaulted by normalization as `model.pooling.learnable_alpha = false`.
- If `learnable_alpha=True`, alpha is clamped to `[1.0, 50.0]`; with the current configs alpha is a fixed buffer and is not clamped during forward.
- There is no pooling `tau_min` in code. The only implemented temperature floor is the router minimum temperature, `LearnedRouter.min_temp = 0.5`.

Code provenance:

- Normalized config defaults: `src/config/normalize.py`.
- Pooling defaults and constructor propagation in `SetOnlyLM`: `src/models/set_only/set_only_lm.py:102-128`.
- Pooling constructor defaults: `src/models/set_only/banks.py:161-185`.
- Learnable alpha clamp: `src/models/set_only/banks.py:193-196`.
- Pooling logits: `src/models/set_only/banks.py:234-244`.
- Router temperature floor: normalized as `model.router.min_temp = 0.5`, passed to `LearnedRouter.min_temp`, and used in router softmax temperature clamping.
- Logged alpha metric: `ausa/pooling_alpha_value` in `src/models/set_only/banks.py:289-298`.
- Logged runtime values: `resolved.pooling_alpha` and `resolved.router_min_temp` in `scripts/run_experiment.py` and `src/train/experiment_logger.py`.

## Router Heads `H_r`

Definition:

- `H_r` is the number of router heads when `router_multihead=True`.
- In code, router heads reuse `num_heads`; there is no separate `router_num_heads`.

Current value:

- LR-normalized dense set config has `H_r = num_heads = 8`.
- Action 1 topocap side/pending config also has `H_r = 8`.

Code provenance:

- Config sets `num_heads=8`, `router_multihead=true`: `configs/paper_lr_norm/set_dense_exact.yaml:8`, `configs/paper_lr_norm/set_dense_exact.yaml:27`.
- Router receives `num_heads=num_heads`: `src/models/set_only/set_only_lm.py:289-295`.
- Multihead router shapes: query/key have shape `[B, H_r, T or M, d_phi]`; `src/models/set_only/router.py:123-127`.
- Set states are split into `[B, H_r, M, d_h]`: `src/models/set_only/router.py:151-153`.

Interface exposure note: if the paper wants `H_r` independent of set-attention heads, code does not currently expose that. Add it to the update plan only if a separate router-head count is required.

## Learned Router Score Implementation

`model.router.score_mode` selects the learned-router score implementation.

- `candidate_gather` is the active default. For each token, it gathers only the
  candidate set indices already supplied by the current bank candidate fiber and
  computes logits on shape `[B, H_r, L, C_max]`, where `C_max` is the padded
  candidate count in `bank.token_to_sets`.
- `dense` preserves the historical implementation: compute dense logits on
  `[B, H_r, L, M]`, then mask all sets outside the candidate fiber.
- Both modes use the same score law
  `score_{t,m}^{(h)} = <q_t^{(h)}, k_m^{(h)}> / sqrt(d_phi)` and the same
  softmax support `m in C_t`, so `candidate_gather` is intended to be
  mathematically equivalent to dense masked routing up to floating-point order.
- Compact router probabilities are passed to diagnostics with their candidate
  indices, so entropy, top-1 weight, candidate-normalized top-1 gap, candidate
  counts, and set-utilization summaries are computed without reconstructing a
  dense `[B,H,L,M]` probability tensor.

Code provenance:

- Normalized default: `model.router.score_mode = candidate_gather` in
  `src/config/normalize.py`.
- Schema/compatibility validation: `src/config/schema.py` and
  `src/config/compatibility.py`.
- Candidate-gather implementation and historical dense fallback:
  `src/models/set_only/router.py`.
- Set-only and hybrid propagation: `src/models/set_only/set_only_lm.py` and
  `src/models/hybrid_token_set_lm.py`.
- Runtime metadata: logged as `resolved.router_score_mode`.

## Output Residual Mode

`model.output_residual_mode` is a named set-only LM output policy applied after routing and before the LM head.

- `anchor_span` is the active set-dictionary policy. It uses the thin,
  non-contextual anchor `h_t^(0)=e(x_t)+p_t` plus the projected routed span.
  With the active config, the token MLP and trained-anchor path are disabled,
  so all dependence on earlier tokens must factor through the routed span.
- `direct` is the default and preserves the A1-A6 implementation: in `strict_past`, the LM head receives `h_t^(0) + r_t`, with `r_t=0` when `C_t=0`.
- `empty_only` is a calibration mode for token-limit and compression-limit experiments: in `strict_past`, the LM head receives `h_t^(0)` only when the supplied candidate fiber is empty, and receives `r_t` when `C_t>0`.
- `none` removes the final token residual in `strict_past` and uses only `r_t`.
- Historical `noncausal` mode remains routed-only, preserving existing behavior.

For strict-past endpoint routing with 1-indexed positions and first endpoint `e_1=w`, exactly the first `w-1` token positions have empty candidate fibers. Under a uniform token-position view, `P(C_t=0)=(w-1)/L`. In the singleton calibration limit `w=1,s=1`, `M=L` and `P(C_t=0)=0`, so `empty_only` and `none` coincide.

Code provenance:

- Normalized config default `model.output_residual_mode = direct`: `src/config/normalize.py`.
- Active set-dictionary config explicitly selects `anchor_span`:
  `configs/set_dictionary/sd9_multiresolution.yaml`.
- Schema/compatibility validation: `src/config/schema.py` and `src/config/compatibility.py`.
- Runtime behavior and metadata: `src/models/set_only/set_only_lm.py`.
- Runner/logger propagation: `scripts/run_experiment.py` and `src/train/experiment_logger.py`.

## Hybrid Token/Set LM Configs

`model.implementation = hybrid_token_set` is a layer-level LM variant used for
post-A1 causal hybrid experiments. It keeps one shared token hidden stream
`X in R^{B x L x d_model}` and applies token-attention layers (`T`) and set
layers (`S`) according to `model.hybrid.pattern`. This is not a separate set
tower: token and set layers update the same token sequence, so information from
early token layers can bridge to later token or set layers through the shared
residual stream.

The hybrid layer pattern is configured, not hardcoded:

- `model.hybrid.pattern` is a string of `T` and `S` with length
  `model.num_layers`.
- `model.hybrid.set_topologies` is a list with one `{window_size, stride}` entry
  per `S` layer, in layer order.
- The current A8 hybrid sparse progressive configs live under
  `configs/a8_hybrid/` and explicitly set model, data, and training
  hyperparameters in YAML.
- Launch scripts for these experiments may override run identity/provenance
  values such as `training.seed`, `training.output_dir`, W&B run name/project,
  and CSV path. They should not duplicate model topology, pooling, router,
  feature, data length, LR, or epoch values as hidden shell constants.

Code provenance:

- Model implementation: `src/models/hybrid_token_set_lm.py`.
- Runner construction: `scripts/run_experiment.py`.
- Config defaults/allow-list/validation: `src/config/normalize.py`,
  `src/config/schema.py`, and `src/config/compatibility.py`.
- Focused tests: `tests/test_hybrid_token_set_lm.py`.

## Candidate Sets and Router Top-1 Metric

Candidate sets are produced by `build_window_bank`.

For autoregressive LM paper mode (`set_causality_mode=strict_past`), sequence length `L`, window `w`, stride `s`:

- starts are `range(0, L - w + 1, s)`;
- `M = floor((L - w) / s) + 1`;
- `S_j = {start_j, ..., start_j + w - 1}`;
- endpoint `e_j = max(S_j)`;
- `token_to_sets[t]` lists the Option-1 candidate fiber `{j : t - w < e_j <= t}`.

The old clipped membership topology remains available as `set_causality_mode=noncausal`, where starts are `range(0, L, s)`, `M = ceil(L / s)`, trailing windows are clipped, and `token_to_sets[t]` lists sets containing token `t`.

Strict-past candidate counts for `L=512`, `w=16`:

| Stride | `M` | min | mean | max |
| --- | ---: | ---: | ---: | ---: |
| `s=3` | 166 | 0 | 5.109375 | 6 |
| `s=4` | 125 | 0 | 3.8359375 | 4 |
| `s=5` | 100 | 0 | 3.072265625 | 4 |
| `s=6` | 83 | 0 | 2.5625 | 3 |
| `s=8` | 63 | 0 | 1.92578125 | 2 |

For the long-context LR-norm headline reference, `L=2048`, `w=16`, `s=8` gives `M = floor((2048 - 16) / 8) + 1 = 255`.

Router top-1 metric:

- For multihead router probabilities `[B, H_r, T, M]`, diagnostics average heads to `Pbar`, renormalize, and compute `confidence = max_j Pbar_{t,j}`.
- `ausa/router_top1_weight` is the epoch mean of that confidence.
- Candidate-normalized top-1 gap is `ausa/router_top1_gap_norm = (top1_c - 1/C) / (1 - 1/C)`, where `C` is the per-token structural candidate count.

Code provenance:

- Candidate construction: `src/models/set_only/banks.py:113-170`.
- Learned-router candidate restriction mask: `src/models/set_only/router.py:129-135` and `src/models/set_only/router.py:166-172`.
- Router top-k application: `src/models/set_only/router.py:137-143` and `src/models/set_only/router.py:174-190`.
- Multihead `Pbar` computation and top-1 weight logging: `src/models/set_only/diagnostics.py:120-127`, `src/models/set_only/diagnostics.py:189-204`.
- Candidate-count and candidate-normalized metrics: `src/models/set_only/diagnostics.py:205-271`.

Diagnostics consume the supplied candidate fiber/mask. In `strict_past`, that fiber is the Option-1 endpoint fiber already stored in `bank.token_to_sets`; in `noncausal`, it is the historical containing-token membership fiber. Diagnostics must not reconstruct containing-token membership internally. Early strict-past tokens with zero candidates are excluded from entropy and gap normalizers so the reported values remain finite.

## `Lambda` Selection

There is no code symbol named `Lambda` for method selection. In current configs and validators, the concrete attention operator is selected by:

- `model.attention_family` in `{dense, sparse, linear}`;
- `model.backend` in `{exact, local_band, landmark}` for the implemented revision schema; the current
  set-dictionary experiment matrix selects `exact` only. `nystrom` is rejected by active config validation,
  and other legacy/deprecated schema values such as `linformer` and `sparse_topk` may still appear outside
  the active paper path;
- `model.backend_params` for backend-specific controls.

Current mappings used by configs:

- dense: `backend=exact`;
- sparse: `backend=local_band`, with `backend_params.radius`;
- historical linear-family label: `backend=landmark`, with `backend_params.landmark_coverage`. This
  coverage-scaled implementation is not active in the current set-dictionary matrix and must not be
  interpreted as asymptotically linear. `nystrom` and `linformer` are legacy/deprecated schema values.

Code provenance:

- Backend instantiation: `src/models/set_only/set_only_lm.py:219-271`.
- Allowed config keys and backend values: `src/config/schema.py:8-50`, `src/config/schema.py:91-101`.
- Human-readable hyperparameter contract: `configs/hyperparameters.md:32-41`, `configs/hyperparameters.md:75-81`.

Interface exposure note: if the manuscript uses `Lambda` as a selectable support/operator family, define it in prose as the config-level pair `(attention_family, backend, backend_params)`. There is no single runtime `Lambda` field to quote.

## Landmark Backend Selection and Blocks

For `backend: landmark`, the current source-of-truth implementation is
`LandmarkAttentionBackend` in `src/set_attention/backends/landmark.py`.

Landmark selection rule:

```python
def _select_landmarks(self, m: int, device: torch.device) -> torch.Tensor:
    K = min(max(round(self.landmark_coverage * m), 2), m)
    if K >= m:
        return torch.arange(m, device=device)
    return torch.tensor(
        [round(i * (m - 1) / (K - 1)) for i in range(K)],
        device=device,
        dtype=torch.long,
    )
```

Therefore landmarks are deterministic, linspace-rounded, and endpoint-anchored
over set index order. They are not sampled, clustered, trained, or selected by
content. If `K >= M`, all sets are landmarks.

Coverage parameterization:

- Canonical config key: `model.backend_params.landmark_coverage`.
- Default: `0.25`.
- Runtime count: `K = max(round(landmark_coverage * M), 2)`, capped at `M`.

For the LR-normalized linear-landmark paper config after A1.6:

- `M = floor((512 - 16) / 8) + 1 = 63`;
- `landmark_coverage = 0.25`;
- `K = max(round(0.25 * 63), 2) = 16`;
- selected landmark set indices are
  `[0, 4, 8, 12, 17, 21, 25, 29, 33, 37, 41, 45, 50, 54, 58, 62]`.

For the long-context LR-norm headline reference:

- `M = floor((2048 - 16) / 8) + 1 = 255`;
- `landmark_coverage = 0.25`;
- `K = max(round(0.25 * 255), 2) = 64`.

Code provenance:

- Config value `landmark_coverage = 0.25`: `configs/paper_lr_norm/set_linear_landmark.yaml`.
- Normalized config default: `src/config/normalize.py`.
- Config validation: `src/config/compatibility.py`.
- Backend instantiation passes `backend_params.landmark_coverage`, defaulting to `0.25` if absent: `src/models/set_only/set_only_lm.py`.
- Landmark index rule: `src/set_attention/backends/landmark.py`.
- Forward pass calls `_select_landmarks`: `src/set_attention/backends/landmark.py:54`.
- Landmark query/key slices: `src/set_attention/backends/landmark.py:55-56`.
- Scores built against landmarks: `scores_mL` and `scores_Lm` at `src/set_attention/backends/landmark.py:58-59`.
- Biases and masks are sliced by the same `landmark_idx`: `src/set_attention/backends/landmark.py:67-100`.
- Two-stage landmark attention is `attn_mL = softmax(m x L)`, `attn_Lm = softmax(L x m)`, then `v_l = attn_Lm @ v`, `out = attn_mL @ v_l`: `src/set_attention/backends/landmark.py:109-115`.
- Logged runtime values: `resolved.landmark_coverage` and `resolved.landmark_count` in `scripts/run_experiment.py` and `src/train/experiment_logger.py`.

Blocks:

- `SetOnlyLM` creates one `SetAttentionBlock` per layer: `src/models/set_only/set_only_lm.py:273-284`.
- Each block owns its own backend instance because `make_backend()` is called inside the list comprehension.
- During forward, all blocks run sequentially over the full set bank `set_states`; the landmark backend does not partition the set bank into local computational blocks. It selects landmarks within each block/layer from the current `m` set positions using the deterministic rule above.
- Block application loop: `src/models/set_only/set_only_lm.py:448-449`.
- Block wrapper call into backend: `src/models/set_only/ska_block.py:30-40`.

Deprecated Nystrom note:

- `NystromBackend` remains in `src/set_attention/backends/nystrom.py` only for historical reference/import compatibility. As of A1.10, constructing it raises `RuntimeError("NystromBackend is deprecated for this revision cycle; use landmark backend.")`, and active config validation rejects `backend: nystrom`.

## Values That Are Hardcoded or Implicit

Add these to the update plan if they need to be first-class config/interface fields or logged in every run JSON:

| Quantity | Current status |
| --- | --- |
| Hashed-count `hash_seed` | canonical `model.feature_params.hash_seed`, default `13`, logged as `resolved.hash_seed` |
| Hashed-count normalization | canonical `model.feature_params.normalize`, default `true`, logged as `resolved.hash_normalize` |
| Hashed-count bins | canonical `model.feature_params.num_bins`, default `128`, logged as `resolved.hash_num_bins` |
| Router `tau_min` / temperature floor | canonical `model.router.min_temp`, default `0.5`, logged as `resolved.router_min_temp` |
| `d_phi` when absent | canonical `model.d_phi = null`, runtime-resolved to `d_model`, logged as `resolved.d_phi` |
| Adapter type when absent | canonical `model.adapter_type = auto`; selected runtime adapter logged as `resolved.adapter_type` |
| `H_r` | implicit reuse of `num_heads`; no separate router-head config |
| `Lambda` | no single field; represented by `attention_family`, `backend`, and `backend_params` |
| Landmark selection method | deterministic linspace-rounded anchored set-index rule via `model.backend_params.landmark_coverage`, default `0.25`; no config for random/content/clustered selection |
| Pooling `alpha` | canonical `model.pooling.alpha`, default `10.0`, logged as `resolved.pooling_alpha` |
