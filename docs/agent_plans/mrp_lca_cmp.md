# MRP-lca-cmp: Long-Context Aggregation / Compressed-Memory Comparison

Status: ACTIVE; batching tests and generator validation passed; calibration launch pending approval

Owner: MRP-lca-cmp mechanism worker

Updated: 2026-07-16 after MRP-3 MQAR completed null/inconclusive, the
research direction shifted from token-precision retrieval to compressed
long-context aggregation, and the batching-preservation guard passed in the
Lizmark `set-attention:latest` container.

## Purpose

MRP-lca-cmp tests the second hybrid-model branch: whether multiresolution
set-dictionary attention helps when the task rewards compressed aggregation
over a long context, rather than exact token-key retrieval. MQAR remains a
completed negative/precision-control result. This task must not be interpreted
as a rerun of MRP-3.

The target comparison is token dense attention versus exact-dense
set-dictionary multiresolution rows under tasks where the answer depends on a
distributed statistic, latent list/state, or count-like aggregate over many
positions. The initial path is synthetic and controlled before any external
benchmark transfer.

## Scientific Questions

1. Can a set-dictionary model match or beat dense token attention on a
   learnable long-context aggregation task at lower peak VRAM?
2. Does the advantage grow with context length when token-precision retrieval
   is not the central requirement?
3. Do mixed-resolution rows allocate coarse/fine groups differently as the
   aggregation horizon grows, while all-fine approaches token-like local
   behavior and all-coarse loses detail?
4. Are any positive results robust to equal optimizer-update budgets and equal
   effective batch size?

## Active Model Boundary

No architecture changes are authorized by this plan, except the single
registered diagnostic probe in "Approved Diagnostic Probes" below.

## Approved Diagnostic Probes

User-approved 2026-07-19: exactly one fiber-diagnosis probe
(`b25`, `L=1024`, `B=4`, seed 0, `candidate_fiber=all_past`, native batching,
`max_updates=2000`, all other settings identical to the completed
`b25|1024|b4|0|native` calibration row). Purpose: test the `endpoint_window`
receptive-field diagnosis in `audit/LCA_calibration_20260718.md` (Gate 2
failure root cause). This probe is **not** part of the calibration or primary
matrix, its numbers are never pooled with matrix rows, and it does **not**
reopen `all_past` (or any other boundary change) for matrix rows. Output is
labeled `allpast_probe` so it cannot be confused with matrix artifacts.

- `model.implementation in {baseline_token,set_only}` only.
- `attention_family=dense`, `backend=exact`; landmark/sparse/fixed-k are not
  active unless a later plan explicitly reopens them.
- Set rows keep `output_residual_mode=anchor_span`,
  `model.token_mlp.enabled=false`, `model.anchor.enabled=false`,
  `model.anchor.teacher.enabled=false`, `candidate_fiber=endpoint_window`,
  `allow_token_token=false`, `multivector_basis.enabled=false`.
- Multiresolution groups remain fine `(w,s)=(2,1)` and coarse `(4,2)`.
- Any row using different task data, optimizer budget, or effective batch is a
  new row, not a continuation of an existing result.

## Batching Infrastructure

Gradient accumulation and evaluation microbatching are allowed only as runtime
memory controls.

- Default behavior is `training.grad_accum_steps=1` and
  `training.eval_microbatch_size=null`; existing SD/MRP configs therefore keep
  one dataloader batch per optimizer update.
- For new LCA rows, `data.batch_size` is the microbatch size and
  `effective_batch_size = data.batch_size * training.grad_accum_steps`.
- Comparisons must hold effective batch size, optimizer-step budget, LR,
  warmup schedule, model shape, and dataset/task seed constant unless the row
  is explicitly labeled as a batching ablation.
- Summaries must report both microbatch size and effective batch size.
- Validation gate: a deterministic unit test must show one full-batch update
  matches the corresponding accumulated microbatches, and eval
  microbatching must preserve metrics.

These controls are intended to avoid OOM and reduce idle GPU time. They are
not a mechanism claim and must not be used to reinterpret prior MRP/SD rows.

## Candidate Tasks

Stage A uses internal synthetic tasks with exact provenance and unlimited
regeneration:

- majority/parity-free count bucket over marked tokens;
- thresholded frequency of one or more latent symbols over the whole context;
- segmented list-state aggregation where the output depends on a compressed
  summary per segment, not a single copied key;
- optional distractor-local tokens so token attention is not rewarded merely
  for copying nearby identifiers.

The first accepted task must meet a learnability gate: token dense and at least
one set row must both move well above random/chance on a short calibration
setting. If the task is too easy, too hard, or degenerate, stop and adjust the
generator before launching scale rows.

## Initial Matrix

Calibration, local/short:

| Context | Rows | Seeds | Purpose |
|---|---|---:|---|
| `L=1024,B=4` | token, `b0`, `b25`, `b50`, `b75`, `b100` | 0,1,2 | generator learnability and metric sanity |
| `L=2048,B=4` | token, `b0`, `b25`, `b50`, `b75`, `b100` | 0,1,2 | compare with the main supported island scale |

Primary compressed-aggregation scale:

| Context | Rows | Seeds | Purpose |
|---|---|---:|---|
| `L=3584,B=4` | token, `b0`, `b25`, `b50`, `b75`, `b100` | 0,1,2 | main exact-dense island with previous set/token frontier evidence |
| `L=4096,B=3` | token, `b0`, `b25`, `b50`, `b75`, `b100` | 0,1,2 | largest full exact-dense feasible island |
| `L=4096,B=4` | selected rows only if preflight admits them | 0,1,2 | feasibility/descriptive stress point; not required for the full grid |

Five-seed extension is authorized only after the 3-seed primary matrix shows a
stable candidate Pareto win and passes the task learnability gate.

## Native-Batch Memory Story (user directive 2026-07-18)

Microbatching and gradient accumulation are runtime memory controls. The paper
story must also state what memory the implementation needs **without** them.
Therefore:

- For every cell that runs with `grad_accum_steps>1` or
  `eval_microbatch_size` set, the pipeline must also record the **native-batch
  reference**: a one-step full-shape preflight of the identical model/task at
  `grad_accum_steps=1`, `eval_microbatch_size=null`, and
  `data.batch_size` equal to the effective batch, logging its peak VRAM or its
  OOM censoring. This is a measurement row, not a trained row.
- Every summary table reports both numbers per cell: peak train VRAM under the
  active batching settings **and** the native-batch peak (or `OOM` censored).
- Memory-advantage claims compare like for like: optimized-vs-optimized and
  native-vs-native, never optimized-set versus native-token.

## OOM Censoring Reconsideration

Microbatching shifts the OOM boundary, so previously censored cells must be
re-examined rather than inherited as infeasible:

- Known prior censoring under the exact-dense paper matrix: `L=4096,B=4`
  token, `b0`, and `b25` (3/3 legacy OOM, pre-2026-07-02, observed legacy
  feasibility outcomes).
- Before any LCA cell is excluded on memory grounds, run the registered
  one-step full-shape preflight twice: native batch and with the intended
  microbatch/accumulation settings. Record both outcomes.
- If a previously censored cell is admitted only under microbatching, it
  enters the matrix as a **new labeled row** (batching-controlled memory
  extension), never as a continuation of the censored native cell, and its
  quality numbers are never pooled with native-batch islands (control-tuple
  rule: effective batch, LR, and update budget held fixed; batching mode is a
  reported stratum).
- `L=4096,B=4` token/`b0`/`b25` reconsideration preflights are registered as
  part of the primary stage, after calibration passes the learnability gate.

## Metrics

Mandatory per row:

- task loss, task accuracy, exact/sequence accuracy when defined;
- bucketed accuracy by aggregation horizon or segment span;
- peak train VRAM under the active batching settings **and** the native-batch
  one-step preflight peak (or explicit OOM censoring), per the native-batch
  memory story above;
- elapsed time, samples/s;
- microbatch size, `grad_accum_steps`, effective batch size;
- set group diagnostics: fine/coarse routing entropy, top-1, effective range,
  and group ablation deltas where the model exposes the hook;
- strict `nan/inf/traceback` log scan and endpoint-valid CSV check.

## Gates

Proceed from calibration to primary only if:

1. token dense reaches the task learnability threshold on the calibration row;
2. at least one set row reaches a nondegenerate accuracy/loss regime;
3. accumulation equivalence and eval microbatch tests pass in the current code;
4. no row uses landmark or changes the model boundary above.

Positive-support language requires a 3-seed CI that places a set row on a
better PPL/task-loss or accuracy versus peak-VRAM frontier than token at the
same context and effective batch. Otherwise report the result as descriptive
or null.

## Current Implementation State

Gradient accumulation and eval microbatching have been added first in the
MQAR-style update/eval helpers because they expose optimizer-update budgets
cleanly. The default path remains `grad_accum_steps=1`, preserving older
SD/MRP optimizer cadence.

Validation recorded on Lizmark in the `set-attention:latest` container:

- `python -m py_compile src/train/mqar.py scripts/run_mqar.py
  src/config/normalize.py src/config/compatibility.py
  src/train/metrics_schema.py tests/test_mqar_metrics.py` passed.
- `python scripts/check_mqar_batching_preservation.py` passed: one full-batch
  update matches two accumulated microbatches, and eval microbatching preserves
  MQAR metrics.
- Loading `configs/set_dictionary/sd9_multiresolution.yaml` resolves
  `training.grad_accum_steps=1` and `training.eval_microbatch_size=null`.

LCA generator scaffold exists (`src/data/lca_cmp.py`, `src/train/lca_cmp.py`,
`scripts/run_lca_cmp.py` with `--dry-run`/`--preflight-one-step`,
`configs/lca_cmp/`). Batching preservation is now covered by a real pytest
module, validated 2026-07-18 in the blue-demon `set-attention` container at
`mrp-lca-cmp-sd@2ded5d1`:

- `tests/test_lca_cmp_batching.py` (4 tests) and
  `tests/test_lca_cmp_generator.py` (6 tests): 10 passed. Accumulated
  microbatches match one full-batch update; eval microbatching preserves
  metrics (relative tolerance for derived PPL); generator digests are
  seed-deterministic and train/validation seeds are disjoint; default configs
  resolve `grad_accum_steps=1`, `eval_microbatch_size=null`.
- `scripts/check_lca_batching_preservation.py` passed.
- `run_lca_cmp.py --dry-run` and `--preflight-one-step` passed
  (`updates=1 train_loss=4.8507 val_loss=4.5152`, untrained chance regime as
  expected).

Next atomic action: launch the registered calibration rows (`L=1024,B=4` and
`L=2048,B=4`, token + b0--b100, seeds 0--2) behind the learnability gate. This
requires explicit user approval; no queue is authorized by this plan alone.
