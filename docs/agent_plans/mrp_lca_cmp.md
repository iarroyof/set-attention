# MRP-lca-cmp: Long-Context Aggregation / Compressed-Memory Comparison

Status: ACTIVE; batching runtime validation passed; LCA generator pending

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

No architecture changes are authorized by this plan.

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

## Metrics

Mandatory per row:

- task loss, task accuracy, exact/sequence accuracy when defined;
- bucketed accuracy by aggregation horizon or segment span;
- peak train VRAM, elapsed time, samples/s;
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

LCA task generator and launch scripts remain pending.

Next atomic action: finish local tests for batching preservation, then
implement the first synthetic LCA generator with a dry-run/preflight path.
