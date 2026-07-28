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
Outcome: OOM-censored on both hosts, including the single microbatch retry —
the candidate-gather router materializes an O(L^2) `T x C` scores tensor
(`router.py:278`). See `audit/LCA_calibration_20260718.md` "Fiber Probe
Outcome".

User-approved 2026-07-20 (option 0): exactly one follow-up probe, identical
to the above except **`router.score_mode=dense`** (dense router computes
`scores = q @ k^T` as `[B,H,T,M]` with masked invalid all-past entries,
avoiding the gathered `[B,H,T,C,d_phi]` key tensor). Precedent:
`audit/SD_8_all_past_dense_routerdense_smoke.md` records that
`score_mode=dense` with `candidate_fiber=all_past` avoids the all-past
candidate-gather OOM. This is a config-level routing implementation choice,
not an architecture change. Label: `allpast_routerdense_probe`; never pooled
with matrix rows. Batching fallback order, as runtime memory controls only:
native B4 → B2 x accum2 → B1 x accum4 (effective batch 4 held constant).
Verdict criteria: if train loss moves clearly below ln 2 / val_acc above
chance, the endpoint_window receptive-field diagnosis is confirmed; if it
fits but fails to learn, the Gate-2 cause is deeper than candidate
reachability. Chunked scoring is deferred unless this probe still OOMs.
Outcome: fits (3216 MiB) but does not learn (val_loss 0.817, val_acc 0.498);
see `audit/LCA_calibration_20260718.md` "Router-Dense Probe Verdict" —
evidence of a set-path/task mismatch (supervision sparsity, pooling, softmax
routing, top-k, no additive accumulator), NOT proof of inherent
architectural disadvantage.

User-approved 2026-07-20 (mechanistic probes; no scale rows): a small probe
series to localize the set-path/task mismatch, all on the `b25`/`L=1024`
family at diagnostic scale, none pooled with matrix rows:

- P0 infrastructure: per-update train-loss curve logging in the LCA runner
  (current `train/loss` is a run mean and can mask late learning).
- P1 `all_past` + `score_mode=dense` + full routing (`router_topk=0`/full
  top-k, as supported by the implementation): tests whether top-k=16 sparse
  selection is the bottleneck vs set compression itself.
- P2 prefix/all-position supervision variant of the same counting task
  (every position predicts its prefix count bucket), run for BOTH token and
  set rows: tests whether sparse final-token supervision is what kills set
  learning; token gets identical supervision so the comparison stays fair.
- P3 oracle set-state sanity check (explicit marker-count feature or true
  window count in the set atom): if the model still fails, the problem is
  routing/readout; if it solves, pooling/set-state learning is the issue.

These probes are diagnostic-only; they do not reopen all_past or any other
boundary change for matrix rows, and no scale rows are authorized.

User-approved 2026-07-20 (blur frontier sweep): a 3-seed diagnostic sweep to
find a set row that **matches token prefix-supervision quality at lower peak
VRAM** at L=1024. Rows: blur families {b25, b50, b75, b100}, all with
`all_past` + `score_mode=dense` + full routing (`router_topk=1023`) +
`data.supervision=prefix`, seeds 0-2, native B4, `max_updates=2000` (12
rows). Rationale: coarser atoms reduce the candidate count, shrinking the
O(L^2) router score tensor — the mechanism by which a set row can sit below
token's 2681 MiB while holding the ~0.90+ quality regime established by the
prefix3 mini calibration. Success criterion: a blur row with 3-seed mean
val_acc within noise of token prefix (0.944±0.032) and 3-seed peak VRAM
strictly below token's 2681 MiB. Diagnostic labels only
(`prefixblur_*`); not matrix rows; no larger-L launch is authorized by this
entry.

User directives 2026-07-23 (sequencing after the blur sweep), revised
2026-07-24 (tightened staging):

1. **Top-k bandwidth sweep on b75 first** (b75 is the frontier row):
   `b75/L1024/prefix/all_past/dense`, `router_topk={16,32,64,128,256,512,1023}`,
   seeds 0-2. Question: how much routing bandwidth does the winning blur
   allocation need before quality collapses? b25 only as secondary control
   if compute is cheap.
2. **L2048 pilot EARLY** (before any top-k x pooling grid; do not
   over-optimize L1024 behavior that may not transfer): rows token prefix,
   b75 full routing, b75 best-sparse-topk (or topk=256 + full if unknown),
   seed 0 smoke first, seeds 0-2 if feasible. Caveat recorded: full routing
   at L2048 may lose the VRAM advantage (dense all-past scores remain
   O(L^2)); if b75 full is near/above token VRAM there, sparse top-k
   bandwidth becomes essential, not optional.
3. **Pooling isolation only if b75 still has a quality gap after a viable
   top-k is chosen**: soft-trimmed Boltzmann (current) vs mean pooling vs
   oracle count control; fixed alpha values allowed, avoid learnable alpha
   until the previous instability is understood.
4. **Highway architecture deferred** until 1-3 are exhausted. If ever
   tested, it is an explicit new branch (`set + causal token highway`) with
   its own memory accounting and ablations: highway off / pointwise only /
   local causal conv / EMA-low-rank. `anchor_span` already provides a
   pointwise gradient path (thin_anchor + routed_repr), but that is not
   context transport — the distinction is recorded so future proposals do
   not conflate them.

User-approved 2026-07-27 (L4096 admission/frontier amendment, staged):

- **Stage A — admission/memory ONLY** (`l4096adm_*` labels, never pooled,
  no scientific claims): rows token prefix and b75 full routing
  (`all_past` + `score_mode=dense` + `router_topk=4095`), L=4096,
  native B4, seed 0, `max_updates=30` (peak-VRAM admission only; Adam
  states and attention peaks materialize within the first updates), on
  Lizmark (>24 GB headroom expected; token dense at L4096 estimated
  ~25-30 GB). The native-B4 peak is the headline number per the standing
  directive to track un-optimized memory. If a row OOMs natively, record
  the OOM, then retry with B2 x accum2 (label suffix `_mb2`) as a
  memory-control fallback — the OOM itself is an admission result.
- **Stage B — frontier rows, GATED**: launched only if Stage A shows a
  real memory asymmetry (b75 materially below token, or token OOMs while
  b75 fits) AND the user confirms after seeing admission numbers. Rows:
  token + b75full, seeds 0-2, `max_updates=4000` (the L2048 budget
  lesson), seed 0 first. Host per admission: rows <=24 GB go to Blue.
- Pooling isolation and highway work remain deferred until L4096 shows
  whether the current b75 mechanism keeps its frontier value at scale.

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
