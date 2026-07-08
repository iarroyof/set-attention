# MRP-3: Synthetic MQAR Mechanism Study

Status: generator/trainer infrastructure READY; launches BLOCKED on MRP-1 completion and explicit approval

Owner: MRP-3 MQAR implementation worker

Updated: 2026-07-07.

## Mission

Test whether the executed fine and coarse set streams make different
contributions as associative-recall lag changes, while preserving the current
model architecture. Synthetic MQAR is trained directly; Pile pretraining is
not part of this task.

The primary claim is a group-by-lag ablation interaction. It is not a claim
that short lag equals high frequency, that long lag is a smooth signal, or that
the coarse stream alone has a longer formal receptive field.

## Required Retrieval Context

1. `../set_dictionary_research_main_plan.md`
2. `mrp0_reproducibility_platform.md`
3. `mrp1_matrix_closure.md`
4. `../set_dictionary_model_provenance_for_math_agent.md`
5. Zoology paper Appendix E:
   `https://arxiv.org/abs/2312.04927`
6. reference generator:
   `https://github.com/HazyResearch/zoology/blob/main/zoology/data/associative_recall.py`
7. `../../src/models/set_only/set_only_lm.py`, especially named group ablation
8. `../../scripts/run_experiment.py`

## Write Scope

- `src/data/mqar.py` (new)
- `src/train/mqar.py` (new)
- `scripts/run_mqar.py` (new)
- `scripts/summarize_mqar.py` (new)
- `scripts/run_mqar_matrix.sh` (new)
- `configs/mqar/` (new)
- focused MQAR tests
- `audit/MRP_3_mqar_mechanism.md`

Use MRP-0 shared APIs. Do not edit the model forward path or add a new
attention architecture.

## Generator Contract

Implement the published multi-query construction:

- vocabulary size `8192`, with disjoint key and value halves;
- unique keys and values per example;
- key/value pairs occupy the initial `2D_kv` positions;
- each key is queried once later in the sequence;
- non-query positions contain deterministic random distractors;
- query labels contain the associated value and all other labels are `-100`;
- query gaps follow the published power-law generator with `power_a=0.01`;
- train, calibration, and test sets use disjoint applied RNG seeds;
- the generator returns exact query positions, matching key positions, and
  lags.

No natural-language or Pile data enters this task.

## Registered Architecture Rows

Use exactly:

- exact token;
- b0;
- b25;
- b50;
- b75;
- b100.

Set rows retain `anchor_span`, token MLP off, anchor off, CE-only,
endpoint-window, `(2,1)` fine and `(4,2)` coarse streams. Model shape remains
`D=384`, `d_ff=1536`, 6 layers, 8 heads.

## Protocol Calibration

Calibration is nonconclusive and uses exact token plus frozen `b*`, seed 0:

- `L=512`;
- `D_kv=64`;
- 100,000 generated training examples, 3,000 calibration examples, and 3,000
  untouched test examples;
- native microbatch `B=16`;
- LR candidates `{1e-4, 3e-4, 1e-3}`;
- maximum 20,000 optimizer updates, with evaluation every 500 updates;
- select the LR and first update at which exact token calibration query accuracy is at
  least `0.99` for two consecutive evaluations.

Select the candidate reaching the condition in the fewest updates. If two
candidates tie, select the smaller LR.

If no token run meets this condition, stop MRP-3 with an incident. Do not
change model width, depth, vocabulary, or generator.

Freeze the selected LR and number of optimizer updates. Train every primary
row for exactly that update count; do not early-stop individual paper rows or
inspect the primary test split during calibration.

## Primary Matrix

Calibration and primary matrices are registrations, not launch authorization.
Do not launch either until approval is recorded by the tracker write owner in
`audit/phase_sd_status.md`.

Use:

- `L=2048`;
- `D_kv=256`;
- all six registered rows;
- applied seeds `0,1,2`;
- the calibrated LR and fixed optimizer-update budget.

Determine the common native microbatch by descending preflight
`B={4,3,2,1}`. Select the first batch at which all six rows complete one
forward/backward/update step with metric logging. This deterministic selection
defines the quality island.

Separately run the one-step capacity preflight for every row at `L=4096,B=4`
and record success or OOM. These preflights are feasibility observations only,
not trained quality results.

## Lag Bins

Aggregate query metrics in fixed bins:

- `[1,32]`;
- `[33,128]`;
- `[129,512]`;
- `[513,1024]`;
- `[1025,2047]`.

Also retain exact lag and the number of queries per bin. Empty bins are
reported and excluded from inferential aggregation.

## Metrics And Probes

Per row and applied seed:

- query-only CE, PPL, and accuracy;
- exact-sequence success rate over all queries;
- metrics by lag bin;
- peak train VRAM and maximum feasible native microbatch;
- fine- and coarse-group ablation delta CE/accuracy by lag bin;
- group-local routing entropy, top-1 concentration, and effective range;
- query count, update count, dataset/config/checkpoint digests;
- clean log scan.

Loss and accuracy denominators count only non-`-100` targets.

## Interaction Gate

For frozen `b*`, let `D_f(s)` and `D_c(s)` be the query-accuracy drop after
fine and coarse group ablation in short bins `[1,128]`, and let `D_f(l)` and
`D_c(l)` be the corresponding drop in long bins `[513,2047]`.

Define

```text
I = (D_f(s) - D_c(s)) + (D_c(l) - D_f(l)).
```

Use a seed-stratified, query-level bootstrap with 10,000 resamples. The
specialization gate passes only when:

1. the 95% CI for `I` is strictly above zero;
2. both short and long aggregates contain at least 1,000 test queries; and
3. unablated b* test accuracy is at least `0.90`.

Disposition is deterministic:

- if support conditions 2--3 pass and the CI condition 1 passes, mark MRP-4
  `NOT_TRIGGERED` because primary specialization is supported;
- if support conditions 2--3 pass but condition 1 fails, trigger exactly
  MRP-4;
- if support condition 2 or 3 fails, mark MRP-4 `NOT_TRIGGERED` because the
  primary task is inadequate for a scale-separation interpretation.

## Tests

1. Key/value and query placement invariants.
2. Unique keys/values and disjoint vocabulary halves.
3. Correct `-100` mask and target shift.
4. Exact lag metadata and bin boundaries.
5. Applied-seed reproducibility and train/test separation.
6. Count-weighted query-only loss and accuracy.
7. Empty-bin behavior.
8. Named group ablation and state restoration.
9. Baseline and set smoke steps remain causal.
10. Metadata and summarizer reject limited or malformed runs.

## Definition Of Done

Calibration is frozen, all 18 primary runs are valid, capacity preflights are
recorded separately, the interaction gate is computed exactly, and
`audit/MRP_3_mqar_mechanism.md` states the result without interpreting lag as
frequency.

## Durable Handoff

Status: generator/trainer infrastructure READY; launches BLOCKED.

Last completed action: implemented non-launched MQAR generator, trainer,
runner, summarizer, guarded matrix wrapper, configs, focused tests, and audit.

Files changed: `src/data/mqar.py`, `src/train/mqar.py`,
`scripts/run_mqar.py`, `scripts/summarize_mqar.py`,
`scripts/run_mqar_matrix.sh`, `configs/mqar/`, focused MQAR tests, this
subplan status/handoff, and `audit/MRP_3_mqar_mechanism.md`.

Commands/tests and outcomes: `python -m py_compile ...` passed for new Python
files; `bash -n scripts/run_mqar_matrix.sh` passed; Blue container pytest for
`tests/test_mqar_generator.py`, `tests/test_mqar_metrics.py`, and
`tests/test_mqar_summarizer.py` reported `7 passed`; `python
scripts/run_mqar.py --config configs/mqar/token_smoke.yaml --dry-run --device
cpu` validated config/generator/provenance; `MQAR_DEVICE=cpu
scripts/run_mqar_matrix.sh --smoke` completed one tiny update; ungated
`scripts/run_mqar_matrix.sh` refused launch with exit code `3`.

Artifacts and digests: no experiment artifacts; audit recorded in
`audit/MRP_3_mqar_mechanism.md`.

Host/PID/log/ETA: local shell only; no training process launched.

Decision or gate result: MRP-0 PASS allows infrastructure. Calibration,
primary, and capacity launches remain blocked on MRP-1 closure and explicit
approval recorded by the shared tracker owner.

Known incident or limitation: active local Python lacks `pytest`, `PyYAML`, and
a complete PyTorch install (`torch.utils` is missing), so runtime validation
uses the Blue project container. Resume and eval-only MQAR checkpoint paths are
not implemented; final checkpoint save uses the MRP-0 checkpoint payload/save
API.

Next atomic action: after MRP-1 closes and explicit launch approval is
recorded, run the registered MQAR calibration protocol; do not launch
calibration, primary, or capacity matrices before that approval.

Inputs required: strict MRP-1 closure and explicit launch approval for any
registered run.
