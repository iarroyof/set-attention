# MRP-3: Synthetic MQAR Mechanism Study

Status: COMPLETE; NULL/INCONCLUSIVE MECHANISM RESULT

Owner: MRP-3 MQAR implementation worker

Updated: 2026-07-15 after primary matrix completion and summarization.

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

Calibration, primary, and capacity preflights are approved as of 2026-07-08.
Run them in the registered order: calibration first, then common-batch
preflight, then primary/capacity according to the frozen calibration result.

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

Status: COMPLETE; NULL/INCONCLUSIVE MECHANISM RESULT.

Last completed action: token LR calibration selected `lr=0.001` at update
`12500`; B4 common-batch preflight passed for the six primary rows; primary
matrix launched on blue-demon GPU0; registered L4096/B4 capacity preflight
completed on lizmark after repairing the clean runtime checkout. The
blue-demon shutdown interrupted the primary matrix after 8/18 rows; the
2026-07-13 resume completed the remaining 10 rows by 2026-07-15. The strict
summarizer accepted all 18 registered endpoint CSVs.

Files changed: `src/data/mqar.py`, `src/train/mqar.py`,
`scripts/run_mqar.py`, `scripts/summarize_mqar.py`,
`scripts/run_mqar_matrix.sh`, `scripts/run_mqar_calibration.sh`,
`configs/mqar/`, focused MQAR tests, this subplan status/handoff, and
`audit/MRP_3_mqar_mechanism.md`.

Commands/tests and outcomes: `python -m py_compile ...` passed for new Python
files; `bash -n scripts/run_mqar_matrix.sh` and
`scripts/run_mqar_calibration.sh` passed; Blue container pytest for
`tests/test_mqar_generator.py`, `tests/test_mqar_metrics.py`, and
`tests/test_mqar_summarizer.py` reported `7 passed`; `python
scripts/run_mqar.py --config configs/mqar/token_smoke.yaml --dry-run --device
cpu` validated config/generator/provenance; `MQAR_DEVICE=cpu
scripts/run_mqar_matrix.sh --smoke` completed one tiny update; ungated
`scripts/run_mqar_matrix.sh` refused launch with exit code `3`; reduced-cadence
calibration validation emitted two update-indexed rows for each token LR
candidate with explicit gate columns.

Artifacts and digests: primary summary is
`out/mqar_primary_B4_lr0p001_u12500/summary.tsv`; capacity preflight artifacts
live under `out/mqar_capacity_preflight_L4096_B4_lr0p001_u12500`; audit
recorded in `audit/MRP_3_mqar_mechanism.md`.

Host/PID/log/ETA: blue-demon isolated checkout
`~/set-attention-anchor-span-sync`; prior primary container/driver exited
before the 2026-07-13 idle-server audit. Log:
`logs/mrp3_mqar_primary_B4_lr0p001_u12500_20260709_083600.log`. Completed rows:
token/b0/b25/b50/b75/b100 seed 0 plus token/b0 seed 1. First incomplete row:
`b25_seed1_B4`, whose CSV is an empty stub with no final checkpoint.

Resume launched on 2026-07-13 using the hardened runner synced into Blue and
completed on 2026-07-15:

| host | GPU | container at health check | rows | log | state |
|---|---:|---|---|---|---|
| blue-demon | 0 | `7faed36a1d47` | seed 1: b25, b50, b75, b100 | `logs/mrp3_mqar_resume_seed1_remaining_gpu0_20260713.log` | complete |
| blue-demon | 1 | `fd1add2829a2` | seed 2: token, b0, b25, b50, b75, b100 | `logs/mrp3_mqar_resume_seed2_all_gpu1_20260713.log` | complete |

Decision or gate result: MRP-0 PASS and MRP-1 PASS are complete. User approval
for registered MRP-3 calibration/primary/capacity preflights was recorded on
2026-07-08 in `audit/phase_sd_status.md`. The completed primary MQAR matrix is
scientifically null/inconclusive: all rows stayed near chance, with frozen
`b25` mean accuracy `0.0002474`, far below the registered `0.90` support
threshold. MRP-4 is therefore `NOT_TRIGGERED` because the primary task is
inadequate for scale-separation interpretation, not because specialization was
confirmed.

Known incident or limitation: the first approved calibration launch was stopped
and quarantined because it evaluated only at the endpoint; see
`audit/incident_mrp3_calibration_eval_cadence_20260708.md`. Active local Python
lacks `pytest`, `PyYAML`, and a complete PyTorch install (`torch.utils` is
missing), so runtime validation uses the project container. Resume and
eval-only MQAR checkpoint paths are not implemented; final checkpoint save uses
the MRP-0 checkpoint payload/save API. The temporary Lizmark
`~/set-attention-anchor-span-sync` checkout was used for recovery provenance
only and is now deprecated. New host launches use the original
`~/set-attention` path after source sync and container validation.

Next atomic action: do not relaunch MRP-3 or trigger MRP-4 from this result.
Proceed to MRP-2 AR-hit evaluation; use MRP-3 only as a completed null
mechanism-probe result in status/paper context.

Inputs required: none for MRP-3.
