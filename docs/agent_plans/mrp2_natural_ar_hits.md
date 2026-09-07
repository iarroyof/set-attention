# MRP-2: Natural Associative-Recall Hit Evaluation

Status: COMPLETE; protocol PASS; scientific result NULL.  The registered local
gate is not supportive, and the separately labeled global-recipe bridge also
finds no set-specific repeated-bigram AR advantage (see
`audit/MRP_2_natural_ar_hits.md`).

Owner: UNASSIGNED

Updated: 2026-08-12 after the global-recipe bridge completed.

## Mission

Measure whether the current exact-dense token and set models differ
specifically on natural repeated-in-context bigram completions. This is a
reduced-scale adaptation of Zoology's Pile AR-hit analysis, not a reproduction
of its 10B-token pretraining result.

## Required Retrieval Context

1. `../set_dictionary_research_main_plan.md`
2. `mrp0_reproducibility_platform.md`
3. `mrp1_matrix_closure.md`
4. `../set_dictionary_model_provenance_for_math_agent.md`
5. Zoology, Section 3.1:
   `https://arxiv.org/abs/2312.04927`
6. `../../src/data/wikitext2.py`
7. `../../scripts/run_experiment.py`

## Write Scope

- `src/data/ar_hits.py` (new)
- `scripts/evaluate_ar_hits.py` (new)
- `scripts/summarize_ar_hits.py` (new)
- `configs/eval/ar_hits/` (new)
- focused AR-hit tests
- `audit/MRP_2_natural_ar_hits.md`

Use MRP-0 checkpoint and metric APIs. Do not edit model forward code.

## Checkpoint Decision

No compatible MRP-1 checkpoints are assumed available for token-level AR-hit
evaluation. Before launch, check the local and host artifact trees once. If no
compatible final checkpoints exist, retrain exactly the registered MRP-2 rows
below with MRP-0 checkpointing and final-validation instrumentation. Existing
CSV summaries are not substitutes for token-level outputs.

## Registered Rows

This matrix is approved as of 2026-07-08. Do not launch until the evaluator,
summarizer, focused tests, and checkpoint inventory/retraining decision are
complete.

At `L=2048,B=4`, exact dense, seeds `0,1,2` applied by MRP-0:

- exact token;
- b0;
- frozen `b*`;
- b100.

Use full WikiText-2, 10 epochs, the current LR/warmup/model shape, and all set
guards. These 12 runs are the complete matrix.

## AR-Hit Definition

At input position `t`, the LM predicts target `y_t=x_{t+1}`. This target is an
AR candidate when the bigram `(x_t,x_{t+1})` occurred at positions
`(j,j+1)` for some `j<t` in the same unbroken context. Enforce this
next-token alignment and never match a future occurrence.

For every candidate record:

- target token;
- most recent and earliest prior-match lag;
- count of that bigram in the ordered training stream;
- document/chunk boundary flags.

WikiText-2 is too small for Zoology's absolute Pile threshold of 1,250
occurrences. Do not reuse that threshold as a binary primary endpoint. Report
all AR candidates stratified by training count:

- `0`;
- `1`;
- `2--5`;
- `6--20`;
- `>20`.

Also report non-AR tokens. A bin is inferential only when it contains at least
1,000 evaluated targets across the official validation split; otherwise label
it descriptive/inconclusive.

## Metrics

Per model and replicate:

- overall, AR-candidate, and non-AR loss/PPL;
- loss/PPL by training-count bin;
- loss/PPL by logarithmic lag bin;
- AR target count and fraction;
- fine- and coarse-group span-ablation delta loss/PPL for b*;
- tokenizer/vocabulary, dataset, checkpoint, and config digests.

Use count-weighted aggregation from token NLLs. Never average per-batch PPL.

## Tests

1. Handcrafted repeated bigrams produce the exact expected mask.
2. Future occurrences do not count.
3. Labels are shifted correctly.
4. Chunk/document boundaries obey the registered context policy.
5. Training-count bins include their endpoints correctly.
6. Empty and sub-1,000 bins are reported without NaN or false inference.
7. Checkpoint/vocabulary mismatch fails closed.
8. Group ablation restores model state after evaluation.

## Interpretation Gate

Support for a natural AR mechanism requires:

1. at least one inferential AR bin;
2. b* has lower paired AR NLL than both b0 and b100, with each 95% CI strictly
   below zero, using 10,000 sequence-block bootstrap resamples nested within
   applied seed; and
3. for each endpoint `e in {b0,b100}`, the difference-in-differences
   `(NLL_b*,AR-NLL_e,AR)-(NLL_b*,nonAR-NLL_e,nonAR)` has a 95% CI strictly
   below zero.

If these conditions fail, record MRP-2 as protocol `PASS` with a
null/inconclusive scientific result. Do not enlarge the dataset inside MRP-2.

## Definition Of Done

The registered rows are evaluated, counts and digests reconcile, tests and log
scans pass, the support gate is applied exactly, and
`audit/MRP_2_natural_ar_hits.md` states whether the result is supportive,
null, or inconclusive.

## Durable Handoff

Status: COMPLETE; protocol PASS; scientific result NULL under the local
candidate fiber.

Last completed action: evaluator/summarizer infrastructure implemented,
focused syntax/smoke validation completed, checkpoint inventory found no
compatible registered final checkpoints across local, blue-demon, and Lizmark
artifact trees, and the targeted 12-row checkpoint retraining completed on
blue-demon.

Files changed: `src/data/ar_hits.py`, `scripts/evaluate_ar_hits.py`,
`scripts/summarize_ar_hits.py`, `scripts/run_mrp2_ar_hit_retrain.sh`,
`configs/eval/ar_hits/`, focused tests, and `audit/MRP_2_natural_ar_hits.md`.

Commands/tests and outcomes: local Python compile passed for AR-hit and MQAR
changed files; `bash -n scripts/run_mrp2_ar_hit_retrain.sh` passed; Blue
container compile passed; Blue container lacks `pytest`, so direct smoke
assertions were used and a plain import/path smoke printed pass.

Artifacts and digests: retrained checkpoints live under
`out/mrp2_ar_hits/retrain/{token,b0,b25,b100}_seed{0,1,2}/checkpoints/final.pt`
on blue-demon. The 2026-07-13 audit found 12/12 rows with epoch-10 CSV
endpoints and final checkpoints.

Host/PID/log/ETA: blue-demon retraining process exited before the 2026-07-13
idle-server audit. Log: `logs/mrp2_ar_hit_retrain_20260709_082300.log`.

Decision or gate result: user approval recorded on 2026-07-08 in
`audit/phase_sd_status.md`; targeted registered retraining is complete, so the
next gate is the registered AR-hit evaluation over those checkpoints.

Known incident or limitation: Blue container currently lacks `pytest`; direct
container smoke checks passed but the focused pytest suite still needs a pytest
environment before final closeout. The first retrain attempt missed the HF
cache mount; the second launch exposed and fixed zero-head companion-group
encoding for b0/b100.

Next atomic action: none inside MRP-2.  The global-recipe AR bridge is complete,
remains separately labeled, and must not be pooled with registered rows.  Its
cross-recipe change is a whole-recipe contrast, not a candidate-fiber-only
effect, and its null does not prove the absence of an internal retrieval
circuit.

Inputs required: MRP-0 checkpoint API, frozen `b*`, applied seeds, full
WikiText-2, and approved host allocation.
