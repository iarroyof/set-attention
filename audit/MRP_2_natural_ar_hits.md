# MRP-2 Natural AR-Hit Evaluation Infrastructure

Status: REGISTERED CHECKPOINT RETRAINING RUNNING ON BLUE-DEMON.

Updated: 2026-07-08.

## Scope Completed

- Added natural repeated-bigram AR-hit detection utilities.
- Added count-weighted NLL/PPL aggregation by AR status, training-count bin,
  and lag bin.
- Added fail-closed checkpoint evaluation script using the MRP-0 checkpoint
  loader's model, dataset, and tokenizer digest checks.
- Added optional fine/coarse group span-ablation evaluation through the public
  `set_span_ablation_mode` hook when exposed by the model.
- Added summarizer validation for registered MRP-2 rows and descriptive
  sub-threshold bins.
- Added focused tests for mask semantics, future-match exclusion, label shift,
  record-boundary policy, bin endpoints, empty bins, checkpoint mismatch, and
  ablation state restoration.
- Added `scripts/run_mrp2_ar_hit_retrain.sh`, a fail-closed launcher for only
  the registered checkpoint-producing rows: token, b0, b25, and b100 at
  `L=2048,B=4`, exact dense, seeds `0,1,2`.

## Launch State

Checkpoint inventory found no compatible registered MRP-2 final checkpoints in
the local tree, the isolated blue-demon checkout, the older blue-demon
checkout, or the Lizmark checkout. Existing paper summary CSVs are not
substitutes because AR-hit evaluation requires token-level checkpoint logits.

Therefore the active launch is targeted retraining with checkpoint saving, not
eval-only reuse. Primary AR-hit evaluation remains blocked until those final
checkpoints exist.

An initial Blue launch attempt at
`logs/mrp2_ar_hit_retrain_20260708_175000.log` failed before training because
the container did not mount the cached WikiText dataset. This is a
launch-environment failure only; it produced no checkpoint and no training
endpoint.

The second launch completed token seed 0, then stopped on b0 because the
launcher encoded all-fine/all-coarse rows with a zero-head companion group.
The config validator correctly rejected `num_heads=0`. The launcher was fixed
to encode b0 as a single fine group and b100 as a single coarse group, matching
the MQAR wrapper convention.

Corrected active launch:

- host: blue-demon;
- checkout: `~/set-attention-anchor-span-sync`;
- GPU: `1`;
- container: `a1181ee11103`;
- docker PID: `2911727`;
- Python PID: `2911878`;
- log: `logs/mrp2_ar_hit_retrain_20260709_082300.log`;
- active row at relaunch check: b0, seed `0`;
- queue: token, b0, b25, b100 for seeds `0,1,2`;
- output root: `out/mrp2_ar_hits/retrain/`.

## Verification

- `python -m py_compile src/data/ar_hits.py scripts/evaluate_ar_hits.py scripts/summarize_ar_hits.py tests/test_ar_hits.py tests/test_ar_hits_summarizer.py`
- Pure direct-load smoke check for AR-hit mask, record-boundary reset, count-bin
  endpoints, and accumulator metrics.
- Pure summarizer smoke check with an incomplete descriptive row.
- Blue container compile passed for the AR-hit evaluator, summarizer, data
  utilities, and focused tests.
- Blue container lacks `pytest`; a plain-Python import/path smoke passed, and
  the full focused assertions were run through a direct harness with exit code
  `0`. Treat this as a functional smoke, not a pytest-suite substitute.
- `bash -n scripts/run_mrp2_ar_hit_retrain.sh` passed.

## Next Atomic Action

Monitor the Blue retraining queue until all 12 final checkpoints exist. Then
run `scripts/evaluate_ar_hits.py` on the same registered rows and
`scripts/summarize_ar_hits.py`.
