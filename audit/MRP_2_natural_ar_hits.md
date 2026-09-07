# MRP-2 Natural AR-Hit Evaluation Infrastructure

Status: COMPLETE; protocol PASS; scientific result NULL under the registered
local-candidate-fiber routing recipe.

Updated: 2026-08-12 after global-recipe bridge closure and interpretation audit.

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

## Historical Launch State

This subsection preserves the state before the registered evaluation.  The
completion record later in this file supersedes it for current decisions.

Checkpoint inventory found no compatible registered MRP-2 final checkpoints in
the local tree, the isolated blue-demon checkout, the older blue-demon
checkout, or the Lizmark checkout. Existing paper summary CSVs are not
substitutes because AR-hit evaluation requires token-level checkpoint logits.

Therefore the action at that time was targeted retraining with checkpoint
saving, not eval-only reuse. Primary AR-hit evaluation was blocked until those
final checkpoints existed; it is now complete.

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

## Historical Next Atomic Action

The next action at that time was to monitor the Blue retraining queue until all
12 final checkpoints existed, then run `scripts/evaluate_ar_hits.py` and
`scripts/summarize_ar_hits.py`. Both steps are complete below.

---

## Registered Evaluation Complete (2026-08-09, blue-demon)

Status: EVALUATION COMPLETE. Protocol PASS. Scientific result: NULL (not
supportive of a natural AR mechanism in b25 under the frozen recipe).

Driver: `scripts/run_mrp2_ar_hit_eval.sh` (commit 17482aa, empty-queue rc
bug fixed after the run; cosmetic only — no eval affected). All 12
registered checkpoints evaluated fail-closed on blue-demon in
`~/set-attention-anchor-span-sync` (8107a7b); evaluator, summarizer,
configs, and `src/data/ar_hits.py` verified byte-identical to the
registered local copies before launch. b25 rows include the registered
fine/coarse group span-ablation. One config fix was required: the token
eval config carried a stray `nhead: 8` key absent from the retrain
config, tripping the fail-closed model-digest check; fixed in b5fb670
and re-run (digest check then passed as designed).

Artifacts: `out/mrp2_ar_hits/eval/{token,b0,b25,b100}_seed{0,1,2}.{json,csv}`,
`out/mrp2_ar_hits/eval/summary.tsv`, driver TSV
`out/mrp2_ar_hits/eval/mrp2_ar_hit_eval_blue.tsv`, logs
`logs/mrp2_ar_hits/blue/` (remote). Summarizer validated 12/12 rows.

### Headline metrics (3-seed mean +- sd; NLL, count-weighted)

| row | AR NLL | non-AR NLL | overall NLL |
|---|---:|---:|---:|
| token | 5.4940+-0.0079 | 6.9544+-0.0007 | 6.8435+-0.0007 |
| b0 | 5.6366+-0.0582 | 6.9549+-0.0180 | 6.8547+-0.0208 |
| b25 (b*) | 5.6311+-0.0375 | 6.9264+-0.0219 | 6.8280+-0.0229 |
| b100 | 5.9248+-0.1028 | 7.3145+-0.0349 | 7.2089+-0.0400 |

48,546 AR targets (7.6% of 639k evaluated), all five training-count bins
inferential (>=1,000 targets); lag bins above 512 empty at L=2048,
lag_129_512 descriptive only (916 targets).

### Interpretation gate (applied exactly)

1. At least one inferential AR bin: PASS (all count bins inferential).
2. b25 lower paired AR NLL than both endpoints, 95% CI strictly below
   zero: FAILS. Paired per-seed diffs vs b0 flip sign
   (+0.066, -0.076, -0.006; mean -0.005) — no CI can sit strictly below
   zero. Vs b100 all seeds negative (mean -0.294).
3. Difference-in-differences vs each endpoint, 95% CI strictly below
   zero: FAILS vs both. DiD vs b0 mean +0.023 (mixed signs); DiD vs
   b100 positive on ALL seeds (+0.034, +0.084, +0.165; mean +0.094) —
   a strictly-below-zero CI is not reachable under seed-nested
   resampling when the per-seed quantity is positive on every seed.

Verdict: gate NOT supportive -> protocol PASS with null scientific
result, per the registered plan. Do not enlarge the dataset.

Bootstrap (added 2026-08-10, literal CIs): the evaluator was extended
to persist per-sequence NLL blocks (`collect_blocks`, be25de3) and the
summarizer now implements the registered 10,000 sequence-block bootstrap
nested within seed (paired draws shared across rows, rng seed 13). All
12 checkpoints were re-evaluated with the extended evaluator; aggregate
metrics reconcile bitwise with the first pass (e.g. b25 seed0 AR NLL
5.6355223 both times), and per-sequence blocks reconcile exactly with
the aggregate target counts. Literal gate CIs:

- vs b0: AR diff -0.0054, 95% CI [-0.0219, +0.0101] (straddles zero,
  cond2 FAIL); DiD +0.0230, 95% CI [+0.0077, +0.0375] (strictly ABOVE
  zero, cond3 FAIL).
- vs b100: AR diff -0.2937, 95% CI [-0.3127, -0.2749] (cond2 PASS);
  DiD +0.0944, 95% CI [+0.0766, +0.1118] (strictly above zero,
  cond3 FAIL).

The literal CIs confirm the null and sharpen it: b25's advantage over
the blur endpoints is significantly SMALLER on AR targets than on
non-AR targets (positive DiD CIs on both sides). Gate verdict:
supportive=False. Machine-readable gate: out/mrp2_ar_hits/eval/gate.json.

### Descriptive findings (informative, not gate outcomes)

1. **Token is the best retriever.** AR NLL token 5.494 vs b25 5.631;
   the gap is consistent on every seed (+0.089..+0.179). Under the
   frozen local-fiber recipe the set path has no retrieval advantage. This is
   direct natural-AR evidence against a set-specific repeated-bigram advantage;
   it does not establish that no internal retrieval mechanism exists.
2. **b25's overall win is a non-AR effect.** b25 beats token on non-AR
   NLL on every seed (-0.004..-0.044) while losing on AR. The WT2
   frontier win of the interior blur point is local-prediction quality,
   NOT associative recall.
3. **Blur is monotone-bad for retrieval.** AR NLL b0 5.637 < b25 5.631
   (tie) << b100 5.925; coarsening destroys AR performance — retrieval
   wants fine resolution, mirroring the LCA finding that aggregation
   wants coarse.
4. **b25's AR ability lives in the fine heads.** Fine-group ablation
   costs +3.53 AR NLL (catastrophic), coarse-group ablation +0.33
   (mild). Non-AR shows the same asymmetry (+2.91 fine vs +0.17 coarse).
5. Count effect is strong and monotone (count_0 NLL 10.49 ->
   count_gt20 3.69 for b25 seed0): models do exploit repetition
   frequency, token row included.

### Consequence for the claim boundary

This first null was measured under the frozen endpoint_window/topk16 recipe,
the same topology the LCA campaign later showed cannot support global
aggregation. That limitation motivated the separately labeled global-recipe
bridge reported below. The bridge is complete, is not part of registered
MRP-2, and must never be pooled with these rows. MRP-2 completion unblocks
MRP-5 per protocol, which still requires explicit transfer approval.

---

## New-Recipe AR Bridge (2026-08-12, blue-demon) — retrieval vs aggregation

Question: does the repaired global recipe (all_past fiber, dense scoring,
full routing topk=2047, dropout 0) change the set path's natural
associative-recall behavior? Bridge rows: token/b0/b25/b75 x seeds 0-2 at
the registered MRP-2 island (L2048/B4, WT2, 10 epochs, lr 1e-4). Bridge
rows carry NO experiment contract and are never pooled with registered
MRP-2. Driver scripts/run_mrp2_ar_hit_bridge.sh; eval configs
configs/eval/ar_hits_bridge/ (digest-matched, fail-closed); same
blocks-bearing evaluator; b25 AND b75 group ablations.

Training (all Blue, 12/12 clean): val PPL token 978.2+-9.6 @ 14677 MiB,
b0 1157.8 @ 16189, b25 1136.1 @ 15679, b75 1215.7 @ 14068. Blur memory
ordering preserved (b75 < token < b25 < b0); global-fiber LM quality
ordering b25 ~= b0 < b75, consistent with the WT2 recipe regression.

AR-hit metrics (3-seed mean +- sd, NLL):

| row | AR | non-AR | overall |
|---|---:|---:|---:|
| token nodrop | 5.5346+-0.0266 | 6.9967+-0.0109 | 6.8856 |
| b0 | 5.7084+-0.0987 | 7.1636+-0.0567 | 7.0531 |
| b25 | 5.6806+-0.0827 | 7.1467+-0.0059 | 7.0353 |
| b75 | 5.7680+-0.0751 | 7.2123+-0.0342 | 7.1026 |

Paired sequence-block bootstrap (10k resamples, nested within seed,
rng 13), key contrasts:

1. **The absolute AR-target gap vs the matched token control persists under
   the global recipe, strictly significant on every set row**: b0-token
   +0.174 CI [+0.155,+0.192]; b25-token +0.146 CI [+0.124,+0.168];
   b75-token +0.233 CI [+0.213,+0.254]. DiDs vs token are near zero for
   b0/b25 (their absolute AR deficits track their non-AR deficits); b75
   has a small positive DiD. This rejects a set-specific AR advantage but
   does not isolate an AR-only architecture deficit.
2. **The global recipe does not repair the AR-target gap and slightly
   worsens it**: b25 new-vs-old AR +0.050 CI [+0.020,+0.078]; b0
   new-vs-old AR +0.072 CI [+0.046,+0.098]. Meanwhile non-AR degrades
   MORE (cross-recipe DiD -0.171 CI [-0.195,-0.148] for b25). This
   contrast changes candidate fiber, score mode, post-score bandwidth,
   and dropout together; it must not be attributed to the fiber alone.
3. Token dropout removal alone: AR +0.041 CI [+0.034,+0.047] (dropout
   mildly helps token retrieval; DiD ~0, uniform effect).
4. **Whole-group ablation follows the head allocation**: b25 loses more
   when its six fine heads are removed (+3.85 vs +0.38 AR delta NLL for
   two coarse heads), while b75 loses more when its six coarse heads are
   removed (+3.42 vs +0.60 for two fine heads). Because group sizes
   differ and the readout is nonlinear, this establishes allocation
   dependence, not a per-head effect or the absence of a retrieval circuit.

Verdict: **the null for demonstrated set-specific repeated-bigram recall
advantage is recipe-robust.** Across two
maximally different operating points (local fiber/topk16/dropout 0.1 and
global fiber/full-routing/nodrop), set-dictionary attention does not beat
the token control on the AR proxy, and the global recipe moves absolute
AR NLL in the wrong direction. Combined with LCA (where global readback
is required for aggregation), this shows that aggregation success does
not transfer to repeated-bigram AR specialization. It does not prove
that every AR label requires retrieval, that no internal retrieval
computation exists, or that candidate fiber alone caused the cross-recipe
change.

Artifacts: out/mrp2_ar_hits_bridge/{retrain,eval}/ (Blue),
TSVs mrp2_ar_hit_bridge_blue.tsv, logs logs/mrp2_ar_hits_bridge/blue/
(remote). All rows single-host (blue-demon); no host mixing.
