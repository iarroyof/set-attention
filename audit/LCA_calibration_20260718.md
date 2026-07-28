# MRP-lca-cmp Calibration (L1024/L2048) — Gate 2 Evaluation

Date: 2026-07-18
Status: COMPLETE — Gate 1 PASS, **Gate 2 FAIL** (all set rows at chance). STOP per
plan `docs/agent_plans/mrp_lca_cmp.md` ("if the task is too easy/too
hard/degenerate, stop and adjust the generator before launching scale rows").
No scale rows launched; no new queue authorized.

## Provenance

- Host: blue-demon (`192.168.241.149`, repo `~/set-attention`), branch
  `mrp-lca-cmp-sd`, host checkout `2ded5d1` (runtime-equivalent to origin
  `91cb271`; diff touches only `audit/`, `docs/`, `tests/`).
- Driver: `scripts/run_lca_cmp_calibration.sh` (36 cells: islands L=1024/B4 and
  L=2048/B4; families token + set b0/b25/b50/b75/b100; seeds 0/1/2;
  `max_updates=2000`, `grad_accum_steps=1`, `eval_microbatch_size=null` —
  native batching, so each row's `train/peak_vram_mib` is the registered
  native-batch peak).
- Queue window: 2026-07-18 11:38–19:23 (~7.7 h, 2 GPU workers round-robin).
- Endpoint validity: 36/36 rows exit 0, complete final CSV
  (`training.seed_applied=True`, finite train/val metrics), 36 `.done`
  markers, strict log scan clean (no traceback/NaN/Inf/OOM in any of the 36
  per-row logs). OOM registry `oom_registry.tsv` is header-only (no censoring,
  no microbatch retries).

## Results Summary (3 seeds per cell; chance = 0.5 acc, ln 2 ≈ 0.693 loss)

| Family | Island | val_acc (seeds 0/1/2) | val_acc mean ± sd | val_loss mean | native peak VRAM (MiB) |
|---|---|---|---|---|---|
| token | L1024 | 0.770 / 0.618 / 0.752 | 0.713 ± 0.083 | 0.599 | 2680.6 |
| b0 (8 fine) | L1024 | 0.502 / 0.500 / 0.499 | 0.500 ± 0.001 | 0.736 | 3349.8 |
| b25 (6f+2c) | L1024 | 0.500 / 0.500 / 0.506 | 0.502 ± 0.004 | 0.799 | 2637.3 |
| b50 (4f+4c) | L1024 | 0.513 / 0.504 / 0.500 | 0.506 ± 0.006 | 0.741 | 2302.5 |
| b75 (2f+6c) | L1024 | 0.500 / 0.505 / 0.500 | 0.502 ± 0.003 | 0.812 | 2014.7 |
| b100 (8 coarse) | L1024 | 0.505 / 0.500 / 0.500 | 0.502 ± 0.003 | 0.829 | 1395.8 |
| token | L2048 | 0.494 / 0.634 / 0.592 | 0.573 ± 0.072 | 0.737 | 9123.9 |
| b0 (8 fine) | L2048 | 0.524 / 0.498 / 0.488 | 0.503 ± 0.018 | 0.715 | 11038.8 |
| b25 (6f+2c) | L2048 | 0.506 / 0.506 / 0.506 | 0.506 ± 0.000 | 0.817 | 8465.1 |
| b50 (4f+4c) | L2048 | 0.502 / 0.500 / 0.483 | 0.495 ± 0.010 | 0.721 | 6891.4 |
| b75 (2f+6c) | L2048 | 0.509 / 0.505 / 0.484 | 0.499 ± 0.014 | 0.705 | 5773.6 |
| b100 (8 coarse) | L2048 | 0.493 / 0.518 / 0.495 | 0.502 ± 0.014 | 0.815 | 3937.6 |

Peak VRAM is identical across seeds within a cell (shape-determined). Memory
ordering note: at L2048 all mixed/coarse set rows (b25–b100) sit below token
(3937.6–8465.1 vs 9123.9 MiB), but b0 exceeds token at both islands; quality
is at chance for every set row, so no Pareto statement is possible.

## Gate Evaluation (plan "Gates" section)

1. **Token dense learnability: PASS at L1024** (0.77/0.62/0.75, all above the
   0.5 chance line at 2000 updates). **Marginal at L2048** (0.49/0.63/0.59;
   one seed at chance).
2. **At least one set row nondegenerate: FAIL.** All 30 set rows (all blur
   levels, both islands, all seeds) sit at val_acc 0.483–0.524 with val_loss
   0.696–1.017 vs 2-class chance loss ln 2 ≈ 0.693.
3. Accumulation/eval-microbatch tests: PASS (validated pre-launch,
   `tests/test_lca_cmp_batching.py`, `tests/test_lca_cmp_generator.py`,
   `scripts/check_lca_batching_preservation.py`).
4. Model boundary respected: PASS (no landmark, no architecture change; set
   rows keep `endpoint_window`, `anchor_span`, `strict_past`).

Gate 2 failure triggers the plan's STOP condition: no scale rows.

## Diagnosis — why set rows stay at chance

The evidence converges on a **structural receptive-field ceiling**, not
optimization budget, generator difficulty, or generalization:

1. **Training loss is flat at chance for the whole run (optimization never
   starts), not a generalization gap.** The logged `train/loss` is the
   token-weighted mean over all 2000 updates
   (`src/train/lca_cmp.py:87-95`). Set rows: mean train loss 0.83–0.87,
   i.e. *at or above* ln 2 for essentially every update; final val loss ≈
   0.70–1.02. Token rows: mean train loss ≈ 0.81 but final val loss
   0.48–0.88, i.e. token loss fell sharply during the run (late-run
   learning). If set rows were merely generalizing poorly, train loss would
   have dropped; it did not move at all.

2. **The routing diagnostics show the set pathway is degenerate by
   construction in this configuration.** Every set row's ausa columns report
   `candidate_count_max = 2.0` (mean ≈ 1.99) for *both* fine and coarse
   groups, `routing_entropy ≈ 0.688 ≈ ln 2` (normalized ≈ 0.993–0.998,
   i.e. uniform over the available candidates), and
   `delta_routing_entropy = 0.0` exactly — the router never specialized in
   2000 updates. `router_topk=16` is a no-op over a 2-candidate fiber.

3. **Root cause in the bank construction.** With
   `candidate_fiber=endpoint_window` + `set_causality_mode=strict_past`
   (`src/models/set_only/banks.py:182-190`), token *t*'s candidate sets are
   exactly the sets whose endpoint falls in `(t − window, t]`. For the fine
   bank (w=2, s=1) that is **2 sets**; for the coarse bank (w=4, s=2) also
   **2 sets** — independently verified by replicating `build_window_bank`
   for L=1024/2048 (num_sets 1023/2047 fine, 511/1023 coarse; candidates per
   token max 2, last-token candidates 2, matching the ausa columns).
   Supervision exists only at the last position (query token;
   `src/data/lca_cmp.py:101,113`), and that query can attend only to sets
   ending at the last ~2–4 positions. Information can only diffuse backward
   ~O(window) positions per layer, so after 6 layers the query's effective
   receptive field is on the order of **tens of tokens** (≲ ~20 fine /
   ~50 coarse), while the count statistic is spread uniformly over 1023/2047
   positions. A marker count visible in a ~20–50-token window is nearly
   uninformative of the global count (window/global count correlation
   ≈ √(w_eff/L) ≲ 0.2), so chance-level accuracy is the structural optimum
   for the set rows. This also explains why b0 (all-fine) fails identically
   to b100 (all-coarse): the bottleneck is the endpoint-window candidate
   fiber, not the fine/coarse resolution mix.

4. **Not a generator-degeneracy issue for token.** Token dense attention
   (full causal) learns at L1024 (0.77/0.62/0.75) and partially at L2048,
   so the task is learnable within budget when the whole context is
   reachable. Note the task is harder than its name suggests: class boundary
   is a single marker count (low ∈ [threshold−8, threshold−1], high ∈
   [threshold, threshold+8], threshold = 123 at L1024), i.e. near-exact
   counting at the boundary — consistent with token's seed variance and its
   weaker L2048 result.

5. **Logs carry no competing explanation.** Strict scan clean; only benign
   warnings: `multiscale disabled; using single-scale bank`
   (`set_only_lm.py:311` — refers to the unimplemented `multiscale` flag,
   unrelated to the multiresolution head groups) and set-Gram-spectrum
   overconstraint notices from diagnostics. Gradient-probe ausa columns are
   NA for all rows (probe not wired for this runner), but `delta_routing_*
   = 0.0` and the flat train loss already localize the failure.

## Recommendation

**Option (c), sharpened: this is a structural task/model mismatch in the set
path, not a budget or generator-difficulty problem — investigate the
endpoint_window receptive-field ceiling before any further calibration
spend.** Concretely:

- Extending the budget (a) cannot help: train loss is flat *from the start*
  and the information the loss requires is not in the query's receptive
  field at any update count.
- Adjusting generator difficulty (b) cannot help either: any task whose
  supervision depends on content beyond ~50 tokens from the last position is
  invisible to the set pathway under the registered boundary; making the
  task easier (wider count separation) keeps it invisible, and making the
  signal local would defeat the purpose of a long-context aggregation
  comparison.
- Do **not** read these 30 rows as evidence about set-dictionary quality
  (option d is not warranted): the rows measure a degenerate
  routing/candidate configuration, not the mechanism's aggregation capacity.
  They are usable as negative calibration evidence about the
  `endpoint_window` fiber itself.
- Suggested next check before any relaunch (needs plan amendment, since the
  Active Model Boundary pins `candidate_fiber=endpoint_window`): repeat one
  set-row calibration (e.g. b25, L1024, seed 0, same budget) with
  `candidate_fiber=all_past` — already implemented
  (`banks.py:176-190`, `set_only_lm.py:199-200`) — which gives each token
  all past set endpoints as candidates and makes `router_topk=16`
  meaningful. If that single probe moves train loss below ln 2, the
  diagnosis above is confirmed end-to-end and the plan can be amended with
  the fiber as a labeled configuration change. Secondary checks: log
  per-update train loss (currently only the run mean is retained) and wire
  the ausa gradient probes for this runner so router-gradient flow is
  directly observable.

No adjustment has been implemented; this audit is diagnosis only, per
instructions.

## Fiber Probe Outcome (2026-07-19/20): all_past is memory-infeasible exact-dense

The user approved exactly one fiber-diagnosis probe (`b25`, `L=1024`, seed 0,
`candidate_fiber=all_past`, `max_updates=2000`, all other settings identical
to the `b25|1024|b4|0|native` row; plan section "Approved Diagnostic Probes").
The probe could not be trained on either host:

| Attempt | Host | Batching | Result |
|---|---|---|---|
| native B4 | blue-demon (24 GiB) | `batch 4 x accum 1` | OOM at first forward: router candidate-gather scores tensor requested 26.97 GiB (`router.py:278`, `(q.unsqueeze(3) * k_g).sum(-1)`) |
| native B4 | lizmark (49 GiB) | `batch 4 x accum 1` | OOM: 29.66 GiB already allocated + same 26.97 GiB request > 47.5 GiB |
| microbatch | lizmark (49 GiB) | `batch 2 x accum 2`, effective batch 4 | OOM: 46.2 GiB allocated + 13.49 GiB request; single registered microbatch retry exhausted |

All three CSVs are header-only (no trained row). Logs:
`logs/lca_cmp/blue/lcacmp_b25_L1024_seed0_allpast_probe.log`,
`logs/lca_cmp/lizmark/lcacmp_b25_L1024_seed0_allpast_probe{,_microbatch}.log`.

Interpretation:

- The receptive-field diagnosis (endpoint_window fiber sees only ~2 candidate
  sets per query) stands as the best explanation of the Gate-2 failure; the
  confirming training probe is blocked by memory, not by ambiguity.
- The block is itself a registered finding: the `all_past` fiber makes the
  multihead candidate-gather router materialize a `T x C` score tensor
  (C = all past set endpoints, ~L/stride), i.e. **O(L^2) router memory in the
  exact-dense implementation**. At `L=1024, D=384` this exceeds a 49 GiB GPU
  even at microbatch 2. The fiber that would give the query access to the
  full context is precisely the one whose router gather is quadratically
  memory-bound — directly relevant to the memory-frontier story.
- This is censoring, not a quality result: nothing about all_past quality can
  be claimed. The registered retry policy (one microbatch retry) is
  exhausted; any further attempt (batch 1 x accum 4, or a chunked candidate
  gather) requires a new explicit decision.

## Router-Dense Probe Verdict (2026-07-20): reachability is not the bottleneck

Following the user's option-0 directive, the same probe cell
(`b25`, `L=1024`, seed 0, `max_updates=2000`, native B4, effective batch 4)
was rerun with `candidate_fiber=all_past` + **`router.score_mode=dense`**
(label `allpast_routerdense_probe`, Lizmark GPU 0, exit 0, strict log scan
clean; precedent `audit/SD_8_all_past_dense_routerdense_smoke.md`).

| Row | Fiber / router | Peak VRAM | train/loss (run mean) | val/loss | val/acc |
|---|---|---:|---:|---:|---:|
| `b25|1024|b4|0|native` | endpoint_window / candidate_gather | 2637.3 MiB | ~0.86 (flat) | 0.943 | 0.500 |
| `allpast_routerdense_probe` | all_past / dense | 3216.0 MiB | 0.814 | 0.817 | 0.498 |

Diagnostics confirm the fiber change did what it should:
`candidate_count_max=1023` (vs 2 with endpoint_window), effective routed
candidates mean ~69-96 per query (router_topk=16 per head now meaningful),
router entropy non-uniform (norm 0.72-0.74). Memory is modest (3216 MiB) —
the candidate-gather OOM was the gathered `[B,H,T,C,d_phi]` key duplication,
not the O(L^2) scores, matching the user's analysis.

Verdict per the pre-registered criteria: **the probe fits but does not
learn** — at the same budget where token dense reached val_acc 0.77, the
set path with full all-past candidate reach remains at chance (val_loss
0.817 > ln 2). Phrasing per user review (2026-07-20): endpoint-window
reachability is definitely broken for global aggregation; all-past+dense
fixes topological reachability but does not by itself make the current set
path learn a global count under sparse final-token supervision. This is
**not** proof of an inherent architecture disadvantage; it is evidence of a
current set-path/task mismatch involving supervision sparsity, pooling,
softmax routing, top-k selection, and the lack of an explicit additive
accumulator. Final-token-only supervision is structurally more damaging for
endpoint-window set attention than for token attention: the token final
query attends directly to all positions, while the set final query reaches
distant markers only through O(window)-per-layer diffusion. Caveats: single
seed, single budget, `train/loss` is the run mean (late learning would be
masked in that column, but endpoint val metrics are unambiguous).

This result does not reopen all_past or score_mode=dense for matrix rows;
both remain diagnostic-only labels. Registered artifacts:
`out/lca_cmp/calibration/b25/L1024/lcacmp_b25_L1024_seed0_allpast_routerdense_probe.{csv,json}`,
logs under `logs/lca_cmp/{blue,lizmark}/lcacmp_b25_L1024_seed0_allpast_*.log`.

## Mechanistic Probe Series (2026-07-20): mismatch localized — NOT architectural

User-approved probe series P0-P3 (plan "Approved Diagnostic Probes") on the
`b25`/`L=1024`/seed 0 cell, 2000 updates, native B4, Lizmark, all strict-scan
clean. P0 added per-update train-loss curve logging
(`<csv_stem>_curve.csv`). Implementation: `data.supervision={endpoint,prefix}`
and `data.oracle_count_token` generator options in `src/data/lca_cmp.py`
(defaults unchanged); driver `scripts/run_lca_cmp_mechanistic_probes.sh`;
11 new tests in `tests/test_lca_cmp_probes.py` (21/21 LCA tests pass in the
Lizmark container).

| Probe | Change vs router-dense probe | val_loss | val_acc | train-loss curve |
|---|---|---:|---:|---|
| baseline: `b25` endpoint_window (calibration) | — | 0.943 | 0.500 | flat ≥ ln 2 |
| baseline: all_past + dense, topk=16 | — | 0.817 | 0.498 | (no curve; run mean 0.814) |
| P1 `allpast_routerdense_fulltopk_probe` | `router_topk=1023` (full routing) | **0.405** | **0.827** | 5.82 → 0.053 |
| P2 `prefixsup_b25_probe` | prefix supervision (topk=16 kept) | **0.433** | **0.787** | 5.24 → 0.474 |
| P2 `prefixsup_token_probe` | prefix supervision, token dense | 0.220 | 0.903 | 5.13 → 0.217 |
| P3 `oracle_b25_probe` | oracle count token (topk=16, endpoint sup.) | **0.126** | **0.985** | 5.68 → 0.00003 |

Verdicts:

- **P1 — top-k=16 sparse selection was a primary bottleneck.** With full
  routing the same set row learns to 0.827 val_acc (vs 0.498), same memory
  (3216 MiB). The failure was not set compression per se.
- **P2 — sparse final-token supervision was an independent bottleneck.**
  Prefix supervision alone (top-k=16 kept) lifts the set row to 0.787.
  Token still leads (0.903), so a residual quality gap remains under equal
  supervision.
- **P3 — routing/readout works when set states carry the count.** Oracle
  count tokens give 0.985; the remaining deficit lives in
  pooling/set-state learning, not in the router or the LM readout.
- Combined conclusion (user formulation): the calibration Gate-2 failure is
  a **set-path/task mismatch** — top-k sparsity + final-token-only
  supervision + learned pooling — and **not** evidence of an inherent
  architectural disadvantage of set-dictionary attention for global
  aggregation. Single seed/budget; no matrix rows affected; all_past and
  score_mode=dense remain diagnostic-only labels.

Artifacts: `out/lca_cmp/calibration/{b25,token}/L1024/*{fulltopk,prefixsup,oracle}*.{csv,json}` +
`*_curve.csv`; logs `logs/lca_cmp/lizmark/lcacmp_*{fulltopk,prefixsup,oracle}*.log`;
driver `scripts/run_lca_cmp_mechanistic_probes.sh`.

## Combined Probe + Prefix 3-Seed Mini Calibration (2026-07-20)

The user-approved combined probe (`all_past` + `score_mode=dense` +
`router_topk=1023` + `data.supervision=prefix`, b25/L1024/seed0, 2000
updates, native B4, Lizmark — launched before the blue-preference rule was
stated) completed clean:

| Row | val_loss | val_acc | Peak VRAM | Train curve |
|---|---:|---:|---:|---|
| token, prefix supervision (P2) | 0.220 | 0.903 | 2678.7 MiB | 5.13 → 0.217 |
| **b25 combined (`combined_probe`)** | **0.141** | **0.936** | 3215.9 MiB | 5.22 → 0.127 |

The two independent fixes compose, and at this budget/diagnostic scale the
set row **exceeds** token dense under identical prefix supervision. This is
a single-seed diagnostic label row, not a matrix result: all_past + full
routing is O(L^2) in scores and the memory-frontier question at larger L is
unresolved.

Follow-up (user-approved): 3-seed mini calibration at L1024 only —
`prefix3_token`, `prefix3_b25_fulltopk`, `prefix3_b25_topk16` (9 rows, same
budget, native B4) launched on blue-demon per the ≤24 GB host-preference
rule. Driver `scripts/run_lca_prefix_3seed.sh`; artifacts under
`out/lca_cmp/prefix3/`, logs `logs/lca_cmp/blue/prefix3_*.log`.

**3-seed results (2026-07-23, all rows exit 0, strict scan clean, Blue idle
after completion):**

| Family | val_acc mean ± sd | val_loss mean ± sd | Peak VRAM |
|---|---|---|---|
| token, prefix | 0.9443 ± 0.0317 | 0.136 ± 0.083 | 2680.6 MiB |
| b25 full routing, prefix | 0.8956 ± 0.0407 | 0.255 ± 0.127 | 3215.9 MiB |
| b25 top-k=16, prefix | 0.8171 ± 0.0114 | 0.383 ± 0.018 | 3216.0 MiB |

Verdict: the single-seed combined result (0.936 > 0.903) was **within seed
noise** — across seeds token prefix leads (0.944 vs 0.896; ~1.9 SEM gap,
overlapping intervals at n=3). The robust findings are: (a) the set row is
now in token's quality regime rather than at chance — the routing/supervision
repairs hold across seeds; (b) full routing beats top-k=16 consistently
(+0.079 acc, tight top-k16 sd 0.011) at identical VRAM; (c) no set-over-token
advantage is claimed. The remaining ~0.05 gap localizes to
pooling/set-state learning (per P3) and possibly the pointwise-only token
residual. The O(L^2) full-routing memory question at L>=2048 remains the
open frontier issue.

## Blur Frontier Sweep (2026-07-24): b75 matches token quality at lower VRAM

User-approved sweep (plan "Approved Diagnostic Probes"): blur families
{b25,b50,b75,b100} x seeds 0-2, all_past + score_mode=dense +
router_topk=1023 + prefix supervision, L=1024, native B4, 2000 updates,
blue-demon. 12/12 rows exit 0, strict scan clean. b25 rows replicate the
prefix3 fulltopk rows bit-for-bit (determinism check).

| Family | val_acc mean ± sd | Peak VRAM | vs token VRAM |
|---|---|---:|---:|
| token prefix (prefix3) | 0.9443 ± 0.0317 | 2680.6 MiB | — |
| b25 | 0.8956 ± 0.0407 | 3215.9 MiB | +20% |
| b50 | 0.8974 ± 0.0435 | 2636.2 MiB | −2% |
| **b75** | **0.9233 ± 0.0157** | **2346.7 MiB** | **−12%** |
| b100 | 0.8379 ± 0.0262 | 1710.7 MiB | −36% |

**Success criterion met**: b75 sits within seed noise of token quality
(gap 0.021 acc ≈ 1.2 SEM with overlapping intervals; tighter sd than token)
at strictly lower peak VRAM (−334 MiB, −12.5%). This is the first
diagnostic-label set row that matches token quality at lower memory on the
LCA aggregation task. Notable non-monotonicity: b75 > b50 ≈ b25 in quality
while VRAM falls with blur — coarse atoms (w4/s2) both shrink the O(L^2)
score tensor and appear to be better evidence units for counting; b100
(all-coarse) then loses too much detail (−0.09 vs b75).

Caveats: L1024 only, 2000 updates, synthetic task, diagnostic labels — not
matrix rows. The L>=2048 frontier question (does b75 hold quality vs token
while token VRAM grows O(L^2) to 9.1+ GiB?) is the decisive next
experiment and requires its own plan amendment. Per user staging, the
top-k bandwidth sweep {16,32,64,128,256,full} precedes any scale-up.

Driver `scripts/run_lca_prefix_blur_sweep.sh`; artifacts
`out/lca_cmp/prefixblur/`; logs `logs/lca_cmp/blue/prefixblur_*.log`.

## Top-k Bandwidth Sweep (2026-07-24): aggregation needs routing bandwidth

User staging item 1: b75/L1024/prefix/all_past/score_mode=dense,
router_topk={16,32,64,128,256,512,1023} x seeds 0-2, native B4,
2000 updates, blue-demon. 21/21 rows exit 0.

| router_topk | val_acc mean ± sd | Peak VRAM mean |
|---:|---:|---:|
| 16 | 0.7836 ± 0.0208 | 2348.5 MiB |
| 32 | 0.8181 ± 0.0032 | 2351.7 MiB |
| 64 | 0.8501 ± 0.0018 | 2360.2 MiB |
| 128 | 0.8509 ± 0.0152 | 2377.7 MiB |
| 256 | 0.8707 ± 0.0301 | 2407.7 MiB |
| 512 | 0.8821 ± 0.0276 | 2374.7 MiB |
| 1023 (full) | 0.9233 ± 0.0157 | 2346.7 MiB |

Findings: (a) quality rises monotonically with routing bandwidth over the
whole range — no plateau before full routing; global counting needs many
weak evidence paths, and top-k selection discards them. (b) Peak VRAM is
**flat** across topk (2347-2408 MiB, spread <3%): in score_mode=dense the
[B,H,T,M] score tensor is materialized regardless of top-k, so top-k
sparsity buys no memory in this router mode. Sparse top-k only becomes a
memory lever if scoring itself is made sparse (candidate-gather without
dense scores, or chunked sparse scoring). (c) topk=1023 replicates the
prefixblur b75 row (0.9233 ± 0.0157 @ 2346.7 MiB) — determinism holds.

Driver `scripts/run_lca_topk_sweep.sh`; TSV
`out/lca_cmp/topksweep/topksweep_blue.tsv` (pulled); per-row CSVs on blue
under `out/lca_cmp/topksweep/b75/L1024/`; logs
`logs/lca_cmp/blue/topksweep_*.log`.

## L2048 Pilot (2026-07-24, Lizmark, seed 0): VRAM advantage holds, quality gap widens

User staging item 2 (moved early per user directive): L=2048, prefix
supervision, all_past + score_mode=dense, native B4, 2000 updates,
seed 0, Lizmark. 3/3 rows exit 0, clean.

| Row | val_acc | val_loss | Peak VRAM |
|---|---:|---:|---:|
| token prefix | 0.9440 | 0.127 | 9123.9 MiB |
| b75 full routing (topk=2047) | 0.8158 | 0.407 | 7201.1 MiB |
| b75 topk=256 | 0.7571 | 0.508 | 7325.4 MiB |

Findings: (a) b75 full routing keeps a **−21% VRAM advantage** over token
at L2048 (7201 vs 9124 MiB) — the blur family scales its memory edge with
L. (b) The quality gap widens vs L1024: 0.816 vs 0.944 (−0.128) compared
to −0.021 at L1024. Token at L2048 is unchanged (0.944 both L), so the
gap is the set row degrading, not token improving. (c) topk=256 at L2048
(0.757) is consistent with the L1024 bandwidth curve — sparsity costs
quality without buying memory in dense-score mode (7325 vs 7201 MiB for
full routing; topk256 is actually *higher* here, within run-to-run noise).

Open questions before interpretation: (1) seed-0 only — L1024 taught us
single seeds mislead by up to ±0.04; seeds 1-2 needed (user pre-approved
"seeds 0-2 if feasible"). (2) Budget: 2000 updates may be short for
L2048 — check `*_curve.csv` for whether b75full loss was still descending
at update 2000. (3) RESOLVED: the RuntimeWarning "multiscale disabled;
using single-scale bank" (`set_only_lm.py:312`) is benign — the
`multiscale` flag is unimplemented (True raises) and unrelated to the
multiresolution head groups; b75's fine/coarse split comes from
`model.multiresolution.groups` (fine w2/s1 2 heads + coarse w4/s2 6
heads), so the fine/coarse story is intact.

Driver `scripts/run_lca_l2048_pilot.sh`; TSV
`out/lca_cmp/l2048pilot/l2048pilot_lizmark.tsv` (pulled); per-row CSVs on
Lizmark under `out/lca_cmp/l2048pilot/`; logs
`logs/lca_cmp/lizmark/l2048pilot_*.log`.

### L2048 pilot seeds 1-2 (2026-07-26, Blue): seed0 gap was partly seed noise

Seeds 1-2 on Blue (same driver, `l2048pilot_blue.tsv`), 6/6 rows exit 0.
3-seed summary at 2000 updates:

| Row | val_acc mean ± sd (n=3) | Peak VRAM |
|---|---:|---:|
| token prefix | 0.9407 ± 0.0066 | 9123.9 MiB |
| b75 full routing | 0.8553 ± 0.0483 | 7201.1 MiB |
| b75 topk=256 | 0.8107 ± 0.0476 | 7327.2 MiB |

The b75 rows show much larger seed variance than token at 2000 updates
(b75full seeds: 0.816 / 0.841 / 0.909) — the signature of an undertrained,
still-descending model rather than a stable quality deficit.

### L2048 budget probe (2026-07-26, Lizmark, 4000 updates, seed 0): gap was mostly budget

Decisive controlled pair, same seed/config as pilot seed0 except
`max_updates=4000` (curves verified 4000 points):

| Row | val_acc @2000 | val_acc @4000 | Peak VRAM |
|---|---:|---:|---:|
| token prefix | 0.9440 | 0.9438 | 9123.9 MiB |
| b75 full routing | 0.8158 | **0.9353** | 7201.1 MiB |

Token is saturated at 2000 updates; b75full gains **+0.12** from doubling
budget and lands within 0.009 of token at **−21% VRAM** (7201 vs 9124
MiB). The "L2048 quality gap" was primarily undertraining, not an
architectural scaling deficit. Caveat: budget pair is seed-0 only;
b75full seeds 1-2 at 4000 updates would confirm variance shrinks too.

**Scientific caution (recorded)**: the top-k bandwidth curve (L1024) and
the pilot sparse row were all measured at 2000 updates. If sparse-top-k
rows train slower than full-routing rows (plausible: less gradient
bandwidth per update), the bandwidth curve may be partially a *training
speed* curve. A budget-matched sparse control (e.g. topk256 at 4000
updates) is required before claiming "aggregation needs routing
bandwidth" as a convergence fact rather than an optimization-speed fact.

Driver `scripts/run_lca_l2048_budget.sh`; TSV
`out/lca_cmp/l2048budget/l2048budget_lizmark.tsv` (pulled); logs
`logs/lca_cmp/lizmark/l2048budget_*.log`.

### L2048 budget-matched confirmation (2026-07-27, Blue+Lizmark, 4000 updates)

User staging 2026-07-26: b75full seeds 1-2 (Blue) and b75topk256 seed 0
(Lizmark), all at 4000 updates, committed driver `cb284ae`, 3/3 rows
exit 0, curves verified 4000 points, CSVs validated and pulled.

Budget-matched L2048 picture (4000 updates unless noted):

| Row | val_acc | Peak VRAM |
|---|---:|---:|
| token prefix | 0.9438 (seed0; saturated — @2000 3-seed 0.9407 ± 0.0066) | 9123.9 MiB |
| b75 full routing | **0.9078 ± 0.0368** (0.9353 / 0.8660 / 0.9221) | 7201.1 MiB |
| b75 topk=256 | 0.8099 (seed0; was 0.7571 @2000) | 7325.4 MiB |

Answers to the two open questions:

1. **Is the L2048 Pareto result seed-stable?** Mostly. b75full @4000
   averages 0.908 at −21% VRAM vs token ≈ 0.941-0.944; the mean gap is
   ~0.033 (≈1.6 SEM at n=3). Two of three seeds reach 0.92-0.94 (token
   regime); seed1 lags at 0.866. Seed variance shrinks vs @2000
   (0.0483 -> 0.0368) but remains well above token's 0.007 — the set row
   is in token's quality regime on average but is less seed-stable. The
   Pareto claim (near-token quality at −21% VRAM) holds on average, not
   uniformly per-seed.

2. **Is sparse top-k bandwidth-limited or just slower to train?** Both.
   topk256 gained +0.053 from the doubled budget (0.7571 -> 0.8099) — so
   part of the @2000 bandwidth curve was indeed training speed. But at
   matched budget it still trails full routing by **−0.125** (0.8099 vs
   0.9353 seed0) — a genuine bandwidth deficit remains. The corrected
   statement: global aggregation needs routing bandwidth; sparse top-k
   both trains slower per update AND converges lower. Since top-k buys
   no VRAM in dense-score mode, full routing dominates it at L2048.

Driver `scripts/run_lca_l2048_budget.sh` (ROWS-parameterized); TSVs
`out/lca_cmp/l2048budget/l2048budget_{blue,lizmark}.tsv` (tracked);
logs `logs/lca_cmp/{blue,lizmark}/l2048budget_*.log`.

## L4096 Stage-A Admission (2026-07-28, Lizmark): memory asymmetry is real

Plan amendment 2026-07-27, Stage A (admission/memory ONLY; 30 updates;
val metrics at 30 updates are meaningless and recorded as such): L=4096,
native B4, prefix supervision, seed 0.

| Row | Peak VRAM (native B4) | Fits Blue 24 GB? |
|---|---:|---|
| token prefix | 33745.8 MiB (33.0 GB) | NO |
| b75 full routing (topk=4095) | 24910.2 MiB (24.3 GB) | NO (marginally over) |

Findings: (a) b75 full routing needs **−26.2%** VRAM vs token at L4096
(−8835.6 MiB) — the memory edge grows with L (−12.5% at L1024, −21.1% at
L2048, −26.2% at L4096), consistent with coarse atoms shrinking the
O(L^2) router score tensor faster than token's O(L^2) attention grows.
(b) Neither row fits Blue's 24 GB natively at L4096 — L4096 scientific
rows are Lizmark-only (token un-runnable on Blue at any microbatch that
keeps B4-native semantics; b75 marginally over). (c) No OOM fallback was
needed on Lizmark. Stage B (token + b75full, seeds 0-2, 4000 updates,
seed 0 first) is gated on user confirmation of these numbers.

Driver `scripts/run_lca_l4096_admission.sh`; TSV
`out/lca_cmp/l4096admission/l4096admission_lizmark.tsv`; logs
`logs/lca_cmp/lizmark/l4096adm_*.log`.

## Artifacts

- Results TSV: `out/lca_cmp/calibration/calibration_runs_blue.tsv` (36 rows)
- Per-row CSV/JSON: `out/lca_cmp/calibration/<family>/L<len>/lcacmp_*.{csv,json}`
- OOM registry (header-only): `out/lca_cmp/calibration/oom_registry.tsv`
- Done markers: `out/lca_cmp/calibration/markers/` (36)
- Logs (pulled from blue-demon): `logs/lca_cmp/blue/lcacmp_*.log` (36),
  `queue.log`, `worker_gpu0.log`, `worker_gpu1.log`
- Plan: `docs/agent_plans/mrp_lca_cmp.md`; matrix: `configs/lca_cmp/matrix.md`
- Host state after completion: both blue-demon GPUs idle (1 MiB, 0% util),
  calibration driver exited, no `lcacmp_*` containers running.
