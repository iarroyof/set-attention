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
