# SD-9.6 Blue-Demon Long-Context Multiresolution Probe

Date: 2026-06-20

Status: probe complete for the corrected blue-demon interval `(8192 downto 2048] = {4096, 2048}`.

## Guard Contract

- CE-only.
- `output_residual_mode=anchor_span`.
- `token_mlp.enabled=false`.
- `anchor.enabled=false`.
- `candidate_fiber=endpoint_window`.
- Landmark backend, `landmark_coverage=0.25`, batch `1`.
- No re-read, all-past, multivector, anchor loss, or topology/backend/coverage changes.

CSV metadata for the primary L=4096/2048 rows matches the guard contract.

## Interval Correction

The user specified the open-upper interval `(8192 downto 2048]`; `L=8192` is already covered on lizmark and is not part of the blue-demon primary sweep. An initial blue-demon support launch accidentally included `L=8192`. Those completed rows are retained only as a capacity sanity check:

- support `L=8192`: mixed65 and all-coarse fit; all-fine OOMed in routing diagnostics.
- blur `L=8192`: mixed25 completed before correction; mixed50 and mixed75 were intentionally killed during correction.

Do not use blue-demon `L=8192` rows in the SD-9.6 operating-point selection.

## Seed-0 Probe Results

Rows are 1 epoch with `data.limit=500`; use these only for capacity and operating-point selection, not final claims.

| L | Variant | Fine/Coarse heads | PPL | Peak VRAM MiB | Fine ΔPPL | Coarse ΔPPL |
|---:|---|---:|---:|---:|---:|---:|
| 2048 | all-fine | 8/0 | 4094.875 | 2196.668 | 5518.556 | NA |
| 2048 | mixed25 | 6/2 | 2292.491 | 1748.192 | 4127.517 | 584.653 |
| 2048 | mixed50 | 4/4 | 1860.434 | 1489.261 | 2508.931 | 1301.443 |
| 2048 | mixed65 | 3/5 | 2832.906 | 1406.022 | 1084.133 | 2636.044 |
| 2048 | mixed75 | 2/6 | 3322.928 | 1332.876 | 994.276 | 3052.073 |
| 2048 | all-coarse | 0/8 | 2889.499 | 1047.611 | NA | 6725.208 |
| 4096 | all-fine | 8/0 | 4229.069 | 7005.175 | 4363.925 | NA |
| 4096 | mixed25 | 6/2 | 4020.689 | 5440.521 | 3114.188 | -250.978 |
| 4096 | mixed50 | 4/4 | 2513.276 | 4070.725 | 2507.795 | 1053.841 |
| 4096 | mixed65 | 3/5 | 3769.553 | 3825.132 | 323.573 | 2960.304 |
| 4096 | mixed75 | 2/6 | 4131.848 | 3614.960 | 115.062 | 3101.798 |
| 4096 | all-coarse | 0/8 | 3953.842 | 2741.296 | NA | 5255.631 |

Effective-range probes separate as intended: fine routes are approximately range `1.0`, coarse routes approximately range `3.0` across mixed rows. Routing entropy and top-1 are also stable across groups in this one-epoch probe.

## Operating-Point Selection

- `L=2048`: in the limited seed-0 probe, mixed50 has the lowest PPL among the tested rows.
- `L=4096`: in the limited seed-0 probe, mixed50 has the lowest PPL among the tested rows.
- These are not full-dataset conclusions and must not be cited as final Pareto or quality wins.
- Use `L=4096` only as the primary blue-demon long-context behavior candidate because it is the largest corrected blue length with all endpoints and mixed rows supported.
- Use `L=2048` only as the scale-comparison candidate to test whether the mixed50 pattern and fine/coarse ablation balance persist under full-dataset, multi-seed runs.

## 5-Seed Cross-Family Plan

Use seed set `{0,1,2,3,4}`. For rows already present at seeds `{0,1,2}`, launch only complementary seeds `{3,4}`.

Group A, short dense reference (`L=512`, blue-demon, exact, batch 16):

- Complete SD-9/SD-9.5 rows to five seeds: mixed25, all-fine, all-coarse.
- Keep dense token baseline as the paper reference; add sparse/linear token baselines only if already configured.

Group B, blue long primary (`L=4096`, landmark coverage 0.25, batch 1):

- Five seeds for all-fine, all-coarse, mixed50, and mixed65.
- mixed50 tests the limited-probe candidate selected from SD-9.6, not an established winner.
- mixed65 preserves the SD-9 long-context coarse-heavy operating point for scale comparison.
- Optional behavior-only: add mixed25 and mixed75 to five seeds if the first four rows confirm the seed-0 pattern and GPUs are free.

Group C, blue long scale anchor (`L=2048`, landmark coverage 0.25, batch 1):

- Five seeds for all-fine, all-coarse, mixed50, and mixed65.
- Purpose: test whether the mixed50 candidate remains competitive as L shrinks.

Group D, token/baseline family controls:

- `L=2048` and `L=4096`: linear landmark token baseline with coverage 0.25, matched D/FF/layers/heads/lr/seeds.
- Sparse local-band token baseline where the existing paper harness supports it.
- Dense exact token only where feasible without changing model/batch; do not force dense O(M^2) for long contexts.

Group E, lizmark frontier:

- Treat `L=8192` as the lizmark reference for SD-9/SD-9.5.
- Treat `L=12288/16384/32768` as frontier/OOM boundary rows from SD-9.5, not blue-demon targets.

## Validation

- Primary L=4096/2048 rows completed with exit code 0.
- `scan_logs` equivalent grep found no NaN/Inf/traceback/OOM findings for the primary rows.
- Non-primary L=8192 findings are recorded above and excluded from the blue-demon decision.

## Lizmark Recovery For Blue-Failed L=8192 Rows

Completed 2026-06-21 on lizmark with `HOST_TAG=lizmark ROW_SET=blur_sweep MODE=probe
LENGTHS="8192" VARIANTS="all_fine mixed50 mixed75"`. This was a recovery probe for rows that failed or
were interrupted on blue-demon. It is **not** a full-dataset experiment: all rows are 1 epoch with
`data.limit=500`.

| L | Host | Variant | Limit | PPL | Peak VRAM MiB | Fine ΔPPL | Coarse ΔPPL |
|---:|---|---|---:|---:|---:|---:|---:|
| 8192 | lizmark | all-fine | 500 | 6507.119 | 25141.455 | 2848.890 | NA |
| 8192 | lizmark | mixed50 | 500 | 4739.952 | 13475.168 | 1782.513 | 1933.865 |
| 8192 | lizmark | mixed75 | 500 | 6771.594 | 11696.928 | 533.603 | 2302.322 |

Guards match the SD-9 contract: landmark backend, coverage `0.25`, batch `1`,
`output_residual_mode=anchor_span`, `token_mlp.enabled=false`, `anchor.enabled=false`, and
`candidate_fiber=endpoint_window`. Log scan found no NaN/Inf/traceback/OOM findings for these recovery
rows.

## Research Direction From Current Evidence

Treat SD-9.6 as a capacity and candidate-selection probe only. The full-dataset evidence remains SD-9
and SD-9.5; the SD-9.6 rows only identify which follow-up rows are worth spending full runs on.

Recommended next validation package:

- `L=4096` blue-demon, full dataset, 5 seeds: all-fine, all-coarse, mixed50, mixed65.
- `L=2048` blue-demon, full dataset, 5 seeds: all-fine, all-coarse, mixed50, mixed65.
- Short `L=512` dense exact: add complementary seeds 3,4 for existing SD-9/SD-9.5 rows.
- `L=8192` lizmark: do not infer a new blur optimum from the limited recovery probe. If the paper needs
  a full blur curve at 8192, run full-dataset mixed50/mixed75 only as additions to the already validated
  SD-9 8192 mixed65/all-fine/all-coarse rows.

Mechanistic pattern worth testing under full runs: fine effective range remains near `1`, coarse near
`3`, while mixed rows split ablation mass across groups. This supports the multiresolution diagnostic
story, but the quality/VRAM frontier must be established only from full-dataset multi-seed rows.
