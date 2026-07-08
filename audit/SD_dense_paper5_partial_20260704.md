# Exact-Dense Five-Seed Matrix: Partial Snapshot

Status: PARTIAL; queues remain active

Snapshot: 2026-07-04 09:57 CST

## Queue State

| Host | Driver | Snapshot result | Action |
|---|---:|---|---|
| blue-demon | 1914084 | 115/120 endpoint-valid complete, 1 partial, 4 not started | GPU 0 is idle; GPU 1 has only five cells including the active row, so no rebalance or second launcher |
| lizmark | 2879441 | 50 complete and 2 partial; 36 endpoint-valid, 14 completed mixed rows require diagnostic retry | Both GPUs active; epoch-scoped probe fix deployed for future containers |

Blue's remaining cells at the status check were `L2048,B3` b100 seeds 1/3
and token seeds 0/2/4. Moving them would require weakening the one-driver
admission policy for little saved time.

Lizmark's post-grid external-workload watcher was stopped so the GPUs remain
available for the required replacement rows after the current driver exits.
See `audit/incident_sd_grid_epoch_gradient_probe_cadence_20260704.md`.

## PPL Matrix

Values are mean +/- 95% Student-t CI, followed by completed applied seeds.
`[R]` is core-contract-valid PPL/VRAM from a row requiring endpoint-diagnostic
replacement.

| row | L512/B16 | L512/B4 | L1024/B4 | L2048/B3 | L2048/B4 | L3584/B4 | L3584/B3 | L4096/B3 | L4096/B4 |
|---|---|---|---|---|---|---|---|---|---|
| token | 815.6 +/- 42.6 (5/5) | 1022.0 +/- 24.7 (5/5) | 929.2 +/- 12.0 (5/5) | 936.7 +/- 250.3 (2/5) | 942.8 +/- 19.4 (5/5) | -- | -- | -- | -- |
| b0 | 881.2 +/- 41.3 (5/5) | 1015.7 +/- 63.3 (5/5) | 941.1 +/- 27.9 (5/5) | 973.5 +/- 35.8 (5/5) | 956.5 +/- 34.0 (5/5) | 894.9 +/- 37.5 (5/5) | -- | -- | -- |
| b25 | 861.6 +/- 30.0 (5/5) | 933.4 +/- 47.3 (5/5) | 968.7 +/- 75.1 (5/5) | 942.5 +/- 40.6 (5/5) | 916.4 +/- 33.7 (5/5) | 875.7 +/- 28.3 (5/5) [R] | -- | -- | -- |
| b50 | 898.3 +/- 55.6 (5/5) | 1004.6 +/- 48.1 (5/5) | 961.0 +/- 63.1 (5/5) | 1014.1 +/- 81.6 (5/5) | 962.1 +/- 51.9 (5/5) | 904.7 +/- 32.7 (5/5) [R] | -- | -- | -- |
| b75 | 972.4 +/- 34.4 (5/5) | 1107.2 +/- 41.8 (5/5) | 1054.2 +/- 55.5 (5/5) | 1077.3 +/- 54.4 (5/5) | 1043.9 +/- 48.8 (5/5) | 1003.4 +/- 44.3 (4/5) [R] | -- | -- | -- |
| b100 | 1260.7 +/- 35.0 (5/5) | 1506.0 +/- 54.1 (5/5) | 1437.2 +/- 27.3 (5/5) | 1363.3 +/- 167.4 (3/5) | 1335.8 +/- 52.3 (5/5) | 1276.1 (1/5) | -- | -- | -- |

The matching peak-VRAM matrix and machine-readable run/cell/pairwise/frontier
tables are under
`out/paper_integrated_evidence/checks/sd_grid_seeded_v1_partial_20260704/`.

## Current Pareto Reading

| Island | Current mean frontier | Targeted conclusion |
|---|---|---|
| L512/B16 | token, b50, b75, b100 | Token dominates b25; b25 still dominates all-fine on means, but its paired PPL delta vs b0 is not resolved |
| L512/B4 | b25, b50, b75, b100 | b50 dominates token on means by 17.4 PPL and 20.3 MiB, but paired PPL CI crosses zero; b25 gains 88.6 +/- 50.7 PPL vs token at +98.0 MiB |
| L1024/B4 | token, b50, b75, b100 | No set-vs-token mean Pareto win; b50 trades +31.8 PPL for -347.9 MiB |
| L2048/B3 | provisional: token, b25, b50, b75, b100 | Token has only 2/5 seeds; no token conclusion. b25 dominates b0 on means by 31.0 PPL and 835.2 MiB |
| L2048/B4 | b25, b50, b75, b100 | b25 dominates token on means by 26.5 PPL and 516.6 MiB, and b0 by 40.2 PPL and 1105.6 MiB; both paired PPL CIs still cross zero |
| L3584/B4 | provisional: b25, b50, b75, b100 | b25 beats b0 by 19.2 +/- 15.6 PPL and 3610.8 MiB; this is the strongest scale signal, but `[R]` replacement and token rows are still required |

The five-seed evidence therefore narrows the claim:

- Mixed resolution repeatedly improves the achievable set frontier and its
  memory advantage over all-fine grows with scale, but it is not universal:
  `L1024,B4` is the counterexample.
- Set-vs-token mean Pareto wins currently appear at `L512,B4` (b50, weak PPL
  margin) and `L2048,B4` (b25), but neither PPL margin excludes zero at 95%.
- At `L512,B16`, token is decisively better than b25
  (`b25-token = +46.1 +/- 24.4` PPL and +387.4 MiB).
- Every available mixed row beats the b0-to-b100 straight interpolation on
  PPL, but consumes more VRAM than that synthetic interpolation. The original
  "not Pareto-better than interpolation" verdict still holds.

This is a progress interpretation, not the final MRP-1 verdict. The remaining
token rows and strict Lizmark replacements determine whether the long-scale
mean wins survive five-seed closure.
