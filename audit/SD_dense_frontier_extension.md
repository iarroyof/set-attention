# Exact-Dense Frontier Extension

Status: COMPLETE; both hosts idle and all 28 planned seed-0 cells validated.

Updated: 2026-06-30.

## Design

Full-data, 10-epoch, seed-0 overlap rectangle:

- L2048/B3 on blue-demon;
- L3584/B4, L3584/B3, and L4096/B3 on lizmark;
- six set blur rows plus matched exact token at every point.

These are full runs, not smoke rows. Seed 0 is a valid replicate but cannot establish a multi-seed
quality claim.

## Blue-demon validation

- Planned cells: 7.
- Completed: 7/7, exit 0, 10/10 epochs.
- Resumable dry run: all seven `SKIP done`.
- Full data: confirmed; no `data.limit`.
- Guards: exact, `anchor_span`, token MLP off, anchor off, endpoint-window, CE-only.
- Peak train VRAM present: 7/7.
- NaN/Inf/traceback/OOM findings: 0.

## Lizmark validation

- Planned cells: 21.
- Completed: 21/21, exit 0, 10/10 epochs.
- Full data: confirmed; no `data.limit` or `data.val_limit`.
- Set guards: exact dense, `anchor_span`, token MLP off, anchor off, endpoint-window, CE-only.
- Token guards: exact dense baseline and no inherited `backend_params`.
- Peak train VRAM present: 21/21.
- Metadata warnings and NaN/Inf/traceback/OOM findings: 0.
- Completion ledger ended at 2026-06-30 00:14 CST; both lizmark GPUs are idle.

## L2048/B3 seed-0 results

| Row | Val PPL | Peak VRAM MiB |
|---|---:|---:|
| b0 | 970.2316 | 14628.2 |
| b25 | 900.7514 | 13797.1 |
| b50 | 977.3015 | 12719.5 |
| b62 | 1013.8007 | 12206.2 |
| b75 | 1028.7704 | 11713.6 |
| b100 | 1336.2179 | 10172.9 |
| token | 885.7863 | 14189.7 |

Within the set family, b25 Pareto-dominates b0 and is the lowest-PPL row. Against token, b25 trades
`+14.97` PPL for `-392.6` MiB.

## Batch-bridge observation

Compared with the corresponding L2048/B4 seed-0 rows:

- token: `931.0960 -> 885.7863` at B4 -> B3;
- b25: `889.9632 -> 900.7514`;
- token peak VRAM: `18633.3 -> 14189.7` MiB;
- b25 peak VRAM: `18097.3 -> 13797.1` MiB.

The seed-0 relative PPL ordering flips across B4/B3. This demonstrates that smaller batch is not merely
a memory control under the current epoch-based training protocol. Keep B3 and B4 as separate islands and
use the overlap points to describe sensitivity. Do not infer a population-level flip until additional
seeds are selected after the lizmark rectangle is known.

## Lizmark seed-0 results

### L3584/B4

| Row | Val PPL | Peak VRAM MiB |
|---|---:|---:|
| b0 | 901.3852 | 42137.0 |
| b25 | 846.9370 | 38552.9 |
| b50 | 927.2816 | 34364.2 |
| b62 | 927.1273 | 32316.7 |
| b75 | 1016.7877 | 30307.6 |
| b100 | 1218.6910 | 24866.1 |
| token | 890.8748 | 41076.2 |

The b25 row Pareto-dominates exact token by `43.94` PPL and `2523.2` MiB in this seed. It also
Pareto-dominates all-fine. The feasible set-family frontier is b25, b62, b75, and b100; b50 is
slightly dominated by b62.

### L3584/B3

| Row | Val PPL | Peak VRAM MiB |
|---|---:|---:|
| b0 | 925.3174 | 31842.2 |
| b25 | 924.9318 | 29159.1 |
| b50 | 934.2197 | 26009.4 |
| b62 | 915.1461 | 24467.7 |
| b75 | 980.7542 | 22967.0 |
| b100 | 1251.8707 | 18878.4 |
| token | 869.0807 | 31035.0 |

The best set row shifts to b62. Relative to token it trades `+46.07` PPL for `-6567.3` MiB. Token is
the lowest-PPL row, while b62, b75, and b100 form the set-family memory frontier.

### L4096/B3

| Row | Val PPL | Peak VRAM MiB |
|---|---:|---:|
| b0 | 902.5021 | 38901.6 |
| b25 | 840.5640 | 35350.1 |
| b50 | 925.3447 | 31277.1 |
| b62 | 942.6251 | 29281.4 |
| b75 | 984.0258 | 27313.9 |
| b100 | 1292.8677 | 22109.6 |
| token | 969.2231 | 37955.4 |

The b25 row Pareto-dominates exact token by `128.66` PPL and `2605.3` MiB in this seed, and
Pareto-dominates all-fine. This is promising but remains a single replicate.

## Interpretation

1. Native B3 is not a memory-only substitute for B4. At L3584, token improves by `21.79` PPL when
   moving B4 to B3 while b25 worsens by `77.99` PPL; the set winner changes from b25 to b62.
2. The same relative shift toward token occurs at the L2048 B4-to-B3 bridge. This is consistent with
   changed optimizer-step count and gradient noise under the epoch-based protocol.
3. The L4096/B3 b25 result reverses that pattern strongly. Because all frontier cells have only seed 0,
   the non-monotonic length/batch interaction may be variance and is not yet a paper claim.
4. B3 supports every L4096 row, so the prepared B2 fallback is unnecessary and remains unlaunched.
5. The registered three-seed B4 matrix remains the primary evidence. The frontier rectangle selects
   confirmation rows; it does not replace the primary matrix.

## Paper-grade follow-up

The earlier winner-only recommendation is superseded by the user-approved regular matrix in
`docs/sd_dense_paper5_matrix.md`: five seeds for `{b0,b25,b50,b75,b100}` plus exact token at every
supported island. Existing b62 rows remain exploratory but are excluded from that regular comparison.

No effective-B4 gradient-accumulation experiment is required for the core feasibility claim; it is a
stronger control only if the paper attempts a matched-quality statement at L4096/B4, where exact token
does not fit.
