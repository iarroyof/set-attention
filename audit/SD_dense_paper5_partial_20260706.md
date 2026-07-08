# Exact-Dense Five-Seed Matrix: 2026-07-06 Snapshot

Status: BLUE COMPLETE; LIZMARK RUNNING; DOWNSTREAM EMPIRICAL LAUNCHES BLOCKED

## Host State

| Host | Corrected state | Validation |
|---|---|---|
| blue-demon | 120/120 complete; driver exited; both GPUs idle | 120 strict endpoint-valid rows, 120 done markers, 120 exit-0 registry rows, clean word-boundary NaN/Inf/runtime/OOM log scan |
| lizmark | 114/135 core-complete, 2 active, 19 not started | 99 endpoint-valid completed rows plus 15 `L3584,B4` mixed diagnostic-retry rows |

At the status check, Lizmark was running `L4096,B3` b100 seed1 and token
seed3. Its remaining first-pass work is the rest of `L4096,B3` plus the 15
supported `L4096,B4` cells. After that, the 15 registered `L3584,B4` mixed
rows still require replacement under the epoch-scoped diagnostic fix.

The fatal-token scan is clean, but every pulled Blue log contains the CuBLAS
nondeterminism warning. Seeds were applied; exact replay, checkpoints, and
data/tokenizer digests were not provided. See
`audit/incident_mrp0_prelaunch_gap_20260706.md`.

## Current Evidence

The complete five-seed b25 comparisons are:

| Island | Delta PPL b25-token, 95% CI | Delta peak VRAM MiB | Reading |
|---|---:|---:|---|
| L2048/B3 | -2.3 +/- 46.9 | -378.5 | mean dominance, quality unresolved |
| L2048/B4 | -26.5 +/- 48.1 | -516.6 | mean dominance, quality unresolved |
| L3584/B3 | -51.8 +/- 54.9 | -1860.0 | mean dominance, quality unresolved |
| L3584/B4 | -20.9 +/- 44.3 | -2517.9 | provisional diagnostic-retry row |

The scale trend in memory is already clear, but the PPL intervals still cross
zero. The `L4096,B3` token seeds therefore remain decision-relevant rather
than redundant.

Against all-fine b0, b25 at `L3584,B3` improves PPL by
`19.9 +/- 9.9` and peak VRAM by `2683.2` MiB. This supports the
multiresolution set-vs-set mechanism at that island, but it does not replace
the matched token comparison.

Full PPL/VRAM matrices and common-seed comparisons:
`out/paper_integrated_evidence/checks/sd_grid_seeded_v1_partial_20260706/`.

## Operating-Point Freeze

The registered selection island `L2048,B4` is complete and strict-valid.
Among the registered interior rows:

| Row | Mean PPL +/- 95% CI | Mean peak VRAM MiB |
|---|---:|---:|
| b25 | 916.351782 +/- 33.660400 | 18116.658203 |
| b50 | 962.147229 +/- 51.945646 | 16679.266602 |
| b75 | 1043.883057 +/- 48.776799 | 15341.291992 |

The deterministic rule selects **b25**. It is frozen now, before any MRP-2,
MRP-3, or MRP-5 outcome exists. Remaining Lizmark rows cannot change this
registered selection because they belong to different `(L,batch)` islands.

## Launch Decision

Do not launch MRP-2 or MRP-3 yet:

1. MRP-1 is not strict-complete; the long matched token controls and
   diagnostic replacements remain.
2. MRP-0 is still PARTIAL: checkpoint/eval-only support, loader
   reproducibility, dataset/tokenizer digests, masked metrics, and the ordered
   token source remain incomplete.
3. MRP-2 likely requires registered retraining because no compatible MRP-1
   checkpoints are known.

Productive work while Lizmark runs is limited to MRP-0 implementation and
tests, Blue final-result analysis, and MRP-6A/B theory. Blue's idle GPUs must
not be used to bypass the registered MRP-1 and MRP-0 gates.
