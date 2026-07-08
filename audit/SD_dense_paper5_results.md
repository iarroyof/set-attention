# Exact-Dense Five-Seed Paper Results

Status: MATRIX RUNNING; OPERATING POINT FROZEN

Frozen: 2026-07-06, before downstream empirical outcomes

## Frozen Operating Point

The registered rule uses the strict-valid exact-dense `L=2048,B=4` island and
selects the minimum mean validation PPL among b25, b50, and b75.

| Row | n | Mean PPL | 95% Student-t CI | Mean peak VRAM MiB |
|---|---:|---:|---:|---:|
| b25 | 5 | 916.351782 | +/- 33.660400 | 18116.658203 |
| b50 | 5 | 962.147229 | +/- 51.945646 | 16679.266602 |
| b75 | 5 | 1043.883057 | +/- 48.776799 | 15341.291992 |

**Frozen `b* = b25`**, corresponding to 6 fine `(2,1)` heads and 2 coarse
`(4,2)` heads.

This selection must not change after MRP-2, MRP-3, or MRP-5 results are
inspected. It does not authorize those launches: MRP-0 PASS and MRP-1 closure
remain separate hard gates.

The corrected rows apply their seeds, but exact same-seed CUDA replay and
checkpoint reuse are not available. This qualification does not change the
registered mean-PPL selection rule. See
`audit/incident_mrp0_prelaunch_gap_20260706.md`.

## Matrix Closure

Blue-demon is complete at 120/120 strict-valid cells. Lizmark remains active;
the final tables, full Pareto verdict, and DONE status will be added only
after all supported cells and the 15 diagnostic replacements pass.

Current analysis: `audit/SD_dense_paper5_partial_20260706.md`.
