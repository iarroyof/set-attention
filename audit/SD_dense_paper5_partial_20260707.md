# Exact-Dense Five-Seed Matrix: 2026-07-07 Post-Lizmark Snapshot

Status: FIRST PASS FINISHED; 15 L3584/B4 MIXED REPLACEMENTS STILL REQUIRED

## Host State

| Host | Corrected state | Validation |
|---|---|---|
| blue-demon | 120/120 complete; driver exited | 120 strict endpoint-valid rows |
| lizmark | driver exited; both GPUs idle at status check | 135/135 core rows complete; 120 endpoint-valid rows; 15 `L3584,B4` mixed rows still fail endpoint gradient diagnostics |

Artifacts and log were pulled from lizmark on 2026-07-07. The queue log ends
with `=== SD-GRID lizmark complete ===`, and the remote process/GPU check found
no active grid driver or training process.

The strict scanner still rejects the same registered mixed `L3584,B4` cells:
`b25`, `b50`, and `b75`, seeds `0..4`. Their epoch-10 CSV rows have PPL, peak
VRAM, ablation, routing, pooling, seed, full-data, and config metadata, but
`ausa/{fine,coarse}/grad_norm_{token_pre_pool,set_post_pool,set_post_blocks}`
remain `NA`. They remain provisional and must be replaced before MRP-1 closes.

Generated snapshot:
`out/paper_integrated_evidence/checks/sd_grid_seeded_v1_post_lizmark_20260707/`.
The strict scanner failure details are saved as
`strict_status_errors.log` in that directory.

Homogenized first-pass table:
`out/paper_integrated_evidence/tables/sd9x_homogenized_runs.tsv`
(`255` full-data rows: `215` set, `40` token; not all endpoint-valid).

## Key Evidence

The complete-valid b25 comparisons are:

| Island | Delta PPL b25-token, 95% CI | Delta peak VRAM MiB | Status |
|---|---:|---:|---|
| L2048/B3 | -2.3 +/- 46.9 | -378.5 | complete_valid |
| L2048/B4 | -26.5 +/- 48.1 | -516.6 | complete_valid |
| L3584/B3 | -51.8 +/- 54.9 | -1860.0 | complete_valid |
| L4096/B3 | -44.7 +/- 44.0 | -2590.3 | complete_valid |

The provisional L3584/B4 b25 comparison is `-20.9 +/- 44.3` PPL and
`-2517.9` MiB versus token, but it cannot be used as endpoint-valid paper
evidence until the replacement rows pass.

The registered operating-point freeze remains unchanged: `L2048,B4` selects
`b*=b25` before MRP-2/3/5 outcomes.

## Replacement Launch

Do not mark MRP-1 complete. The replacement wave was launched after this
snapshot:

1. the external workload was stopped again on lizmark;
2. the 15 invalid `L3584,B4` mixed records were archived outside the corrected
   root;
3. the unchanged `paper5` manifest dry-run planned 15 replacement cells and
   skipped 120 valid lizmark cells;
4. replacement driver PID `3940226` was launched.

After it finishes, run the strict scanner again and accept MRP-1 only if all 255 cells are
   endpoint-valid or registered terminal OOM.
