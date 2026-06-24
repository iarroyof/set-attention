# SD-9.7 Gap Completion Log

Timestamp: 2026-06-23 13:09 (raptor-mini agent)

Summary
-------
- Ran `python3 scripts/normalize_sd9x_runs.py out/paper_mechanisms` on both hosts and fetched the produced TSVs.
- Local artifacts created:
  - `out/blue_sd9x_homogenized_runs.tsv`
  - `out/lizmark_sd9x_homogenized_runs.tsv`
  - `out/merged_sd9x_homogenized_runs.tsv` (concat)
  - `out/aggregation_sd9x_summary.tsv` (per-(model_kind,variant,seq_len) means)
  - `out/missing_cells_sd9x.tsv` (per-(variant,seq_len) missing seeds)

Quick counts (both hosts combined)
- `2048`: 53 rows
- `4096`: 25 rows
- `8192`: 30 rows
- `12288`: 1 row
- `16384`: 3 rows

Completed cells (full-data)
- Set runs: L=2048 and L=4096 — `all_fine`, `mixed50`, `mixed62`, `all_coarse` — seeds 0–2: MARKED DONE (see `audit/phase_sd_status.md` and `audit/SD_9_7_handoff.md`).
- Set runs: L=8192 — `all_fine`, `mixed62`, `all_coarse` — seeds 0–2: MARKED DONE (per earlier notes / homogenizer corpus).

Pending / missing cells (actions required)
- G2: `mixed50` seeds 0–2 @ L=8192 on lizmark — PENDING (blocks the blur-optimum argmin claim).
- G3b / G4b: additional 8192 set sweeps (mixed25/mixed75 & 5-seed collections) — PENDING on lizmark.
- Db: matched landmark token baseline for some lengths (L=4096 running on lizmark; L=2048 done on blue) — await D-4096 completion and homogenizer re-run.
- A: short dense L=512 5-seed (seeds 3,4) — PENDING on blue; use `scripts/run_sd9_multiresolution.sh` with seeds 3,4.
- G5: scale-L frontier at L≥16384 — PENDING (smoke then full, lizmark only; watch OOMs).

Provenance / launcher notes
- Blue queue launcher: `logs/sd9_6_blue_long_blur_sweep_launch.log` (launcher running; launched remaining blue-supported full queue on 2026-06-23).
- Lizmark migrated G4 launcher: `logs/sd9_7_G4_migrated_lizmark_launch.log`.
- Homogenizer outputs located on hosts: `out/paper_integrated_evidence/tables/sd9x_homogenized_runs.tsv` (blue, lizmark).

Next recommended actions
1. Wait for lizmark D-4096 token baseline to finish; run homogenizer on lizmark and blue again and re-run aggregation.
2. After homogenizer shows `mixed50@8192` seeds 0–2 present, flip G2/G3b/G4b rows to `DONE` in `audit/SD_9_7_handoff.md` and update `audit/phase_sd_status.md` with timestamps, host/launcher PID, and log paths.
3. If user approves, launch `A` (L=512 short dense seeds 3,4) on blue via `scripts/run_sd9_multiresolution.sh` (health check then stop polling).

Recorded by: raptor-mini agent
