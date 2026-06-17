# A4.1 Long-Context Smoke Audit

Status: PASS

## Scope

- Sequence length: `L=2048`.
- Models: dense baseline (baseline_token) + dense SKA (set_only, w=16, s=8).
- 1 seed (seed=0), 10 epochs.
- M (set tokens at L=2048): `M=255`.
- batch_size=16 on single RTX 4090 (~15.6 GB peak fp32 -- within 24 GB budget).
- Purpose: verify OOM-free execution and finite convergence at L=2048.

## Commands / Scripts

- `bash scripts/run_a41_smoke.sh`
- `python scripts/summarize_a41_smoke.py`

## Prelaunch State

- Branch: `paper/final-results-bundle`
- HEAD: `1174947643b001647c4fb92cc48e88c75f044be4`
- A3.3 manifest: `pass` with `24` / `24` runs.
- A3.3 handoff: `Status: PASS`

## Failures / Retries

- None.

## Run Artifacts

| slug | impl | seed | lr | L | w | s | M | rows | final_val_loss | final_val_ppl | peak_vram_mib | time_per_epoch_s | candidate_count_mean | set_causality_mode | config | csv_path | source_csv_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_dense | baseline_token | 0 | 1e-4 | 2048 | NA | NA | NA | 10 | 6.817228977496807 | 913.4501342773438 | 18620.482421875 | 53.46615982055664 | NA | NA | configs/a4_long_context/baseline_dense_lc.yaml | out/paper_mechanisms/a41_smoke/a41_smoke_baseline_dense_L2048/a41_baseline_dense_D384_FF1536_L2048_lr1e-4_seed0.csv | c64b82b6134901afec7b70f74eacb040481de0d7b21aba080963a8c162880893 |
| set_dense | set_only | 0 | 1e-4 | 2048 | 16 | 8 | 255 | 10 | 7.3167787698599005 | 1505.346923828125 | 11043.11181640625 | 54.798749923706055 | 1.9814453125 | strict_past | configs/a4_long_context/set_dense_lc.yaml | out/paper_mechanisms/a41_smoke/a41_smoke_set_dense_L2048/a41_set_dense_D384_FF1536_L2048_w16_s8_lr1e-4_seed0.csv | b8425fb039eeb816540ceff1bc054dbba20fc743e6acafa7ec964fe61005de8d |

## Generated Artifacts

- `out/paper_integrated_evidence/tables/a41_smoke_all_runs.tsv`
- `out/paper_integrated_evidence/checks/a41_smoke_manifest.json`
