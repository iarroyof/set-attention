# A4.3 Convergence Panel

Status: PASS

Expected runs: 6
Validated runs: 6

## Provenance

- Branch: `paper/final-results-bundle`
- HEAD: `1174947643b001647c4fb92cc48e88c75f044be4`
- Dirty entries: 68
- Seed: `0`
- LR selection: all families use `1e-4`, selected by lowest mean A2/A2.4 val PPL across seeds.

## Validation

- All expected CSV/JSON artifacts exist and completed at least 30 epochs.
- CSV metrics are finite.
- Logs contain no OOM, traceback, standalone nan/inf, or W&B step warnings.
- Set-only runs record strict_past with w=16, s=8, M=63.
- Landmark runs record landmark_coverage=0.25 and resolved landmark_count.

## Summary Artifacts

- `out/paper_integrated_evidence/tables/a4_convergence_all_runs.tsv`
- `out/paper_integrated_evidence/tables/a4_convergence_summary.tsv`
- `out/paper_integrated_evidence/checks/a4_convergence_manifest.json`

## Results

- baseline_dense_exact: final val PPL 1140.8690185546875, train PPL 57.15745162963867, epoch 30, `out/paper_mechanisms/a43_convergence/a43_convergence_baseline_dense_exact_D384_FF1536_L512/a43_baseline_dense_exact_D384_FF1536_L512_w16_s8_lr1e-4_seed0_ep30.csv`
- baseline_sparse_local_band: final val PPL 1116.221435546875, train PPL 47.81793975830078, epoch 30, `out/paper_mechanisms/a43_convergence/a43_convergence_baseline_sparse_local_band_D384_FF1536_L512/a43_baseline_sparse_local_band_D384_FF1536_L512_w16_s8_lr1e-4_seed0_ep30.csv`
- baseline_linear_landmark: final val PPL 1331.915771484375, train PPL 64.56221008300781, epoch 30, `out/paper_mechanisms/a43_convergence/a43_convergence_baseline_linear_landmark_D384_FF1536_L512/a43_baseline_linear_landmark_D384_FF1536_L512_w16_s8_lr1e-4_seed0_ep30.csv`
- set_dense_exact: final val PPL 2439.735595703125, train PPL 74.7025146484375, epoch 30, `out/paper_mechanisms/a43_convergence/a43_convergence_set_dense_exact_D384_FF1536_L512/a43_set_dense_exact_D384_FF1536_L512_w16_s8_lr1e-4_seed0_ep30.csv`
- set_sparse_local_band: final val PPL 2251.05419921875, train PPL 80.03376770019531, epoch 30, `out/paper_mechanisms/a43_convergence/a43_convergence_set_sparse_local_band_D384_FF1536_L512/a43_set_sparse_local_band_D384_FF1536_L512_w16_s8_lr1e-4_seed0_ep30.csv`
- set_linear_landmark: final val PPL 2138.69287109375, train PPL 81.48432159423828, epoch 30, `out/paper_mechanisms/a43_convergence/a43_convergence_set_linear_landmark_D384_FF1536_L512/a43_set_linear_landmark_D384_FF1536_L512_w16_s8_lr1e-4_seed0_ep30.csv`
