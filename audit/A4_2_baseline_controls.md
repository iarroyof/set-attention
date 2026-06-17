# A4.2 Baseline Controls

Status: PASS

Expected runs: 6
Validated runs: 6

## Provenance

- Branch: `paper/final-results-bundle`
- HEAD: `1174947643b001647c4fb92cc48e88c75f044be4`
- Dirty entries: 66

## Validation

- All expected CSV/JSON artifacts exist and completed 10 epochs.
- CSV metrics are finite.
- Logs contain no OOM, traceback, standalone nan/inf, or W&B step warnings.
- Baseline sparse records causal local_band radius=4.
- Baseline linear records landmark_coverage=0.25 and resolved landmark_count.

## Summary Artifacts

- `out/paper_integrated_evidence/tables/a4_long_context_baseline_controls_all_runs.tsv`
- `out/paper_integrated_evidence/tables/a4_long_context_baseline_controls_summary.tsv`
- `out/paper_integrated_evidence/checks/a4_long_context_baseline_controls_manifest.json`

## Run Artifacts

- A4.2-control baseline_sparse_local_band seed=0 lr=1e-4 w=16 L=2048 val_ppl=838.1158447265625 `out/paper_mechanisms/a42_baseline_controls/a42_baseline_controls_baseline_sparse_local_band_D384_FF1536_L2048/a42_controls_baseline_sparse_local_band_D384_FF1536_L2048_w16_s8_lr1e-4_seed0.csv` sha256=c5215260009235e0929956251a917b04881593b68e430fab3682e1cb883ecef7
- A4.2-control baseline_sparse_local_band seed=1 lr=1e-4 w=16 L=2048 val_ppl=811.8572998046875 `out/paper_mechanisms/a42_baseline_controls/a42_baseline_controls_baseline_sparse_local_band_D384_FF1536_L2048/a42_controls_baseline_sparse_local_band_D384_FF1536_L2048_w16_s8_lr1e-4_seed1.csv` sha256=ed2700389b26dddccda1f2839905efbece88d499128a8239d1fc3e12d76ee7d3
- A4.2-control baseline_sparse_local_band seed=2 lr=1e-4 w=16 L=2048 val_ppl=792.5055541992188 `out/paper_mechanisms/a42_baseline_controls/a42_baseline_controls_baseline_sparse_local_band_D384_FF1536_L2048/a42_controls_baseline_sparse_local_band_D384_FF1536_L2048_w16_s8_lr1e-4_seed2.csv` sha256=84d6948b3a3ea3d73394e84eb01d115beae1eeb825bc886bf6ac715d1855becb
- A4.2-control baseline_linear_landmark seed=0 lr=1e-4 w=16 L=2048 val_ppl=1006.4029541015625 `out/paper_mechanisms/a42_baseline_controls/a42_baseline_controls_baseline_linear_landmark_D384_FF1536_L2048/a42_controls_baseline_linear_landmark_D384_FF1536_L2048_w16_s8_lr1e-4_seed0.csv` sha256=680aa04c15f1423cd02c6f0e5bab103f348368e0e4061ebf9ca2bf29a0325e3f
- A4.2-control baseline_linear_landmark seed=1 lr=1e-4 w=16 L=2048 val_ppl=959.36669921875 `out/paper_mechanisms/a42_baseline_controls/a42_baseline_controls_baseline_linear_landmark_D384_FF1536_L2048/a42_controls_baseline_linear_landmark_D384_FF1536_L2048_w16_s8_lr1e-4_seed1.csv` sha256=e9af8d5ec1d3df63fdca9afab033f1baaa1fdfa4a196f2148adacf0cb6c7bb28
- A4.2-control baseline_linear_landmark seed=2 lr=1e-4 w=16 L=2048 val_ppl=988.320068359375 `out/paper_mechanisms/a42_baseline_controls/a42_baseline_controls_baseline_linear_landmark_D384_FF1536_L2048/a42_controls_baseline_linear_landmark_D384_FF1536_L2048_w16_s8_lr1e-4_seed2.csv` sha256=071c86366922a0f56ea32403c7809edc6d195fac1f8d77a66b78d5213d762aae
