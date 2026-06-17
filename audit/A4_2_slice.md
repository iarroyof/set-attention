# A4.2 Long-Context Family Slice Audit

Status: PASS

## Scope

- Sequence length: `L=2048`.
- LR-norm headline reference: D=384, d_ff=1536, w=16, s=8.
- Families: baseline_token (dense), set_dense (exact), set_sparse (local_band), set_linear (landmark).
- Seeds: [0, 1, 2].
- Total runs: 12 (4 families × 3 seeds).
- M (set tokens at L=2048, w=16, s=8): `M=255`.
- Landmark count (coverage=0.25): `64`.
- batch_size=4 (confirmed OOM-free in A4.1 for fp32 dense at L=2048).

## Commands / Scripts

- `bash scripts/run_a42_slice.sh`
- `python scripts/summarize_a42_slice.py`

## Prelaunch State

- Branch: `paper/final-results-bundle`
- HEAD: `1174947643b001647c4fb92cc48e88c75f044be4`
- A4.1 manifest: `pass` with `2` / `2` runs.
- A4.1 handoff: `Status: PASS`

## Failures / Retries

- None.

## Per-Family val_ppl Summary (mean ± std over seeds)

| family | n | mean_val_ppl | std_val_ppl |
| --- | --- | --- | --- |
| baseline_token | 3 | 928.165 | 40.993 |
| set_dense | 3 | 1511.542 | 90.482 |
| set_sparse | 3 | 1429.39 | 27.337 |
| set_linear | 3 | 1434.663 | 30.928 |

## Run Artifacts

| family_slug | impl | backend | seed | lr | L | w | s | M | landmark_count | rows | final_val_loss | final_val_ppl | peak_vram_mib | time_per_epoch_s | candidate_count_mean | set_causality_mode | config | csv_path | source_csv_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_token | baseline_token | dense | 0 | 1e-4 | 2048 | NA | NA | NA | NA | 10 | 6.771603162472065 | 872.7098388671875 | 18620.482421875 | 53.506200313568115 | NA | NA | configs/a4_long_context/baseline_dense_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_baseline_token_D384_FF1536_L2048/a42_baseline_token_D384_FF1536_L2048_lr1e-4_seed0.csv | 6e8809bfccaaa9c8d3b13fe88d943ad765644240d1fc264d05c3157904315877 |
| baseline_token | baseline_token | dense | 1 | 1e-4 | 2048 | NA | NA | NA | NA | 10 | 6.877838684962346 | 970.5264282226562 | 18620.482421875 | 53.470654010772705 | NA | NA | configs/a4_long_context/baseline_dense_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_baseline_token_D384_FF1536_L2048/a42_baseline_token_D384_FF1536_L2048_lr1e-4_seed1.csv | e1f2a83cd2c9ffa4b4deb60660e3c982017415e3403e0509fcbfadfd70bc253d |
| baseline_token | baseline_token | dense | 2 | 1e-4 | 2048 | NA | NA | NA | NA | 10 | 6.847219283764179 | 941.2600708007812 | 18620.482421875 | 53.660264015197754 | NA | NA | configs/a4_long_context/baseline_dense_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_baseline_token_D384_FF1536_L2048/a42_baseline_token_D384_FF1536_L2048_lr1e-4_seed2.csv | 39927de7f6fa5c47d81cec5f89e111795e45584b996916b78345234ef97f464d |
| set_dense | set_only | exact | 0 | 1e-4 | 2048 | 16 | 8 | 255 | NA | 10 | 7.371598207033598 | 1590.1732177734375 | 11043.11181640625 | 49.64252948760986 | 1.9814453125 | strict_past | configs/a4_long_context/set_dense_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_set_dense_D384_FF1536_L2048/a42_set_dense_D384_FF1536_L2048_w16_s8_lr1e-4_seed0.csv | b7fcd561159f2f496f43f90f53bff909b7103b99057085cdd4a8870159b8a355 |
| set_dense | set_only | exact | 1 | 1e-4 | 2048 | 16 | 8 | 255 | NA | 10 | 7.233311304679284 | 1384.80029296875 | 11043.11181640625 | 50.290828227996826 | 1.9814453125 | strict_past | configs/a4_long_context/set_dense_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_set_dense_D384_FF1536_L2048/a42_set_dense_D384_FF1536_L2048_w16_s8_lr1e-4_seed1.csv | a7271ebcf16b165644da308659ca0e439d5d117c3cd146b1487d8223abc45211 |
| set_dense | set_only | exact | 2 | 1e-4 | 2048 | 16 | 8 | 255 | NA | 10 | 7.3522177292750435 | 1559.6514892578125 | 11043.11181640625 | 50.02620553970337 | 1.9814453125 | strict_past | configs/a4_long_context/set_dense_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_set_dense_D384_FF1536_L2048/a42_set_dense_D384_FF1536_L2048_w16_s8_lr1e-4_seed2.csv | 1638780b3fa7f9778e1d78d0cad5dd731eb20a203246135eb0f97cda440866e9 |
| set_sparse | set_only | local_band | 0 | 1e-4 | 2048 | 16 | 8 | 255 | NA | 10 | 7.239774410541241 | 1393.779296875 | 11043.11181640625 | 55.080618381500244 | 1.9814453125 | strict_past | configs/a4_long_context/set_sparse_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_set_sparse_D384_FF1536_L2048/a42_set_sparse_D384_FF1536_L2048_w16_s8_lr1e-4_seed0.csv | e56e34737b6bb21ededeba7b61b5a69535d086e89a11a9a035fd28f4544a2e1d |
| set_sparse | set_only | local_band | 1 | 1e-4 | 2048 | 16 | 8 | 255 | NA | 10 | 7.268336020983183 | 1434.1617431640625 | 11043.11181640625 | 64.98688292503357 | 1.9814453125 | strict_past | configs/a4_long_context/set_sparse_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_set_sparse_D384_FF1536_L2048/a42_set_sparse_D384_FF1536_L2048_w16_s8_lr1e-4_seed1.csv | ff4f1aa6ad01d26099af2d558fd80f836b75e4a62d3308bbf1abac594f90ecdb |
| set_sparse | set_only | local_band | 2 | 1e-4 | 2048 | 16 | 8 | 255 | NA | 10 | 7.286348544634306 | 1460.2286376953125 | 11043.11181640625 | 54.767115116119385 | 1.9814453125 | strict_past | configs/a4_long_context/set_sparse_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_set_sparse_D384_FF1536_L2048/a42_set_sparse_D384_FF1536_L2048_w16_s8_lr1e-4_seed2.csv | 955bb2cb0818a7c0ae55f17e4499b53326841148c345a3e4b2a67086a92abcfa |
| set_linear | set_only | landmark | 0 | 1e-4 | 2048 | 16 | 8 | 255 | 64 | 10 | 7.29476261138916 | 1472.5672607421875 | 10996.30126953125 | 57.09276294708252 | 1.9814453125 | strict_past | configs/a4_long_context/set_linear_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_set_linear_D384_FF1536_L2048/a42_set_linear_D384_FF1536_L2048_w16_s8_lr1e-4_seed0.csv | 8b3a4fc38da406620aa623c15a5e41cccc769929593477d08257492a2e4bb886 |
| set_linear | set_only | landmark | 1 | 1e-4 | 2048 | 16 | 8 | 255 | 64 | 10 | 7.24194596363948 | 1396.8092041015625 | 10996.30126953125 | 55.89674472808838 | 1.9814453125 | strict_past | configs/a4_long_context/set_linear_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_set_linear_D384_FF1536_L2048/a42_set_linear_D384_FF1536_L2048_w16_s8_lr1e-4_seed1.csv | c3509127fd153cf206a6d24cb411bca6e8192ec8b7229e6c632906a887bf291b |
| set_linear | set_only | landmark | 2 | 1e-4 | 2048 | 16 | 8 | 255 | 64 | 10 | 7.268650660148034 | 1434.6131591796875 | 10996.30126953125 | 63.81236791610718 | 1.9814453125 | strict_past | configs/a4_long_context/set_linear_lc.yaml | out/paper_mechanisms/a42_slice/a42_slice_set_linear_D384_FF1536_L2048/a42_set_linear_D384_FF1536_L2048_w16_s8_lr1e-4_seed2.csv | 196c413473ea2f7f8d470d9a903ec17e2030bd3b76fb02bf2a45083bdf220256 |

## Generated Artifacts

- `out/paper_integrated_evidence/tables/a42_slice_all_runs.tsv`
- `out/paper_integrated_evidence/checks/a42_slice_manifest.json`
