# A3.3 Stride Sweep Audit

Status: PASS

## Scope

- Fixed window: `w=16`.
- Strides: `s in {4, 8, 12, 16}`.
- Families: SetDense/exact, SetSparse/local_band, SetLinear/landmark.
- Seeds: endpoint strides (4, 16) use seeds 0,1,2; interior strides (8, 12) use seed 0.
- **Demoted complement**: M changes with s (confounding). Caption accordingly.
  M values: s=4→M=125, s=8→M=63, s=12→M=42, s=16→M=32.
- A2.2/A2.3 remain the locked s=8 LR-normalized headline/family grid; not overridden.

## Commands / Scripts

- `bash scripts/run_a3_stride_sweep.sh`
- `python scripts/summarize_a3_stride_sweep.py`

## Prelaunch State

- Branch: `paper/final-results-bundle`
- HEAD: `1174947643b001647c4fb92cc48e88c75f044be4`
- A3.2 manifest: `pass` with `27` / `27` runs.
- A3.2 handoff: `Status: PASS`

## Failures / Retries

- None.

## Run Artifacts

| family | backend | seed | lr | w | s | M | rows | final_val_ppl | time_per_epoch_s | candidate_count_mean | config | csv_path | source_csv_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Set Dense | exact | 0 | 1e-4 | 16 | 4 | 125 | 10 | 1588.8892822265625 | 59.15751671791077 | 3.835937500949875 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_dense_exact_D384_FF1536_w16/a3_stride_dense_exact_D384_FF1536_w16_s4_lr1e-4_seed0.csv | fc1c98c30d7d4917939dc2e1fbc6bc88a02276e668f4fece96f7e2dec0f509d5 |
| Set Dense | exact | 1 | 1e-4 | 16 | 4 | 125 | 10 | 1503.0682373046875 | 59.64940690994263 | 3.835937500949875 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_dense_exact_D384_FF1536_w16/a3_stride_dense_exact_D384_FF1536_w16_s4_lr1e-4_seed1.csv | ea6515116b0b88d09b79cb7edef6a95e1883b841c1a5c466853fe01f2ec7a360 |
| Set Dense | exact | 2 | 1e-4 | 16 | 4 | 125 | 10 | 1457.2314453125 | 60.46311402320862 | 3.835937500949875 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_dense_exact_D384_FF1536_w16/a3_stride_dense_exact_D384_FF1536_w16_s4_lr1e-4_seed2.csv | acc8c5bb54f665ec56166901b6cfef519293fbf40182b4c5997aa3ce51b571ea |
| Set Dense | exact | 0 | 1e-4 | 16 | 8 | 63 | 10 | 1469.400390625 | 50.94200778007507 | 1.9257812504749374 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_dense_exact_D384_FF1536_w16/a3_stride_dense_exact_D384_FF1536_w16_s8_lr1e-4_seed0.csv | 40a06ff49f7127f22baedaa709617ed7c2f3de5684b42def60704009469572ad |
| Set Dense | exact | 0 | 1e-4 | 16 | 12 | 42 | 10 | 1563.3856201171875 | 45.38905954360962 | 1.291015625 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_dense_exact_D384_FF1536_w16/a3_stride_dense_exact_D384_FF1536_w16_s12_lr1e-4_seed0.csv | d390331b5805615f3722a936043f6259711e5d3f7d65e7b86e2da2a278750387 |
| Set Dense | exact | 0 | 1e-4 | 16 | 16 | 32 | 10 | 1558.739990234375 | 31.510679960250854 | 0.9707031252374687 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_dense_exact_D384_FF1536_w16/a3_stride_dense_exact_D384_FF1536_w16_s16_lr1e-4_seed0.csv | f1af8dfb94fa19e2d70249a7539487537239a513302dd11f3614aa230e9545f8 |
| Set Dense | exact | 1 | 1e-4 | 16 | 16 | 32 | 10 | 1442.417724609375 | 32.14507174491882 | 0.9707031252374687 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_dense_exact_D384_FF1536_w16/a3_stride_dense_exact_D384_FF1536_w16_s16_lr1e-4_seed1.csv | 6b0e8496bc79eacb6f92aba70cc6fd195d893838e165005bf827b871cf4109ee |
| Set Dense | exact | 2 | 1e-4 | 16 | 16 | 32 | 10 | 1553.0992431640625 | 31.81311297416687 | 0.9707031252374687 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_dense_exact_D384_FF1536_w16/a3_stride_dense_exact_D384_FF1536_w16_s16_lr1e-4_seed2.csv | aa728808ea2872bb167c41270c71168ac3de523c15a4ae49ea00bd09553ce8df |
| Set Linear | landmark | 0 | 1e-4 | 16 | 4 | 125 | 10 | 1597.0853271484375 | 66.34721803665161 | 3.835937500949875 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_linear_landmark_D384_FF1536_w16/a3_stride_linear_landmark_D384_FF1536_w16_s4_lr1e-4_seed0.csv | 01da3f7b0d8ccb7563c2cea19df5276801de25b961370cc6399a51c0bb9ed648 |
| Set Linear | landmark | 1 | 1e-4 | 16 | 4 | 125 | 10 | 1582.466064453125 | 66.63434910774231 | 3.835937500949875 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_linear_landmark_D384_FF1536_w16/a3_stride_linear_landmark_D384_FF1536_w16_s4_lr1e-4_seed1.csv | 96290632f6c45048aa1daafbd3ccd814f0bdbc398c549e85c2955af33881ecdb |
| Set Linear | landmark | 2 | 1e-4 | 16 | 4 | 125 | 10 | 1378.9002685546875 | 66.27790188789368 | 3.835937500949875 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_linear_landmark_D384_FF1536_w16/a3_stride_linear_landmark_D384_FF1536_w16_s4_lr1e-4_seed2.csv | 1f0ec354efa6b5f83a66086d920f959403cc49b4f621001de18b2f090706d727 |
| Set Linear | landmark | 0 | 1e-4 | 16 | 8 | 63 | 10 | 1400.873291015625 | 45.37909197807312 | 1.9257812504749374 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_linear_landmark_D384_FF1536_w16/a3_stride_linear_landmark_D384_FF1536_w16_s8_lr1e-4_seed0.csv | cfd1229ab9b16256776222f345f4d31492c97af893e5474ae4bf6448df9151c2 |
| Set Linear | landmark | 0 | 1e-4 | 16 | 12 | 42 | 10 | 1417.940673828125 | 37.380030393600464 | 1.291015625 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_linear_landmark_D384_FF1536_w16/a3_stride_linear_landmark_D384_FF1536_w16_s12_lr1e-4_seed0.csv | abebec5f2bce54b99904f7b1df114ffd410596a192e3e458cb7f74747159d119 |
| Set Linear | landmark | 0 | 1e-4 | 16 | 16 | 32 | 10 | 1430.8734130859375 | 33.845250368118286 | 0.9707031252374687 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_linear_landmark_D384_FF1536_w16/a3_stride_linear_landmark_D384_FF1536_w16_s16_lr1e-4_seed0.csv | a6eaa96d18350fd59026fb55849c9af8ceafc13e4d473ad4b9a5b427919648cd |
| Set Linear | landmark | 1 | 1e-4 | 16 | 16 | 32 | 10 | 1492.43408203125 | 33.819056272506714 | 0.9707031252374687 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_linear_landmark_D384_FF1536_w16/a3_stride_linear_landmark_D384_FF1536_w16_s16_lr1e-4_seed1.csv | a977de6c0556a5971d395ec4d13744e26f62d59832fe4b27ee81946d0ee048fc |
| Set Linear | landmark | 2 | 1e-4 | 16 | 16 | 32 | 10 | 1494.6981201171875 | 38.56081962585449 | 0.9707031252374687 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_linear_landmark_D384_FF1536_w16/a3_stride_linear_landmark_D384_FF1536_w16_s16_lr1e-4_seed2.csv | b38e3e42073af3f7bee277b95501be2e04306c28510d44fc79142c0b38be3b12 |
| Set Sparse | local_band | 0 | 1e-4 | 16 | 4 | 125 | 10 | 1430.948486328125 | 66.998610496521 | 3.835937500949875 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_sparse_local_band_D384_FF1536_w16/a3_stride_sparse_local_band_D384_FF1536_w16_s4_lr1e-4_seed0.csv | 82c6d157e6ac2614cc9c65e85f7a060146e44f45dc946c62ab3f299891f8c9a7 |
| Set Sparse | local_band | 1 | 1e-4 | 16 | 4 | 125 | 10 | 1436.7655029296875 | 65.88850474357605 | 3.835937500949875 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_sparse_local_band_D384_FF1536_w16/a3_stride_sparse_local_band_D384_FF1536_w16_s4_lr1e-4_seed1.csv | d08060d64382e1648c23bc8941c37b7faa1c3274617c013c214ffee12d226ca1 |
| Set Sparse | local_band | 2 | 1e-4 | 16 | 4 | 125 | 10 | 1345.6754150390625 | 66.32667565345764 | 3.835937500949875 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_sparse_local_band_D384_FF1536_w16/a3_stride_sparse_local_band_D384_FF1536_w16_s4_lr1e-4_seed2.csv | 42a4448d4f9bce30383361f129e841bfb9a2212de4bc6283c4f49f8df416eff9 |
| Set Sparse | local_band | 0 | 1e-4 | 16 | 8 | 63 | 10 | 1386.0950927734375 | 45.66507434844971 | 1.9257812504749374 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_sparse_local_band_D384_FF1536_w16/a3_stride_sparse_local_band_D384_FF1536_w16_s8_lr1e-4_seed0.csv | 7d9b1bbc509049472378e8aa0f949df6defd3a3357b488a4b422d2a056a603f4 |
| Set Sparse | local_band | 0 | 1e-4 | 16 | 12 | 42 | 10 | 1443.8173828125 | 39.09066081047058 | 1.291015625 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_sparse_local_band_D384_FF1536_w16/a3_stride_sparse_local_band_D384_FF1536_w16_s12_lr1e-4_seed0.csv | 49b055dde310da110fe022754efccc5e181b27e3bd4ba1597d6be50cc0e0ec39 |
| Set Sparse | local_band | 0 | 1e-4 | 16 | 16 | 32 | 10 | 1518.40771484375 | 36.085367918014526 | 0.9707031252374687 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_sparse_local_band_D384_FF1536_w16/a3_stride_sparse_local_band_D384_FF1536_w16_s16_lr1e-4_seed0.csv | 34f4fbb59be602ee036d8746ca8169332e23bfe813f0506c7e85065d7ff4ae29 |
| Set Sparse | local_band | 1 | 1e-4 | 16 | 16 | 32 | 10 | 1593.6551513671875 | 34.23415732383728 | 0.9707031252374687 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_sparse_local_band_D384_FF1536_w16/a3_stride_sparse_local_band_D384_FF1536_w16_s16_lr1e-4_seed1.csv | 8242ed6d7fbf179c269b41126b80745761f76ea35353a132308713e6b387bbcc |
| Set Sparse | local_band | 2 | 1e-4 | 16 | 16 | 32 | 10 | 1407.481201171875 | 33.80473065376282 | 0.9707031252374687 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_stride_sweep/a3_stride_sweep_sparse_local_band_D384_FF1536_w16/a3_stride_sparse_local_band_D384_FF1536_w16_s16_lr1e-4_seed2.csv | e6cf085b77ddbc0cdef430f11dcfedf730869c02a20ad18d6e8038d0fc0d6bd4 |

## Summary

| family | backend | w | s | M | runs | seeds | val_ppl_mean | val_ppl_std | candidate_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Set Dense | exact | 16 | 12 | 42 | 1 | 0 | 1563.385620 | 0.000000 | 1.291016 |
| Set Dense | exact | 16 | 16 | 32 | 3 | 0,1,2 | 1518.085653 | 53.554838 | 0.970703 |
| Set Dense | exact | 16 | 4 | 125 | 3 | 0,1,2 | 1516.396322 | 54.569069 | 3.835938 |
| Set Dense | exact | 16 | 8 | 63 | 1 | 0 | 1469.400391 | 0.000000 | 1.925781 |
| Set Linear | landmark | 16 | 12 | 42 | 1 | 0 | 1417.940674 | 0.000000 | 1.291016 |
| Set Linear | landmark | 16 | 16 | 32 | 3 | 0,1,2 | 1472.668538 | 29.568067 | 0.970703 |
| Set Linear | landmark | 16 | 4 | 125 | 3 | 0,1,2 | 1519.483887 | 99.586632 | 3.835938 |
| Set Linear | landmark | 16 | 8 | 63 | 1 | 0 | 1400.873291 | 0.000000 | 1.925781 |
| Set Sparse | local_band | 16 | 12 | 42 | 1 | 0 | 1443.817383 | 0.000000 | 1.291016 |
| Set Sparse | local_band | 16 | 16 | 32 | 3 | 0,1,2 | 1506.514689 | 76.469026 | 0.970703 |
| Set Sparse | local_band | 16 | 4 | 125 | 3 | 0,1,2 | 1404.463135 | 41.636974 | 3.835938 |
| Set Sparse | local_band | 16 | 8 | 63 | 1 | 0 | 1386.095093 | 0.000000 | 1.925781 |

## Generated Artifacts

- `out/paper_integrated_evidence/tables/a3_stride_sweep_all_runs.tsv`
- `out/paper_integrated_evidence/tables/a3_stride_sweep_summary.tsv`
- `out/paper_integrated_evidence/checks/a3_stride_sweep_manifest.json`

## Note on Confounding

Stride and M are not independently controlled here. Interpret this sweep as showing how ppl and candidate-count jointly vary as the number of sets (M) increases with decreasing stride at fixed window. Do not caption this as an isolated stride effect.
