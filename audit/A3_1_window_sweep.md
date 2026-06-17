# A3.1 Window-Size Sweep Audit

Status: PASS

## Scope

- Fixed stride: `s=4`.
- Windows: `w in {6, 8, 12, 16, 20, 24}`.
- Families: SetDense/exact, SetSparse/local_band, SetLinear/landmark.
- Seeds: endpoints/reference `w in {6,16,24}` use seeds 0,1,2; interiors use seed 0.
- A2.2/A2.3 remain the locked `s=8` LR-normalized headline/family grid; they were not rerun or overridden.

## Commands / Scripts

- `bash scripts/run_a3_window_sweep.sh`
- `python scripts/summarize_a3_window_sweep.py`

## Prelaunch State

- Branch: `paper/final-results-bundle`
- HEAD: `1174947643b001647c4fb92cc48e88c75f044be4`
- A2 manifest: `pass` with `153` / `153` runs.
- A2 handoff: `Status: PASS`

## Failures / Retries

- None.

## Run Artifacts

| family | backend | seed | lr | w | s | M | rows | final_val_ppl | time_per_epoch_s | candidate_count_mean | config | csv_path | source_csv_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Set Dense | exact | 0 | 1e-4 | 6 | 4 | 127 | 10 | 1344.8857421875 | 69.06046891212463 | 1.4824218754749374 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w6_s4_lr1e-4_seed0.csv | 1096985680aca9019726715407bea5b868b2ca89bfddc39fd29428dbf5702c88 |
| Set Dense | exact | 1 | 1e-4 | 6 | 4 | 127 | 10 | 1338.0286865234375 | 60.196810483932495 | 1.4824218754749374 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w6_s4_lr1e-4_seed1.csv | 66eaa8df3b3fdd09fe233439ad0db3b882b67a5318f50251e4198e63b10336a9 |
| Set Dense | exact | 2 | 1e-4 | 6 | 4 | 127 | 10 | 1285.390380859375 | 59.888635873794556 | 1.4824218754749374 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w6_s4_lr1e-4_seed2.csv | 0fe08e445302291773a8bd82d667b5571827365245a6537fd3efc02895c478c6 |
| Set Dense | exact | 0 | 1e-4 | 8 | 4 | 127 | 10 | 1383.9183349609375 | 59.40695118904114 | 1.9648437504749374 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w8_s4_lr1e-4_seed0.csv | ed5b1a355ab9d84fa7b62587e020b269bd5275ed71c2ff397ea5dbeeb907e84f |
| Set Dense | exact | 0 | 1e-4 | 12 | 4 | 126 | 10 | 1414.0616455078125 | 60.177886724472046 | 2.912109375949875 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w12_s4_lr1e-4_seed0.csv | 036fa95c4879345b0ef5652a3772b66ef2ed0afb44842da3939a275542e52c74 |
| Set Dense | exact | 0 | 1e-4 | 16 | 4 | 125 | 10 | 1440.5948486328125 | 59.07196593284607 | 3.835937500949875 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w16_s4_lr1e-4_seed0.csv | 9402e8d79b761910922fc400bbe3749e1fd148777ae2dc929e279d47af2417b3 |
| Set Dense | exact | 1 | 1e-4 | 16 | 4 | 125 | 10 | 1422.940673828125 | 59.026061058044434 | 3.835937500949875 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w16_s4_lr1e-4_seed1.csv | 0a2dedc5e433ed5ab095b88b2d30f5c6034fd457a9bdd3ca25388121ff05b0fe |
| Set Dense | exact | 2 | 1e-4 | 16 | 4 | 125 | 10 | 1470.1663818359375 | 62.23021674156189 | 3.835937500949875 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w16_s4_lr1e-4_seed2.csv | 5d918b1131b1397ebf7808673fe22b662a58ff3028e99ed4857f0ea8a5e9bca8 |
| Set Dense | exact | 0 | 1e-4 | 20 | 4 | 124 | 10 | 1436.7943115234375 | 61.293506145477295 | 4.736328125 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w20_s4_lr1e-4_seed0.csv | 30bd1c17072c40c9bfc9f91f553fa214068c745900e52fbbbb18c428a5af6ebb |
| Set Dense | exact | 0 | 1e-4 | 24 | 4 | 123 | 10 | 1410.8551025390625 | 58.31687045097351 | 5.61328125189975 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w24_s4_lr1e-4_seed0.csv | 9aa68976f34709da504fd537c7a0ddaedc8b336d672eb829b15d2af96adc1a6c |
| Set Dense | exact | 1 | 1e-4 | 24 | 4 | 123 | 10 | 1495.803955078125 | 59.13285970687866 | 5.61328125189975 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w24_s4_lr1e-4_seed1.csv | a55a5776d513d04089bb55d3ce848b6e92006b7f199f86af4b0ba913a884fba2 |
| Set Dense | exact | 2 | 1e-4 | 24 | 4 | 123 | 10 | 1470.9671630859375 | 60.76392698287964 | 5.61328125189975 | configs/paper_complements/family_dense_exact.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_dense_exact_D384_FF1536_s4/a3_window_dense_exact_D384_FF1536_w24_s4_lr1e-4_seed2.csv | 3be26815cdfe094516a66f58abdfc9dfecbd642027c43afac7dd45324585be78 |
| Set Linear | landmark | 0 | 1e-4 | 6 | 4 | 127 | 10 | 1434.046875 | 71.56747055053711 | 1.4824218754749374 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w6_s4_lr1e-4_seed0.csv | 3e240630177774b9e10b95f659a482b02a8c0e51ad1e4f1d8da0d6b59839a501 |
| Set Linear | landmark | 1 | 1e-4 | 6 | 4 | 127 | 10 | 1496.353271484375 | 67.75098752975464 | 1.4824218754749374 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w6_s4_lr1e-4_seed1.csv | b21c7c15e3120f15587ea8f26b3086956082da6e891c674a1f3431d2b907ec48 |
| Set Linear | landmark | 2 | 1e-4 | 6 | 4 | 127 | 10 | 1449.0302734375 | 67.85268545150757 | 1.4824218754749374 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w6_s4_lr1e-4_seed2.csv | b3158e7bbafa7b3d6ad6913f3be0339c8aea0a33e3f6bf0731f469a7eba0605a |
| Set Linear | landmark | 0 | 1e-4 | 8 | 4 | 127 | 10 | 1411.4417724609375 | 66.86591005325317 | 1.9648437504749374 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w8_s4_lr1e-4_seed0.csv | 90bf2c475182bd88da84631f1c9fa5574feee85b67f68b843f1675934b7eeeac |
| Set Linear | landmark | 0 | 1e-4 | 12 | 4 | 126 | 10 | 1545.87841796875 | 70.53544640541077 | 2.912109375949875 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w12_s4_lr1e-4_seed0.csv | ffeab7dabaf84ad85b63e26c741c74f50bd9964ecbf96c92d7a08b4f99ca46d3 |
| Set Linear | landmark | 0 | 1e-4 | 16 | 4 | 125 | 10 | 1537.080078125 | 67.85912752151489 | 3.835937500949875 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w16_s4_lr1e-4_seed0.csv | 8626a33e04e1917aaf91efe889c24996a02ccff2f8ad13682eb927e235ae1cf0 |
| Set Linear | landmark | 1 | 1e-4 | 16 | 4 | 125 | 10 | 1457.2030029296875 | 67.02644348144531 | 3.835937500949875 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w16_s4_lr1e-4_seed1.csv | 52a327425c7ad90c908dd65ce7766f8d679e6a1de35c206fe8876c6f6d0311ec |
| Set Linear | landmark | 2 | 1e-4 | 16 | 4 | 125 | 10 | 1436.3662109375 | 67.29468989372253 | 3.835937500949875 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w16_s4_lr1e-4_seed2.csv | e3b38b02970408a4d646bad3bd8d43d9c1059a7d86befa388c25f00feb46e337 |
| Set Linear | landmark | 0 | 1e-4 | 20 | 4 | 124 | 10 | 1448.1971435546875 | 69.6754379272461 | 4.736328125 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w20_s4_lr1e-4_seed0.csv | 1a5105ecf057225f163f4234d0b04a07028a64db1df7cfca8fec7ec11c4bc97b |
| Set Linear | landmark | 0 | 1e-4 | 24 | 4 | 123 | 10 | 1496.872802734375 | 66.97526669502258 | 5.61328125189975 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w24_s4_lr1e-4_seed0.csv | ef2af4e26e0819e59bd49e750b48f5c146a3652835b0d9be0a7bf6bfd5108ebd |
| Set Linear | landmark | 1 | 1e-4 | 24 | 4 | 123 | 10 | 1639.2142333984375 | 66.80062127113342 | 5.61328125189975 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w24_s4_lr1e-4_seed1.csv | 7d85f88b7f748d037e09c18ca5cff2d90238edc94c92addaab0adec9f1e97e9d |
| Set Linear | landmark | 2 | 1e-4 | 24 | 4 | 123 | 10 | 1444.478515625 | 69.8208556175232 | 5.61328125189975 | configs/paper_complements/family_linear_landmark.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_linear_landmark_D384_FF1536_s4/a3_window_linear_landmark_D384_FF1536_w24_s4_lr1e-4_seed2.csv | aa03c03003fb64cd8ecdb5a3e14a430ab389c30227f59ee5409dd4868a330add |
| Set Sparse | local_band | 0 | 1e-4 | 6 | 4 | 127 | 10 | 1275.9007568359375 | 58.834800243377686 | 1.4824218754749374 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w6_s4_lr1e-4_seed0.csv | bb0e7f844568f637d30dc259a0ee761fe3c91c323a77578a70cb8b4f366846e4 |
| Set Sparse | local_band | 1 | 1e-4 | 6 | 4 | 127 | 10 | 1327.4932861328125 | 60.033164262771606 | 1.4824218754749374 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w6_s4_lr1e-4_seed1.csv | 7956a77d0973cee43315a37543d11f87eb6458c8fb37331f2271bec889e2e1e7 |
| Set Sparse | local_band | 2 | 1e-4 | 6 | 4 | 127 | 10 | 1372.36376953125 | 60.596951961517334 | 1.4824218754749374 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w6_s4_lr1e-4_seed2.csv | bb95bc5bab4bc9d5a4f647276350316c921e049176ef53b71b2842298aab54ae |
| Set Sparse | local_band | 0 | 1e-4 | 8 | 4 | 127 | 10 | 1349.8958740234375 | 63.159308433532715 | 1.9648437504749374 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w8_s4_lr1e-4_seed0.csv | 76f1f2dea11b8162fc2dfe3cdc243e2b346aad6cf02bb49f02a8b2771aee9e85 |
| Set Sparse | local_band | 0 | 1e-4 | 12 | 4 | 126 | 10 | 1348.1610107421875 | 60.362205028533936 | 2.912109375949875 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w12_s4_lr1e-4_seed0.csv | cd0239e85e7672764be44b10a3ba19927222baa022bfa3283d787ec17ee1017b |
| Set Sparse | local_band | 0 | 1e-4 | 16 | 4 | 125 | 10 | 1404.224609375 | 59.368715047836304 | 3.835937500949875 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w16_s4_lr1e-4_seed0.csv | 5d891468682468b1f1f7921fe2f98ac91f57170c4b81b1698c394ac92b5de9bb |
| Set Sparse | local_band | 1 | 1e-4 | 16 | 4 | 125 | 10 | 1405.9296875 | 58.66859173774719 | 3.835937500949875 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w16_s4_lr1e-4_seed1.csv | 7ede842d813b0ed2ddd504b8eaba40057a347a910d3eff09a4e0eee6b9dfe190 |
| Set Sparse | local_band | 2 | 1e-4 | 16 | 4 | 125 | 10 | 1549.0439453125 | 59.544370889663696 | 3.835937500949875 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w16_s4_lr1e-4_seed2.csv | 925e4ab165da6a2b9745ef6f035a54ff1f63f661553273012543d744c28a3092 |
| Set Sparse | local_band | 0 | 1e-4 | 20 | 4 | 124 | 10 | 1468.4912109375 | 60.49320650100708 | 4.736328125 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w20_s4_lr1e-4_seed0.csv | ee7435667cc0b68d17a3c8c9e9ee5dbfd9c44193861ce7d16de3efa9f6f48614 |
| Set Sparse | local_band | 0 | 1e-4 | 24 | 4 | 123 | 10 | 1430.170166015625 | 59.134567737579346 | 5.61328125189975 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w24_s4_lr1e-4_seed0.csv | fd4be8ca15199ef46e543104bd13a2e4d7f4c744660a44f424e440eef50cdbfd |
| Set Sparse | local_band | 1 | 1e-4 | 24 | 4 | 123 | 10 | 1504.6220703125 | 60.015965700149536 | 5.61328125189975 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w24_s4_lr1e-4_seed1.csv | 511bebd97fbffb81beff8dc3808a43f948aa964f78a785471d2f7d74d5edd06c |
| Set Sparse | local_band | 2 | 1e-4 | 24 | 4 | 123 | 10 | 1407.9195556640625 | 61.36257600784302 | 5.61328125189975 | configs/paper_complements/family_sparse_local_band.yaml | out/paper_mechanisms/a3_window_sweep/a3_window_sweep_sparse_local_band_D384_FF1536_s4/a3_window_sparse_local_band_D384_FF1536_w24_s4_lr1e-4_seed2.csv | a98ae0ae72c694c7d081b1cdc46647f71d61c05dfe470a9b3d27f3c1ed1ba3d2 |

## Summary

| family | backend | w | s | M | runs | seeds | val_ppl_mean | val_ppl_std | candidate_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Set Dense | exact | 12 | 4 | 126 | 1 | 0 | 1414.061646 | 0.000000 | 2.912109 |
| Set Dense | exact | 16 | 4 | 125 | 3 | 0,1,2 | 1444.567301 | 19.483363 | 3.835938 |
| Set Dense | exact | 20 | 4 | 124 | 1 | 0 | 1436.794312 | 0.000000 | 4.736328 |
| Set Dense | exact | 24 | 4 | 123 | 3 | 0,1,2 | 1459.208740 | 35.662981 | 5.613281 |
| Set Dense | exact | 6 | 4 | 127 | 3 | 0,1,2 | 1322.768270 | 26.577995 | 1.482422 |
| Set Dense | exact | 8 | 4 | 127 | 1 | 0 | 1383.918335 | 0.000000 | 1.964844 |
| Set Linear | landmark | 12 | 4 | 126 | 1 | 0 | 1545.878418 | 0.000000 | 2.912109 |
| Set Linear | landmark | 16 | 4 | 125 | 3 | 0,1,2 | 1476.883097 | 43.407375 | 3.835938 |
| Set Linear | landmark | 20 | 4 | 124 | 1 | 0 | 1448.197144 | 0.000000 | 4.736328 |
| Set Linear | landmark | 24 | 4 | 123 | 3 | 0,1,2 | 1526.855184 | 82.278824 | 5.613281 |
| Set Linear | landmark | 6 | 4 | 127 | 3 | 0,1,2 | 1459.810140 | 26.554044 | 1.482422 |
| Set Linear | landmark | 8 | 4 | 127 | 1 | 0 | 1411.441772 | 0.000000 | 1.964844 |
| Set Sparse | local_band | 12 | 4 | 126 | 1 | 0 | 1348.161011 | 0.000000 | 2.912109 |
| Set Sparse | local_band | 16 | 4 | 125 | 3 | 0,1,2 | 1453.066081 | 67.870169 | 3.835938 |
| Set Sparse | local_band | 20 | 4 | 124 | 1 | 0 | 1468.491211 | 0.000000 | 4.736328 |
| Set Sparse | local_band | 24 | 4 | 123 | 3 | 0,1,2 | 1447.570597 | 41.351544 | 5.613281 |
| Set Sparse | local_band | 6 | 4 | 127 | 3 | 0,1,2 | 1325.252604 | 39.412720 | 1.482422 |
| Set Sparse | local_band | 8 | 4 | 127 | 1 | 0 | 1349.895874 | 0.000000 | 1.964844 |

## Generated Artifacts

- `out/paper_integrated_evidence/tables/a3_window_sweep_all_runs.tsv`
- `out/paper_integrated_evidence/tables/a3_window_sweep_summary.tsv`
- `out/paper_integrated_evidence/checks/a3_window_sweep_manifest.json`

## Recommendation For Figure 1 / B5

Use `a3_window_sweep_summary.tsv` for the fixed-stride candidate-count mechanism figure and `a3_window_sweep_all_runs.tsv` for provenance/error bars.
