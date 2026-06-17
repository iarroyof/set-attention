# A6.1 d_phi Capacity Sweep

Status: PASS

## Scope

A6.1 tests whether increasing SKA interface capacity via `model.d_phi` improves performance while holding token model width fixed.

Fixed setup: D=384, d_ff=1536, L=512, w=16, s=8, M=63, strict_past, 10 epochs, seeds 0/1/2, LR=1e-4. Linear uses landmark_coverage=0.25.

## Summary

| family | backend | d_phi | n | mean val PPL | std | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Set Dense | exact | 384 | 3 | 1422.839884 | 45.807377 | 1389.087036 | 1487.601929 |
| Set Dense | exact | 512 | 3 | 1525.559408 | 71.125017 | 1437.025269 | 1611.171753 |
| Set Dense | exact | 768 | 3 | 1540.967814 | 45.675280 | 1506.620117 | 1605.518066 |
| Set Linear | landmark | 384 | 3 | 1515.922607 | 135.277873 | 1324.885376 | 1620.314575 |
| Set Linear | landmark | 512 | 3 | 1484.101359 | 58.565485 | 1408.439941 | 1551.110596 |
| Set Linear | landmark | 768 | 3 | 1496.534342 | 58.278889 | 1426.882812 | 1569.519531 |
| Set Sparse | local_band | 384 | 3 | 1527.470581 | 47.135482 | 1469.192993 | 1584.634766 |
| Set Sparse | local_band | 512 | 3 | 1447.257406 | 30.278309 | 1413.460327 | 1486.926147 |
| Set Sparse | local_band | 768 | 3 | 1512.030151 | 78.823175 | 1412.012695 | 1604.665405 |

## Best d_phi by Family

| family | backend | best d_phi | mean val PPL |
| --- | --- | ---: | ---: |
| Set Dense | exact | 384 | 1422.839884 |
| Set Linear | landmark | 512 | 1484.101359 |
| Set Sparse | local_band | 512 | 1447.257406 |

## Artifacts

- All runs TSV: `out/paper_integrated_evidence/tables/a6_dphi_capacity_all_runs.tsv`
- Summary TSV: `out/paper_integrated_evidence/tables/a6_dphi_capacity_summary.tsv`
- Manifest: `out/paper_integrated_evidence/checks/a6_dphi_capacity_manifest.json`

## Validation

- Expected runs: 27
- Validated runs: 27
- Log failures: 0
- Failures: 0
