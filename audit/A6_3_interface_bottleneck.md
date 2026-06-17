# A6.3 Set-Token Interface Bottleneck Sweep

Status: PASS

## Hypothesis

`d_phi` may bottleneck wider set states by limiting the router query/key interface when `set_state_dim > d_phi`. This is an interface bottleneck, not a direct value bottleneck: routed values remain `set_state_dim` before the final projection back to token width.

## Implementation Math

For multihead learned routing, token queries and set descriptors are projected into `d_phi`: `q_t = W_q h_t`, `k_m = W_k d_m`, and logits are `q_t k_m^T / sqrt(d_phi)`. The values read by the router are set states reshaped into heads with total width `set_state_dim`; the routed context is then projected back to `d_model` before the residual LM head path.

## Summary

| family | backend | set_state_dim | d_phi | d_phi/setdim | n | mean val PPL | delta vs d_phi=384 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Set Dense | exact | 384 | 384 | 1.000000 | 3 | 1422.839884 | 0.000000 |
| Set Dense | exact | 512 | 384 | 0.750000 | 3 | 1433.853027 | 0.000000 |
| Set Dense | exact | 512 | 512 | 1.000000 | 3 | 1425.729655 | -8.123372 |
| Set Dense | exact | 768 | 384 | 0.500000 | 3 | 1520.750610 | 0.000000 |
| Set Dense | exact | 768 | 512 | 0.666667 | 3 | 1471.637817 | -49.112793 |
| Set Dense | exact | 768 | 768 | 1.000000 | 3 | 1556.770671 | 36.020060 |
| Set Linear | landmark | 384 | 384 | 1.000000 | 3 | 1515.922607 | 0.000000 |
| Set Linear | landmark | 512 | 384 | 0.750000 | 3 | 1521.928304 | 0.000000 |
| Set Linear | landmark | 512 | 512 | 1.000000 | 3 | 1496.791056 | -25.137248 |
| Set Linear | landmark | 768 | 384 | 0.500000 | 3 | 1554.769491 | 0.000000 |
| Set Linear | landmark | 768 | 512 | 0.666667 | 3 | 1427.530314 | -127.239176 |
| Set Linear | landmark | 768 | 768 | 1.000000 | 3 | 1584.157796 | 29.388306 |
| Set Sparse | local_band | 384 | 384 | 1.000000 | 3 | 1527.470581 | 0.000000 |
| Set Sparse | local_band | 512 | 384 | 0.750000 | 3 | 1407.610067 | 0.000000 |
| Set Sparse | local_band | 512 | 512 | 1.000000 | 3 | 1421.859660 | 14.249593 |
| Set Sparse | local_band | 768 | 384 | 0.500000 | 3 | 1530.087484 | 0.000000 |
| Set Sparse | local_band | 768 | 512 | 0.666667 | 3 | 1516.979899 | -13.107585 |
| Set Sparse | local_band | 768 | 768 | 1.000000 | 3 | 1579.366455 | 49.278971 |

## Interpretation Rule

The bottleneck hypothesis is supported when raising `d_phi` at fixed `set_state_dim` lowers validation PPL relative to the `d_phi=384` reference. It is weakened when matched `d_phi=set_state_dim` fails to recover the wider set-state degradation.

## Artifacts

- All runs TSV: `out/paper_integrated_evidence/tables/a6_interface_bottleneck_all_runs.tsv`
- Summary TSV: `out/paper_integrated_evidence/tables/a6_interface_bottleneck_summary.tsv`
- Manifest: `out/paper_integrated_evidence/checks/a6_interface_bottleneck_manifest.json`

## Validation

- Total expected rows: 54
- Total validated rows: 54
- New expected runs: 27
- New validated runs: 27
- Reused rows: 27
- Log failures: 0
- Failures: 0
