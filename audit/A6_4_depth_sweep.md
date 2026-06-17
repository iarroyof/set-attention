# A6.4 Set-Stack Depth Sweep

Status: PASS

## Hypothesis

`num_layers` (set-processing stack depth) may be a bottleneck for SKA families, particularly those with restricted intra-set communication (sparse local-band, linear landmark backends). Each SetAttentionBlock applies Z^{l+1} = FFN(Attn_backend(Z^l, bias) + Z^l). With sparse/linear backends, only a subset of set states can communicate per layer, so information that requires multi-hop propagation through set space requires more layers. Wider set_state_dim (capacity) cannot substitute for depth (reach).

## Implementation Math

SetOnlyLM builds `num_layers` SetAttentionBlocks: `self.blocks = nn.ModuleList([SetAttentionBlock(...) for _ in range(num_layers)])`. Each block: Z^{l+1} = FFN(Attn_backend(Z^l) + Z^l) where Z^l ∈ R^{M × set_state_dim}. The router reads routed context (still set_state_dim-wide) and projects back to d_model. This sweep fixes (set_state_dim, d_phi) at the two A6.4 capacity pairs and varies depth in {6,8,10} to test whether more layers recover the PPL gap.

## Summary

| family | backend | set_state_dim | d_phi | num_layers | n | mean val PPL | delta vs depth=6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Set Dense | exact | 384 | 384 | 6 | 3 | 1422.839884 | 0.000000 |
| Set Dense | exact | 384 | 384 | 8 | 3 | 1545.552653 | 122.712769 |
| Set Dense | exact | 384 | 384 | 10 | 3 | 1568.778564 | 145.938680 |
| Set Dense | exact | 768 | 512 | 6 | 3 | 1471.637817 | 0.000000 |
| Set Dense | exact | 768 | 512 | 8 | 3 | 1618.666260 | 147.028442 |
| Set Dense | exact | 768 | 512 | 10 | 3 | 1556.348755 | 84.710938 |
| Set Linear | landmark | 384 | 384 | 6 | 3 | 1515.922607 | 0.000000 |
| Set Linear | landmark | 384 | 384 | 8 | 3 | 1533.850586 | 17.927979 |
| Set Linear | landmark | 384 | 384 | 10 | 3 | 1582.233602 | 66.310994 |
| Set Linear | landmark | 768 | 512 | 6 | 3 | 1427.530314 | 0.000000 |
| Set Linear | landmark | 768 | 512 | 8 | 3 | 1611.043213 | 183.512899 |
| Set Linear | landmark | 768 | 512 | 10 | 3 | 1619.697550 | 192.167236 |
| Set Sparse | local_band | 384 | 384 | 6 | 3 | 1527.470581 | 0.000000 |
| Set Sparse | local_band | 384 | 384 | 8 | 3 | 1549.152547 | 21.681966 |
| Set Sparse | local_band | 384 | 384 | 10 | 3 | 1594.315877 | 66.845296 |
| Set Sparse | local_band | 768 | 512 | 6 | 3 | 1516.979899 | 0.000000 |
| Set Sparse | local_band | 768 | 512 | 8 | 3 | 1585.583211 | 68.603312 |
| Set Sparse | local_band | 768 | 512 | 10 | 3 | 1599.544027 | 82.564128 |

## Interpretation Rule

The depth bottleneck hypothesis is supported when increasing `num_layers` lowers validation PPL, especially for sparse/linear backends relative to dense. If depth gain is larger for sparse/linear, this indicates restricted per-layer communication (not just set-state width) limits representation quality.

## Artifacts

- All runs TSV: `out/paper_integrated_evidence/tables/a64_depth_sweep_all_runs.tsv`
- Summary TSV: `out/paper_integrated_evidence/tables/a64_depth_sweep_summary.tsv`
- Manifest: `out/paper_integrated_evidence/checks/a64_depth_sweep_manifest.json`

## Validation

- Total expected rows: 54
- Total validated rows: 54
- New expected runs: 36
- New validated runs: 36
- Reused rows (depth=6 from A6.3): 18
- Log failures: 0
- Failures: 0
