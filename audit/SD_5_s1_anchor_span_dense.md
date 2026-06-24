# SD-5 S1 Anchor-Span Dense

Status: PASS

Mode: `full`
Validated runs: 6 / 6

Contract:
- `output_residual_mode=anchor_span`
- `anchor.enabled=false` (CE only)
- dense exact backend
- `candidate_fiber=endpoint_window`
- `token_mlp.enabled=false`
- deferred knobs disabled (`multivector_basis=false`, `r=1`, `set_diversity.lambda_div=0`)

Topology summary:

| w | s | n | seeds | mean val PPL | std | mean span-ablation delta PPL |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 4 | 2 | 3 | 0,1,2 | 1297.866252 | 10.168949 | 46449.991821 |
| 16 | 8 | 3 | 0,1,2 | 1510.899740 | 82.833538 | 41341.360677 |

Runs table: `out/paper_integrated_evidence/tables/sd5_s1_anchor_span_dense_runs.tsv`
Summary table: `out/paper_integrated_evidence/tables/sd5_s1_anchor_span_dense_summary.tsv`
Manifest: `out/paper_integrated_evidence/checks/sd5_s1_anchor_span_dense_manifest.json`

Notes:
- Span-ablation metrics are evaluated during the trained run by zeroing `span_t` at validation.
- S1 has no anchor pre-encoder and should not log anchor auxiliary losses.
