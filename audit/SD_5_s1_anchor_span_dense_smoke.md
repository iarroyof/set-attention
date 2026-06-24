# SD-5 S1 Anchor-Span Dense

Status: PASS

Mode: `smoke`
Validated runs: 1 / 1

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
| 4 | 2 | 1 | 0 | 643.824585 | 0.000000 | -128.934143 |

Runs table: `out/paper_integrated_evidence/tables/sd5_s1_anchor_span_dense_smoke_runs.tsv`
Summary table: `out/paper_integrated_evidence/tables/sd5_s1_anchor_span_dense_smoke_summary.tsv`
Manifest: `out/paper_integrated_evidence/checks/sd5_s1_anchor_span_dense_smoke_manifest.json`

Notes:
- Span-ablation metrics are evaluated during the trained run by zeroing `span_t` at validation.
- S1 has no anchor pre-encoder and should not log anchor auxiliary losses.
