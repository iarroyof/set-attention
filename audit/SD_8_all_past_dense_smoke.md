# SD-8 All-Past Dense

Status: PASS

Mode: `smoke`
Validated runs: 1 / 1

Contract:
- `output_residual_mode=anchor_span`
- `anchor.enabled=false` (CE only)
- dense exact backend
- `candidate_fiber=all_past`
- `token_mlp.enabled=false`
- deferred knobs disabled (`multivector_basis=false`, `r=1`, `set_diversity.lambda_div=0`)

Topology summary:

| w | s | n | seeds | mean val PPL | std | mean span-ablation delta PPL |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 4 | 2 | 1 | 0 | 1038.765625 | 0.000000 | -304.949158 |

Runs table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_smoke_runs.tsv`
Summary table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_smoke_summary.tsv`
Manifest: `out/paper_integrated_evidence/checks/sd8_all_past_dense_smoke_manifest.json`

Notes:
- Span-ablation metrics are evaluated during the trained run by zeroing `span_t` at validation.
- SD-8 all_past CE-only has no anchor pre-encoder and should not log anchor auxiliary losses.
