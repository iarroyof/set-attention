# SD-6 S2 Anchoring Rescue

Status: PASS

Mode: `full`
Validated runs: 6 / 6

Contract:
- `output_residual_mode=anchor_span`
- `anchor.enabled=true`, `anchor.target=pre_encoder`, `anchor.pre_encoder_layers=2`
- `anchor.lambda_h=0.1`, `anchor.detach_target=true`, `anchor.norm=layernorm`
- `anchor.teacher.enabled=false`
- dense exact backend, `candidate_fiber=endpoint_window`
- `token_mlp.enabled=false`, `multivector_basis=false`, `r=1`, `set_diversity.lambda_div=0`

Topology summary:

| w | s | n | seeds | mean val PPL | std | final recon error | mean span-ablation delta PPL |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 2 | 3 | 0,1,2 | 1276.605265 | 12.040938 | 1.363129 | 44247.962443 |
| 16 | 8 | 3 | 0,1,2 | 1437.192017 | 68.947326 | 1.386870 | 39426.043660 |

Branch verdict:

| w | s | branch | PPL delta vs S1 | combined 95% CI | final recon | last-3 recon slope | recommendation |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 4 | 2 | MIXED | -21.260987 | 17.834625 | 1.363129 | -0.002250 | manual review before any downstream launch |
| 16 | 8 | B | -73.707723 | 121.957204 | 1.386870 | -0.000484 | SD-8: D-fiber all_past CE-only first; r=2 only secondary |

Runs table: `out/paper_integrated_evidence/tables/sd6_s2_anchoring_runs.tsv`
Summary table: `out/paper_integrated_evidence/tables/sd6_s2_anchoring_summary.tsv`
Epoch trajectory table: `out/paper_integrated_evidence/tables/sd6_s2_anchoring_epoch_trajectory.tsv`
Topology trajectory table: `out/paper_integrated_evidence/tables/sd6_s2_anchoring_topology_trajectory.tsv`
Verdict table: `out/paper_integrated_evidence/tables/sd6_s2_anchoring_verdict.tsv`
Manifest: `out/paper_integrated_evidence/checks/sd6_s2_anchoring_manifest.json`

Notes:
- `anchor/recon_error_norm` is emitted per epoch in the trajectory tables.
- Span-ablation metrics are evaluated during the trained run by zeroing `span_t` at validation.
- No SD-7 or SD-8 follow-up was launched by this summarizer.
