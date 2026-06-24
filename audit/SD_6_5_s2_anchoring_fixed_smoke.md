# SD-6.5 Fixed S2 Anchoring Rescue

Status: PASS

Mode: `smoke`
Validated runs: 1 / 1

Contract:
- `output_residual_mode=anchor_span`
- `anchor.enabled=true`, `anchor.target=pre_encoder`, `anchor.pre_encoder_layers=2`
- `anchor.lambda_h=0.1`, `anchor.lambda_pre=1.0`, `anchor.pre_encoder_head=true`
- `anchor.detach_target=true`, `anchor.norm=layernorm`
- `anchor.teacher.enabled=false`
- dense exact backend, `candidate_fiber=endpoint_window`
- `token_mlp.enabled=false`, `multivector_basis=false`, `r=1`, `set_diversity.lambda_div=0`

Topology summary:

| w | s | n | seeds | mean val PPL | std | final recon error | mean span-ablation delta PPL |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 2 | 1 | 0 | 611.192810 | 0.000000 | 1.326884 | -86.711060 |

Runs table: `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_smoke_runs.tsv`
Summary table: `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_smoke_summary.tsv`
Epoch trajectory table: `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_smoke_epoch_trajectory.tsv`
Topology trajectory table: `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_smoke_topology_trajectory.tsv`
Manifest: `out/paper_integrated_evidence/checks/sd6_5_s2_anchoring_fixed_smoke_manifest.json`

Notes:
- `anchor/recon_error_norm` is emitted per epoch in the trajectory tables.
- Anchor validity guard: final topology mean `recon_error_norm` must be < 1.2.
- Span-ablation metrics are evaluated during the trained run by zeroing `span_t` at validation.
- No SD-7 or SD-8 follow-up was launched by this summarizer.
