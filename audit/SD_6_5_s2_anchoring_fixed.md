# SD-6.5 Fixed S2 Anchoring Rescue

Status: PASS

Mode: `full`
Validated runs: 6 / 6

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
| 4 | 2 | 3 | 0,1,2 | 1288.973145 | 65.889471 | 1.184113 | 44441.713053 |
| 16 | 8 | 3 | 0,1,2 | 1510.373413 | 32.844301 | 1.176980 | 40151.144816 |

Branch verdict:

| w | s | branch | PPL delta vs S1 | combined 95% CI | final recon | guard pass | last-3 recon slope | recommendation |
| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| 4 | 2 | B | -8.893107 | 75.443707 | 1.184113 | True | 0.004992 | SD-8: D-fiber all_past CE-only first; r=2 only secondary |
| 16 | 8 | B | -0.526327 | 100.834607 | 1.176980 | True | 0.002703 | SD-8: D-fiber all_past CE-only first; r=2 only secondary |

Runs table: `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_runs.tsv`
Summary table: `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_summary.tsv`
Epoch trajectory table: `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_epoch_trajectory.tsv`
Topology trajectory table: `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_topology_trajectory.tsv`
Verdict table: `out/paper_integrated_evidence/tables/sd6_5_s2_anchoring_fixed_verdict.tsv`
Manifest: `out/paper_integrated_evidence/checks/sd6_5_s2_anchoring_fixed_manifest.json`

Notes:
- `anchor/recon_error_norm` is emitted per epoch in the trajectory tables.
- Anchor validity guard: final topology mean `recon_error_norm` must be < 1.2.
- Span-ablation metrics are evaluated during the trained run by zeroing `span_t` at validation.
- No SD-7 or SD-8 follow-up was launched by this summarizer.
