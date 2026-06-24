# SD-8.1 d_phi=768 set_state_dim=768 w4s2

Status: PASS

Mode: `smoke`
Validated runs: 1 / 1

Contract:
- `(w,s)=(4,2)`, `M=255` for full mode
- `d_model=384`, `d_phi=768`, `set_state_dim=768` in full mode
- `output_residual_mode=anchor_span`
- `anchor.enabled=false` (CE only)
- dense exact backend, `candidate_fiber=all_past`, `router.score_mode=dense`
- deferred knobs disabled (`multivector_basis=false`, `r=1`, `set_diversity.lambda_div=0`)

Summary:

| n | seeds | mean val PPL | std | mean peak VRAM MiB | std VRAM | mean span-ablation delta PPL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 542.580627 | 0.000000 | 24.177246 | 0.000000 | 7.455627 |

Comparison:

| label | mean val PPL | mean peak VRAM MiB | delta PPL vs new | delta VRAM MiB vs new |
| --- | ---: | ---: | ---: | ---: |
| SD-8.1 d_phi=768 setdim=768 all_past | 542.580627 | 24.177246 | 0.000000 | 0.000000 |

Runs table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_dphi768_w4s2_smoke_runs.tsv`
Summary table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_dphi768_w4s2_smoke_summary.tsv`
Comparison table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_dphi768_w4s2_smoke_comparison.tsv`
Manifest: `out/paper_integrated_evidence/checks/sd8_all_past_dense_dphi768_w4s2_smoke_manifest.json`

Notes:
- Peak VRAM is raw `train/peak_vram_mib`, consistent with `audit/vram_overhead_audit.md`.
- The Set Dense empty_only VRAM reference is the recorded old near-2 compressed point, not rerun here.
