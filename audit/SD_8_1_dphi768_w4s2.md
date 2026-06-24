# SD-8.1 d_phi=768 set_state_dim=768 w4s2

Status: PASS

Mode: `full`
Validated runs: 3 / 3

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
| 3 | 0,1,2 | 1241.522583 | 44.528320 | 12764.080078 | 0.000000 | 71758.745646 |

Comparison:

| label | mean val PPL | mean peak VRAM MiB | delta PPL vs new | delta VRAM MiB vs new |
| --- | ---: | ---: | ---: | ---: |
| SD-8.1 d_phi=768 setdim=768 all_past | 1241.522583 | 12764.080078 | 0.000000 | 0.000000 |
| Dense token baseline | 781.109436 | 13407.220703 | -460.413147 | 643.140625 |
| Set Dense empty_only old ref | 1273.600000 | 11807.300000 | 32.077417 | -956.780078 |
| Best SD so far: SD-8 all_past d_phi=384 setdim=384 | 1288.603190 | 11913.547852 | 47.080607 | -850.532226 |
| Fixed S2 anchor_span d_phi=384 setdim=384 | 1288.973145 | NA | 47.450562 | NA |

Interpretation:
- Doubling the dictionary atom width improves the best SD `(4,2)` result so far by `47.080607` PPL
  (`1288.603190` -> `1241.522583`).
- The improvement costs `850.532226` MiB more peak train VRAM than SD-8 all_past at
  `d_phi=set_state_dim=384` (`11913.547852` -> `12764.080078` MiB).
- The new run also beats the old Set Dense empty_only near-2 reference by `32.077417` PPL, but uses
  `956.780078` MiB more peak train VRAM.
- It still trails the dense token baseline by `460.413147` PPL, while using `643.140625` MiB less peak
  train VRAM than that token baseline.
- Span-ablation delta remains very large, so the model is still span-carried rather than using a token
  bypass.

Runs table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_dphi768_w4s2_runs.tsv`
Summary table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_dphi768_w4s2_summary.tsv`
Comparison table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_dphi768_w4s2_comparison.tsv`
Manifest: `out/paper_integrated_evidence/checks/sd8_all_past_dense_dphi768_w4s2_manifest.json`

Notes:
- Peak VRAM is raw `train/peak_vram_mib`, consistent with `audit/vram_overhead_audit.md`.
- The Set Dense empty_only VRAM reference is the recorded old near-2 compressed point, not rerun here.
