# SD-8 All-Past Dense-Router

Status: PASS

Mode: `full`
Validated runs: 6 / 6

Contract:
- `output_residual_mode=anchor_span`
- `anchor.enabled=false` (CE only)
- dense exact backend
- `candidate_fiber=all_past`
- `router.score_mode=dense` (same causal support, avoids all_past candidate-gather OOM)
- `token_mlp.enabled=false`
- deferred knobs disabled (`multivector_basis=false`, `r=1`, `set_diversity.lambda_div=0`)

Topology summary:

| w | s | n | seeds | mean val PPL | std | mean span-ablation delta PPL |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 4 | 2 | 3 | 0,1,2 | 1288.603190 | 15.208598 | 62723.872070 |
| 16 | 8 | 3 | 0,1,2 | 1363.931559 | 8.742478 | 48865.647868 |

Verdict:
- SD-8 all_past CE-only is a validated PASS run package, not an adoption PASS.
- `(16,8)` improves materially relative to S1 / fixed S2 (`~1510` PPL) and the old SKA dense direct
  reference (`1422.8` PPL), but remains far above the dense token baseline (`781.1` PPL).
- `(4,2)` is essentially flat relative to fixed S2 (`1288.97` PPL) and still worse than the old set dense
  empty_only reference (`1273.6` PPL).
- Span-ablation deltas remain very large, so prediction is still carried by the span path and there is no
  token bypass.

Recommended next step for user review:
- Stop here. Do not launch `window_plus_landmarks`, `r=2`, SD-7, `lambda_h=1.0`, or multivector follow-ups
  without explicit user approval.
- If continuing the capacity branch, the next pre-registered SD-8 follow-up to consider is
  `window_plus_landmarks` CE-only; `r=2` remains secondary.

Runs table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_routerdense_runs.tsv`
Summary table: `out/paper_integrated_evidence/tables/sd8_all_past_dense_routerdense_summary.tsv`
Manifest: `out/paper_integrated_evidence/checks/sd8_all_past_dense_routerdense_manifest.json`

Notes:
- Span-ablation metrics are evaluated during the trained run by zeroing `span_t` at validation.
- SD-8 all_past CE-only has no anchor pre-encoder and should not log anchor auxiliary losses.
