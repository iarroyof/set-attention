# SD-9 Multi-Resolution Frontier Test

> **Mixed historical/current evidence.** The short exact-dense result remains reference evidence. The
> long landmark arm is historical only. Current cells and claims are governed by
> `docs/sd_dense_paper5_matrix.md` and `audit/phase_sd_status.md`.

Date: 2026-06-20
Status: PASS validation package, negative Pareto verdict

## Contract

- Short context: L=512, batch=16, dense exact backend on blue-demon.
- Long context: L=8192, batch=1, landmark backend on lizmark with `landmark_coverage=0.25`.
- Rows per context: mixed plus all-fine `(2,1)` and all-coarse `(4,2)` uniform extremes.
- Seeds: 0, 1, 2.
- CE-only: `anchor.enabled=false`.
- Guarded residual contract: `output_residual_mode=anchor_span`, `token_mlp.enabled=false`, `candidate_fiber=endpoint_window`.
- SD-9 is a set-vs-set frontier test, not a token-attention claim.

## Guard Verification

The SD-9 launch/config path pins `output_residual_mode=anchor_span` in both the YAML and launcher overrides. The summarizer asserts both:

- `model.output_residual_mode == "anchor_span"`
- `resolved.output_residual_mode == "anchor_span"`

The summarizer also asserts `model.token_mlp.enabled == false`, `model.anchor.enabled == false`, and `model.candidate_fiber == "endpoint_window"` for every validated run.

## Completion

Both full matrices completed:

- Short: 9/9 rows exited 0 on blue-demon.
- Long: 9/9 rows exited 0 on lizmark.

Validation passed with no log-scan failures in the per-context manifests.

## Summary

| Context | Variant | Backend | Heads fine/coarse | Mean val PPL | Mean peak VRAM MiB | Mean span-ablation delta PPL |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| short | all-fine | exact | 8/0 | 912.9089 | 13933.3740 | 58367.9036 |
| short | mixed-25 | exact | 6/2 | 862.1083 | 13790.4800 | 64488.9126 |
| short | all-coarse | exact | 0/8 | 1267.7873 | 11941.9321 | 46346.7570 |
| long | all-fine | landmark | 8/0 | 1033.1019 | 27927.6885 | 59190.1363 |
| long | mixed-65 | landmark | 3/5 | 1008.9868 | 20192.8843 | 57981.5002 |
| long | all-coarse | landmark | 0/8 | 1431.3822 | 15160.3140 | 48334.0371 |

## Pareto Verdict

Short context:

- Mixed-25 is better than the fine-to-coarse interpolation on PPL: `862.1083` vs interpolated `1001.6285`.
- Mixed-25 is worse than the interpolation on peak VRAM: `13790.4800` MiB vs interpolated `13435.5135` MiB.
- Verdict: not Pareto-better than the interpolation.

Long context:

- Mixed-65 is better than the fine-to-coarse interpolation on PPL: `1008.9868` vs interpolated `1282.0271`.
- Mixed-65 is worse than the interpolation on peak VRAM: `20192.8843` MiB vs interpolated `19948.0794` MiB.
- Verdict: not Pareto-better than the interpolation.

## Artifacts

- Short runs: `out/paper_integrated_evidence/tables/sd9_multiresolution_short_runs.tsv`
- Short summary: `out/paper_integrated_evidence/tables/sd9_multiresolution_short_summary.tsv`
- Short verdict: `out/paper_integrated_evidence/tables/sd9_multiresolution_short_verdict.tsv`
- Short manifest: `out/paper_integrated_evidence/checks/sd9_multiresolution_short_manifest.json`
- Long runs: `out/paper_integrated_evidence/tables/sd9_multiresolution_long_runs.tsv`
- Long summary: `out/paper_integrated_evidence/tables/sd9_multiresolution_long_summary.tsv`
- Long verdict: `out/paper_integrated_evidence/tables/sd9_multiresolution_long_verdict.tsv`
- Long manifest: `out/paper_integrated_evidence/checks/sd9_multiresolution_long_manifest.json`

## Reference Context

Reference only, not adoption criteria:

- L=512 token baseline: 781.1.
- A8.3 L=8192 landmark rows: set 2181.3 / baseline 1048.4.
