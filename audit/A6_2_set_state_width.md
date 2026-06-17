# A6.2 Explicit Set-State Dimensionality Sweep

Status: PASS

## Scope

A6.2 explicitly tests `model.set_state_dim`, the width of pooled set states, set-attention blocks, set backend value states, and routed set context before projection back to token width. Token width is held fixed at D=384 and `d_phi` is held fixed at 384.

Matched token controls are reused from validated A2/A2.4 artifacts. SKA `set_state_dim=384` rows are reused from A6.1 `d_phi=384` artifacts. New runs cover SKA `set_state_dim` in {512,768}.

## Summary

| family | backend | set_state_dim | n | mean val PPL | std | delta vs anchor | mean VRAM MiB | sec/epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline Dense | exact | NA | 3 | 781.109436 | 37.144728 | 0.000000 | 13407.220703 | 31.245398 |
| Baseline Linear | landmark | NA | 3 | 973.942179 | 13.738719 | 0.000000 | 12596.476562 | 27.456766 |
| Baseline Sparse | local_band | NA | 3 | 740.160116 | 3.803057 | 0.000000 | 13408.720703 | 33.152757 |
| Set Dense | exact | 384 | 3 | 1422.839884 | 45.807377 | 0.000000 | NA | NA |
| Set Dense | exact | 512 | 3 | 1433.853027 | 49.209999 | 11.013143 | NA | NA |
| Set Dense | exact | 768 | 3 | 1520.750610 | 24.492327 | 97.910726 | NA | NA |
| Set Linear | landmark | 384 | 3 | 1515.922607 | 135.277873 | 0.000000 | NA | NA |
| Set Linear | landmark | 512 | 3 | 1521.928304 | 41.431678 | 6.005697 | NA | NA |
| Set Linear | landmark | 768 | 3 | 1554.769491 | 57.671969 | 38.846883 | NA | NA |
| Set Sparse | local_band | 384 | 3 | 1527.470581 | 47.135482 | 0.000000 | NA | NA |
| Set Sparse | local_band | 512 | 3 | 1407.610067 | 53.793346 | -119.860514 | NA | NA |
| Set Sparse | local_band | 768 | 3 | 1530.087484 | 11.288768 | 2.616903 | NA | NA |

## Best SKA set_state_dim

| family | backend | best set_state_dim | mean val PPL |
| --- | --- | ---: | ---: |
| Set Dense | exact | 384 | 1422.839884 |
| Set Linear | landmark | 384 | 1515.922607 |
| Set Sparse | local_band | 512 | 1407.610067 |

## Artifacts

- All runs TSV: `out/paper_integrated_evidence/tables/a6_set_state_width_all_runs.tsv`
- Summary TSV: `out/paper_integrated_evidence/tables/a6_set_state_width_summary.tsv`
- Manifest: `out/paper_integrated_evidence/checks/a6_set_state_width_manifest.json`

## Validation

- Total expected rows: 36
- Total validated rows: 36
- New expected runs: 18
- New validated runs: 18
- Reused rows: 18
- Log failures: 0
- Failures: 0
