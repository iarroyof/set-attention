# A7 Empty-Only Calibration

Status: PASS

## Baseline Provenance

Dense token baseline rows are reused from A2, filtered by exact match: `Baseline token`, `backend=exact`, `D=384`, `d_ff=1536`, `L=512`, `epochs=10`, `lr=1e-4`, seeds `0,1,2`.

## Artifacts

- All runs: `out/paper_integrated_evidence/tables/a7_empty_only_calibration_all_runs.tsv`
- Summary: `out/paper_integrated_evidence/tables/a7_empty_only_calibration_summary.tsv`
- Manifest: `out/paper_integrated_evidence/checks/a7_empty_only_calibration_manifest.json`

## Validation

- New set-side runs: 24/24
- Reused baseline runs: 3/3
- Log scan failures: 0

## Candidate-Count Extension

- `w=2,s=1`: completes the original one-seed point to three seeds.
- `w=3,s=1`: tests a valid high-`M/L` topology with mean candidate count near three.
- `w=2,s=2`: controls for the same window size under non-overlapping endpoint topology.
- `w=8,s=4`: completes the remaining original one-seed topology to three seeds.

## Interpretation Guardrail

A7 tests empirical convergence under the calibrated `empty_only` residual policy. It should not be described as exact Transformer equivalence; the set-side path still uses set pooling, set-stack processing, and routing projections.
