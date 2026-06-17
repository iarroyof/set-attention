# A7.5 Targeted Seed Extension

Status: PASS

## Scope

Seeds 3 and 4 were added for the convergence-critical A7 operating points: matched dense/sparse/linear token baselines and dense/sparse/linear set `empty_only` families at `(w,s)={(1,1),(2,1),(3,1)}`.

The compressed A7 points outside this set remain at three seeds because their degradation is large and not the limiting uncertainty for the empirical convergence claim.

## Validation

- New runs validated: 24/24
- Reused base A7 rows: 81
- Augmented all-run rows: 105
- Augmented summary rows: 27
- Failures: 0

## Artifacts

- All runs: `out/paper_integrated_evidence/tables/a7_backend_family_empty_only_augmented_all_runs.tsv`
- Summary: `out/paper_integrated_evidence/tables/a7_backend_family_empty_only_augmented_summary.tsv`
- Manifest: `out/paper_integrated_evidence/checks/a7_seed_extension_manifest.json`
