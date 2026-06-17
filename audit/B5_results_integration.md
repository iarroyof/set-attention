# B5 Results Integration Audit

Status: PASS

## Scope

Integrated the completed A1-A6 evidence into the current NeurIPS final bundle after the final A5 handoff passed, then completed the first B1-B4/B6 consistency pass over the paper-facing definitions and claims.

## Inputs

- `audit/A5_4_handoff.md`
- `out/paper_integrated_evidence/checks/final_reproducibility_manifest.json`
- `out/paper_integrated_evidence/checks/final_artifact_index.tsv`
- A2/A2.4 matched-control summaries
- A4.2/A4.3 long-context and convergence summaries
- A6.1-A6.4 capacity, interface, and depth summaries

## Modified Files

- `out/final_paper_bundle/overleaf_ready/example_paper.tex`: replaced stale Tier-C Results text in the current NeurIPS bundle with artifact-backed Results covering matched controls, mechanism sweeps, long-context memory/quality, convergence, and A6 capacity diagnostics.
- `out/final_paper_bundle/overleaf_ready/example_paper.tex`: reconciled the paper-facing architecture/theory text with the implemented strict-past endpoint bank, T1 dropped trailing windows, direct residual path, candidate-fiber diagnostics, and pre-shifted loss normalization.
- `out/final_paper_bundle/overleaf_ready/example_paper.tex`: removed stale historical appendix tables containing unreconciled pre-A1 values and replaced them with provenance pointers to the final A5 artifact index.
- `scripts/validate_a5_handoff.py`: extended the final handoff validator to include A6.1-A6.4 manifests, audits, and source CSV SHA256 checks.
- `audit/A5_4_handoff.md`: regenerated from the validator with A6.1-A6.4 included.
- `out/paper_integrated_evidence/checks/final_artifact_index.tsv`: regenerated final artifact index.
- `out/paper_integrated_evidence/checks/final_reproducibility_manifest.json`: regenerated final reproducibility manifest.

## Validation

- Blue-demon A5 validator: PASS
  - manifests: 14
  - source CSVs: 428
  - source JSONs: 428
  - indexed artifacts: 61
- Local stale-reference grep over the paper and final handoff artifacts: PASS
  - no stale historical-results labels, old set-count claims, deprecated landmark-count keys, or active Nyström-backend claims in the updated Results/handoff surface.
- `git diff --check` for touched B5 files: PASS.
- `python3 -m py_compile scripts/validate_a5_handoff.py`: PASS.
- `bash scripts/compile_paper_bundle.sh /mnt/d/UserFolders/Documents/GitHub/set-attention example_paper.tex`: PASS.
  - Output PDF: `out/final_paper_bundle/checks/compile_logs/run_bX7PyO/example_paper.pdf`
  - Final PDF pages: 42
  - Fatal LaTeX errors: none
  - Fatal/undefined-control grep: none
  - Remaining warnings: layout warnings and empty bibliography warning because the current draft has no citation commands.

## Paper Claim State

- Matched backend controls are written conservatively: token dense/sparse/linear baselines outperform SKA at the LR-normalized `L=512` reference.
- Long-context claim is limited to memory advantage with worse PPL, not quality improvement.
- A4.3 convergence is written as unfavorable to SKA.
- A6 capacity diagnostics are written as ruling out a single simple bottleneck explanation; moderate `d_phi` helps some rows, explicit set-state widening helps SetSparse, and additional set-stack depth hurts across tested settings.
- Abstract and introduction no longer claim validation-perplexity dominance over matched token baselines.
- Main model/theory definitions use `M=floor((L-w)/s)+1`, strict-past endpoint candidates, and direct residual `h_t^(0)+r_t`.

## Blue-Demon Sync

- `out/final_paper_bundle/overleaf_ready/example_paper.tex` is the current paper source.
- Final handoff artifacts and `scripts/validate_a5_handoff.py` match local/blue-demon SHA256 values after sync.
