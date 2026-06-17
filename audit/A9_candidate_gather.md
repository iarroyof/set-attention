# A9 Candidate-Gather Router Comparison

Status: PASS

Goal: remove dense token-to-set router scores/probabilities while preserving A7 empty_only semantics.

Matrix: set dense/sparse/linear, topologies `(4,2)` and `(8,4)`, seeds `0,1,2`, `D=384`, `d_ff=1536`, `L=512`, LR `1e-4`, strict-past causal LM.

Only intended change versus the A7 references: `model.router.score_mode=candidate_gather`.

Artifacts:
- `out/paper_integrated_evidence/tables/a9_candidate_gather_all_runs.tsv`
- `out/paper_integrated_evidence/tables/a9_candidate_gather_summary.tsv`
- `out/paper_integrated_evidence/checks/a9_candidate_gather_manifest.json`

Summary:
- Set Dense empty_only + candidate_gather `(w,s)=(4,2)`: PPL 1326.825155 (delta vs dense-router reference 53.245484), VRAM 11834.985840 MiB (delta 27.719238 MiB).
- Set Dense empty_only + candidate_gather `(w,s)=(8,4)`: PPL 1349.026204 (delta vs dense-router reference 37.325683), VRAM 11190.640137 MiB (delta 118.781739 MiB).
- Set Linear empty_only + candidate_gather `(w,s)=(4,2)`: PPL 1314.316325 (delta vs dense-router reference -41.838623), VRAM 11648.323730 MiB (delta 27.719238 MiB).
- Set Linear empty_only + candidate_gather `(w,s)=(8,4)`: PPL 1393.340495 (delta vs dense-router reference -1.600179), VRAM 11151.353027 MiB (delta 118.781738 MiB).
- Set Sparse empty_only + candidate_gather `(w,s)=(4,2)`: PPL 1404.808757 (delta vs dense-router reference -33.649454), VRAM 11834.985840 MiB (delta 27.719238 MiB).
- Set Sparse empty_only + candidate_gather `(w,s)=(8,4)`: PPL 1481.668172 (delta vs dense-router reference 56.261515), VRAM 11190.640137 MiB (delta 118.781739 MiB).
