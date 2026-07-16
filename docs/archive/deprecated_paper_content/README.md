# Deprecated Paper Content

Status: provenance only; not active paper evidence.

This directory indexes manuscript content that was useful for the earlier SKA
and set-attention revision history but is deprecated for the current
`set-dictionary/anchor-span` paper direction.

Deprecated content is not deleted from project history and may remain in audit
or artifact files. It must not be used to override the current exact-dense
multiresolution matrix, launch new experiments, or support active claims.

## Deprecated From Active Results

| Old section or artifact family | Why deprecated for current paper | Replacement |
|---|---|---|
| `Matched Backend Controls at the LR-Normalized Reference` with dense/sparse/landmark SKA rows | It answers a historical backend-family question and mixes inactive sparse/landmark families with older SKA implementations. | Exact-dense matched token/set-dictionary control within each `(L,batch)` island. |
| A7 `empty_only` singleton and near-token calibration figures/tables | It tests a different single-resolution residual/readout path and should not be used as the current set-dictionary token-limit story. | b0 all-fine is the current high-resolution set control; b25 beating b0 is the active interior-allocation result. |
| A-series LR, window, pooling, sparse, and landmark mechanism figures in active Results | They are historical diagnostics for older model families and can confuse reviewers about the current architecture. | Fine/coarse ablation, effective range, routing entropy/top1, and rarity/position diagnostics from `sd_grid_seeded_v1`. |
| Any landmark/linear efficiency prose | Coverage-scaled landmark is not the intended linear/subquadratic implementation and is inactive. | Exact-dense constant-factor score-memory allocation only. |
| Old single-stream/direct-residual claims | The current branch uses multiresolution streams and `anchor_span` output. | MRP-6D multiresolution theory and MRP-1 exact-dense results. |

## Active Replacement Evidence

- `audit/SD_dense_paper5_final_20260708.md`
- `docs/sd_dense_paper5_matrix.md`
- `out/paper_integrated_evidence/checks/sd_grid_seeded_v1_final_20260708/`
- `out/final_paper_bundle/overleaf_ready/example_paper.tex`

## Rule For Future Agents

If a legacy section contains an idea still worth discussing, adapt the idea to
the current exact-dense multiresolution rows. Do not copy the old result as if
it were current evidence.
