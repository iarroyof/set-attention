# MRP-lca-cmp Follow-up Report (L1024/L2048/L4096) — 2026-07-30

Date: 2026-07-30  
Status: SUMMARY ONLY — this report re-summarizes rows that were already completed
before 2026-07-30 (topksweep, l2048budget, l4096admission, l4096stageb). No new
runs were launched by this report; both hosts were GPU-idle at verification time.
The approved NEW next steps (periodic-eval trajectory probe, data-scale probe)
are still pending and are NOT covered here.  
AMENDMENT 2026-07-31: the trajectory probe (l4096tj) has since run and the
"overfitting" interpretation referenced below is RETRACTED — the stageb8k
endpoint 0.7570 was a trough sample of a ±0.15 validation oscillation, not a
generalization trend. See `audit/LCA_calibration_20260718.md` (L4096 trajectory
probe section) and `audit/LCA_research_story_20260727.md` for the corrected
record. The recommendation section below is likewise superseded: the top lever
is now an lr-schedule probe plus best-of-trajectory reporting, then L4096
seeds 1-2 with periodic eval; the 40k data-scale probe is deprioritized.  
Branch: `mrp-lca-cmp-sd`  
Source revision: `8b0754e`

## Provenance

- Hosts: Blue (`iarroyof@192.168.241.149`) and Lizmark (`iarroyof@192.168.241.205`)
- Drivers:
  - `scripts/run_lca_topk_sweep.sh`
  - `scripts/run_lca_l2048_budget.sh`
  - `scripts/run_lca_l4096_admission.sh`
  - `scripts/run_lca_l4096_stageb.sh`
- Artifacts:
  - TSV summaries under `out/lca_cmp/{topksweep,l2048budget,l4096admission,l4096stageb}/`
  - Per-row CSV/JSON outputs under `out/lca_cmp/{topksweep,l2048budget,l4096admission,l4096stageb}/`
  - Logs under `logs/lca_cmp/{blue,lizmark}/`
- Completion basis: completed `.done` markers were present on the remote hosts for the Lizmark rows, and the corresponding local TSV/CSV artifacts were present in the workspace for Blue and Lizmark.

## Setup

The follow-up work targeted the approved continuation path from the MRP-lca-cmp plan:

1. Top-k bandwidth sweep on the b75/L1024 frontier with prefix supervision and dense router scoring.
2. L2048 budget-matched confirmation rows for token dense and b75 full/top-k controls.
3. L4096 admission and Stage-B frontier rows to check whether the b75 mechanism still preserves a memory advantage at larger context length.

The runs used the same LCA comparison scaffolding as the earlier calibration work, with the approved settings for each row and no architecture change.

## Results Summary

### 1) Top-k sweep at L=1024 (b75 / prefix / dense router)

The sweep confirms that the b75 row becomes substantially better as routing bandwidth increases. The strongest row is `topk=1023`, which reaches near-token quality while staying below token VRAM at the same island.

| Row | val_acc | val_loss | peak VRAM (MiB) |
|---|---:|---:|---:|
| `topk=16` | 0.759–0.799 | 0.418–0.488 | 2347–2349 |
| `topk=32` | 0.816–0.822 | 0.376–0.389 | 2351.7 |
| `topk=64` | 0.848–0.852 | 0.322–0.332 | 2360.2 |
| `topk=128` | 0.841–0.868 | 0.289–0.344 | 2377.7 |
| `topk=256` | 0.836–0.889 | 0.244–0.367 | 2407.7 |
| `topk=512` | 0.861–0.913 | 0.195–0.296 | 2374.7 |
| `topk=1023` | 0.905–0.935 | 0.146–0.205 | 2346.7 |

Interpretation: the b75 row is sensitive to routing bandwidth, and the full-routing variant is the clear best performer in this sweep.

### 2) L=2048 budget-matched confirmation rows

At L=2048, the b75 full-routing row remains competitive with token dense while using less peak VRAM, and the sparse top-k=256 control is clearly weaker than the full-routing b75 row.

| Row | val_acc | val_loss | peak VRAM (MiB) |
|---|---:|---:|---:|
| token dense | 0.9438 | 0.1195 | 9123.9 |
| b75 full routing | 0.9353 | 0.1464 | 7201.1 |
| b75 top-k=256 | 0.8099 | 0.4080 | 7325.4 |

Interpretation: the budget-matched confirmation supports the L1024 trend that the b75 row can stay near token-quality at this longer context, but the sparse-control row is not a strong substitute for full routing at this scale.

### 3) L=4096 admission and Stage-B frontier rows

The L4096 rows make the memory frontier explicit. The b75 full-routing row fits at roughly 24.9 GiB, while the token dense row reaches roughly 33.7 GiB under the same native-batch admission settings.

| Row | val_acc | val_loss | peak VRAM (MiB) |
|---|---:|---:|---:|
| token dense (admission) | 0.4856 | 0.8695 | 33745.8 |
| b75 full routing (admission) | 0.5311 | 0.7718 | 24910.2 |
| token dense (Stage-B, 4000 updates) | 0.9407 | 0.1331 | 33745.8 |
| b75 full routing (Stage-B, 4000 updates) | 0.8382 | 0.3480 | 24915.9 |
| b75 full routing + 8000-update extension | 0.7570 | 0.8960 | 24915.9 |

Interpretation: the memory asymmetry is real at L4096, but the Stage-B quality gap remains significant for the seed-0 frontier row. The 8000-update extension DEGRADES the b75 row's validation accuracy relative to the 4000-update run (0.8382 → 0.7570) while validation loss rises sharply (0.348 → 0.896) and train loss keeps falling (0.2486 → 0.1907): this is the overfitting verdict recorded in `audit/LCA_calibration_20260718.md` and `audit/LCA_research_story_20260727.md`. The L2048 budget-rescue pattern does NOT repeat at L4096; the failure mode at scale is generalization, not optimization budget.

## Takeaways

- The b75 set row remains a credible frontier candidate at larger context length.
- The strongest evidence is still the L1024 top-k sweep: higher routing bandwidth materially improves the b75 row without sacrificing the VRAM advantage.
- At L2048, the b75 full-routing row remains close to token dense quality while using less memory.
- At L4096, the set row clearly wins on memory admission, but the quality gap remains meaningful under the launched frontier condition.
- These results support a continued focus on targeted ablations around routing bandwidth and frontier budget rather than broad reruns.

## Artifacts

- Top-k sweep summary: `out/lca_cmp/topksweep/topksweep_blue.tsv`
- L2048 budget summaries: `out/lca_cmp/l2048budget/l2048budget_blue.tsv`, `out/lca_cmp/l2048budget/l2048budget_lizmark.tsv`
- L4096 admission summary: `out/lca_cmp/l4096admission/l4096admission_lizmark.tsv`
- L4096 Stage-B summary: `out/lca_cmp/l4096stageb/l4096stageb_lizmark.tsv`
- Logs: `logs/lca_cmp/blue/` and `logs/lca_cmp/lizmark/`

## Recommendation

The approved next steps (pre-registered in `audit/LCA_research_story_20260727.md`, Part IV 2c) follow from the L4096 overfitting verdict and are generalization diagnostics, in order:

1. Periodic-eval support in the LCA runner (endpoint-only validation today — the val peak between updates 4000 and 8000 is unobserved) plus one L4096 trajectory probe to locate it.
2. A data-scale probe (40k train examples vs the current 20k).
3. A regularization probe (dropout/weight-decay).

These decide whether L4096 parity is a recipe problem or an operator problem; only then does operator work (sum/additive routing probe, then an explicitly-labeled hybrid branch) become the priority. A best-top-k row at L2048/L4096 is NOT recommended: the top-k sweep above shows sparse top-k is memory-neutral in dense-score mode and is dominated by full routing at equal memory, so it cannot close the L4096 gap.
