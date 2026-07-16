# SD Dense Paper5 Final Analysis

Status: PASS; MRP-1 exact-dense five-seed matrix closed on 2026-07-08.

Artifacts:

- Results root: `out/paper_mechanisms/sd_grid_seeded_v1/`
- Strict status: `out/paper_integrated_evidence/checks/sd_grid_seeded_v1_final_20260708/status_all.tsv`
- Cell summaries: `out/paper_integrated_evidence/checks/sd_grid_seeded_v1_final_20260708/cells.tsv`
- Pairwise deltas: `out/paper_integrated_evidence/checks/sd_grid_seeded_v1_final_20260708/pairwise.tsv`
- Frontier rows: `out/paper_integrated_evidence/checks/sd_grid_seeded_v1_final_20260708/frontier.tsv`
- Paper table: `out/final_paper_bundle/overleaf_ready/tables/sd_grid_final_matrix.tex`

## Validation

The pulled Lizmark replacement wave has ended. Lizmark reported no active grid
driver or worker process, both GPUs were idle, and
`logs/sd_grid_lizmark_paper5_seeded_v1_replacements_20260707.log` ended with
`=== SD-GRID lizmark complete ===`.

The strict scanner accepted the full corrected namespace:

```bash
env SD_GRID_REQUIRE_CONTRACT=sd_grid_seeded_v1 \
  python3 scripts/sd_grid_status.py out/paper_mechanisms/sd_grid_seeded_v1
```

Endpoint-valid CSV count is 255. Both formal OOM registries in the corrected
root are header-only. The remaining `L4096/B4` holes are therefore not new
replacement-wave failures: they are the registered exact-dense capacity
boundary inherited from the legacy all-seed OOM observation for token, b0, and
b25, with no corrected endpoint CSV.

All accepted rows use full WikiText-2, no `data.limit`, 10 epochs, exact dense
backend, applied seeds 0--4, `output_residual_mode=anchor_span`, token MLP off,
trained anchor off, CE-only, endpoint-window candidate fiber, and the
`sd_grid_seeded_v1`/`current_matrix_v1` contracts.

## Final Matrix

PPL is validation perplexity. VRAM is the logged peak training VRAM in MiB.
Each populated cell is a five-seed mean plus sample standard deviation.

### Short calibration and batch sensitivity

| Island | Token | b0 all-fine | b25 | b50 | b75 | b100 | Mean frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| L512/B4 | 1022.0 +/- 19.9; 3983 | 1015.7 +/- 51.0; 4138 | 933.4 +/- 38.1; 4081 | 1004.6 +/- 38.7; 3963 | 1107.2 +/- 33.7; 3877 | 1506.0 +/- 43.6; 3645 | b25, b50, b75, b100 |
| L512/B16 | 815.6 +/- 34.3; 13419 | 881.2 +/- 33.2; 13955 | 861.6 +/- 24.2; 13807 | 898.3 +/- 44.8; 13333 | 972.4 +/- 27.7; 12963 | 1260.7 +/- 28.2; 11958 | token, b50, b75, b100 |
| L1024/B4 | 929.2 +/- 9.7; 8001 | 941.1 +/- 22.4; 8302 | 968.7 +/- 60.5; 8047 | 961.0 +/- 50.8; 7653 | 1054.2 +/- 44.7; 7307 | 1437.2 +/- 22.0; 6640 | token, b50, b75, b100 |

The short regime is not a monotone win for set attention. At L512/B4, b25 is
the best quality row and is close to the token VRAM point, while b50/b75/b100
trace the expected memory-saving branch. At L512/B16, however, the token
baseline is clearly the best-quality row; b50 becomes the first memory-saving
set point but pays about +83 PPL. At L1024/B4, token is again the best-quality
row, but b50/b75/b100 form a clean memory branch. This is useful because it
prevents the paper from overstating the result: mixed resolution is not a
universal short-context PPL improvement. The interesting behavior emerges as
the exact-dense score tensors become the dominant constraint.

### Main exact-dense frontier islands

| Island | Token | b0 all-fine | b25 | b50 | b75 | b100 | Mean frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| L2048/B3 | 944.8 +/- 28.3; 14189 | 973.5 +/- 28.8; 14646 | 942.5 +/- 32.7; 13811 | 1014.1 +/- 65.7; 12727 | 1077.3 +/- 43.9; 11726 | 1354.8 +/- 49.3; 10184 | b25, b50, b75, b100 |
| L2048/B4 | 942.8 +/- 15.6; 18633 | 956.5 +/- 27.4; 19222 | 916.4 +/- 27.1; 18117 | 962.1 +/- 41.8; 16679 | 1043.9 +/- 39.3; 15341 | 1335.8 +/- 42.1; 13288 | b25, b50, b75, b100 |
| L3584/B3 | 945.3 +/- 18.8; 31035 | 913.4 +/- 23.2; 31858 | 893.5 +/- 26.8; 29175 | 934.8 +/- 42.5; 26017 | 1006.4 +/- 26.9; 22979 | 1277.4 +/- 55.8; 18886 | b25, b50, b75, b100 |
| L3584/B4 | 896.6 +/- 19.7; 41076 | 894.9 +/- 30.2; 42169 | 875.7 +/- 22.8; 38570 | 904.7 +/- 26.3; 34379 | 993.5 +/- 32.6; 30321 | 1238.9 +/- 53.2; 24880 | b25, b50, b75, b100 |
| L4096/B3 | 909.3 +/- 28.8; 37955 | 898.5 +/- 33.8; 38918 | 864.6 +/- 21.7; 35365 | 921.7 +/- 38.9; 31292 | 996.1 +/- 35.9; 27324 | 1288.1 +/- 18.1; 22123 | b25, b50, b75, b100 |

This is the central paper result. Across every populated main island, b25 is on
the mean Pareto frontier and is the best set row by PPL. More importantly, it
is not merely better than the all-coarse memory extreme; it also improves over
all-fine b0. The all-fine row is a set-mediated high-resolution control, so
the b25 advantage argues that the useful object is not "make sets as close to
tokens as possible." A small coarse allocation appears to regularize or
complement the fine set stream while reducing the exact score-memory
coefficient.

The pairwise token deltas are directionally strongest at the larger contexts:

- L2048/B4 b25 vs token: -26.5 PPL, -517 MiB.
- L3584/B3 b25 vs token: -51.8 PPL, -1860 MiB.
- L3584/B4 b25 vs token: -20.9 PPL, -2506 MiB.
- L4096/B3 b25 vs token: -44.7 PPL, -2590 MiB.

The paired PPL confidence intervals still overlap zero, so the defensible
language is "mean Pareto wins under this five-seed matrix," not "statistically
settled universal PPL superiority." The VRAM side is much cleaner: moderate
blur consistently lowers reported peak memory relative to both token and
all-fine once L is large enough.

### Dense capacity boundary

| Island | Token | b0 all-fine | b25 | b50 | b75 | b100 | Mean frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| L4096/B4 | OOM/missing | OOM/missing | OOM/missing | 893.2 +/- 39.3; 41387 | 976.8 +/- 33.3; 36102 | 1267.8 +/- 21.9; 29182 | b50, b75, b100 |

This row should be interpreted as a capacity-boundary result, not as a direct
PPL comparison against missing token/b0/b25 endpoints. The scientifically
interesting point is that the first supported B4 allocation is not the
low-quality all-coarse model. b50 keeps a PPL near the best B3/B4 main rows
while making the exact-dense L4096/B4 configuration trainable. That is the
strongest evidence for the score-memory allocation story: increasing blur is
not just a cosmetic topology sweep; it moves an otherwise unsupported operating
point into the feasible set while preserving usable language-model quality.

## Mechanism Patterns

The ablation diagnostics are large in absolute PPL because zeroing routed span
contributions removes the main contextual prediction path. They are most useful
comparatively. In b25 and b50 rows, fine ablation costs far more than coarse
ablation, so the fine stream remains the principal carrier of detailed
prediction. The coarse stream has smaller single-group ablation deltas, but its
presence coincides with lower PPL and lower memory than all-fine. This is the
right qualitative pattern for a mixed-resolution model: coarse heads do not
replace fine heads; they change the feasible allocation and appear to provide a
complementary context summary.

Routing diagnostics also separate the streams. Fine effective range is below
one token-center unit in the main rows, while coarse effective range is near
three. Coarse routing entropy is higher and coarse top-1 probability is lower
than fine routing, matching the intended blur interpretation. At L3584/B4, for
example, b25 has fine effective range 0.638, coarse effective range 2.992,
fine entropy 0.294, coarse entropy 0.672, fine top-1 0.882, and coarse top-1
0.576. This is not merely a head-count label in the config; the two groups
learn measurably different routing behavior.

The token-type stratification does not by itself identify the causal benefit
source. Rare-token losses are much higher than frequent-token losses in every
row, and late/early halves are close within each frequency bucket. The useful
role of this diagnostic is negative control: the b25 gains are not obviously
explained by a single late-context or rare-token artifact. The next natural
stage is therefore targeted AR-hit/MQAR probing rather than another broad WT2
blur sweep.

## Research Direction

MRP-1 now supports three paper claims:

1. exact-dense multiresolution set-dictionary attention has an interior
   quality/memory frontier;
2. b25 is the registered main operating point for supported comparisons, frozen
   from L2048/B4 before downstream tasks;
3. heavier blur, especially b50, extends the feasible exact-dense context/batch
   boundary at L4096/B4.

It does not support three stronger claims:

1. it does not prove universal token-attention superiority;
2. it does not prove subquadratic or linear scaling;
3. it does not prove that coarse heads carry long-range associative recall.

The next empirical step is not to use Lizmark immediately for more dense WT2
sweeps. Lizmark can be paused/released until a new approved stage needs it.
The plan is now unblocked for MRP-2 natural AR-hit evaluation and MRP-3
synthetic MQAR, both using the frozen b25 winner plus the registered token,
all-fine, and all-coarse controls. Those tasks are needed to turn the frontier
observation into a mechanism claim about what the fine and coarse streams are
doing.
