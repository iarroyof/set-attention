# WT2 recipe regression bridge (wt2rr) — 2026-08-06

Driver `scripts/run_wt2_recipe_regression.sh` (commit `a51119f`, contract
fix `a35aa0a`). Question (user-framed): does the repaired LCA recipe
(`all_past` + dense router scoring + full routing + zero dropout) transfer
back to language modeling, or does it change the WikiText-2 quality-memory
operating point measured under the old local-fiber recipe
(`endpoint_window` + `candidate_gather` + top-k=16 + dropout 0.1)?

Interpretation guard (user, binding): if the old frontier does not
survive, the phrasing is "the original WT2 frontier was measured under the
local-fiber recipe; the repaired global-fiber recipe changes the
quality-memory operating point" — NOT "the frontier was an artifact of
the local fiber". The old result remains valid for local-context LM.

Island: the GOLD controlled operating point of the registered matrix —
exact / B16 / L512, 10 epochs, lr 1e-4, warmup 1000, d_model 384, 6
layers, 8 heads, deterministic=true to match the matrix protocol. Rows
carry NO experiment_contract on purpose (sd_grid_seeded_v1 pins the old
recipe). Labels `wt2rr_*`; never pooled with the registered sd_grid
matrix. Prefix supervision does not apply to LM (next-token loss is
already dense all-position supervision).

Setup notes: Blue's `.hf/datasets` cache was empty (LCA is synthetic) —
synced the 47 MB WT2 cache from local before launch.

## Seed0 first pass (2026-08-06, all Blue)

| row | val_ppl | train_ppl | peak MiB |
|---|---:|---:|---:|
| tokennodrop | 834.3 | 229.2 | 12343 |
| b25nodrop | 1033.8 | 334.3 | 13030 |
| b75nodrop | 1208.1 | 343.2 | 12463 |
| b75drop (new fiber, dropout 0.1) | 1209.2 | 346.9 | 13019 |

Registered matrix references (same island, old recipe, 5 seeds):
- token: 815.6+-34.3 (seed0 802.7), peak 13419 MiB
- b25: 861.6+-24.2, peak 13807 MiB
- b75: 972.4+-27.7, peak 12963 MiB

## 3-seed verdict (2026-08-06, all Blue) — seed0 findings CONFIRMED

| row | val_ppl (3 seeds) | peak MiB | old recipe (5 seeds) |
|---|---:|---:|---:|
| tokennodrop | 857.9+-22.3 | 12343 | 815.6+-34.3 @ 13419 |
| b25nodrop | 1070.9+-39.2 | 13030 | 861.6+-24.2 @ 13807 |
| b75nodrop | 1155.5+-70.7 | 12463 | 972.4+-27.7 @ 12963 |
| b75drop (new fiber, dropout 0.1) | 1151.9+-72.7 | 13019 | — |

Verdict: the repaired recipe does NOT transfer to WT2 at the GOLD island
— confirmed at 3 seeds. b25 +24% PPL, b75 +19% PPL vs the old recipe;
the b75drop attribution row sits within 0.3% of b75nodrop, so the whole
degradation is fiber/scoring/routing, not dropout. No memory edge (set
~ token peak). Token mildly prefers dropout (+5% PPL without it,
overlapping spreads — mild overfitting). Blur ordering preserved.

Findings (seed0, all confirmed by the 3-seed set):
1. **The repaired recipe does NOT transfer to WT2 at this island**: set
   PPL degrades materially — b25 861.6 -> 1033.8 (+20%), b75 972.4 ->
   1208.1 (+24%) — far beyond the 5-seed spread (+-24..28). Train PPL
   also worsens (b75 283.8 -> 343.2), so this is not overfitting: the
   global fiber/full routing makes the set model fit WORSE on a
   local-structure-dominated task. Consistent with the established
   picture: k=16/local fiber was ample for WT2; global reachability adds
   routing noise without benefit.
2. **No memory edge under the new recipe at this island**: token nodrop
   12343 vs b25nodrop 13030 (+5.6%) and b75nodrop 12463 (+1.0%). The old
   recipe's b75 edge was already thin here (-3.4%); the repaired recipe
   erases it. Attribution: b75drop (new fiber, dropout 0.1) peaks at
   13019 — the fiber/scoring change costs ~+556 MiB over old b75; dropout
   removal saves ~556 MiB (b75) and ~1076 MiB (token).
3. **Dropout is near-neutral for b75 on WT2** (1208.1 vs 1209.2 — 0.1%),
   unlike LCA where it was strongly harmful. Token mildly prefers
   dropout (old seed0 802.7 vs nodrop 834.3; train PPL 229 vs ~253 old —
   mild token overfitting without dropout). The dropout-free recipe is
   therefore task-dependent, not universal: confirmed for LCA, neutral
   for the set row on WT2, slightly harmful for token on WT2.
4. Blur ordering is preserved across recipes: b25 < b75 in PPL on WT2
   under both (861.6 < 972.4 old; 1033.8 < 1208.1 new) — the WT2 blur
   optimum stays b25, reinforcing the task-dependence of the blur
   optimum (LCA: b75).

Plan consequences: seeds 1-2 launched 2026-08-06 14:19 (approved
protocol: material change -> 3 seeds). If confirmed, the paper wording
uses the guard phrasing: the LCA-repaired recipe is task-specific; the
WT2 frontier stands as measured under the local-fiber recipe, and the
bridge documents that the two recipes are different operating points for
different task classes. No WT2 matrix re-run under the new recipe is
warranted — the direction is already decisive at seed0.

Artifacts: `out/wt2_recipe_regression/{token,set}/L512/wt2rr_*_seed0.csv`,
`out/wt2_recipe_regression/wt2_recipe_regression_blue.tsv`, logs
`logs/wt2_recipe_regression/blue/` (remote). References:
`out/paper_mechanisms/sd_grid_seeded_v1/{token,set}/L512/*_b16_*.csv`.

## L3584/B3 island (seed0 launched 2026-08-08, Lizmark)

User-directed extension: the longest complete direct comparison and the
island closest to the WT2 capacity boundary (L4096/B4 set rows are
censored in the registered matrix). Rows: tokennodrop, b25nodrop,
b75nodrop, and b25drop (attribution switched from b75drop to b25drop —
the question here is "does the old LM set winner survive the repaired
fiber?", b25 being the old WT2 winner). Per-GPU sequential queues keep
both Lizmark GPUs continuously fed: gpu0 = [b25nodrop, tokennodrop],
gpu1 = [b75nodrop, b25drop].

Host decision (data-backed): NO row of this island fits Blue's 24 GB —
registered old-recipe peaks are token 31035 / b25 29175 / b75 22979 MiB,
and the repaired fiber only adds memory (b75 old already exceeds Blue's
admission headroom). All L3584/B3 rows therefore run on Lizmark, which
also keeps host-consistency with the registered L3584/B3 reference rows
(also Lizmark).

CORRECTION (2026-08-08, user prompt): the "only adds memory" reasoning
was wrong — dropout removal LOWERS VRAM and at this island it dominates.
Measured new-recipe seed0 peaks (Lizmark): b75nodrop 20297 MiB
(old 22979, -11.7%), b25nodrop 23791 (old 29175, -18.5%), tokennodrop
22072 (old 31035, -28.9%). b75nodrop therefore DOES fit Blue (4.3 GB
headroom). tokennodrop does NOT in practice: it OOM'd on Blue in
backward at 23.53 GiB capacity despite 2.5 GB nominal headroom
(cross-host allocator difference) — the >=4 GB headroom rule is now in
memory/launch-provenance-and-vram-hygiene.md.

5-seed wave (user-directed 2026-08-08, comparable to the 5-seed old
recipe; seed0 stays on Lizmark for all four rows — HOST-MIXING CAVEAT
that must travel into any paper wording):
- Blue: b75nodrop seeds 1-4 (gpu1: seeds 1,3 sequential via first
  driver; gpu0: seeds 2,4 via second driver). tokennodrop seeds 1-4
  moved OFF Blue after the empirical OOM.
- Lizmark: chained driver starts when the seed0 pass completes:
  gpu0 = tokennodrop + b25nodrop per seed, gpu1 = b25drop per seed,
  seeds 1-4.

Seed0 partial results (Lizmark; b25drop pending at time of writing):
b75nodrop val_ppl 1119.0 @ 20297 MiB; b25nodrop 1117.5 @ 23791;
tokennodrop 973.3 @ 22072. vs old recipe: b25 +25% PPL (and the blur
ordering INVERTS at this island under the new recipe: b25 ~= b75),
b75 +11%, token +3%. Under the new recipe b75 keeps a -8% VRAM edge
vs token here (20297 vs 22072).

Registered references (old recipe, 5 seeds, Lizmark):
- token: 945.3 mean val_ppl @ 31035 MiB
- b25: 893.5 mean val_ppl @ 29175 MiB (old frontier edge: -52 PPL AND
  -6% VRAM vs token — the island where the local-fiber set row wins)
- b75: 1006.4 mean val_ppl @ 22979 MiB

## L3584/B3 5-seed FINAL (2026-08-09) — complete picture

All 20 rows complete. Host provenance (must travel into paper wording):
seed0 all rows Lizmark; seeds 1-4 token/b25nodrop/b25drop Lizmark,
b75nodrop Blue. tokennodrop seeds 1-4 attempted on Blue first — OOM in
backward at 23.53 GiB (2.5 GB nominal headroom insufficient cross-host);
moved to Lizmark. Blue s14 TSV carries those FAIL rows as provenance.

| row (5 seeds) | val_ppl | train_ppl | peak MiB | vs old recipe |
|---|---:|---:|---:|---|
| tokennodrop | 990.6+-15.0 | 313.6 | 22072 | +4.8% PPL, -28.9% VRAM |
| b25nodrop | 1125.7+-46.0 | 419.9 | 23791 | +26.0% PPL, -18.5% VRAM |
| b75nodrop | 1120.5+-28.7 | 422.2 | 20297 | +11.3% PPL, -11.7% VRAM |
| b25drop | 1125.0+-42.5 | 421.1 | 31154 | — |

Findings:
1. **Dropout is quality-neutral for the set row at L3584 too**:
   b25nodrop 1125.7 vs b25drop 1125.0 (0.07%). The entire b25
   degradation (+26%) is fiber/scoring/routing, as at L512. But dropout
   is memory-expensive: b25drop peaks +31% over b25nodrop (31154 vs
   23791) — on WT2 dropout buys the set row nothing and costs a third
   more memory.
2. **The blur ordering inverts under the repaired recipe**: old recipe
   b25 893.5 << b75 1006.4; new recipe b25 1125.7 ~= b75 1120.5. The
   old local-fiber recipe monetized b25's six fine heads; under
   all_past+full routing the fine-head advantage vanishes (b25 degrades
   +26%, b75 only +11%).
3. **The old WT2 frontier win does not survive the repaired recipe**:
   old b25 beat token on BOTH axes (-52 PPL AND -6% VRAM). New recipe's
   best set row (b75) is +13.1% PPL vs tokennodrop at -8.0% VRAM —
   cheaper but substantially worse. Guard phrasing applies verbatim:
   the original WT2 frontier was measured under the local-fiber recipe;
   the repaired global-fiber recipe changes the quality-memory
   operating point. Both are valid; they answer different task classes.
4. **Cross-island consistency**: L512 and L3584 tell the same story
   (b25 +24%/+26%, b75 +19%/+11%, dropout-neutral set rows, token
   mildly dropout-preferring). The L512 bridge 3-seed and L3584 5-seed
   sets mutually reinforce.
5. New-recipe memory ordering at L3584: b75 20297 < token 22072 < b25
   23791 << b25drop 31154. All nodrop peaks below the old-recipe peaks.

Artifacts: `out/wt2_recipe_regression/{token,set}/L3584/wt2rr_*.{csv,json}`,
TSVs `wt2_recipe_regression_{blue,lizmark}.tsv`, logs
`logs/wt2_recipe_regression/{blue,lizmark}/` (remote).
