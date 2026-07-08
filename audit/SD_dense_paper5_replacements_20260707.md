# Exact-Dense Five-Seed Matrix: L3584/B4 Replacement Launch

Status: RUNNING

Launched: 2026-07-07 10:18 CST on lizmark.

## Why

The corrected first pass completed all 255 rows, but the strict endpoint
scanner rejected the 15 mixed `L3584,B4` rows:

- `b25` (`f6c2`), seeds `0..4`;
- `b50` (`f4c4`), seeds `0..4`;
- `b75` (`f2c6`), seeds `0..4`.

The rejected first-pass rows had PPL/VRAM and most diagnostics, but epoch-10
`ausa/{fine,coarse}/grad_norm_{token_pre_pool,set_post_pool,set_post_blocks}`
were `NA`. They remain archived provisional evidence only.

## Prelaunch Checks

- `cancer_rl_agent__deferred_until_sd_grid_release` was stopped again before
  launch.
- Both GPUs were idle at `1/49140 MiB`, `0%` utilization.
- Remote source contains the gradient-probe reset fix
  (`_reset_gradient_probe_schedule()` in `src/models/set_only/set_only_lm.py`).
- Invalid artifacts were archived outside the corrected root:
  `out/_archive/sd_grid_seeded_v1_invalid_l3584_b4_mixed_20260707_1018/`.
- Archive move count: 60 filesystem entries covering 15 CSVs, 15 JSONs,
  15 output directories, 15 done markers, plus logs where present.
- Strict status after archive accepted 120 lizmark endpoint-valid rows.
- Dry run of the unchanged `paper5` manifest:
  - `15` planned replacement runs;
  - `120` skipped endpoint-valid rows;
  - planned cells exactly matched the 15 registered replacement rows.

## Launch

Command class:

```bash
GRID_PROFILE=paper5 SEEDS="0 1 2 3 4" HOST_TAG=lizmark \
  GRID_NAMESPACE=sd_grid_seeded_v1 RUN_TAG=seeded_v1 \
  REQUIRE_APPLIED_SEED=1 TRAINING_DETERMINISTIC=true \
  REQUIRE_EXCLUSIVE_GPU=1 ALLOW_GPU_CORESIDENCY=0 \
  GPU0=0 GPU1=1 nohup bash scripts/run_sd_grid.sh
```

Remote driver state after launch:

- parent PID `3940224`;
- driver PID `3940226`;
- worker PIDs `3940654`, `3940655`;
- launch log:
  `logs/sd_grid_lizmark_paper5_seeded_v1_replacements_20260707.log`;
- first active containers:
  - `sdgrid_lizmark_set_exact_3584_f6c2_b4_0`;
  - `sdgrid_lizmark_set_exact_3584_f6c2_b4_1`.

Initial GPU use after launch: both GPUs at about `41923/49140 MiB`.

## Evidence Decision Before Replacement

Observed complete-valid mean Pareto wins on PPL and peak VRAM already exist at:

- `L2048/B4`: b25 vs token, delta PPL `-26.5 +/- 48.1`, delta VRAM `-516.6 MiB`;
- `L3584/B3`: b25 vs token, delta PPL `-51.8 +/- 54.9`, delta VRAM `-1860.0 MiB`;
- `L4096/B3`: b25 vs token, delta PPL `-44.7 +/- 44.0`, delta VRAM `-2590.3 MiB`.

These are observed mean Pareto wins, not statistically separated PPL wins,
because the paired PPL intervals still cross zero. The provisional `L3584/B4`
b25 row also mean-dominates token, but it cannot count as endpoint-valid paper
evidence until this replacement wave passes.

Mechanism evidence that depends on endpoint gradient diagnostics cannot use
the rejected `L3584/B4` mixed first-pass rows. PPL/VRAM conclusions can be
reported as provisional for that island; full PPL+VRAM+diagnostic claims must
wait for replacement validation.

## Next

Do one status/pull/strict-validation pass after the replacement driver exits.
Do not launch MRP-2/3/5 or other architecture work while MRP-1 replacement and
MRP-0 validation gates remain open.
