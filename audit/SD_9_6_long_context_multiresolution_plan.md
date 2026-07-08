# SD-9.6 Long-Context Multiresolution Plan

> **Historical landmark-era plan. Do not execute.** Current experiments are exact dense and live in
> `docs/sd_dense_paper5_matrix.md`; current status is `audit/phase_sd_status.md`.

Date: 2026-06-20

Purpose: extend SD-9/SD-9.5 from short `L=512` and lizmark-owned `L=8192+` scale probes into a controlled open-upper-interval long-context ladder on blue-demon, then define the 5-seed cross-family follow-up needed for paper-grade comparisons.

## Guards

- CE-only.
- `output_residual_mode=anchor_span`.
- `token_mlp.enabled=false`.
- `anchor.enabled=false`.
- `candidate_fiber=endpoint_window`.
- `model.multiresolution.enabled=true` for multiresolution rows.
- Long-context set rows use `backend=landmark`, `landmark_coverage=0.25`, batch `1`.
- Do not change topology, backend, coverage, or batch to avoid OOM. If a row OOMs, record it as a result.

## Blue-Demon Support Probe

Run first on blue-demon, highest to lowest, for the requested interval `(8192 downto 2048]`.
`L=8192` is excluded from the blue-demon primary sweep because it is already covered on lizmark.

| L | Backend | Batch | Rows | Seeds | Mode |
|---:|---|---:|---|---|---|
| 4096 | landmark | 1 | mixed65, all-fine, all-coarse | 0 | 1 epoch, `data.limit=500` |
| 2048 | landmark | 1 | mixed65, all-fine, all-coarse | 0 | 1 epoch, `data.limit=500` |

This determines which rows blue-demon can support before launching full sweeps. The queue is implemented by `scripts/run_sd9_6_blue_long_multires_queue.sh` with:

```bash
ROW_SET=support_probe MODE=probe LENGTHS="4096 2048" IMAGE=set-attention:latest bash scripts/run_sd9_6_blue_long_multires_queue.sh
```

Correction note, 2026-06-20: an initial blue-demon support probe was accidentally launched with
`L=8192` included. Its completed rows may be retained as a capacity sanity check only; SD-9.6 blue
operating-point selection and follow-on combination sweeps should use `L=4096,2048`.

## Multiresolution Behavior Sweeps

At each supported `L`, run a blur-fraction sweep:

| Variant | Fine heads `(2,1)` | Coarse heads `(4,2)` | % coarse |
|---|---:|---:|---:|
| all-fine | 8 | 0 | 0 |
| mixed25 | 6 | 2 | 25 |
| mixed50 | 4 | 4 | 50 |
| mixed65 | 3 | 5 | 62.5 |
| mixed75 | 2 | 6 | 75 |
| all-coarse | 0 | 8 | 100 |

Primary diagnostics:

- PPL and peak train VRAM.
- Fine/coarse span-ablation delta PPL.
- Effective range per group.
- Routing entropy and top-1 per group.
- Token-type stratified loss: early/late context and frequent/rare target buckets.
- PPL/VRAM position of each mixed row relative to all-fine/all-coarse and the fine-to-coarse interpolation.

Interpretation targets:

- Whether the optimal coarse fraction shifts upward as `L` grows.
- Whether coarse-head ablation becomes more important at larger `L`.
- Whether effective range separates cleanly into short fine routes and longer coarse routes.
- Whether mixed rows improve PPL and/or VRAM relative to all-fine at the same `L`.

## 5-Seed Cross-Family Plan

Use seed set `{0,1,2,3,4}`. For rows already run with seeds `{0,1,2}`, enqueue only complementary seeds `{3,4}`.

### Current Complementary Seeds

Short SD-9.5 has seeds `{0,1,2}` for:

- mixed25
- all-fine
- all-coarse

Complementary short runs should use the short dense-exact protocol (`L=512`,
batch `16`, exact backend), not the long-context landmark queue. Add a small
short complement launcher or extend `scripts/run_sd9_5_probes.sh` with a seed
selector before running these rows.

For long contexts, most rows currently have seed `0` only or are still probing. Once a length/row is supported, run seeds `1,2,3,4` for paper-grade 5-seed estimates using the landmark long-context queue.

### Cross-Family Groups

At relevant operating points, compare:

1. Token baselines:
   - dense exact where feasible, especially `L=512` and possibly `L=2048`.
   - sparse local-band token attention.
   - linear landmark token attention, coverage `0.25`.
2. Set-only single-resolution references:
   - all-fine `(2,1)`.
   - all-coarse `(4,2)`.
   - existing paper set-family rows where topology/backend match.
3. Set-dictionary multiresolution rows:
   - mixed25, mixed50, mixed65, mixed75.
4. Existing SD controls:
   - SD-9 short/long frontier.
   - SD-9.5 mechanism probes.
   - long-context OOM rows as frontier boundaries.

### Operating Points

Use a scale ladder, not every possible length:

| Operating point | Role |
|---:|---|
| 512 | short-context reference and 5-seed completion; dense exact feasible |
| 2048 | first long-context paper operating point; compare against existing A4 long-context evidence |
| 4096 | blue-demon mid-scale if support probe passes |
| 8192 | lizmark reference only for this branch; exclude from blue-demon SD-9.6 primary sweeps |
| 12288/16384/32768 | lizmark frontier/OOM boundary rows |

### Baseline Convergence Comparisons

For each supported `L`, plot learning curves against the relevant matched token baseline:

- final PPL and PPL-per-VRAM.
- epoch-wise convergence slope.
- whether mixed-resolution catches up to, approaches, or diverges from matched token attention.
- whether benefits are memory-only, quality-only, or Pareto improvements.

Baseline selection:

- `L=512`: dense token baseline plus sparse/linear baselines where already available.
- `L>=2048`: sparse local-band and linear landmark token baselines are the main matched controls; dense exact token is included only if feasible without changing batch/model.

## Launch Rule

Do not start full 5-seed blur sweeps at an `L` until the blue support probe has classified that `L` by row:

- `PASS`: run full seeds.
- `OOM`: record as boundary; do not retry by changing batch/backend/coverage.
- `SLOW/PENDING`: leave running. Do not repeatedly poll while the user is waiting.

Detached-launch monitoring policy:

- After a detached `nohup` launch, run exactly one compact health check to confirm the process is alive
  and expected early logs/artifacts exist.
- Once that first status is healthy, disconnect and stop polling. Do not run repeated SSH polling or
  local sleep loops.
- Wait for the user's explicit notification that the sweep ended before final validation, artifact sync,
  summarization, or launching the next sweep.
- Break this wait rule only if the first health check shows the job died, artifacts are missing/incomplete,
  or logs show OOM/NaN/traceback/W&B failures; then write an incident and report the blocker.

## Lizmark Recovery Rows

If a row is scientifically relevant but fails on blue-demon due to capacity, rerun it on lizmark with
the same guards and `HOST_TAG=lizmark`; do not change backend, coverage, batch, topology, or objective.
Current recovery target from the accidental blue `L=8192` probe is:

```bash
HOST_TAG=lizmark ROW_SET=blur_sweep MODE=probe LENGTHS="8192" VARIANTS="all_fine mixed50 mixed75" \
  IMAGE=set-attention:latest bash scripts/run_sd9_6_blue_long_multires_queue.sh
```
