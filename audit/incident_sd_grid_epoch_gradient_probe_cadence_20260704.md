# Incident: Epoch Gradient-Probe Cadence

Status: MITIGATED IN SOURCE; 15 L3584/B4 MIXED RERUNS REQUIRED

Discovered: 2026-07-04 during the user-requested MRP-1 queue check.

## Finding

The strict Lizmark scan rejected completed mixed `L=3584,B=4` rows because
their epoch-10 CSV rows had `NA` for:

- `ausa/{fine,coarse}/grad_norm_token_pre_pool`;
- `ausa/{fine,coarse}/grad_norm_set_post_pool`;
- `ausa/{fine,coarse}/grad_norm_set_post_blocks`.

The metrics are present at epoch 1. The model used one global probe counter
with interval 200, and validation plus span-ablation forwards advanced that
counter. At this operating point, later training epochs could contain no probe
step, while diagnostics were reset after every epoch.

## Impact

- The PPL, peak-VRAM, ablation, routing, pooling, seed, full-data, and config
  fields are present. This is an endpoint instrumentation defect, not a model
  architecture or optimization change.
- Blue rows and Lizmark `L=2048,B=4` rows pass the full endpoint scanner and
  are unaffected.
- The 15 mixed `L=3584,B=4` cells (`b25`, `b50`, and `b75`, seeds 0--4) must
  be replaced before they count as contract-valid paper rows.
- Their current PPL/VRAM may be shown only as provisional, never as completed
  MRP-1 evidence.

## Mitigation

`SetOnlyLM.get_diagnostics()` now resets the gradient-probe schedule after
each epoch's diagnostics are emitted. The first training batch of the next
epoch therefore records the probe regardless of how many validation or
ablation forwards occurred.

The patch changes instrumentation only. It does not alter the forward output,
loss, optimizer, routing, pooling, or architecture.

Focused Lizmark container tests passed: `13 passed`.

The patch is deployed in both hosts' bind-mounted source. Runs started after
deployment use it. Runs already inside a container at deployment retain the
old imported code and remain subject to strict validation.

## Recovery

1. Allow the existing single Lizmark driver to finish; do not add a second
   launcher.
2. Keep the external `cancer_rl_agent` workload deferred. Watcher PID
   `3049751` was stopped on 2026-07-04.
3. After the driver exits, archive the 15 affected CSV/JSON/log records
   outside the corrected aggregation root.
4. Relaunch the unchanged `paper5` manifest. Atomic identity and the strict
   scanner must skip valid cells and rerun only the archived invalid cells
   plus any still-pending cells.
5. Require endpoint diagnostics and rerun the full strict scanner before
   accepting the replacement rows.

## 2026-07-07 Verification

Lizmark PID `2879441` exited normally and
`logs/sd_grid_lizmark_paper5_seeded_v1.log` ends with
`=== SD-GRID lizmark complete ===`. Artifacts were pulled locally.

The full first pass now has PPL/VRAM for all 255 corrected rows, but the strict
scanner still rejects all 15 registered mixed `L3584,B4` cells for the same
endpoint gradient diagnostics listed above. Therefore the affected first-pass
rows remain provisional only.

Remote source contains the `_reset_gradient_probe_schedule()` fix, but the
2026-07-07 check also found `cancer_rl_agent__deferred_until_sd_grid_release`
running again without current GPU allocation. That blocked replacement launch
until the container was stopped again.

The external container was stopped again on 2026-07-07. The 15 invalid records
and markers were archived outside the corrected root, a dry run planned
exactly 15 replacement cells and skipped 120 valid lizmark cells, and the
replacement driver was launched as PID `3940226` with workers `3940654` and
`3940655`. Launch audit:
`audit/SD_dense_paper5_replacements_20260707.md`.
