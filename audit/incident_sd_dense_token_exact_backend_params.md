# Dense token exact-backend config incident

Status: RESOLVED; corrected baseline queues completed.

Date verified: 2026-06-29.

## Failure

Every newly scheduled exact dense-token cell exited before model construction with:

`config.schema.ConfigError: exact backend forbids backend_params`

The grid used the landmark token YAML and attempted to clear its parameters with
`model.backend_params={}`. The override parser preserved a nonempty mapping by the time compatibility
validation ran.

## Scope and scientific impact

- No failed token row trained or produced a usable epoch, so none can enter an aggregate.
- The set branch uses the set-dictionary YAML and never executes the token-only override. It is unaffected.
- All completed dense set rows were revalidated as full-data, 10-epoch, exact-backend runs with the SD-9
  guards and peak VRAM. Completed set logs had no NaN/Inf/traceback findings.
- Set-vs-set blur curves and the dense OOM frontier remain valid.
- Dense set-vs-token gains are not established until the corrected token controls finish. Any earlier
  cross-family number based on heterogeneous historical token buckets remains reference-only.

## Resolution

`scripts/run_sd_grid.sh` now selects a backend-native token config:

- exact: `configs/paper_lr_norm/baseline_dense_exact.yaml`
- landmark: `configs/paper_lr_norm/baseline_linear_landmark.yaml`
- local-band/sparse: `configs/paper_lr_norm/baseline_sparse_local_band.yaml`

The exact branch no longer tries to erase inherited landmark parameters. `scripts/sd_grid_status.py`
also reads metrics at `SD_GRID_TARGET_EPOCHS` (10 by default), so a protocol-matched run that continued
to 30 epochs contributes its epoch-10 row rather than its final row.

Follow-up consistency fix: `cell_inflight()` now selects the backend-native token config signature too.
Before this correction, exact token training used the right dense YAML but process duplicate detection
still searched for the landmark YAML name. The host-level grid lock protected the active retry, so this
did not duplicate or alter a result; the fix restores the documented cross-launcher guard.

Validation completed:

- `bash -n scripts/run_sd_grid.sh`
- epoch-selection regression check in `tests/test_sd_grid_status.py`
- exact dense-token config load in the training container on blue-demon and lizmark
- token-only dry run on both hosts
- one post-launch health check on both corrected queues

Final retry outcome:

- blue-demon: 8/8 requested exact-token cells completed 10 epochs;
- lizmark: L2048 3/3 completed 10 epochs; L4096 3/3 produced expected CUDA OOM before epoch 1;
- metadata failures and unexpected completed-log findings: 0;
- matched verdict: `audit/SD_dense_matched_results.md`.
