# A1.5 Cache Fingerprint and W&B Step Audit

Date: 2026-05-08

## Scope

A1.5 checks were limited to the active A1.1-A1.4 code paths. No training was
run. The blue-demon Docker service was used for runtime checks because the
checks import project runtime modules.

## Mechanisms inspected

- Config duplicate-run fingerprint:
  `src/config/compatibility.py`
  - `validate_compatibility()` computes `cfg["_fingerprint"]`.
  - `_fingerprint()` strips transient paths including `logging` and
    `training.seed`.
  - `_record_fingerprint()` writes JSONL entries to
    `SET_ATTENTION_FINGERPRINT_PATH`, or
    `training.output_dir/metrics/config_fingerprints.jsonl` if the environment
    variable is unset.
  - Reusing an existing fingerprint emits:
    `Config fingerprint <fp> already seen; run may be redundant.`

- Experiment logging fingerprint and W&B step path:
  `src/train/experiment_logger.py`
  - `ExperimentLogger` computes a flattened config fingerprint for run
    metadata and CSV naming.
  - W&B initialization defines `epoch` as the step metric and maps train,
    validation, efficiency, AUSA, and model metric namespaces to it.
  - `log_epoch()` writes the CSV `epoch` column and calls W&B logging with
    `step=epoch`.

- Active experiment entrypoint:
  `scripts/run_experiment.py`
  - Creates `ExperimentLogger` after attaching resolved model metadata.
  - Calls `logger.log_epoch(epoch, ...)` once per training epoch.

An older artifact helper exists in `src/set_attention/data/artifact_cache.py`,
but the active `scripts/run_experiment.py` path does not call
`artifact_root()`, `assert_meta_compatible()`, or
`require_cached_artifacts()`. The runtime checks therefore target the active
config fingerprint and logger paths.

## Blue-demon Docker checks

All checks were run in `~/set-attention` inside the Docker Compose service:

```bash
docker compose exec -T set-attention env PYTHONPATH=src python /tmp/a1_5_checks.py cache-hit
docker compose exec -T set-attention env PYTHONPATH=src python /tmp/a1_5_checks.py cache-miss-wandb
```

The temporary check script loaded
`configs/paper_lr_norm/set_dense_exact.yaml` through the normal config stack.
It used `SET_ATTENTION_FINGERPRINT_PATH` to isolate fingerprint JSONL output
under `/tmp`.

## Results

### Cache-hit path

- Same normalized config loaded twice.
- Fingerprint was stable: `f4bf77b05f68`.
- Fingerprint JSONL remained at one entry.
- Second load emitted the expected duplicate warning:
  `Config fingerprint f4bf77b05f68 already seen; run may be redundant.`

Result: pass.

### Cache-miss path

- Base config fingerprint: `68608145238f`.
- Changed config used `model.router.min_temp=0.75`.
- Changed config fingerprint: `6252765abbad`.
- Fingerprint JSONL contained two entries.
- No duplicate warning was emitted for the changed config.

Result: pass.

### W&B and CSV step monotonicity

The check used a fake in-process W&B module to avoid network access while
exercising the logger's W&B branch.

- CSV path inspected:
  `/tmp/a1_5_cache_miss_wandb/logger/metrics.csv`
- Metadata JSON inspected:
  `/tmp/a1_5_cache_miss_wandb/logger/metrics.json`
- CSV epochs: `[1, 2, 3]`
- W&B log steps: `[1, 2, 3]`
- W&B logged epoch values: `[1, 2, 3]`
- Defined W&B metrics:
  `epoch`, `train/*`, `val/*`, `efficiency/*`, `ausa/*`, `model/*`,
  with namespace metrics using `step_metric="epoch"`.

Result: pass. Logged steps are monotonic for the exercised path.

## Blockers

None for A1.5.
