# MRP-3 Calibration Eval-Cadence Incident

Date: 2026-07-08.

## Summary

The first approved MRP-3 token calibration launch on blue-demon was stopped
before completion because the runner evaluated only once after the full
`training.max_updates=20000` endpoint. The registered MRP-3 protocol requires
evaluation every 500 optimizer updates and selection of the first LR/update
where calibration query accuracy reaches at least `0.99` for two consecutive
evaluations.

## Affected Run

- Host: blue-demon
- Checkout: `~/set-attention-anchor-span-sync`
- Log: `logs/mrp3_mqar_calibration_token_20260708_172459.log`
- Container: `4b037ced678e`
- Docker PID: `1930288`
- Python PID: `1930444`
- Row active at stop: token, `lr=1e-4`

The container was stopped intentionally. The partial endpoint-only trace must
not be used to select the MRP-3 LR/update budget.

## Fix

Instrumentation/control only, no architecture change:

- `src/train/mqar.py` now exposes `train_mqar_update_block()` so calibration
  can train in finite update blocks without restarting the data iterator.
- `scripts/run_mqar.py` now supports `training.eval_every_updates`; when set,
  it logs evaluation rows at that cadence, records consecutive calibration
  hits, and stops when the registered gate passes.
- `scripts/run_mqar_calibration.sh` now sets:
  - `training.eval_every_updates=500`;
  - `training.calibration_accuracy_threshold=0.99`;
  - `training.calibration_consecutive_evals=2`.
- `src/train/metrics_schema.py` now includes calibration trace columns.

## Validation Required Before Relaunch

Run the one-step / reduced-cadence Blue container validation and confirm the
CSV contains multiple update-indexed rows with:

- `train/completed_updates`;
- `val/calibration_consecutive_hits`;
- `val/calibration_gate_passed`;
- `val/calibration_selected_update`.

Only then relaunch the registered token LR calibration sweep.
