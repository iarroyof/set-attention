# Server Copy Provenance And Launch-Safety Audit

Date: 2026-07-09; updated 2026-07-13 after Blue MRP-3 resume.

## Current Rule

Do not launch new work from a server checkout unless this file or
`audit/phase_sd_status.md` says that checkout is launch-ready for the specific
stage. Runtime files must not be changed underneath an active shell driver.

## Active Jobs

| Stage | Host | GPU | Container / PID | Checkout | Log | State |
|---|---|---:|---|---|---|---|
| MRP-3 MQAR primary resume | blue-demon | 0 | `7faed36a1d47` at health check | `~/set-attention-anchor-span-sync` | `logs/mrp3_mqar_resume_seed1_remaining_gpu0_20260713.log` | live seed 1 b25/b50/b75/b100 queue |
| MRP-3 MQAR primary resume | blue-demon | 1 | `fd1add2829a2` at health check | `~/set-attention-anchor-span-sync` | `logs/mrp3_mqar_resume_seed2_all_gpu1_20260713.log` | live seed 2 token/b0/b25/b50/b75/b100 queue |

## Copy Status

| Copy | Role | Status |
|---|---|---|
| local workspace `/mnt/d/userfolders/documents/github/set-attention` | editing and audit source | dirty; paper assets and status updated; no commit made |
| blue-demon `~/set-attention-anchor-span-sync` | active runtime copy | authoritative runtime for the current MRP-3 resume |
| lizmark `~/set-attention` | not active for MRP-2/MRP-3 | stale/partial source from earlier attempts; do not launch MRP work from this path |
| lizmark `~/set-attention-anchor-span-sync` | clean runtime copy | launch-ready for MRP-3 capacity preflight only after 2026-07-09 full source sync and container validation |

## Critical SHA State

Runtime Python/data files match local and blue-demon:

- `scripts/run_mqar.py`
- `src/train/mqar.py`
- `src/train/metrics_schema.py`
- `src/data/mqar.py`
- `src/data/ar_hits.py`
- `scripts/evaluate_ar_hits.py`
- `scripts/summarize_ar_hits.py`
- `scripts/run_mrp2_ar_hit_retrain.sh`

`scripts/run_mqar_matrix.sh` no longer intentionally differs for the active
MRP-3 path. The old primary shell driver exited before the 2026-07-13
idle-server audit, so the hardened local runner was synced into Blue before
the resume. The active runner fail-closes unless calibrated `MQAR_LR` and
`MQAR_MAX_UPDATES` are explicitly set, supports `MQAR_SEEDS`/`MQAR_ROWS`, and
skips rows with existing final checkpoints.

Lizmark `~/set-attention-anchor-span-sync` was created from a fresh staged
source sync on 2026-07-09. The first staging pass accidentally excluded
`src/data`; it was discarded and replaced with a corrected sync that excludes
only root-level `/data`, not package source. Container validation then passed:

```text
bash -n scripts/run_mqar_matrix.sh scripts/run_mrp2_ar_hit_retrain.sh scripts/run_mqar_calibration.sh
python -m py_compile scripts/run_mqar.py scripts/summarize_mqar.py scripts/evaluate_ar_hits.py scripts/summarize_ar_hits.py src/data/mqar.py src/data/ar_hits.py src/train/mqar.py src/train/metrics_schema.py
```

The stale Lizmark `~/set-attention` path remains deprecated because it still
mixes old partial sync state with prior container artifacts.

## Lizmark Capacity Preflight Result

The needed non-overlapping Lizmark job was the registered MRP-3 one-step
capacity preflight for `L=4096,B=4,D_kv=512`, using the frozen calibration
settings `lr=0.001` and `max_updates=12500`. It was split across both Lizmark
GPUs from the clean checkout and completed with no NaN/Inf, traceback, or OOM.

| Row | Seed | GPU | Peak train VRAM MiB | State |
|---|---:|---:|---:|---|
| token | 0 | 0 | 34239.0 | complete |
| b0 | 0 | 0 | 39922.4 | complete |
| b25 | 0 | 0 | 30342.1 | complete |
| b50 | 0 | 1 | 23539.8 | complete |
| b75 | 0 | 1 | 18919.1 | complete |
| b100 | 0 | 1 | 12880.6 | complete |

Logs:

- `logs/mrp3_capacity_L4096_B4_gpu0_20260709_090500.log`
- `logs/mrp3_capacity_L4096_B4_gpu1_20260709_090500.log`

Output root:

- `out/mqar_capacity_preflight_L4096_B4_lr0p001_u12500`

## Incidents Closed In This Audit

- Root-level stale snapshots `a41_status.txt` and `a42_status.txt` were moved
  to `audit/archive/legacy_status_snapshots/`.
- The MRP-3 matrix wrapper no longer has unsafe old LR/update defaults in the
  local editing copy.
- The MRP-3 matrix wrapper now has restart skip semantics keyed by final
  checkpoint.
- The MRP-2 retrain wrapper encodes all-fine/all-coarse rows as one positive
  group, not a zero-head companion group.

## Required Before Next Launch

1. Wait for active Blue MRP-3 resume containers to exit or explicitly stop them.
2. Do not modify Blue runtime files underneath the active resume containers.
3. Run container syntax/import checks on Blue before any new launch family.
4. Lizmark MRP work must use `~/set-attention-anchor-span-sync`; do not use
   stale `~/set-attention`.
5. Update this file and `audit/phase_sd_status.md` in the same turn as any
   launch or stop action.
