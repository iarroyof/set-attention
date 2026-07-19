# Server Copy Provenance And Launch-Safety Audit

Date: 2026-07-09; updated 2026-07-17 after auditing GitHub push state and
Blue/Lizmark checkout safety.

## Current Rule

Do not launch new work from a server checkout unless this file or
`audit/phase_sd_status.md` says that checkout is launch-ready for the specific
stage. Runtime files must not be changed underneath an active shell driver.

GitHub is now the authoritative branch source for new syncs:

- `set-dictionary/anchor-span` at `3e7edb4`
- `mrp-lca-cmp-sd` at `d0f7ae8` (pushed 2026-07-17); hosts additionally carry
  the unpushed validation commit `2ded5d1` via verified bundle.

The earlier local branch `mrp-lca-cmp` was created from `origin/main` and is a
false-start validation branch. Do not launch from it and do not pull it to
GPU hosts for current set-dictionary work.

The 2026-07-17 host repair is complete (see
`audit/incident_branch_host_context_20260717.md`, Resolution section). Both
`~/set-attention` directories are now clean git checkouts on
`mrp-lca-cmp-sd@2ded5d1`, compose-validated (compile, LCA dry-run, batching
preservation, in-container write test). The pre-repair directories are
preserved at `~/repo_audit_copies/set-attention_pre_mrp_lca_cmp_sd_repair_20260717_175927`
on each host. Lizmark launches must set `UID`/`GID` env for docker compose
(host uid 1001 vs image default 1000).

Temporary alternate directories created during recovery, including
`~/set-attention-anchor-span-sync` and `~/set-attention-mrp0-validation`, are
deprecated audit copies. They must be kept for provenance but not used for new
launches.

## Active Jobs

| Stage | Host | GPU | Container / PID | Checkout | Log | State |
|---|---|---:|---|---|---|---|
| MRP-3 MQAR primary resume | blue-demon | 0 | `7faed36a1d47` at health check | `~/set-attention-anchor-span-sync` | `logs/mrp3_mqar_resume_seed1_remaining_gpu0_20260713.log` | live seed 1 b25/b50/b75/b100 queue |
| MRP-3 MQAR primary resume | blue-demon | 1 | `fd1add2829a2` at health check | `~/set-attention-anchor-span-sync` | `logs/mrp3_mqar_resume_seed2_all_gpu1_20260713.log` | live seed 2 token/b0/b25/b50/b75/b100 queue |
| MRP-lca-cmp calibration | blue-demon | 0 | driver exited | `~/set-attention` | `logs/lca_cmp/blue/{queue,worker_gpu0}.log` | COMPLETE 2026-07-18 19:23: 36/36 rows endpoint-valid, OOM registry header-only; Gate 2 FAIL (all set rows at chance) — see `audit/LCA_calibration_20260718.md` |
| MRP-lca-cmp calibration | blue-demon | 1 | same driver, second worker | `~/set-attention` | `logs/lca_cmp/blue/worker_gpu1.log` | COMPLETE 2026-07-18 19:23: same queue, round-robin; both GPUs idle |

## Copy Status

| Copy | Role | Status |
|---|---|---|
| local workspace `/mnt/d/userfolders/documents/github/set-attention` | editing and audit source | canonical documentation/analysis mirror |
| blue-demon `~/set-attention` | active runtime path | launch-ready 2026-07-17: clean checkout `mrp-lca-cmp-sd@2ded5d1`, compose-validated; pre-repair tree archived under `~/repo_audit_copies/` |
| lizmark `~/set-attention` | active runtime path | launch-ready 2026-07-17: clean checkout `mrp-lca-cmp-sd@2ded5d1`, compose-validated; launches require `UID`/`GID` env set; pre-repair tree archived under `~/repo_audit_copies/` |
| blue-demon `~/set-attention-anchor-span-sync` | deprecated audit copy | historical runtime copy for July MRP recovery jobs; do not launch new work here |
| lizmark `~/set-attention-anchor-span-sync` | deprecated audit copy | historical runtime copy for July MRP recovery/capacity jobs; do not launch new work here |
| blue-demon `~/set-attention-mrp0-validation` | deprecated audit copy | MRP-0 validation snapshot; do not launch new work here |

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

This was a temporary recovery path. It is now deprecated for new launches; the
original `~/set-attention` path has been restored as the active runtime copy.

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
4. New Blue/Lizmark MRP work must use the original `~/set-attention` path. Do
   not launch from `~/set-attention-anchor-span-sync` or
   `~/set-attention-mrp0-validation`.
5. Update this file and `audit/phase_sd_status.md` in the same turn as any
   launch or stop action.
