# Incident: Configured Training Seed Was Not Applied

Status: MITIGATING. Seed application and provenance guards are implemented
locally and deployed only to the new blue-demon deterministic rerun. Full MRP-0
checkpoint/data/loader remediation remains open.

Discovered: 2026-06-30 during restart-safe program planning.

## Finding

The unified runner records `training.seed` in run names and metadata but does
not call `src/common/repro.py:set_seed()` or another RNG initialization path
before constructing datasets, models, loaders, or optimizers.

The grid launcher passes `training.seed=<value>`, but no launcher,
container-environment, or runner-level seed application was found.

## Scope

This affects current and historical runs executed through
`scripts/run_experiment.py`.

Likely behavior is that separate processes receive independent framework RNG
states. Completed rows therefore remain usable as stochastic replicates, but:

- the configured seed labels are not reproducible RNG states;
- cross-model rows with the same label are not paired;
- paired-seed tests and exact reruns are invalid;
- model initialization and loader order cannot be reconstructed from current
  artifacts.

This does not by itself show duplicated models or invalidate unpaired
per-cell means. It is a reproducibility and statistical-pairing defect.

## Verified Impact

The defect affected every completed paper5 run executed through the old
`scripts/run_experiment.py` path. The labels `0..4` did not select those RNG
states. This does not invalidate the rows as independent stochastic
replicates, but it does invalidate exact-seed and paired-seed interpretations.

The blue-demon legacy paper5 queue ended normally before remediation. The
lizmark legacy paper5 queue was still active on 2026-07-01 and was deliberately
left unmodified.

## Current Queue Policy

Do not rewrite, delete, or relabel legacy artifacts. Summarize them only as
unpaired stochastic replicates.

Corrected runs use the isolated namespace
`out/paper_mechanisms/sd_grid_seeded_v1`. The launcher scans only that
namespace, so legacy completion metadata cannot skip a corrected cell. It also
rejects completion unless the CSV records:

- `training.seed_applied=true`;
- `training.applied_seed == training.seed`;
- `training.torch_initial_seed == training.seed`;
- `training.deterministic=true`;
- `training.benchmark_mode=false`.

The corrected blue-demon paper5 matrix contains 120 cells: seeds `0..4` for
all six rows at each of `L512/B16`, `L512/B4`, `L1024/B4`, and `L2048/B3`.
Its dry run reported `PLAN=120`, `SKIP=0`, and it launched on 2026-07-01.

Do not deploy the runner hotfix to lizmark until its legacy queue exits. The
corresponding corrected lizmark matrix has 135 supported cells and remains
pending.

Do not launch MRP-2, MRP-3, or MRP-5 until MRP-0 applies and logs RNG seeds.

## Remediation

MRP-0 must:

1. apply the configured seed before all stochastic construction; **implemented**
2. log requested and applied seeds; **implemented**
3. add same-seed/different-seed regression tests; **implemented for model initialization**
4. add checkpoint and data/tokenizer provenance;
5. prove deterministic loader batches and complete the canonical platform audit.

## Resolution Criteria

The incident closes only after MRP-0 tests prove reproducible initialization
and loader batches and new-run metadata records the applied RNG state.
