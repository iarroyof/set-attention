# MRP-0: Reproducibility And Evaluation Platform

Status: PASS

Owner: UNASSIGNED

Updated: 2026-07-07 after Blue container validation.

## Mission

Provide one tested checkpoint, RNG, masked-metric, and artifact-provenance
contract for MRP-2, MRP-3, and MRP-5 without changing model architecture.

Do not deploy broad MRP-0 changes into the active Lizmark MRP-1 queue.
Corrected `sd_grid_seeded_v1` rows apply their configured seeds and satisfy a
specialized MRP-1 contract, but they do not satisfy this broader checkpoint,
digest, loader, masked-metric, or exact-replay contract.

## Required Retrieval Context

1. `../set_dictionary_research_main_plan.md`
2. `../../audit/phase_sd_status.md`
3. `../../scripts/run_experiment.py`
4. `../../src/common/repro.py`
5. `../../src/train/loop.py`
6. `../../src/train/experiment_logger.py`
7. `../../src/train/metrics_schema.py`
8. `../../src/config/normalize.py`
9. `../../src/config/compatibility.py`

## Write Scope

- `scripts/run_experiment.py`
- `src/train/checkpoints.py` (new)
- `src/data/ordered_text.py` (new shared ordered-token interface)
- `src/train/loop.py`
- `src/train/experiment_logger.py`
- `src/train/metrics_schema.py`
- `src/config/normalize.py`
- `src/config/schema.py`
- `src/config/compatibility.py`
- focused tests under `tests/`
- `configs/hyperparameters.md`
- `audit/MRP_0_reproducibility_platform.md`
- the seed incident file

Do not modify set/model forward implementations.

## Current Implementation State

The original seed defect is mitigated for corrected MRP-1 rows: the runner
calls `set_seed()` before data/model construction and logs requested,
applied, and Torch seeds. Legacy rows remain unseeded replicates.

The missing platform is now implemented locally:
`src/train/checkpoints.py`, `src/data/ordered_text.py`, checkpoint/eval-only
configuration, digest provenance, loader-generator tests, masked LM metrics,
registered metric columns, and strict deterministic mode are present.

Historical MRP-1 used non-fail-closed deterministic requests. New
reproducibility-certified configs use `training.strict_deterministic=true`,
validate `CUBLAS_WORKSPACE_CONFIG`, disable TF32, and call deterministic
algorithms with `warn_only=false`.

MRP-0 passed focused suite, duplicate strict smoke, checkpoint replay, and
eval-only immutability validation in the Blue project container on 2026-07-07.
One-step full-shape preflights are still required immediately before selected
R1 retraining launches.

Incidents:

- `audit/incident_training_seed_not_applied_20260630.md`;
- `audit/incident_mrp0_prelaunch_gap_20260706.md`.

## Implementation Contract

1. Call `set_seed()` once after config resolution and before dataset, model,
   sampler, or optimizer construction.
2. Log:
   - requested seed;
   - `torch.initial_seed()` after application;
   - deterministic/benchmark flags;
   - dataset-generator and loader-generator seeds.
3. Introduce a checkpoint schema containing:
   - model state;
   - resolved config and fingerprint;
   - epoch/global step;
   - requested/applied RNG seed;
   - dataset fingerprint;
   - ordered vocabulary or tokenizer artifact and SHA-256 digest;
   - optimizer/scheduler state for resumable training checkpoints;
   - source commit identifier when available.
4. Save final and explicitly requested intermediate checkpoints atomically.
5. Add eval-only loading with `map_location`, strict state-dict loading, config
   compatibility checks, and tokenizer/dataset digest rejection.
6. Make metric accumulation count only non-ignored targets. `-100` positions
   must contribute neither loss denominator nor accuracy denominator.
7. Permit registered task-specific metric columns without silently dropping
   them from CSV/JSON outputs.
8. Mark MQAR as causal next-token prediction in normalization and
   compatibility logic.
9. Provide a generic ordered-token source interface with sample/document
   offsets, persisted vocabulary/tokenizer metadata, and stable dataset digest.
   Implement the WikiText-2 adapter here so MRP-2 and MRP-5 do not both modify
   `src/data/wikitext2.py`.

## Tests

All must pass:

1. Same config and seed produce identical initial parameter tensors.
2. Different seeds change at least one initial parameter tensor.
3. Same seed produces identical first two loader batches.
4. Checkpoint save/load reproduces logits exactly on a fixed CPU input.
5. A tokenizer or dataset digest mismatch fails closed.
6. Eval-only mode constructs no optimizer and mutates no checkpoint.
7. Masked loss and accuracy equal a hand-computed example with `-100` labels.
8. Existing WikiText-2 exact token and set smoke paths still run.
9. Config metadata and output fingerprints contain all provenance fields.
10. The ordered WikiText-2 adapter preserves token order and emits stable
    offsets/digests across repeated loads.

Run project/PyTorch tests in the project container. Record exact commands and
outputs in the audit.

## Definition Of Done

- all tests pass;
- the incident impact on MRP-1 is documented;
- MRP-2/3/5 have one stable checkpoint/evaluation interface;
- config documentation is updated;
- no executable source is changed while that host has active training;
- the tracker owner accepts the handoff and records MRP-0 `PASS` in
  `audit/phase_sd_status.md`.

## Durable Handoff

Status: PASS.

Last completed action: validated the checkpoint, eval-only, ordered
provenance, digest, loader-seed, masked-metric, registered-metric, MQAR task,
and strict deterministic paths in an isolated Blue container checkout.

Files changed: runner/config/logger/metric/data/reproducibility code, new
checkpoint and ordered-text modules, focused tests, validation scripts,
config documentation, this subplan, and MRP-0 audits.

Commands/tests and outcomes: `scripts/validate_mrp0_platform.sh` passed in
`set-attention-dev:cu124` on Blue-demon after mounting the offline WikiText
cache from `~/set-attention/.hf`. The script reported `34 passed`, duplicate
strict token and b25 smokes with identical train/validation losses, exact
cross-checkpoint tensor/logit replay for both families, and eval-only loading
without checkpoint mutation.

Artifacts and digests: pulled to
`out/mrp0_validation_blue_20260707/20260707_171240/`. Replay summaries:
`token_strict_replay.json` and `b25_strict_replay.json` both report
`cross_checkpoint_tensors_exact=true` and `same_checkpoint_logits_exact=true`.

Host/PID/log/ETA: Blue MRP-1 ended and is idle. Lizmark MRP-1 first pass also
ended; 15 `L3584,B4` mixed replacement rows remain pending. Broad MRP-0
source should still not be deployed into any replacement launch tree until
MRP-1 replacement handling is explicitly coordinated.

Decision or gate result: MRP-0 no longer blocks MRP-2/3/5 infrastructure.
Those workstreams remain subject to their separate MRP-1 closure and explicit
launch-approval gates.

Known incident or limitation: legacy labels were not applied RNG states and
corrected MRP-1 checkpoints do not exist. Historical CUDA replay is not
guaranteed. The MRP-0 platform validates new checkpointed runs only.

Next atomic action: run one-step full-shape preflights for any selected R1
configs immediately before launching them. Do not deploy broad MRP-0 source
into the active Lizmark replacement process tree.

Inputs required: current runner/config/logger/data code and an isolated local
container test environment.
