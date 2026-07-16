# MRP-3 MQAR Mechanism Infrastructure Audit

Updated: 2026-07-15.

Status: COMPLETE as a protocol run; scientific result is null/inconclusive
because primary MQAR accuracy stayed near chance for every row.

## Scope

Implemented the non-launched MQAR generator/trainer infrastructure for the
set-dictionary/anchor-span branch after MRP-0 PASS. No model forward code or new
architecture was edited.

## Files

- `src/data/mqar.py`: Zoology-compatible MQAR generator, deterministic
  distractors, exact query/key/lag metadata, split seed helpers, dataset
  provenance, and stable digests.
- `src/train/mqar.py`: query-only loss/accuracy helpers, fixed lag-bin metrics
  with empty-bin reporting, exact-sequence accuracy, update-based trainer,
  evaluator, and public-hook fine/coarse span-ablation evaluation.
- `scripts/run_mqar.py`: MQAR-specific runner using shared config loading,
  MRP-0 seed application, experiment logging, checkpoint payload/save APIs, and
  dry-run/preflight modes.
- `scripts/summarize_mqar.py`: strict registered-matrix summarizer that rejects
  word-boundary NaN/Inf, smoke/limited rows, malformed seeds, metadata gaps, and
  wrong-backend rows.
- `scripts/run_mqar_matrix.sh`: launch-guarded matrix wrapper. It refuses the
  registered matrix unless both `MRP3_MQAR_LAUNCH=approved` and `--launch` are
  present. `--smoke` remains CPU-local.
- `configs/mqar/`: smoke and primary templates for exact token and set rows.
- `tests/test_mqar_generator.py`, `tests/test_mqar_metrics.py`,
  `tests/test_mqar_summarizer.py`: focused CPU tests for generator invariants,
  split reproducibility, lag bins, masked query metrics, public-hook group
  ablation restoration, and summarizer rejection.

## Generator Contract

The generator follows the published HazyResearch/Zoology `multiquery_ar`
construction:

- vocabulary defaults to `8192`;
- keys and values are sampled without replacement from disjoint vocabulary
  halves;
- initial key/value pairs occupy the first `2 * D_kv` input positions;
- every key is queried once later in the sequence;
- non-query labels are `-100`;
- query labels are the associated values at the shifted next-token target
  positions;
- gap sampling uses `power_a=0.01` by default and samples without replacement;
- distractors are generated deterministically from the applied generator seed;
- exact query positions, matching key positions, lags, query keys, and query
  values are retained.

Train, calibration/validation, and test seed helpers are disjoint and offset
away from the shared MRP-0 loader seed defaults.

## Metrics And Ablations

Loss, PPL, and accuracy denominators count only non-`-100` query targets through
the shared masked-loss contract. Fixed lag bins are:

- `[1,32]`;
- `[33,128]`;
- `[129,512]`;
- `[513,1024]`;
- `[1025,2047]`.

Empty bins are emitted with count `0` and metric values `None`/`NA`, and are
therefore available for exclusion from inferential aggregation.

Named fine/coarse group ablation is implemented only through existing public
model hooks: `set_span_ablation_mode`, `multiresolution_group_metadata`, and
state restoration to the prior mode. No forward-path edits were made.

## Validation

Commands run locally:

```text
python -m py_compile src/data/mqar.py src/train/mqar.py scripts/run_mqar.py scripts/summarize_mqar.py tests/test_mqar_generator.py tests/test_mqar_metrics.py tests/test_mqar_summarizer.py
python - <<'PY' ... summarize_mqar lightweight rejection checks ... PY
bash -n scripts/run_mqar_matrix.sh
scripts/run_mqar_matrix.sh
```

Outcomes:

- Python compilation passed.
- Summarizer accepted a complete synthetic registered token row and rejected
  smoke, non-finite, and malformed-seed rows.
- Matrix shell syntax passed.
- Ungated matrix invocation refused to launch with exit code `3`.

Commands attempted but blocked by the current shell environment:

```text
pytest -q tests/test_mqar_generator.py tests/test_mqar_metrics.py tests/test_mqar_summarizer.py
python scripts/run_mqar.py --config configs/mqar/token_smoke.yaml --dry-run --device cpu
```

Blocked outcomes:

- `pytest` is not installed in the active Python environment.
- active Python is `3.13.9`; it exposes an incomplete namespace `torch` without
  `torch.utils`, so PyTorch-backed generator/runner tests cannot execute here.
- `PyYAML` is also unavailable, so config loading cannot be validated in this
  shell.

Blue container validation completed after the local environment block:

```text
python -m pytest -q \
  tests/test_mqar_generator.py \
  tests/test_mqar_metrics.py \
  tests/test_mqar_summarizer.py

7 passed in 0.66s

python scripts/run_mqar.py --config configs/mqar/token_smoke.yaml \
  --dry-run --device cpu

Dry run: MQAR config, generator, seeds, and provenance validated.
train_digest=06219eeea98010b45084d2b8bea3ed9227e70e5f69060fbc6a35fa5df69052f4
validation_digest=60535b68bba508fb0fd9af29af1eb2c75de2e4fce6d55a2bda683d463092f835
dataset_digest=6569d026e945dd2ef21197dddc91b686382e70bd622bfd6747953c86d0b81014

MQAR_DEVICE=cpu scripts/run_mqar_matrix.sh --smoke

[task] detected mqar (cfg.task)
updates=1 train_loss=9.4373 val_loss=9.3868 val_acc=0.0000
```

The matrix launch guard was also validated in the container: an ungated
`scripts/run_mqar_matrix.sh` invocation refused to launch and exited with code
`3`.

## Launch Guard

The registered matrix wrapper is fail-closed by default and requires an
explicit environment flag plus command argument. After launch, the local
editing copy was further hardened to require explicit calibrated `MQAR_LR` and
`MQAR_MAX_UPDATES`; do not sync that wrapper over Blue's active primary shell
driver until it exits.

## Known Limitations

- Resume/eval-only checkpoint paths are not implemented in `scripts/run_mqar.py`;
  the required MRP-0 final checkpoint save path is implemented for new MQAR
  training runs.
- Capacity preflights are represented by the one-step preflight mode and guarded
  matrix wrapper. The registered `L=4096,B=4` capacity observations were run on
  lizmark on 2026-07-09 and are feasibility/VRAM observations only, not trained
  quality results.

## Approval And Calibration Relaunch State

User approval was clarified on 2026-07-08. The first registered token
calibration LR launch on blue-demon was intentionally stopped because the
existing runner evaluated only once at the full-update endpoint. That trace is
invalid for selecting the registered first LR/update satisfying two consecutive
`0.99` calibration evaluations; see
`audit/incident_mrp3_calibration_eval_cadence_20260708.md`.

The cadence fix is instrumentation/control only:

- `src/train/mqar.py` exposes finite update-block training.
- `scripts/run_mqar.py` logs calibration evaluations every
  `training.eval_every_updates`.
- `scripts/run_mqar_calibration.sh` pins the registered cadence and gate:
  `eval_every_updates=500`, `calibration_accuracy_threshold=0.99`,
  `calibration_consecutive_evals=2`.

Reduced-cadence Blue container validation passed with `max_updates=2` and
`eval_every_updates=1`: each token LR candidate emitted two update-indexed CSV
rows containing `train/completed_updates`,
`val/calibration_consecutive_hits`, `val/calibration_gate_passed`, and
`val/calibration_selected_update`.

The registered token LR calibration sweep selected `lr=0.001` at update
`12500`: `lr=1e-4` failed by update `20000`, `lr=3e-4` passed at update
`14000`, and `lr=1e-3` passed first at update `12500`.

B4 common-batch preflight passed on blue-demon GPU0 for all six rows at seed 0
with one update and metric logging:

- token peak train VRAM: `9365.9` MiB;
- b0 peak train VRAM: `10951.7` MiB;
- b25 peak train VRAM: `8826.3` MiB;
- b50 peak train VRAM: `7396.5` MiB;
- b75 peak train VRAM: `6050.4` MiB;
- b100 peak train VRAM: `3969.1` MiB.

The initial B4 preflight wrapper was over-broad and began seed 1 after the
registered six-row seed-0 gate had passed; those preflight containers were
stopped, and `scripts/run_mqar_matrix.sh` now accepts `MQAR_SEEDS`.

Historical primary launch:

- host: blue-demon;
- checkout: `~/set-attention-anchor-span-sync`;
- GPU: `0`;
- container: `f60c262402a5`;
- docker PID: `2934320`;
- python PID: `2934464`;
- log: `logs/mrp3_mqar_primary_B4_lr0p001_u12500_20260709_083600.log`;
- first active row: token seed `0`;
- matrix: token, b0, b25, b50, b75, b100; seeds `0,1,2`;
- frozen training: `B=4`, `lr=0.001`, `max_updates=12500`.

Post-launch hardening note: the local editing copy of
`scripts/run_mqar_matrix.sh` was hardened after this primary shell driver
started. It now fail-closes without explicit calibrated `MQAR_LR` and
`MQAR_MAX_UPDATES`, supports `MQAR_SEEDS`, and skips completed checkpoint rows
on restart. The interrupted primary driver is no longer running; the hardened
copy is synced to Blue and was used for the 2026-07-13 resume. See
`audit/server_copy_provenance_20260709.md`.

2026-07-13 resume launch:

| host | GPU | container at health check | rows | log | state |
|---|---:|---|---|---|---|
| blue-demon | 0 | `7faed36a1d47` | seed 1: b25, b50, b75, b100 | `logs/mrp3_mqar_resume_seed1_remaining_gpu0_20260713.log` | completed on 2026-07-14 |
| blue-demon | 1 | `fd1add2829a2` | seed 2: token, b0, b25, b50, b75, b100 | `logs/mrp3_mqar_resume_seed2_all_gpu1_20260713.log` | completed on 2026-07-15 |

## Primary Matrix Result

On 2026-07-15 both blue-demon GPUs were idle and no `run_mqar.py`,
`run_experiment.py`, `sdgrid`, or set-attention training process remained. The
18 registered primary endpoint CSVs were pulled locally from
`~/set-attention-anchor-span-sync/out/mqar_primary_B4_lr0p001_u12500`.

Validation:

```text
python scripts/summarize_mqar.py out/mqar_primary_B4_lr0p001_u12500/*.csv \
  --out out/mqar_primary_B4_lr0p001_u12500/summary.tsv
```

accepted all 18 registered rows. A word-boundary PCRE2 scan over the original
primary log and both resume logs found no NaN, Inf, traceback, OOM, error,
failed, or permission-denied marker. The logs contain nonfatal runtime
warnings about near-hard routing, set Gram spectrum overconstraint, and one
pooling-collapse warning; these are diagnostic observations, not launch
failures.

Summary over three applied seeds:

| Row | Accuracy mean | Accuracy sd | Loss mean | Loss sd | Exact-sequence accuracy | Query count |
|---|---:|---:|---:|---:|---:|---:|
| token | 0.0002626 | 0.0000173 | 8.3424 | 0.0020 | 0.0 | 2304000 |
| b0 | 0.0002470 | 0.0000172 | 8.5166 | 0.0010 | 0.0 | 2304000 |
| b25 | 0.0002474 | 0.0000079 | 8.4891 | 0.0018 | 0.0 | 2304000 |
| b50 | 0.0002452 | 0.0000176 | 8.4958 | 0.0091 | 0.0 | 2304000 |
| b75 | 0.0002509 | 0.0000158 | 8.5118 | 0.0031 | 0.0 | 2304000 |
| b100 | 0.0002630 | 0.0000179 | 8.5226 | 0.0004 | 0.0 | 2304000 |

The registered mechanism gate cannot be evaluated as positive evidence:
support condition 3 requires unablated frozen-row `b25` test accuracy at least
0.90, but the observed `b25` mean is 0.0002474. Therefore MRP-4 is
`NOT_TRIGGERED` by the deterministic rule because the primary task is
inadequate for scale-separation interpretation. Do not use these rows as
evidence for fine/coarse specialization; use them only as a completed null
mechanism probe and as evidence that this MQAR setup did not train to the
support regime under the frozen protocol.

This null interpretation is not a state-of-the-art performance requirement.
These are small prototype models, so absolute MQAR accuracy is expected to be
far below specialized large-model or task-tuned results. The reason the gate
fails is narrower: the registered ablation/lag mechanism test requires the
model to solve enough of the synthetic task that group-removal effects can be
interpreted as recall specialization. Descriptively, the relative comparison is
still useful:

- token has the best mean loss (`8.3424`);
- b25 has the best set-row mean loss (`8.4891`) and lower peak VRAM than token
  and b0;
- b100 has accuracy numerically tied with token but the worst set loss
  (`8.5226`), so its accuracy should not be read as better task modeling;
- b25 improves loss over b0 and b100, matching the multiresolution frontier
  pattern qualitatively, but it does not beat token on this MQAR setup.

Thus MRP-3 contributes a descriptive prototype comparison, not a positive
mechanism attribution.

## L4096/B4 Capacity Preflight

Lizmark was repaired by creating a clean dedicated runtime checkout at
`~/set-attention-anchor-span-sync`, replacing an incorrect first staging pass
that had excluded `src/data`. Container syntax/import validation passed before
launch. The old Lizmark `~/set-attention` checkout remains deprecated and was
not used.

The registered one-step capacity preflight completed on 2026-07-09 under
`out/mqar_capacity_preflight_L4096_B4_lr0p001_u12500`. It used
`L=4096,B=4,D_kv=512`, seed `0`, `lr=0.001`, and the frozen `12500`-update
primary budget metadata. The row split used both Lizmark GPUs and completed
without NaN/Inf, traceback, or OOM.

| Row | Peak train VRAM MiB |
|---|---:|
| token | 34239.0 |
| b0 | 39922.4 |
| b25 | 30342.1 |
| b50 | 23539.8 |
| b75 | 18919.1 |
| b100 | 12880.6 |

Capacity interpretation is limited to feasibility and memory scaling. These
rows used tiny train/validation counts for one update and must not be treated
as MQAR quality evidence.
