# MRP-3 MQAR Mechanism Infrastructure Audit

Updated: 2026-07-07.

Status: infrastructure READY; calibration, primary, and capacity experiments
not launched.

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
- `scripts/run_mqar_matrix.sh`: approval-gated matrix wrapper. It refuses the
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

No calibration, primary, or capacity experiment was launched. The registered
matrix wrapper is fail-closed by default and requires an explicit environment
flag plus command argument.

## Known Limitations

- Resume/eval-only checkpoint paths are not implemented in `scripts/run_mqar.py`;
  the required MRP-0 final checkpoint save path is implemented for new MQAR
  training runs.
- Capacity preflights are represented by the one-step preflight mode and guarded
  matrix wrapper; the actual `L=4096,B=4` capacity observations were not run.

## Next Atomic Action

After MRP-1 closes and explicit launch approval is recorded, run the registered
calibration protocol. Do not launch calibration, primary, or capacity matrices
before that approval.
