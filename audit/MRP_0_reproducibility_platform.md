# MRP-0 Reproducibility And Evaluation Platform

Status: PASS

Updated: 2026-07-07.

## Implemented

- atomic checkpoint schema with model/config/epoch/global-step provenance;
- optimizer, RNG, and DataLoader-generator state for resume;
- strict model/data/tokenizer digest checks on load;
- eval-only checkpoint loading before optimizer construction;
- checkpoint SHA-256 sidecars and manifests;
- ordered WikiText token-stream, record-offset, sample-offset, and vocabulary
  digests;
- explicit dataset/train-loader/validation-loader seeds;
- masked `-100` LM loss, accuracy, and token denominators;
- preregistered additional logger metric columns;
- MQAR task normalization as strict-past causal prediction;
- strict deterministic mode with fail-closed PyTorch algorithms,
  CuBLAS workspace validation, and TF32 disabled;
- source-commit and runtime deterministic provenance.

## Added Tests

- `tests/test_mrp0_checkpoints.py`;
- `tests/test_mrp0_ordered_data.py`;
- `tests/test_mrp0_masked_metrics.py`;
- `tests/test_mrp0_determinism.py`;
- `tests/test_mrp0_config_contract.py`;
- `scripts/verify_checkpoint_replay.py`;
- `scripts/validate_mrp0_platform.sh`.

Local checks pass:

- `bash -n scripts/validate_mrp0_platform.sh`;
- `python3 -m py_compile` on the MRP-0 runner, checkpoint, data, metric, and
  focused test files;
- `git diff --check` on the same implementation, validation, and audit files;
- dependency-free ordered-provenance smoke
  (`dataset_digest=fc4c7a43727e640d964285ee0fb2d07beffba5e170882a80f2881851a7248281`).

The local environment does not contain functional PyTorch, PyYAML, pytest, or
Docker, so project validation was run in an isolated Blue-demon container
checkout rather than the local shell.

## Container Evidence

`scripts/validate_mrp0_platform.sh` passed in the Blue-demon project container
on 2026-07-07 using isolated checkout
`~/set-attention-mrp0-validation`, image `set-attention-dev:cu124`, and
offline WikiText cache mounted from `~/set-attention/.hf`.

Command shape:

```text
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -e REPO_ROOT=/workspace \
  -e OUT_ROOT=/workspace/out/mrp0_validation \
  -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -v $HOME/set-attention-mrp0-validation:/workspace \
  -v $HOME/set-attention/.hf:/tmp/.hf \
  set-attention-dev:cu124 \
  bash -lc 'python -m pip install -q --target /tmp/pytest_pkgs pytest && \
            bash /workspace/scripts/validate_mrp0_platform.sh'
```

Observed outcome:

```text
34 passed, 15 warnings
token duplicate smoke: train_loss=6.3323, val_loss=6.1910 twice
b25 duplicate smoke: train_loss=6.4187, val_loss=6.0845 twice
token strict replay: cross_checkpoint_tensors_exact=true, same_checkpoint_logits_exact=true
b25 strict replay: cross_checkpoint_tensors_exact=true, same_checkpoint_logits_exact=true
eval_only checkpoint load: val_loss=6.1910, no checkpoint mutation
MRP-0 validation PASS: /workspace/out/mrp0_validation/20260707_171240
```

Pulled local artifacts:
`out/mrp0_validation_blue_20260707/20260707_171240/`.

This evidence satisfies:

1. pass all focused and existing grid/config/diagnostic tests;
2. train identical strict token smokes twice and compare final tensors/logits;
3. train identical strict b25 smokes twice and compare final tensors/logits;
4. perform eval-only loading without checkpoint mutation or optimizer output;
5. emit stable dataset/tokenizer/checkpoint digests;
6. contain no nondeterminism, NaN/Inf, traceback, OOM, or W&B-step failure.

Before launching the 12 selected R1 cells, still run one-step full-shape
preflights for those selected configs.

Selected matrix:
`docs/mrp_reproducible_selected_matrix.md`.
