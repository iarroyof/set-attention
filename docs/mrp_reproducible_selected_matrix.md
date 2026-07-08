# Selected Reproducible Retraining Matrix

Status: REGISTERED; LAUNCH BLOCKED UNTIL MRP-0 CONTAINER PASS

Updated: 2026-07-06.

## Purpose

Do not rerun the full 255-cell MRP-1 grid. Preserve it as the five-seed
empirical matrix, with its documented exact-replay qualification.

Retrain only cells that directly support a remaining scientific target, using
strict deterministic execution, ordered-data/tokenizer digests, explicit
loader seeds, atomic final checkpoints, and exact checkpoint replay tests.

## Validation Ladder

No full-data selected cell may launch until all stages pass:

1. CPU unit tests:
   - checkpoint round-trip and immutability;
   - model/data/tokenizer mismatch rejection;
   - optimizer/RNG/loader resume;
   - ordered offsets and stable digests;
   - identical first two loader batches;
   - hand-computed masked `-100` loss and accuracy;
   - strict deterministic configuration;
   - registered metric-column retention.
2. Blue container smoke:
   - exact token trained twice with the same seed at `L=32`, limited data;
   - exact equality of final state tensors and fixed-input CPU logits;
   - b25 trained twice with the same seed at `L=32`, limited data;
   - exact equality of b25 final state tensors and fixed-input CPU logits;
   - eval-only load creates no checkpoint and leaves the source checkpoint
     SHA-256 unchanged;
   - token and b25 config/diagnostic regression tests.
3. Full-data preflight:
   - one forward/backward/update plus checkpoint write/read for token, b0,
     b25, and b100 at `L=2048,B=4`;
   - no CuBLAS nondeterminism warning;
   - identical dataset/tokenizer digests for all rows.

Any failure stops the launch and creates an incident. A smoke row is never
used as scientific evidence.

## R1: MRP-2 Checkpoint Matrix

Host: blue-demon after MRP-0 PASS.

Exact dense, full WikiText-2, `L=2048,B=4`, 10 epochs, seeds `0,1,2`:

| Row | Heads | Runs | Scientific use |
|---|---|---:|---|
| token | token attention | 3 | matched token control and MRP-2 checkpoint |
| b0 | 8 fine | 3 | uniform-fine endpoint |
| b25 | 6 fine + 2 coarse | 3 | frozen `b*` and primary MRP-2 model |
| b100 | 8 coarse | 3 | uniform-coarse endpoint |

Total: 12 runs. This is exactly the registered MRP-2 retraining path, so no
separate pre-MRP-2 WikiText retraining is added.

## R2: Long-Scale Reproducibility Certificate

Host: lizmark only after current MRP-1 and diagnostic replacements close.

Exact dense, full WikiText-2, `L=3584,B=3`, 10 epochs, seeds `0,1,2`:

| Row | Runs | Scientific use |
|---|---:|---|
| token | 3 | matched long-scale token control |
| b0 | 3 | all-fine endpoint |
| b25 | 3 | selected multiresolution point |
| b100 | 3 | all-coarse endpoint |

Total: 12 runs. Existing b50/b75 five-seed rows retain Pareto-shape evidence;
they are not retrained because they are neither the selected point nor a
uniform endpoint.

## Remaining MRPs

- MRP-3 MQAR trains its registered synthetic rows directly through the
  validated MRP-0 APIs; no additional WikiText checkpoint row is needed.
- MRP-5 uses the same checkpoint/digest/metric contract on its registered
  tokenizer-matched datasets.
- Existing MRP-1 rows are never pooled with selected reproducibility rows to
  manufacture a larger seed count. Report them as separate five-seed matrix
  and reproducibility-certified sensitivity packages.
