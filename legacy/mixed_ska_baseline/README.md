# Deprecated Mixed SKA/Baseline Implementation

This directory contains the original hybrid implementation that mixed
baseline token attention with SKA set attention. It is preserved for:

1. Reproducing results from experiments run before migration.
2. Reference during the set-only rewrite.
3. Verifying equivalence against the new implementation.

## DO NOT USE FOR NEW EXPERIMENTS

Use `src/models/set_only/` instead.

## To run (reproduction only)

```bash
cd legacy/mixed_ska_baseline
python scripts/train_toy_lm_banked.py --help
```
