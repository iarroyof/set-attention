# Incident: A2.0 Baseline Diagnostics Delta NaN

Date: 2026-05-09

Phase/task: A2.0 pre-launch smoke gate.

Category: logging/diagnostics finite-metadata bug.

## Failure

The first A2.0 dense baseline 10-epoch smoke completed training, but the CSV
and offline W&B summary contained non-finite baseline delta metrics:

- `baseline/delta_attention_entropy`
- `baseline/delta_attention_confidence`

The values were `nan` on every epoch, despite finite train/validation losses and
perplexities. This violates the A2.0 gate requirement that runs complete without
NaN and produce well-formed CSV/JSON metadata.

## Fix

Minimal fix: `src/models/baseline_token/diagnostics.py` now mirrors the A1.9
finite-delta policy used by set-only diagnostics. First-observation deltas are
reported as `0.0`, and later deltas fall back to `0.0` if either side is not a
finite number.

The affected baseline smoke must be rerun before the A2.0 gate can pass.
