# Incident: A1.9 Diagnostics Delta NaN

Date: 2026-05-09

Phase/task: A1.9 consolidated audit and smoke gate.

Category: cross-task interaction / surface bug.

## Failure

The first A1.9 Docker gate run failed the finite-diagnostics checks. Tiny
strict-past forward smokes produced finite logits, but `SetDiagnostics` emitted
NaN first-epoch delta metrics:

- `ausa/delta_routing_entropy`
- `ausa/delta_set_variance`
- `ausa/delta_router_confidence`

These are epoch-to-epoch deltas. On the first epoch or first forward-only smoke
there is no previous epoch baseline, so returning NaN violates the A1.9
requirement that diagnostics are finite.

## Fix

Minimal fix: keep the metric keys, but report `0.0` when no finite previous and
current value pair exists. This preserves the interpretation that the first
observed point has no measured change from a prior point, and keeps run metadata
finite.

Regression coverage was added to `tests/test_diagnostics_option1.py`.
