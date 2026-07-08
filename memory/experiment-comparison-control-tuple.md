---
name: experiment-comparison-control-tuple
description: Never compare runs that differ in batch/LR/backend/etc.; fix a control tuple, vary ONE factor. Check this before every launch, summary, or claim.
metadata:
  type: feedback
---

Before launching, aggregating, or claiming ANY result, fix a **control tuple** and ensure the runs being
compared differ in EXACTLY the one factor under study. Control tuple =
`{dataset, D/layers/heads/d_ff, backend, landmark_coverage_or_k, batch_size, lr, warmup_steps, epochs,
data.limit, output_residual_mode, token_mlp, candidate_fiber, hardware_class}`.
Two runs are comparable iff they match on the tuple except the studied axis
(blur, or L, or coverage). Plotting a curve across L while batch/LR vary is a
confounded curve.

**Why:** ~4 days of compute (2026-06-22..25) were partly wasted because two crucial facts were not held in
immediate memory and applied to every decision: (1) `landmark@coverage` is NOT sub-quadratic — it is
`0.25*M^2` and the content-bias adapter allocates full `[B,H,M,M]` (see [[set-attention-research-direction]]);
(2) batch size varied silently (batch16 dense L=512 vs batch1 landmark L>=2048) at a fixed nominal LR=1e-4,
so effective LR / gradient noise changed across the sweep. Result: cross-stratum PPL comparisons (e.g.
gluing batch16 L=512 onto the batch1 long-context curve) were misleading. **Within-stratum** gains
(mixed-vs-fine at the SAME L, batch, backend, lr) remain valid.

**How to apply:**
- Stratify all runs by the control tuple; make comparisons ONLY within a stratum. batch16 and batch1 are
  SEPARATE strata — never on one axis. Same for dense vs landmark, coverage vs fixed-k.
- A varying batch across L is itself a confound (tokens/step = batch*L). The current trainer does
  **not** implement gradient accumulation: `training.grad_accum_steps` may appear in metadata/configs,
  but `src/train/loop.py` steps once per microbatch. Do not use that field as a control until the
  trainer implementation and parity tests exist.
- Re-using mixed-batch history: keep each stratum as its own panel; if a cross-stratum axis is wanted, add a
  small "bridge" run in the target stratum rather than re-running everything. A finding that reproduces
  across strata (e.g. mixed>fine at both batch16 and batch1) is a ROBUSTNESS asset, not waste.
- Encode the control tuple into the run cell-id (batch is now in the sd_grid cell-id) so iso-config runs
  aggregate and cross-config runs never silently collide.
- Reviewer rule: never write "linear/sub-quadratic" for landmark@coverage; never compare two PPLs whose
  control tuples differ. State the tuple in every results table.
- Current branch rule: compare only exact-dense set/token rows from the regular blur matrix
  `{b0,b25,b50,b75,b100}` and keep B3/B4 as separate islands.
- Record per-cell GPU occupancy separately from the model control tuple.
  Current corrected Lizmark rows require exclusive occupancy at start and end;
  co-resident attempts are quarantined and cannot enter any paper analysis.
  An OOM is a capacity observation only when admission telemetry proves
  exclusivity; otherwise it is a retry, not evidence.
