# MRP-4: MQAR Scale-Separation Sensitivity

Status: NOT_TRIGGERED by completed MRP-3 support gate

Owner: UNASSIGNED

Updated: 2026-07-15 after MRP-3 primary MQAR summary.

## Mission

Determine whether an inconclusive MRP-3 group-by-lag interaction is caused by
the modest resolution separation in the primary `(2,1)` fine / `(4,2)` coarse
model.

This is an appendix-only architecture sensitivity test. It cannot be used as
evidence that the primary LM-matched model learned specialization.

## Deterministic Trigger

Execute this subplan if and only if MRP-3 has adequate short/long query support
and unablated b* accuracy, but the 95% interaction CI is not strictly above
zero. If primary specialization passes, or if support/accuracy is inadequate,
set this subplan to `NOT_TRIGGERED`, record the reason, and run nothing.

The trigger does not authorize a launch. After it fires, obtain separate
explicit approval recorded by the tracker write owner before running the three
sensitivity replicates.

## Required Retrieval Context

1. `../set_dictionary_research_main_plan.md`
2. `mrp3_mqar_mechanism.md`
3. `../../audit/MRP_3_mqar_mechanism.md`
4. `../set_dictionary_model_provenance_for_math_agent.md`

## Write Scope

- sensitivity configs under `configs/mqar/scale_separation/`
- the existing MQAR launcher/summarizer only where generic row support is
  required
- focused config/metadata tests
- `audit/MRP_4_scale_separation.md`

## Registered Change

Keep the fine stream `(w,s)=(2,1)`. Replace only the coarse stream by
`(w,s)=(16,8)`, preserving coarse overlap ratio `w/s=2`. Use frozen `b*`,
the MRP-3 model shape, generator, LR, optimizer-update budget, applied seeds,
and common quality-island batch.

Run exactly:

- primary LM-matched b* `(2,1)+(4,2)`, reusing MRP-3 artifacts;
- sensitivity b* `(2,1)+(16,8)`, seeds `0,1,2`.

No token, endpoint, alternative head split, or natural-language run is added.

## Metrics And Gate

Compute the same lag-bin, group-ablation, routing, accuracy, and VRAM metrics as
MRP-3. Recompute interaction `I` with the same bootstrap.

The sensitivity hypothesis is supported only if:

1. the sensitivity model has `I>0` with 95% CI above zero; and
2. its mean `I` exceeds the primary model mean `I`; and
3. unablated test accuracy remains at least `0.90`.

Regardless of outcome, stop after these three runs. Do not introduce another
resolution.

## Definition Of Done

The trigger is recorded, all sensitivity artifacts validate, the fixed gate is
applied, and the audit states either that larger separation sharpens the
interaction or that scale separation did not explain the null result.

## Durable Handoff

Status: NOT_TRIGGERED.

Last completed action: deterministic trigger, sole topology change, and
sensitivity gate registered. The completed MRP-3 primary MQAR matrix failed
support condition 3: frozen `b25` mean accuracy was `0.0002474`, far below the
required `0.90`. Therefore MRP-4 is not triggered because the primary task is
inadequate for scale-separation interpretation.

Files changed: this subplan only during registration.

Commands/tests and outcomes: none.

Artifacts and digests: none.

Host/PID/log/ETA: none.

Decision or gate result: not triggered by the completed MRP-3 support gate.

Known incident or limitation: this result can support only an appendix
scale-separation sensitivity statement.

Next atomic action: none. Do not launch the MRP-4 scale-separation rows from
the current MQAR result.

Inputs required: none unless a future approved MQAR protocol replaces the null
MRP-3 result and explicitly reopens this gate.
