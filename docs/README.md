# Documentation Index

Updated: 2026-06-30.

## Active Set-Dictionary Documents

Read in this order:

1. `set_dictionary_research_main_plan.md` -- canonical research program,
   dependencies, memory tiers, and agent ownership.
2. `../audit/phase_sd_status.md` -- current operational state.
3. The active task under `agent_plans/`.
4. `sd_dense_paper5_matrix.md` -- current MRP-1 cells while its queues run.
5. `set_dictionary_dev_agent_prompt.md` -- standing implementation guards.
6. `set_dictionary_model_provenance_for_math_agent.md` -- code-grounded current
   model for formal and manuscript work.
7. `revision_source_of_truth_definitions.md` -- code-backed definitions.

The active research direction is exact-dense multiresolution set dictionary versus matched
exact-dense token attention. Current blur rows are `{b0,b25,b50,b75,b100}`. Backend capabilities
described in config documentation are not launch approval.

`agent_plans/README.md` defines subplan status, ownership, and durable handoff
rules. The current program includes matrix closure, reproducibility platform,
natural AR-hit evaluation, synthetic MQAR, PG-19 transfer, and a four-part
theory/proof integration track.

## Historical Documents

- `ska_pat_feedback_revision_plan_v2_6_locked.md` is the original Phase-A/v2.7 lock and is historical
  for current branch execution.
- `archive/brainstorms/` contains superseded generated plans and untitled drafts.
- `archive/deferred/` contains unapproved future designs such as fixed-k landmark work.
- `archive/legacy_context/` contains replaced environment/Phase-A context.
- `example_paper_*.tex` and `icml2026_notation_cleanup.tex` are old manuscript working files. The
  canonical manuscript is `../out/final_paper_bundle/overleaf_ready/example_paper.tex`.
- `SKA Vs Baseline VRAM allocation.pdf` is a historical diagnostic artifact, not current matrix
  evidence.

Never execute a launch instruction from `archive/`. Historical backend references remain useful for
provenance but cannot override the current matrix or tracker.
