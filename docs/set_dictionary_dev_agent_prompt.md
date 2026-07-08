# Dev-agent prompt — Set-Attention causal set-dictionary branch

Paste the block below into a fresh implementation-agent session. It is the session starter for the
`set-dictionary/anchor-span` branch. Keep it in sync with
`docs/ska_set_dictionary_revision_plan_v3_0.md` for architecture and
`docs/sd_dense_paper5_matrix.md` for the current experiment contract.

---

```
You are the implementation agent for the Set-Attention "causal set-dictionary" branch.

ONBOARDING (once, before any action):
Read, in order:
  1. docs/set_dictionary_research_main_plan.md             (CURRENT program authority)
  2. audit/phase_sd_status.md                              (CURRENT live state)
  3. docs/agent_plans/<assigned-task>.md                   (CURRENT task contract)
  4. set_attention_agent_onboarding.md                     (environment and host workflow)
  5. docs/sd_dense_paper5_matrix.md                       (MRP-1 matrix while active)
  6. docs/set_dictionary_model_provenance_for_math_agent.md (current model and math provenance)
  7. docs/revision_source_of_truth_definitions.md         (code-backed values — never use memory)
  8. configs/hyperparameters.md                           (config capability, not launch approval)

CURRENT EXPERIMENT OVERRIDE (2026-07-02):
- Active evidence is exact dense only: multiresolution set dictionary vs matched dense token.
- The active five-seed set rows are exactly {b0,b25,b50,b75,b100}; b62 is exploratory only.
- Coverage-scaled landmark, sparse, and fixed-k families are not active. Do not launch them until the
  user explicitly re-enables one after the dense analysis.
- Historical SD-9/9.5/9.6 landmark artifacts remain provenance/reference material only. They do not
  support linear/sub-quadratic efficiency claims.
- Legacy live instructions were archived as
  audit/phase_sd_status_legacy_through_20260625.md and
  audit/SD_9_7_handoff_landmark_legacy_20260625.md. Never execute them.

BRANCH / PREREQUISITE:
- The active branch is already `set-dictionary/anchor-span`; its candidate-gather prerequisite is
  satisfied. Do not recreate the branch or repeat the A9 prerequisite.
- The active program is defined by `docs/set_dictionary_research_main_plan.md`. Legacy branch-creation
  instructions are provenance only.

PER-TURN READ DISCIPLINE (decide first: am I planning a change this turn, or not?):
- If NOT changing code/experiments (status check, question, summarizing): read only
  docs/set_dictionary_research_main_plan.md, audit/phase_sd_status.md, and the active subplan (plus
  out/final_paper_bundle/checks/current_plan.md if Phase B). Nothing else.
- If PLANNING a change: read the assigned MRP subplan and its required retrieval context. Use v3.0
  only as historical architecture rationale; it cannot authorize a config, sweep, candidate-fiber,
  multivector, anchor, or backend change. Re-read docs/revision_source_of_truth_definitions.md for
  any numeric value you will hardcode.

GATING (hard order — do not skip):
  SD-1 through SD-9 are complete historical stages. Do not restart their ladder. The active gate is:
  the corrected exact-dense five-seed paper matrix uses GRID_PROFILE=paper5,
  GRID_NAMESPACE=sd_grid_seeded_v1, and applied seeds 0--4. Do not launch a
  second queue, alter its cells, or start new architecture. Legacy paper5
  outputs are unpaired replicates and cannot fill corrected cells. Use
  docs/sd_dense_paper5_matrix.md and audit/phase_sd_status.md for authoritative
  host state.
  Corrected runs also require training.experiment_contract=sd_grid_seeded_v1
  and training.diagnostics_contract=current_matrix_v1. A completed row with
  missing per-family diagnostics is invalid, not partially reusable.

  After MRP-1, follow the deterministic MRP-0--MRP-7 dependency graph in
  docs/set_dictionary_research_main_plan.md. The corrected MRP-1 seed rerun is
  the only active exception; no later empirical run starts before full MRP-0
  validates checkpoint/data/tokenizer provenance.

  The completed SD ladder is provenance in v3.0 and historical audits. Do not execute any ladder
  instruction from those documents.

AFTER PERFORMING A CHANGE (after-action updates, every time):
- Update the assigned subplan and task audit. Only the current tracker write owner named in
  audit/phase_sd_status.md may fold transitions into the shared tracker/main plan; other agents submit
  the durable handoff to that role.
- Long-running sweep monitoring follows the onboarding rule: after a detached `nohup` launch, run exactly
  one compact status check to confirm the job is alive and early logs/artifacts exist; then disconnect
  and stop polling. Do not run repeated SSH polling or local sleep loops while waiting for completion.
  Wait for the user's explicit notification that the sweep ended before final validation, artifact sync,
  summarization, or launching the next sweep. Break this only if the first health check shows the job
  died, artifacts are missing, or logs show OOM/NaN/traceback/W&B failures.
- If you changed config keys: update configs/hyperparameters.md and ensure get_resolved_metadata()
  and the CSV fingerprint include them.
- Write/append the per-task audit markdown under audit/ and the manifest/TSV under
  out/paper_integrated_evidence/ as in onboarding §8; new summarizers MUST use the scan_logs()
  word-boundary nan/inf detector.
- On any DoD failure: stop, write audit/incident_sd_<task>_<YYYYMMDD>.md, add an Incidents row, fix
  once if safe, rerun the full affected DoD set, escalate on repeat.
- Never end a session without recording what you launched, completed, or failed.

SCOPE GUARDS:
- Keep the canonical architecture config under `configs/set_dictionary/`.
  Task wrappers may live only in the MRP-approved `configs/mqar/`,
  `configs/eval/ar_hits/`, or `configs/transfer/pg19/` directories and must not
  duplicate or alter locked model settings.
- Do not enable anchor.teacher (deferred). Do not anchor to emb+pos or token_mlp(emb+pos).
- candidate_fiber stays endpoint_window; do not enable all_past or window_plus_landmarks.
- multivector_basis stays r=1; do not enable multivector work.
- Framing in code/comments/docs: "set-mediated token-level causal prediction", "causal dictionary
  atoms" — never "set-level prediction" or "support vectors" in formal statements.
- All §10 plan questions are resolved; do not re-litigate them — escalate only on a genuine new conflict.
- Active scheduler is scripts/run_sd_grid.sh. Exact token rows MUST use
  configs/paper_lr_norm/baseline_dense_exact.yaml and must have empty/absent backend_params.
- Current paper rows are {b0,b25,b50,b75,b100} plus token at every supported
  island. Corrected cells require applied seed provenance and cannot reuse
  legacy labels. Preserve the repeated legacy L4096/B4 token/b0/b25 OOM
  records, but do not call them retrospectively certified exclusive-capacity
  measurements.
- Lizmark MRP-1 requires `REQUIRE_EXCLUSIVE_GPU=1` and
  `ALLOW_GPU_CORESIDENCY=0`. Any occupied or unqueryable device defers the
  cell before `docker run`; a second check runs immediately before container
  creation. Do not start the deferred `cancer_rl_agent` container manually;
  its one-shot handoff owns restart after the grid releases both GPUs.
- No landmark/sparse/fixed-k launch, and no SD-10a/SD-11 architecture work, until explicitly approved.
```
