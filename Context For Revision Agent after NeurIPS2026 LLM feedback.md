# Current Revision-Agent Context

Updated: 2026-07-02.

This file is the environment overlay for the active `set-dictionary/anchor-span` branch. The previous
long Phase-A/v2.7 context is archived at
`docs/archive/legacy_context/Context_For_Revision_Agent_legacy_through_20260629.md` and is not launch
guidance.

## Read Order

For current set-dictionary work, read:

1. `docs/set_dictionary_research_main_plan.md`
2. `audit/phase_sd_status.md`
3. the assigned file under `docs/agent_plans/`
4. `docs/set_dictionary_dev_agent_prompt.md`
5. `docs/sd_dense_paper5_matrix.md` while MRP-1 is active
6. `docs/set_dictionary_model_provenance_for_math_agent.md`
7. `docs/revision_source_of_truth_definitions.md`

Use `out/final_paper_bundle/overleaf_ready/example_paper.tex` for manuscript edits. It is a manuscript
artifact, not experiment-launch authority.

## Current Research Direction

- Active evidence: exact-dense multiresolution set dictionary versus matched exact-dense token
  attention.
- Active five-seed blur matrix: `{b0,b25,b50,b75,b100}` plus token at every supported `(L,batch)`
  island.
- Corrected confirmation rows apply seeds `0..4` and live only under
  `sd_grid_seeded_v1`. Legacy labels are unpaired stochastic replicates and
  cannot fill corrected cells.
- Set guards: `output_residual_mode=anchor_span`, `token_mlp.enabled=false`,
  `anchor.enabled=false`, CE-only, `candidate_fiber=endpoint_window`, no re-read/all-past/multivector.
- Full WikiText-2 only, no `data.limit`; 10 epochs; LR `1e-4`; `D=384`, `d_ff=1536`, 6 layers,
  8 heads.
- Coverage-scaled landmark, sparse, fixed-k, Nyström, SD-10a, and SD-11 are inactive. Completed
  landmark artifacts are historical quality references only and cannot support linear or
  sub-quadratic claims.
- Never execute files under `docs/archive/` or the legacy status snapshots under `audit/`.

The active scheduler is `scripts/run_sd_grid.sh` with `GRID_PROFILE=paper5`,
`GRID_NAMESPACE=sd_grid_seeded_v1`, `REQUIRE_APPLIED_SEED=1`, and
`TRAINING_DETERMINISTIC=true`. Exact-token rows use
`configs/paper_lr_norm/baseline_dense_exact.yaml` and must not carry
`backend_params`. Corrected rows require the `sd_grid_seeded_v1` experiment
contract and `current_matrix_v1` diagnostics contract.

## Hosts

| Host | Address | Current ownership |
|---|---|---|
| blue-demon | `iarroyof@192.168.241.149` | supported short/medium exact-dense rows assigned by the matrix |
| lizmark | `iarroyof@192.168.241.205` | supported large-memory exact-dense rows assigned by the matrix |

Both host artifact trees can be authoritative for their assigned cells. The local workspace is the
canonical documentation and analysis mirror.

Authentication uses `sshpass -f ~/.ssh/.sshpass`; do not place passwords in commands, logs, or repo
files. Run project/PyTorch commands in the `set-attention:latest` Docker image with the repo mounted at
`/workspace`. Hugging Face and W&B runs remain offline.

## Job Discipline

1. Dry-run the host-specific matrix and reconcile pending cells.
2. Run only one grid driver per host.
3. Before syncing executable source/config/scripts, check the host once for
   active grid/training processes. If any are active, stop and do not sync.
4. After launch, perform one compact health check.
5. Stop polling until the user explicitly requests status.
6. The tracker write owner records every transition, PID, log, and ETA in
   `audit/phase_sd_status.md`; concurrent task agents submit subplan/audit
   handoffs instead of editing the shared tracker.
7. Reuse only corrected rows whose requested/applied/torch seeds and
   deterministic flags pass. Preserve legacy artifacts and repeated OOM
   records, but never use legacy labels as corrected seeds.
8. Lizmark MRP-1 requires `REQUIRE_EXCLUSIVE_GPU=1` and
   `ALLOW_GPU_CORESIDENCY=0`. Join every cell to
   `gpu_admission_lizmark.tsv` and reject any row whose start/end occupancy is
   not exclusive. The deferred external container is restarted only after the
   grid releases both GPUs.

Do not infer live state from prose in archived audits. Use `audit/phase_sd_status.md`.
Do not infer program state from chat history. Use the main plan and active
subplan, and write a durable handoff before stopping.
