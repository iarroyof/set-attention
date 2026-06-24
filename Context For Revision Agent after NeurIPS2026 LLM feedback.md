**Context For Revision Agent**

Locked revision plan:
- Use `docs/ska_pat_feedback_revision_plan_v2_6_locked.md` as the authoritative PAT feedback revision plan. It supersedes pasted v2.4/v2.5 plan text and all older planning notes for this revision cycle. As of 2026-05-13 it contains the v2.7 baseline-control amendment.
- The locked architectural choices are: Option-1 endpoint-based strict-past routing, R1 direct embedding residual, T1 drop partial trailing windows, landmark-only linear backend, `landmark_coverage=0.25`, no pydantic assumption, and inference cache claims framed as proposed design unless implemented.
- The v2.7 amendment restores matched token-backend baseline controls as required final evidence: `baseline_sparse_local_band` and `baseline_linear_landmark` must be run before final B5/A5 backend-family comparisons. Existing completed A2/A3/A4 set-family artifacts remain valid but are incomplete for backend attribution until those controls are added.
- New chats should open, in order:
  1. `docs/ska_pat_feedback_revision_plan_v2_6_locked.md`;
  2. `docs/revision_source_of_truth_definitions.md`;
  3. this context file;
  4. `audit/phase_a_status.md` -- **read this before doing any Phase A work**; it is the authoritative task/run status tracker for all of Phase A;
  5. `out/final_paper_bundle/overleaf_ready/example_paper.tex` -- **current NeurIPS paper source**;
  6. `configs/hyperparameters.md`.
- Do **not** use `docs/example_paper_working_agent.tex`, `docs/example_paper_from_zip.tex`, or `docs/example_paper_patched.tex` as current paper sources. Those are stale ICML-template drafts kept for history only. Paper updates should target `out/final_paper_bundle/overleaf_ready/example_paper.tex` and compile through `scripts/compile_paper_bundle.sh`, unless the user explicitly directs otherwise.

Definition source of truth:
- Use `docs/revision_source_of_truth_definitions.md` for code-backed definitions and values for hashed-count features, geometry/content-bias notation, `d_phi`, pooling `alpha` / temperature defaults, router heads, candidate counts, `Lambda`/backend selection, and router top-1 metrics. Do not fill those gaps from prose memory.

Active experimental evidence policy:
- From 2026-06-11 onward, reviewer-facing experimental history for this revision is the post-A1 causal LM record only: strict-past routing, T1 dropped trailing windows, explicit residual policy, and validated manifests under `out/paper_integrated_evidence/checks/`.
- Pre-A1, noncausal, and causality-unverified artifacts are legacy/internal audit material only. Do not mix them into active tables, figures, claims, or summaries unless they are explicitly rebuilt and revalidated under the post-A1 causal LM protocol.
- Common legacy locations include `out/paper_bundle/`, `out/paper_complements_bundle/`, `out/paper_complements/`, older `out/metrics/paper_action*` artifacts, and dense-only mechanism plots generated before the A1 strict-past correction.
- Do not use the old typo for causal status in docs, paper text, audit notes, captions, or scripts. Use "post-A1 causal LM", "causal LM", or "strict-past causal LM" as appropriate.
- The noncausal implementation mode may remain available for bidirectional/non-AR research, but it is not part of the active autoregressive LM evidence record.
- VRAM reporting rule: report raw `train/peak_vram_mib` unless a dedicated audit identifies a non-architectural GPU allocation with an exact per-row correction formula and provenance. If such an artifact is found later, report both raw and adjusted VRAM, document the formula, and apply it only to affected rows; do not apply blanket subtraction across token and set families. Current audit `audit/vram_overhead_audit.md` found no justified subtraction for A7: near-token set overhead is dominated by token+set streams and dense token-to-set routing/set-attention tensors.

Phase A status tracking:
- `audit/phase_a_status.md` is the master status tracker for all Phase A tasks (A0–A5).
- **Read it at session start before doing any Phase A work.** It disambiguates what is done, running, or pending so you do not re-run completed phases or skip required ones.
- **Update it whenever any of the following happen:**
  - A phase or sub-task changes status (pending → running → done / fail).
  - A long-running job is launched (record PID, ETA, launch time, and which bat/script triggers the sync).
  - A phase audit file is written (record its path in the tracker row).
  - An incident is created (add a row to the Incidents table).
  - A blocking dependency resolves (update the dependency chain comment).
- Update format: edit the relevant row's Status column and Notes column in-place; append a "Last updated" timestamp at the top of the file.
- The tracker is a local file (`audit/phase_a_status.md`). It is untracked by Git (under `audit/`, which is in `.gitignore`). Sync it to blue-demon alongside any related script changes if the remote agent needs it.
- Do not confuse this file with `out/final_paper_bundle/checks/current_plan.md`, which tracks Phase B (paper writing) actions only.
- The pattern mirrors the Phase B `progress_log.md` / `current_plan.md` discipline: always read before acting, always update after acting.

Long-running sweep monitoring:
- After launching a long-running sweep on blue-demon with `nohup`, run exactly one compact status check to confirm the job is alive and producing expected early artifacts/logs.
- Once that first healthy status is confirmed, disconnect and stop polling. Do not start local `sleep` loops or repeated SSH polling while the user is waiting for the sweep to finish.
- Wait for the user's explicit notification that the sweep ended before running final validation, summarization, artifact sync, or launching the next sweep.
- Only break this wait rule if the first status check shows the job died, artifacts are missing/incomplete, or logs show OOM/NaN/traceback/W&B step failures; then capture the incident and report the blocker.

Repository/source of truth:
- Primary experiment server: `blue-demon:~/set-attention`
- Local paper/workspace mirror may exist at:
  `/mnt/d/UserFolders/Documents/GitHub/set-attention`
- Treat blue-demon as the source of truth for completed experiments and generated artifacts.
- Preserve the noncausal implementation as an optional configurable mode. Do not delete it; it is a valid set-attention behavior for bidirectional/non-AR tasks and diagnostics, although not valid as evidence for autoregressive LM claims.

Access and credentials:
- Blue-demon SSH credentials are available locally via the existing SSH/password files already used by Codex. Prefer the configured SSH entry or password file, and never paste credential contents into logs or committed files.
  - Canonical credential file (path only, never the secret): `../blue-demon.txt` relative to the repo root, i.e. `D:\UserFolders\Documents\GitHub\blue-demon.txt` (WSL: `/mnt/d/UserFolders/Documents/GitHub/blue-demon.txt`). It has 4 `key: value` lines: a label, `local IP address: 192.168.241.149`, `user: iarroyof`, `pass: <password>`. Parse the password with `grep -i '^pass' "$F" | sed 's/^[^:]*:[[:space:]]*//'` and pass it via `sshpass -e` (export `SSHPASS`). The same password is also mirrored at WSL `~/.ssh/.sshpass`. Run remote ops through `wsl -d Ubuntu-24.04 -u iarroyof` using a pre-written `.sh` (inline PowerShell→wsl→ssh quoting breaks); key-based ssh is not configured, so `sshpass -f` on a structured file or a bare `ssh` will fail auth.
- GitHub credential/token file on the Windows host: `C:\Users\nachi\Documents\github_toke.txt`.
- Use credential files only when needed for sync operations. Do not commit credential files, token contents, copied secrets, shell history containing tokens, or generated logs that expose them.
- Keep the three repo copies synced when source/config/script/context changes are made:
  - local WSL copy: `/mnt/d/UserFolders/Documents/GitHub/set-attention`;
  - remote GitHub branch: `origin/paper/final-results-bundle`;
  - blue-demon dev copy: `blue-demon:~/set-attention`.
- Use Git for tracked source files. Use provenance-preserving artifact sync for ignored generated outputs under `out/` only when those artifacts are needed across machines.

Highest-value workflow rule:
- Prefer a blue-demon-authoritative workflow for tracked source changes until the local WSL `/mnt/d` Git metadata issue is fully fixed.
- It is valid to edit a local WSL copy as a scratch/work buffer when that is faster, but the edit is not authoritative until it has been copied to `blue-demon:~/set-attention`, checked there, and committed there.
- The local WSL file contents can be correct while local `.git/` metadata is stale or read-only. In that state, local `git status` may show modified files even when the file content already matches the pushed commit.
- Do not waste time trying to commit from local WSL if `.git/index.lock` or `.git/config.lock` fails with read-only or permission errors.
- Preferred tracked-file workflow:
  1. Edit on blue-demon directly when convenient; otherwise edit the local WSL copy as scratch.
  2. Copy the edited tracked file to the exact same repo-relative path under `blue-demon:~/set-attention`.
  3. Inspect the diff and commit on blue-demon.
  4. Push from blue-demon if GitHub credentials are available there.
  5. If blue-demon cannot push, create a Git bundle on blue-demon, copy it to local `/tmp`, import it into a temporary clone such as `/tmp/set-attention-push`, and push from there using the Windows GitHub token file.
- Remote sync guardrail, added after repeated accidental basename copies:
  - Never run `scp file1 file2 ... blue-demon:~/set-attention/` for source/config/script/context syncs; that drops every file into the repo root by basename and creates misleading untracked files.
  - Do not use multi-source `scp` to `blue-demon:~/set-attention/` even when the files are already known. If syncing several files, either run one exact destination command per file or use `rsync -avR` so repo-relative paths are preserved.
  - Use one of these exact-path patterns instead:
    - `scp local/path/file.py blue-demon:~/set-attention/local/path/file.py`
    - `rsync -avR local/path/file.py local/path/other.py blue-demon:~/set-attention/`
  - Before syncing multiple files, print the intended repo-relative file list. After syncing, verify with `sha256sum` or `git diff -- path` on blue-demon for those exact paths.
  - If a bad basename copy is discovered in `~/set-attention`, record it and clean it intentionally; do not leave future agents guessing whether root-level copies are authoritative.
- When using the Windows token file, extract only the actual token line beginning with `ghp_` or `github_pat_`; the file may contain explanatory text and blank lines.
- Verify sync by comparing:
  - blue-demon `git rev-parse HEAD`;
  - GitHub `origin/paper/final-results-bundle`;
  - `sha256sum` of the relevant file across local and the temporary/blue-demon copy.
- Do not rely solely on local WSL `git status` while the `.git/` mount issue persists.

Blue-demon experiment environment:
- SSH target: `iarroyof@192.168.241.149`, repo path `~/set-attention`.
- The host repo is the Docker Compose control directory. Do not run experiment Python with host `/usr/bin/python3`; it may not have `torch` installed.
- Run all model tests, smoke checks, and experiments inside the Docker Compose service `set-attention`.
- `pytest` is currently unavailable in the `set-attention` container. For focused revision tests, use direct function execution inside the container unless/until pytest is installed there.
- Verified runtime:
  - Compose service: `set-attention`
  - Container working directory: `/workspace`
  - Python inside container: `/usr/bin/python`
  - Python version: `3.11.0rc1`
  - PyTorch: `2.5.1+cu124`
  - CUDA available: yes
  - GPU count: 2
  - GPU model: NVIDIA GeForce RTX 4090
- Use this command pattern for Python checks:
```bash
sshpass -f ~/.ssh/.sshpass ssh iarroyof@192.168.241.149 \
  'cd ~/set-attention && docker compose exec -T set-attention python - <<PY
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY'
```
- Use this command pattern for experiments:
```bash
sshpass -f ~/.ssh/.sshpass ssh iarroyof@192.168.241.149 \
  'cd ~/set-attention && docker compose exec -T \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_DATASETS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    set-attention \
    python scripts/run_experiment.py --config <config.yaml> --csv-path <out.csv>'
```
- Existing blue-demon run scripts already use this pattern, for example `scripts/gpu0_run_lrnorm_headline_pairs.sh`.
- For Git operations, summaries, and file transfer, use the host shell from `~/set-attention`; for any code path importing project modules, `torch`, CUDA, datasets, or training utilities, use the `set-attention` container.

**Critical Finding**

The current set-only LM implementation has an autoregressive causality leak.

Verified files:
- `src/models/set_only/banks.py`
- `src/models/set_only/set_only_lm.py`
- `src/models/set_only/router.py`
- `src/train/loop.py`
- `src/data/wikitext2.py`

Current behavior:
- `build_window_bank()` builds forward windows `[start, start + window_size)`.
- `Bank.pool()` pools all valid tokens in each set with only padding masks.
- `token_to_sets` includes every set containing token `t`.
- `SetOnlyLM.causal=True` only applies an inter-set causal mask after pooling.
- `LearnedRouter` and `UniformRouter` route over `token_to_sets` without enforcing `max S_m <= t`.
- WikiText-2 labels are next-token shifted, but this does not prevent leakage because `z_t` can already contain future input tokens through pooled set states.

Therefore:
- Current AR LM perplexity claims are not causally valid.
- Current non-causal set-only results may be retained only as non-causal / bidirectional / diagnostic appendix evidence until evaluated on tasks where non-causal access is appropriate.
- The revision should not defend the old AR headline as-is.

---

## Branches To Implement Or Support

Implement the fork cleanly so we can select the strongest path later.

### Branch 0: Current Non-Causal Architecture

Purpose:
- Preserve current implementation for diagnostics, bidirectional tasks, and appendix comparisons.
- Keep this as a first-class, named behavior, not as a failed or negative-result variant.

Suggested config:
```yaml
model:
  set_causality_mode: noncausal
```

Semantics:
- Current window bank.
- Current pooling.
- Current routing over all containing sets.
- Must be explicitly labeled non-causal when used in AR contexts.
- If this mode is selected with an autoregressive LM objective, the code should emit a clear warning in logs and run manifests.
- Paper drafts that show AR-LM results from this mode must include visible draft annotations warning that the numbers are non-causal and not valid headline AR evidence.

### Our model would be comparable to token-level causal LM only if you verify one of these:

- Causal pooling: for token `t`, each set contributing to its prediction excludes tokens `> t`.
- Causal bank construction: each set is truncated causally relative to the prediction point
- Prefix-time recomputation: set states are recomputed from prefixes only
- Equivalent proof: a formal argument that the current routing/pooling pipeline cannot leak future token information to token `t`.

Without one of those, I would not claim fairness against standard token-causal baselines.

Analyze the problem and suggest the best option, not necessarily limited to the branches below, that jointly optimizes efficiency, expressivity, causality, and ease of integration with the existing implementation.

The branch-selection analysis should include a theory-motivated comparison against the current non-causal operator as a reference. A Lipschitz-style or operator-perturbation argument is acceptable if a stronger equivalence proof is not available: characterize what information or candidate support is lost when moving from the non-causal reference to a causal branch, and connect that argument to measurable diagnostics such as candidate count, pooling support, entropy, transport, and validation loss.

Meanwhile the following are options quickly suggested.

### Branch 1b: End-Aligned Causal Bank

This is the likely strongest defensible path if we want to preserve efficiency.

Semantics:
- Each set is a past-facing or end-aligned window.
- Token `t` may route only to sets whose maximum token index is `<= t`.
- Add fallback singleton/current-token sets so early positions do not have empty candidate families.

Suggested config:
```yaml
model:
  set_causality_mode: end_aligned
  causal_fallback: singleton_current
```

Required implementation:
- Add bank metadata:
  - `set_starts`
  - `set_ends`
  - `set_indices`
  - `token_to_sets_causal`
  - candidate count per token
- Routing must use `token_to_sets_causal` when `set_causality_mode=end_aligned`.
- Pooling can remain one pooled state per set.
- Set-attention stack can remain set-level with causal set-to-set mask.
- Add explicit tests that no token representation depends on future tokens.

Paper math:
```text
C_t^- = {m : max S_m <= t},
z_t^(u) = sum_{m in C_t^-} pi_{t,m}^(u) * s_tilde_m^(u).
```

### Branch 1a: Causal Prefix Pooling

This is the cleanest semantic fix but likely weakens the efficiency story.

Semantics:
```text
s_{m,t}^{(0)}
=
sum_{u in S_m, u <= t} omega_{u,m,t} h_u^{(0)}.
```

Implementation implications:
- Pooling becomes token-conditioned.
- Routing consumes per-token causal set summaries.
- More expensive and more invasive.
- Implement only if selected explicitly or after 1b underperforms badly.

Suggested config:
```yaml
model:
  set_causality_mode: prefix_pooling
```

### Branch 2: Bidirectional / Encoder Pivot

Purpose:
- Keep current architecture as valid by changing task framing.
- Suitable tasks: MLM, classification, retrieval, LRA-style tasks, bidirectional sequence diagnostics.

Suggested config:
```yaml
model:
  set_causality_mode: bidirectional
task:
  objective: mlm
```

### Branch 3: Theory + Diagnostic Study

Purpose:
- Remove competitive AR LM claim.
- Keep Tier C mechanisms as the contribution:
  - candidate-count collapse
  - pooling concentration
  - gradient transport
  - cross-family dense/sparse/linear behavior

No architecture fix strictly required, but all AR wording must be removed or caveated.

---

## Required Code Workstreams

### A. Modeling Code

Files to inspect/edit:
- `src/models/set_only/banks.py`
- `src/models/set_only/set_only_lm.py`
- `src/models/set_only/router.py`
- `src/config/schema.py`
- `scripts/run_experiment.py`

Add:
- `set_causality_mode`
- `causal_fallback`
- clear validation errors for incompatible settings
- metadata-rich `Bank` object

Suggested enum:
```python
SetCausalityMode = Literal[
    "noncausal",
    "end_aligned",
    "prefix_pooling",
    "bidirectional",
]
```

Keep old behavior exactly reachable through `noncausal`.

### B. Dataset / Objective

Files:
- `src/data/wikitext2.py`
- `src/train/loop.py`

Tasks:
- Confirm AR shift remains correct for causal branches.
- Add optional leak-test dataset or utility where future tokens can be perturbed while past tokens are fixed.
- For bidirectional branch, add MLM-style masking only if Branch 2 is selected.

### C. Diagnostics

Add causality diagnostics:
1. Perturbation test:
   - Compute `logits_t` for a sequence.
   - Change tokens `x_{>t}` only.
   - Confirm `logits_t` is unchanged under causal modes.

2. Gradient test:
   - Compute loss at position `t`.
   - Confirm gradients w.r.t. embeddings/tokens at positions `>t` are zero or numerically negligible.

3. Candidate-family audit:
   - For every `t`, assert all routed sets satisfy `max S_m <= t` in end-aligned mode.

Output diagnostics to CSV/JSON:
```text
causality_violation_max_abs
future_grad_norm
num_invalid_future_candidates
mean_causal_candidate_count
min_causal_candidate_count
```

### D. Interface

Expose through:
- YAML configs
- CLI overrides
- experiment manifests
- result CSVs

Every run must record:
```text
set_causality_mode
causal_fallback
window_size
stride
mean_candidate_count
min_candidate_count
max_candidate_count
pooling_mode
router_type
router_topk
router_multihead
backend
backend_params
seed
git_commit
config_sha256
source_config_path
```

---

## Latest Main-Paper Grid To Preserve As Historical / Appendix Evidence

Current reference operating recipe used in the latest main-paper experiments:

```text
Dataset: WikiText-2 LM
Sequence length: 512
Batch size: 16
Epochs: 10
Warmup: 1000
Model width: D=384
Layers: 6
Heads: 8
FFN width: 1536
Dropout family: 0.1
Seed: historically seed 0, latest normalized runs may include more
Vocabulary size: 76618
Architecture: causal transformer_lm for baseline
Set features: hashed_counts
Router: learned
router_topk: 16
router_multihead: true
pooling: soft_trimmed_boltzmann
tau_pool: 0.1
pooling_multihead: false
window size: 16
stride: 8
Dense set backend: dense_exact
Sparse set backend: local_band, radius=4
Historical linear set backend: landmark, num_landmarks=24
Current A1.6 linear set backend: landmark, landmark_coverage=0.25
```

Important:
- These current set-only AR LM results were produced under the leaking set-only architecture.
- They may be kept only as historical/non-causal appendix evidence unless rerun under a causal branch.
- Do not use them as headline AR evidence.

---

## Strong Defensible Experiment Plan

Given dedicated dual RTX 4090 workstation, do not use relaxed evidence. Use strong seeding.

### Minimum Causal AR Grid

Use Branch 1b first.

Seeds:
```text
0, 1, 2, 3, 4
```

Learning rates:
```text
5e-5, 1e-4, 2e-4, 5e-4
```

Families:
```text
baseline_dense_exact
baseline_sparse_local_band
baseline_linear_landmark
set_dense_exact_end_aligned (or the selected causal branch)
set_sparse_local_band_end_aligned (or the selected causal branch)
set_linear_landmark_end_aligned (or the selected causal branch)
```

Reference size:
```text
L=512
B=16
D=384
layers=6
heads=8
d_ff=1536
epochs=10 initially, then extend best settings to convergence
```

Topology sweeps:
```text
window_size: 8, 16, 32, 64
stride: 4, 8, 16, 32
```

Pooling temperature:
```text
tau_pool: 0.03, 0.05, 0.1, 0.2, 0.5, 1.0
```

Router temperature:
```text
tau_router: 0.5, 1.0, 2.0
```

Longer context extension:
```text
L: 1024, 2048, 4096
```

For long context:
- Use best few operating points only.
- Include baselines where feasible.
- If baselines become infeasible, report memory/time limits explicitly.

### Required Comparisons

At minimum:
- Dense causal Transformer baseline.
- Sparse causal Transformer baseline.
- Linear causal Transformer baseline.
- Causal Set Dense.
- Causal Set Sparse.
- Causal Set Linear.

If the token-baseline sparse or linear backends do not already exist, implement them rather than omitting them, provided the implementation can be validated with the same causality tests and smoke-training gates. Name these clearly as token-attention baselines with backend `dense_exact`, `local_band`, or `landmark`, distinct from Set Attention families.

Make this matrix effective across all relevant parts of the plan. If time permits, add one published efficient-attention baseline:
- Performer
- Longformer
- Routing Transformer

Do not claim broad efficient-attention superiority without at least one published efficient-attention comparator.

---

## Provenance Requirements

Reuse existing provenance style, but tighten it.

Current artifact sync state:
- Source branch for this revision work: `paper/final-results-bundle`.
- `main` has been verified separately and should not be mixed into the paper branch except by an explicit merge/rebase decision.
- Ignored artifacts under `out/` were non-destructively synced between local and blue-demon with sync ID `20260506_130440`.
- Preserve existing sync manifests and conflict archives under:
  - `out/_artifact_sync_manifests/20260506_130440/`
  - `out/_artifact_sync_conflicts/20260506_130440/`
- Do not bulk-commit `out/`. Use manifests, summaries, copied paper assets, or explicit tracked provenance files when information must be shared by Git.

For every run, write:
```text
out/runs/<run_id>/
  config.yaml
  resolved_config.yaml
  metrics.csv
  diagnostics.csv
  causality_audit.json
  manifest.json
  stdout.log
  stderr.log
```

`manifest.json` must contain:
```json
{
  "run_id": "...",
  "git_commit": "...",
  "git_status_short": "...",
  "host": "blue-demon",
  "gpu_name": "...",
  "cuda_version": "...",
  "python_version": "...",
  "torch_version": "...",
  "config_sha256": "...",
  "source_config_path": "...",
  "seed": 0,
  "set_causality_mode": "end_aligned",
  "started_at": "...",
  "ended_at": "...",
  "metrics_csv": "...",
  "causality_audit": "..."
}
```

Summary scripts must never reconstruct numbers from memory. They must read only:
- metrics CSVs
- diagnostics CSVs
- manifest JSONs
- causality audit JSONs

---

## Paper Revision Guidance

Until causal reruns are complete:
- Do not silently delete old AR-LM claims; instead mark them clearly in draft form as non-causal historical evidence and demote them away from headline claims.
- Before any camera-ready or submission PDF, remove visible draft notes and ensure old non-causal set-only AR results appear only as historical/non-causal appendix evidence, with explicit non-causality clarification.
- Keep Tier C theory and diagnostics if framed architecture-independently; where a statement depends on causality, annotate the draft and update the math after the causal branch is selected.
- State that causal variants are under evaluation only after results exist, and only in draft comments where appropriate.

If Branch 1b, or another approved causal approach, succeeds:
- Main paper can return to AR LM framing.
- Rebuild Table 1 and headline figure only from causal runs.
- Add causality audit table in main or appendix.

If Branch 1b fails:
- Do not force the narrative.
- Pivot to Branch 2 or Branch 3, or another approved approach that meets the above requirements.

---

## Immediate First Tasks For New Codex Chat

1. Inspect current code and add a minimal regression test that reproduces the leak under `set_causality_mode=noncausal` in an AR-LM setting.
2. Add or verify `set_causality_mode=noncausal`, preserving exact old behavior for valid non-causal/bidirectional use cases.
3. Add warning logs, run-manifest warnings, and draft paper annotations whenever `noncausal` is paired with an AR-LM objective.
4. Produce a short branch-selection memo comparing `end_aligned`, `prefix_pooling`, and any better proposed causal design against the non-causal reference using efficiency, expressivity, implementation risk, and measurable diagnostics.
5. Implement the selected causal bank/routing mode, with `end_aligned` as the default starting point unless the memo justifies another choice.
6. Add causality audit tests and diagnostics:
   - non-causal AR mode should fail the leak test by design;
   - causal AR modes must pass perturbation, gradient, and candidate-family audits.
7. Verify or implement token-attention baselines for dense exact, sparse local-band, and linear landmark backends.
8. Make sure the hyperparameter space is sufficiently, but not excessively, sampled to show informative ranges to a meticulous reader.
9. Generate reference configs for:
   - token baseline dense exact;
   - token baseline sparse local-band;
   - token baseline linear landmark;
   - causal set dense exact;
   - causal set sparse local-band;
   - causal set linear landmark;
   - non-causal set variants for appropriate non-AR tasks/datasets.
10. Add run scripts for dual RTX 4090 scheduling on blue-demon.
11. Add provenance manifests for every run.
12. Run all smoke tests and short production-readiness checks on blue-demon before launching expensive experiments.
13. Only after tests, diagnostics, manifests, and smoke runs pass, launch the seed/LR grid on blue-demon.
