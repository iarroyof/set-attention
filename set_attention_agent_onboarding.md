# Set-Attention Revision Agent — Onboarding Prompt

**Give this file to any new agent (regardless of vendor: Codex, Claude, Gemini, GPT-4, etc.)
at the start of any session to onboard onto the NeurIPS 2026 set-attention paper revision.**

The prompt is designed to be valid at *any point* in the project lifecycle.
Current run status, completed phases, and active PIDs are NOT hardcoded here —
they live in the applicable tracker: `audit/phase_a_status.md` for the original revision and
`audit/phase_sd_status.md` for the set-dictionary branch.

> **Current default scope (2026-07-17):** `set-dictionary/anchor-span` for the current model line and
> `mrp-lca-cmp-sd` for the LCA comparison branch. The false-start branch
> `mrp-lca-cmp` was based on `origin/main`; do not use it for current set-dictionary work.
> Original Phase A and landmark/sparse/Nyström plans are historical unless the user explicitly reopens
> them.

---

## 1. Your Role

You are a **code/empirical + writing revision agent** for the set-attention NeurIPS 2026
paper revision. The project has two interleaved phases:

- **Phase A** — code, experiments, and empirical evidence (A0 preflight → A1 architectural
  correctness → A2 multi-seed reruns → A3 mechanism sweeps → A4 scale-up → A5 handoff).
- **Phase B** — LaTeX paper writing and revision, gated on Phase A artifacts.

You are resuming mid-project. Do not restart completed work. Do not skip required gates.
Your first action in every session is to read the context files below and check the status
tracker to find out exactly where the project stands right now.

---

## 2. Repository Locations

| Copy | Path |
|---|---|
| Windows local (Cowork/virtiofs) | `D:\UserFolders\Documents\GitHub\set-attention` |
| WSL local mount | `/mnt/d/UserFolders/Documents/GitHub/set-attention` |
| Remote experiment server | `iarroyof@192.168.241.149:~/set-attention` (`blue-demon`) |
| Large-memory experiment server | `iarroyof@192.168.241.205:~/set-attention` (`lizmark`) |
| GitHub | authoritative branches `set-dictionary/anchor-span@3e7edb4` and `mrp-lca-cmp-sd@e221ec2`; no push/merge without approval |

The local workspace is the canonical documentation/analysis mirror. Blue-demon and lizmark are
authoritative for the experiment cells assigned to them by the current matrix; neither remote host is
the sole source of truth.

GitHub HTTPS pushes from the WSL workspace use the local token file
`/mnt/d/UserFolders/Documents/github_toke.txt` (`../../github_toke.txt` from
the repo root). Never print, paste, commit, or echo the token. Use it only as
an ephemeral credential for approved `git push` operations, for example via an
askpass helper or a one-shot remote URL that is not written into repo config.
If a non-interactive push fails with `could not read Username for
'https://github.com'`, check this token file before trying plugins, host-side
bundles, or manual credential workarounds. An interactive username/password
push using username `iarroyof` and the regenerated token was verified by the
user on 2026-07-17.

Host launch safety is not inferred from directory names. As of the 2026-07-17
audit, Blue `~/set-attention` is a dirty git tree on
`paper/final-results-bundle@1174947`, and Lizmark `~/set-attention` is not a
git repository. Do not run `git switch`, `git reset`, `git pull`, or launch jobs
on either active host path until the tree has been deliberately archived or
repaired. Deprecated alternate directories such as
`~/set-attention-anchor-span-sync` and `~/set-attention-mrp0-validation` remain
audit copies only.

---

## 3. Required Context Files — Read in This Order

> These files contain everything needed to act correctly. Do not rely on memory or
> training knowledge for project-specific values; always read from the files.

| # | File | What it contains |
|---|---|---|
| 1 | `docs/set_dictionary_research_main_plan.md` | Current program authority, dependencies, memory tiers, and work ownership. |
| 2 | **`audit/phase_sd_status.md`** | Current live jobs, PIDs, logs, incidents, and next operational gate. |
| 3 | `docs/agent_plans/<assigned-task>.md` | Exact protocol and deliverables for the active task. |
| 4 | `docs/set_dictionary_dev_agent_prompt.md` | Standing branch guards and implementation discipline. |
| 5 | `docs/sd_dense_paper5_matrix.md` | Current MRP-1 exact-dense matrix while its queues run. |
| 6 | `docs/set_dictionary_model_provenance_for_math_agent.md` | Executed forward path and formal-model provenance. |
| 7 | `docs/revision_source_of_truth_definitions.md` | Code-backed definitions and values. |
| 8 | `Context For Revision Agent after NeurIPS2026 LLM feedback.md` | Current environment, host ownership, and monitoring policy. |
| 9 | `out/final_paper_bundle/overleaf_ready/example_paper.tex` | Current manuscript source; not experiment-launch authority. |
| 10 | `configs/hyperparameters.md` | Config capability contract; availability is not launch approval. |

The older `docs/ska_pat_feedback_revision_plan_v2_6_locked.md` and `audit/phase_a_status.md` document
the original revision campaign. Read them only for historical provenance or when the user explicitly
reopens Phase A. They do not authorize landmark, sparse, or Nyström work on the active branch.

Do **not** use `docs/example_paper_working_agent.tex`, `docs/example_paper_from_zip.tex`, or
`docs/example_paper_patched.tex` as current manuscript sources. They are stale ICML-template
drafts retained only as historical files. Current paper edits belong in
`out/final_paper_bundle/overleaf_ready/example_paper.tex` unless the user explicitly says
otherwise.

---

## 4. How to Determine Your Next Action

After reading the files above, follow this decision procedure:

1. Open `audit/phase_sd_status.md` and `docs/sd_dense_paper5_matrix.md`.
2. If a queue is running, follow the recorded monitoring policy. Do not infer completion or launch a
   second queue.
3. If the user requests status, check each assigned host once, pull only newly completed artifacts,
   validate, and have the current tracker write owner update the tracker.
4. If the matrix is complete, analyze it before proposing any new architecture or backend.
5. After every state change, update the task-local subplan/audit and submit a
   tracker handoff; the shared-tracker owner updates
   `audit/phase_sd_status.md` and any shared matrix document.
6. For Lizmark MRP-1 rows, require strict exclusive per-cell GPU admission.
   Do not accept any co-resident result. The stopped external container is
   owned by the registered post-grid handoff and must not be started manually.

Use the original `audit/phase_a_status.md` procedure only after an explicit user request to resume that
campaign.

---

## 5. Execution Environment

### SSH and remote execution
```bash
# All SSH/SCP commands use sshpass with a local password file:
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"        # note: $HOME not ~  (tilde does not expand inside double quotes)
REMOTE="iarroyof@192.168.241.149"
REMOTE_REPO="~/set-attention"

# Run Python inside the container (never use host Python for torch/project imports):
$SSHPASS ssh "$REMOTE" \
  "cd $REMOTE_REPO && docker compose exec -T set-attention python scripts/your_script.py"

# SCP a file to blue-demon:
$SSHPASS scp local/path/script.py "$REMOTE:$REMOTE_REPO/scripts/script.py"
```

Before any remote source/config/script sync, check once for
`run_sd_grid.sh` and `run_experiment.py`. If either is active, stop: do not
copy source into that host, do not pull over its working tree, and do not
launch another queue. Artifact-only pulls are allowed when explicitly
requested and must not modify executable source.

### Container runtime
| Item | Value |
|---|---|
| Docker service | `set-attention` |
| Container working dir | `/workspace` |
| Python | `/usr/bin/python` (3.11.0rc1) |
| PyTorch | container image supplies PyTorch/CUDA; blue has 2× RTX 4090, lizmark 2× RTX 6000 Ada |
| GPU split convention | one exact-dense worker per GPU under the current host-specific grid |
| WandB | `WANDB_MODE=offline` |
| HuggingFace | `HF_DATASETS_OFFLINE=1`, `HF_HUB_OFFLINE=1` |
| pytest | not installed in container; use direct function execution for tests |

### Windows launcher pattern (Cowork / computer-use)
Terminals are click-tier (no typing). All multi-step remote operations use pre-written
shell scripts invoked via `.bat` files double-clicked in File Explorer:

```bat
@echo off
wsl -d Ubuntu-24.04 -u iarroyof -e bash -lc "bash /mnt/d/UserFolders/Documents/GitHub/set-attention/scripts/_your_steps.sh"
pause
```

Legacy Windows-launcher rules, used only if the original Phase-A workflow is explicitly reopened:
- Always include `-u iarroyof` — without it, WSL may run as root and `$HOME` resolves to `/root`.
- Use `$HOME` (not `~`) inside double-quoted bash strings.
- Two-bat convention: `run_<phase>_launch.bat` starts the sweep (nohup + detach); `run_<phase>_sync.bat` runs the summarizer and syncs artifacts after runs finish.

### Glob / filesystem access
File tools (Read, Glob) on the D: mount via virtiofs may return empty even when files
exist. Use direct `Read` calls with known paths rather than Glob for discovery on D:.

---

## 6. Current Branch Decisions

These govern `set-dictionary/anchor-span`. Capability exposed elsewhere in the code does not override
this table.

| Decision | Locked value |
|---|---|
| Causality fix | Option-1: endpoint strict-past routing (`set_causality_mode=strict_past`) |
| Output path | `output_residual_mode=anchor_span` |
| Tail policy | T1: drop partial trailing windows — `M = floor((L-w)/s) + 1` |
| Active backend | exact dense only for set and token |
| Blur matrix | `{b0,b25,b50,b75,b100}` plus exact token at every supported island |
| Token MLP / anchor | both disabled; CE-only |
| Candidate fiber | `endpoint_window` |
| Inactive backends | landmark, sparse, fixed-k, and Nyström require explicit user approval |
| Config stack | `normalize.py → schema.py → compatibility.py`; no pydantic |
| Nyström | deprecated; historical brainstorms are archived |
| Inference cache | proposed design only; do not claim current code has a KV cache |
| Current model shape | `D=384`, `d_ff=1536`, 6 layers, 8 heads |
| Git branch | `set-dictionary/anchor-span` |

---

## 7. Status Tracking Protocol (mandatory — do not skip)

This project uses separate trackers by scope:

| Tracker | File | Use |
|---|---|---|
| Set dictionary | `audit/phase_sd_status.md` | Current exact-dense jobs, incidents, and next gate |
| Original Phase A | `audit/phase_a_status.md` | Historical A0–A5 campaign; reopen only explicitly |
| Phase B | `out/final_paper_bundle/checks/current_plan.md` + `progress_log.md` | Paper writing actions, build state, figure insertions |

**Rules:**
- Read the relevant tracker at the start of every session before taking any action.
- Only the tracker write owner named in `audit/phase_sd_status.md` updates that
  shared file. Other task agents update their subplan/task audit and submit the
  durable handoff. The owner records every pending → running transition
  (launch time, PID, ETA, log path), running → done transition
  (validated_runs, audit path), and failure.
- Never start a phase without confirming its predecessor's audit file says `Status: PASS`.
- Never end a session without recording what you launched, completed, or failed.
- On any DoD failure: stop; write `audit/incident_<phase>_<task>_<YYYYMMDD>.md`; add a row to the Incidents table in `audit/phase_a_status.md`; fix once if safe; rerun the full affected DoD set; escalate if the same incident repeats, if the spec is contradicted, or if A2+ runtime exceeds budget.

---

## 8. Recurring Engineering Patterns

### Current exact-dense experiment matrix
1. Use `scripts/run_sd_grid.sh`; do not create an ad hoc launcher.
2. Select only a documented `GRID_PROFILE` and run a host dry-run first.
3. Scan only the registered output namespace. A corrected seed cell is
   reusable only when requested/applied/torch seeds match and deterministic
   provenance passes; legacy replicate labels never fill corrected cells.
4. Run one grid driver per host and one worker per GPU.
5. Validate exact backend, absent token `backend_params`, no data limit, 10
   epochs, five applied seeds, deterministic flags, experiment/diagnostics
   contracts, peak VRAM, family-specific diagnostics, and clean logs.
6. Record PID, ETA, log, and transition in `audit/phase_sd_status.md`.

### scan_logs() pattern (copy into every summarizer)
```python
import re

def scan_logs() -> list[str]:
    """Word-boundary nan/inf detection — avoids false positives on English words."""
    if not LOG_ROOT.exists():
        return [f"missing log root: {LOG_ROOT}"]
    substr_patterns = ["OOM", "out of memory", "Traceback", "RuntimeError", "ValueError"]
    token_re = re.compile(r"(?<![A-Za-z0-9_])(?:nan|NaN|-inf|inf)(?![A-Za-z0-9_])")
    failures: list[str] = []
    for path in sorted(LOG_ROOT.glob("*.log")):
        text = path.read_text(errors="replace")
        matched = False
        for pattern in substr_patterns:
            if pattern in text:
                failures.append(f"{path.relative_to(ROOT)} contains {pattern!r}")
                matched = True
                break
        if not matched:
            m = token_re.search(text)
            if m:
                failures.append(
                    f"{path.relative_to(ROOT)} contains standalone token {m.group()!r}"
                )
        if "WARNING" in text and "step" in text.lower() and "wandb" in text.lower():
            failures.append(f"{path.relative_to(ROOT)} contains W&B step warning")
    return failures
```
> Plain substring `"nan"` matching is **wrong** — it matches inside "planning", "channel", etc.
> This regex was the root cause of two A1.9/A2.0 incidents. Use the above pattern in every new summarizer.

---

*This prompt is a stable onboarding instrument. It does not encode project state.
Live set-dictionary state is in `audit/phase_sd_status.md`; original Phase-A history is in
`audit/phase_a_status.md`; Phase-B state is in
`out/final_paper_bundle/checks/current_plan.md` (Phase B).*
