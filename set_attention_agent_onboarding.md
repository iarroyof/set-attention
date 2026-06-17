# Set-Attention Revision Agent — Onboarding Prompt

**Give this file to any new agent (regardless of vendor: Codex, Claude, Gemini, GPT-4, etc.)
at the start of any session to onboard onto the NeurIPS 2026 set-attention paper revision.**

The prompt is designed to be valid at *any point* in the project lifecycle.
Current run status, completed phases, and active PIDs are NOT hardcoded here —
they live in `audit/phase_a_status.md`, which you will read as step 4 below.

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
| GitHub | branch `origin/paper/final-results-bundle` |

Blue-demon is the **authoritative source of truth** for completed experiments and generated
artifacts. The local copy is a working mirror. Git operations should prefer blue-demon
(see context file for the workaround when the local `.git/` mount is read-only).

---

## 3. Required Context Files — Read in This Order

> These files contain everything needed to act correctly. Do not rely on memory or
> training knowledge for project-specific values; always read from the files.

| # | File | What it contains |
|---|---|---|
| 1 | `docs/ska_pat_feedback_revision_plan_v2_6_locked.md` | **Locked** PAT feedback revision plan. Architectural decisions (Option-1, R1, T1, landmark backend). Full Phase A and B task breakdown with DoD. Supersedes all older pasted plans. |
| 2 | `docs/revision_source_of_truth_definitions.md` | Code-backed definitions and values: M formula, candidate fiber, landmark indices, pooling alpha, router min_temp, adapter resolution. Never fill these from prose memory. |
| 3 | `Context For Revision Agent after NeurIPS2026 LLM feedback.md` | Execution environment, blue-demon SSH/Docker workflow, repo sync rules, credential paths, Phase A/B tracking discipline, causality finding. |
| 4 | **`audit/phase_a_status.md`** | **Master Phase A status tracker.** Rows for every A0–A5 sub-task: status (done/running/pending), run counts, audit file pointer, notes. Also contains the incident log and blocking dependency chain. **This is how you determine what to do next.** |
| 5 | `docs/example_paper_working_agent.tex` | Current LaTeX working draft. Read before any Phase B writing. |
| 6 | `configs/hyperparameters.md` | Public config contract: canonical YAML keys, defaults, range checks. |

---

## 4. How to Determine Your Next Action

After reading the files above, follow this decision procedure:

1. **Open `audit/phase_a_status.md`.** Find the first row whose Status is not `✅ DONE`.
2. **Check its prerequisites.** The dependency chain in the tracker shows what must pass first. Do not start a phase whose predecessor audit file does not have `Status: PASS`.
3. **If a phase is `🔄 RUNNING`**: check the log on blue-demon (the tracker row will contain the log path and the relevant bat/script). Determine if it is still running or has finished. If finished, run the corresponding `_sync` bat file to collect artifacts, run the summarizer, and verify the manifest. Then update the tracker row to `✅ DONE`.
4. **If a phase is `⏳ PENDING`**: create the run script + summarizer following the established pattern (see existing scripts in `scripts/run_a3_*.sh` and `scripts/summarize_a3_*.py` as templates). Use the two-bat pattern: `run_<phase>_launch.bat` to start the sweep on blue-demon detached via nohup, `run_<phase>_sync.bat` to collect artifacts after completion. Update the tracker when you launch and again when you confirm PASS.
5. **If all Phase A tasks are `✅ DONE`**: proceed to Phase B. Read `out/final_paper_bundle/checks/current_plan.md` and `progress_log.md` to determine the Phase B state.
6. **After every action that changes project state**, update `audit/phase_a_status.md` (see tracking protocol in section 7).

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

### Container runtime
| Item | Value |
|---|---|
| Docker service | `set-attention` |
| Container working dir | `/workspace` |
| Python | `/usr/bin/python` (3.11.0rc1) |
| PyTorch | 2.5.1+cu124, CUDA available, 2× RTX 4090 |
| GPU split convention | GPU 0 → dense family; GPU 1 → sparse + linear families |
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

Key rules:
- Always include `-u iarroyof` — without it, WSL may run as root and `$HOME` resolves to `/root`.
- Use `$HOME` (not `~`) inside double-quoted bash strings.
- Two-bat convention: `run_<phase>_launch.bat` starts the sweep (nohup + detach); `run_<phase>_sync.bat` runs the summarizer and syncs artifacts after runs finish.

### Glob / filesystem access
File tools (Read, Glob) on the D: mount via virtiofs may return empty even when files
exist. Use direct `Read` calls with known paths rather than Glob for discovery on D:.

---

## 6. Locked Architectural and Config Decisions

These are frozen. Do not change them without an explicit user instruction and a new
locked-plan document superseding `docs/ska_pat_feedback_revision_plan_v2_6_locked.md`.

| Decision | Locked value |
|---|---|
| Causality fix | Option-1: endpoint strict-past routing (`set_causality_mode=strict_past`) |
| Token access | R1: direct embedding residual — `final_t = h_t^(0) + r_t` |
| Tail policy | T1: drop partial trailing windows — `M = floor((L-w)/s) + 1` |
| Linear backend | landmark only; `landmark_coverage=0.25`; linspace-rounded indices |
| Config stack | `normalize.py → schema.py → compatibility.py`; no pydantic |
| Nyström | deprecated; hard-fails construction; active YAMLs moved to `configs/_deprecated/` |
| Inference cache | proposed design only; do not claim current code has a KV cache |
| LR-norm headline reference | `D=384, d_ff=1536, w=16, s=8, M=63` at `L=512` |
| Anchor topology reference | `D=384, d_ff=1536, w=16, s=4, M=125` at `L=512` |
| Git branch | `paper/final-results-bundle` |

---

## 7. Status Tracking Protocol (mandatory — do not skip)

This project uses two parallel trackers, one per phase:

| Tracker | File | Use |
|---|---|---|
| Phase A | `audit/phase_a_status.md` | All A0–A5 task/run statuses, incidents, dependency chain |
| Phase B | `out/final_paper_bundle/checks/current_plan.md` + `progress_log.md` | Paper writing actions, build state, figure insertions |

**Rules:**
- Read the relevant tracker at the start of every session before taking any action.
- Update the tracker after every status transition: pending → running (record launch time, PID, ETA, log path), running → done (record validated_runs, audit file path), or any failure.
- Never start a phase without confirming its predecessor's audit file says `Status: PASS`.
- Never end a session without recording what you launched, completed, or failed.
- On any DoD failure: stop; write `audit/incident_<phase>_<task>_<YYYYMMDD>.md`; add a row to the Incidents table in `audit/phase_a_status.md`; fix once if safe; rerun the full affected DoD set; escalate if the same incident repeats, if the spec is contradicted, or if A2+ runtime exceeds budget.

---

## 8. Recurring Engineering Patterns

### New experiment sweep (any phase)
1. Create `scripts/run_<phase>_<sweep>.sh` — runs on blue-demon, Docker exec, two GPU workers (dense on 0, sparse+linear on 1), `record_prelaunch()` at top, `nohup` not needed here (called via launch script).
2. Create `scripts/summarize_<phase>_<sweep>.py` — validates CSVs + JSONs, checks `strict_past`, checks landmark coverage, verifies finite values, writes TSVs + manifest JSON + audit markdown to `out/paper_integrated_evidence/` and `audit/`.
3. Create `scripts/_run_<phase>_launch.sh` — local WSL script: `mkdir -p` log dirs on blue-demon, SCP run script, start via `nohup ... &`, print PID.
4. Create `scripts/_run_<phase>_steps.sh` — local WSL script: SCP summarizer, run it in container, SCP 4 artifacts back, verify manifest with Python.
5. Create `run_<phase>_launch.bat` and `run_<phase>_sync.bat` in repo root.
6. Record PID + ETA in `audit/phase_a_status.md` when launching; update to DONE after sync.

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
All live state is in `audit/phase_a_status.md` (Phase A) and
`out/final_paper_bundle/checks/current_plan.md` (Phase B).*
