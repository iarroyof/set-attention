# Agent memory & status — global vs host-specific (read on session start)

This directory holds **GLOBAL, shared agent context** (committed, read by ANY agent on this branch — local,
blue-demon, lizmark, raptor-mini, dev agents). It exists so a fresh agent lands with the right context and
knows where to start.

## Files (global project memory; intended for version control)
- `../docs/set_dictionary_research_main_plan.md` — canonical program-level
  authority and dependency graph.
- `../docs/agent_plans/` — task-local immediate memory and handoff state.
- `set-attention-research-direction.md` — current exact-dense research direction and claim boundary.
- `verified-before-proposed.md` — the user's working preferences/feedback (tested-vs-untested attribution;
  hold untested architecture).
- `blue-demon-access.md` — current blue/lizmark host map and `sshpass -f ~/.ssh/.sshpass` workflow.
- `MEMORY.md` — one-line index of the above.
- `archive/` — superseded research syntheses retained only for provenance.

## Convention (how to split global vs host-specific)
- **GLOBAL → here, committed:** research context, decisions, preferences, credential *pointers* (never the
  secret itself). Any agent on any host reads these via git.
- **HOST-SPECIFIC / TRANSIENT → `audit/phase_sd_status.md`:** live PIDs, GPU occupancy, which host is running
  which sweep, launch logs. The tracker is the single source of truth for "where are we / what's running",
  recorded as a dated snapshot at commit time. Read the program plan first, then this tracker, then the
  assigned subplan. Task-specific operating procedures live in `docs/agent_plans/`.
- **SECRETS → never in the repo.** Use the external `~/.ssh/.sshpass` file. Never paste a plaintext
  password into a command, log, file, or commit.

So: shared context + status travel with the branch via git (the main plan, subplans, this directory,
and the tracker); per-host live state is just the tracker's snapshot. Never execute archived plans or
infer current jobs from historical memory. If a future need arises for durable per-host notes, add
`memory/hosts/<hostname>.md` (committed, clearly host-scoped) — but keep secrets out.
