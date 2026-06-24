# Agent memory & status — global vs host-specific (read on session start)

This directory holds **GLOBAL, shared agent context** (committed, read by ANY agent on this branch — local,
blue-demon, lizmark, raptor-mini, dev agents). It exists so a fresh agent lands with the right context and
knows where to start.

## Files (all global, all committed)
- `set-attention-research-direction.md` — running research synthesis: theory direction, empirical state
  (SD-1…9 outcomes), and locked decisions. The "what we know / what we decided" memory.
- `verified-before-proposed.md` — the user's working preferences/feedback (tested-vs-untested attribution;
  hold untested architecture).
- `blue-demon-access.md` — POINTER to credentials + the remote workflow. Stores the *path* `../blue-demon.txt`
  (repo-parent, outside the repo) and the `sshpass -e` pattern — **never a secret**.
- `MEMORY.md` — one-line index of the above.

## Convention (how to split global vs host-specific)
- **GLOBAL → here, committed:** research context, decisions, preferences, credential *pointers* (never the
  secret itself). Any agent on any host reads these via git.
- **HOST-SPECIFIC / TRANSIENT → `audit/phase_sd_status.md`:** live PIDs, GPU occupancy, which host is running
  which sweep, launch logs. The tracker is the single source of truth for "where are we / what's running",
  recorded as a dated snapshot at commit time. **Read it first every session.** The handoff
  `audit/SD_9_7_handoff.md` is the standing playbook (gap matrix + guards).
- **SECRETS → never in the repo.** Credentials live only in `../blue-demon.txt` (repo-parent) and the GitHub
  token file; reference them by path, parse at use time, pass via `sshpass -e`. Never paste a plaintext
  password into any command, log, file, or commit.

So: shared context + status travel with the branch via git (this dir + the tracker + the handoff); per-host
live state is just the tracker's snapshot. If a future need arises for durable per-host notes, add
`memory/hosts/<hostname>.md` (committed, clearly host-scoped) — but keep secrets out.
