---
name: launch-provenance-and-vram-hygiene
description: Never predict VRAM fit across recipe changes from old peaks; record per-row host/seed provenance and failed launches in the audit at launch AND completion; verify before reporting.
metadata:
  type: feedback
---

VRAM fit and provenance details are exactly the details that get lost between chat and repo, and losing
them degrades confidence in every downstream claim (user, 2026-08-08). Two concrete failures motivated
this rule: (1) I claimed "no L3584/B3 row fits Blue" from OLD-recipe registered peaks — but dropout
deactivation LOWERS VRAM (L512: -556 MiB set, -1076 MiB token), and at L3584/B3 the new-recipe peaks came
out LOWER than old (b75 22979 -> 20297 MiB, token 31035 -> 22072, b25 29175 -> 23791), so b75nodrop did
fit Blue; (2) the empirical Blue OOM of tokennodrop (Lizmark peak 22072 MiB, ~2.5 GB nominal headroom,
still OOM in backward) showed that even measured cross-host peaks need real headroom.

**Why:** recipe knobs that change memory (dropout, candidate_fiber, score_mode, topk, batch, L) invalidate
old peak numbers in BOTH directions. Dropout=0 removes mask/backward buffers (large saving, grows with
activation size); all_past+dense scoring adds score/gather memory (grows with L and M). The net sign is
not predictable by inspection. Allocator behavior also differs across hosts: a peak measured on one host
needs >= ~4 GB headroom on another (the grid policy's GPU_ADMISSION_HEADROOM_MIB=4096 exists for this).

**How to apply:**
- Before any fit claim: identify every memory-relevant knob that differs from the reference rows. If any
  differs, either run a short admission probe or state the estimate as unverified with the knob direction
  (dropout removal lowers, fiber/dense scoring raises).
- Headroom rule: a row fits a host only if measured peak + 4 GB <= host capacity, measured on that host or
  with explicit cross-host caution. 2.5 GB nominal headroom is NOT a fit (tokennodrop Blue OOM, 2026-08-08).
- Every launch records in the wave's audit section, BEFORE the user-facing report: host per row, seed set,
  config-guard deviations vs the reference protocol, and any failed launch attempts with their cause
  (contract pin, missing dataset cache, OOM). Every completion report records which seeds ran on which
  host. If a wave is host-mixed, the caveat travels into the paper wording, not just the audit.
- Never report git/sync/job status without verifying it in the same turn (HEAD on each copy, driver log
  tails, docker ps). A lost local watcher does not imply a lost remote driver — check before concluding.
- When a queued wave spans hosts, the driver TSV `host` column is the authoritative per-row provenance;
  audits summarize it, papers footnote it.
