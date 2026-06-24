---
name: verified-before-proposed
description: User prefers consolidating/measuring verified results over bolting on untested architecture; wants clear per-change attribution
metadata:
  type: feedback
---

On the set-attention project the user is wary of proposing architectural modifications that haven't been ablated, and dislikes accumulating untested changes in the plan. They want a clear ledger of which changes were tested and whether each helped/hurt/was neutral (with numbers), separated from untested proposals.

**Why:** with many experiments run, it's easy to lose track of what actually moved the metric vs what was speculative; untested architecture bolted on speculatively wastes compute and muddies attribution.

**How to apply:** lead with a verified-vs-proposed ledger when recommending next steps. Distinguish measurement/scaling of an already-verified model (safe, can't hurt perf) from genuine architecture changes (untested → unknown). Mark untested architecture as HOLD requiring explicit per-task go, not auto-gated. Default to consolidating the verified winner ([[set-attention-research-direction]] — SD-9 multi-resolution is the only clear win) + write-up over launching new architecture. Only the multi-resolution mix clearly helped; anchoring didn't, wider fiber/atom-width gave plateauing partial gains. Re-read (SD-10a) / full latent (SD-11) are untested hypotheses, held.
