---
name: verified-before-proposed
description: User prefers consolidating/measuring verified results over bolting on untested architecture; wants clear per-change attribution
metadata:
  type: feedback
---

On the set-attention project the user is wary of proposing architectural modifications that haven't been ablated, and dislikes accumulating untested changes in the plan. They want a clear ledger of which changes were tested and whether each helped/hurt/was neutral (with numbers), separated from untested proposals.

**Why:** with many experiments run, it's easy to lose track of what actually moved the metric vs what was speculative; untested architecture bolted on speculatively wastes compute and muddies attribution.

**How to apply:** lead with a verified-vs-proposed ledger when recommending next steps. Distinguish
measurement/scaling of the existing exact-dense multiresolution model from genuine architecture
changes. The active work is the five-seed regular blur matrix `{b0,b25,b50,b75,b100}` plus matched
token controls. Re-read (SD-10a), full latent (SD-11), landmark, sparse, fixed-k, and Nyström work are
untested/inactive hypotheses and require explicit user approval.
