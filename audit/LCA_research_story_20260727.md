# LCA Research Story: From the Set-Dictionary Matrix to the L2048 Pareto Point

Status document, 2026-07-27. Purpose: keep the full experimental pathway
traceable — every question posed, every answer obtained, how the nodes
connect, and what remains open. Authoritative numbers live in
`audit/LCA_calibration_20260718.md`; this document is the narrative map.

## 0. Where we started (pre-LCA, the reported matrix)

The set-dictionary (SD) full experimental matrix (MRP-1, paper5) was a
WikiText LM matrix: blur families {b0,b25,b50,b75,b100} + exact token,
islands L∈{512,1024,2048,4096} × B∈{3,4}, seeds 0-2 (255 strict
endpoint-valid rows). It froze `b*=b25` at the L2048/B4 selection island
(see `audit/SD_dense_paper5_final_20260708.md`). MRP-3 (MQAR) came back
null. Architecturally, every set row in that matrix ran with
`router_topk=16` and `router.score_mode=candidate_gather` — frozen
defaults inherited from the earliest set-only configs, never sweep
dimensions.

What the matrix established: set-dictionary attention's LM quality/VRAM
frontier for next-token prediction. What it could not establish: whether
the compressed set pathway retains and aggregates *global* information —
LM perplexity is dominated by local context, so a global-aggregation
failure would be invisible in it.

## 1. The motivating question (why LCA exists)

Does set-dictionary attention lose the ability to aggregate evidence
distributed across the whole context — and if so, where exactly is the
loss: reachability, routing, supervision, pooling, or the architecture
itself? The LCA (long-context aggregation) task was built as the minimal
probe: count markers distributed over L=1024/2048 positions, bucketed
answer, synthetic, seed-deterministic, with a matched token baseline.

## 2. The pathway, node by node

### Node 1 — Calibration and the Gate-2 failure (2026-07-18/19)
Registered calibration matrix (36 rows: token + b0..b100 × L{1024,2048}
× seeds 0-2, `endpoint_window` fiber, final-token loss, 2000 updates).
**Result: token 0.77 at L1024; every set row at chance (Gate 2 FAIL).**
Diagnosis: with `endpoint_window`, each query routes only to the ~2 set
atoms adjacent to its position — distant markers are topologically
unreachable from the supervised final token, and 6 layers of local
diffusion cannot bridge L=1024. Question refined: reachability first.

### Node 2 — Fiber probe: all_past is OOM-censored in candidate_gather
(2026-07-19). Switching `candidate_fiber=all_past` fixes reachability in
principle, but the candidate-gather router materializes
`[B,H,T,C,d_phi]` (keys duplicated over T×C×d_phi) — OOM on both hosts,
including microbatch retry. Recorded as an implementation boundary, not
a scientific result.

### Node 3 — Router-dense probe: reachability alone is not sufficient
(2026-07-20, option 0). `all_past` + `router.score_mode=dense` (scores
`q@k^T` as `[B,H,T,M]`, masking invalid entries) fits in memory (3216
MiB, `candidate_count_max=1023`) — but still does not learn
(val_acc 0.498 vs token 0.77). **Topology fixed; learning still
absent.** This killed the "it's just reachability" hypothesis and opened
the mechanistic phase.

### Node 4 — Mechanistic probes: localizing the mismatch (2026-07-20)
Three cheap probes at b25/L1024:
- P1 full routing (`router_topk=1023`): **0.827** — top-k=16 sparse
  selection was a bottleneck.
- P2 prefix supervision (every position predicts its prefix count):
  **0.787** (token 0.903) — final-token-only supervision was an
  independent bottleneck; the gradient-path asymmetry vs token attention
  is real but repairable.
- P3 oracle count token: **0.985** — routing/readout can solve the task
  when the count is represented; the residual deficit is
  pooling/set-state learning.
**Verdict: set-path/task mismatch (supervision sparsity + routing
sparsity + pooling), NOT an inherent architectural disadvantage.**

### Node 5 — Combined probe and the seed-noise lesson (2026-07-20)
Full routing + prefix supervision composed: 0.936 seed0, apparently
above token (0.903). The prefix3 3-seed mini-calibration showed this was
seed noise: token 0.9443±0.0317, b25-full 0.8956±0.0407, b25-topk16
0.8171±0.0114. Standing lesson, applied ever since: **no single-seed
conclusions; set rows carry higher seed variance than token.**

### Node 6 — Blur frontier sweep: the first Pareto point (2026-07-24)
Blur families {b25,b50,b75,b100} × seeds 0-2, all_past + dense +
full routing + prefix, L1024. **b75: 0.9233±0.0157 @ 2346.7 MiB vs token
0.9443±0.0317 @ 2680.6 MiB — within seed noise at −12.5% VRAM.** Quality
is non-monotone in blur (b75 > b50 ≈ b25 > b100): coarse atoms (w4/s2)
both shrink the O(L²) score tensor and are better counting units;
all-coarse loses detail. This was the first diagnostic set row matching
token quality at strictly lower memory.

### Node 7 — Top-k bandwidth sweep (2026-07-24)
b75/L1024, topk={16..1023} × seeds 0-2 (21 rows): quality **monotone, no
plateau** (0.784 → 0.923). Simultaneously: **VRAM flat across top-k**
(2347-2408 MiB) — in dense-score mode the `[B,H,T,M]` tensor is
materialized regardless of k, so top-k sparsity is not a memory lever in
this router mode. Provisional claim ("aggregation needs bandwidth")
immediately flagged as confounded by training budget (see Node 9).

### Node 8 — L2048 pilot: the gap that wasn't (2026-07-24/26)
Seed 0 at 2000 updates: token 0.944 @ 9124 MiB, b75full 0.816 @ 7201
MiB — VRAM edge holds (−21%) but an apparent quality gap (−0.128).
Seeds 1-2 (Blue) gave b75full 0.8553±0.0483 — high seed variance, the
undertraining signature.

### Node 9 — Budget probe and confirmation: the current frontier
(2026-07-26/27). Controlled pairs at 4000 updates:
- token: 0.9438 (saturated already at 2000).
- b75full seed0: 0.8158 → **0.9353** (+0.12 — the gap was mostly budget).
- b75full 3-seed @4000: **0.9078±0.0368 @ −21.1% VRAM** — Pareto claim
  holds on average; seed1 (0.866) keeps it from being a per-seed win.
- b75topk256 seed0: 0.7571 → 0.8099 (+0.053 from budget) but still
  −0.125 below full at matched budget: **the sparse-routing deficit is
  both training speed AND genuine bandwidth.** Since top-k buys no VRAM
  in dense-score mode, full routing dominates topk256 at L2048.

### Node 10 — L4096 admission/frontier (2026-07-27/28)
Stage A admission/memory only: token vs b75full at L4096, native B4
headline numbers, OOM = admission result. **Result: asymmetry confirmed —
token 33745.8 MiB vs b75full 24910.2 MiB (−26.2%)**; the b75 memory edge
grows with L (−12.5% L1024, −21.1% L2048, −26.2% L4096). Neither row
fits Blue's 24 GB at L4096 — scientific rows are Lizmark-only. Stage B
(scientific rows, seeds 0-2, 4000 updates) is gated on user
confirmation of the admission numbers.

## 3. How the nodes connect (the logic of the path)

Calibration (N1) found the failure → fiber probes (N2-N3) separated
reachability from learning → mechanistic probes (N4) split the remaining
failure into routing sparsity / supervision sparsity / pooling, each with
a dedicated control → the combination of fixes (N5) established the
quality regime, and 3-seed discipline calibrated our noise floor → only
then did frontier questions become meaningful: blur (N6) found the
memory-quality sweet spot, top-k (N7) mapped the bandwidth dimension,
scale-up (N8-N9) tested transfer and exposed budget as a confound →
L4096 (N10) tests whether the mechanism keeps paying at scale. Each node
either eliminated a hypothesis or converted a concern into a measured
parameter. No dead ends were wasted: the OOM (N2) forced the dense-score
mode that the entire frontier line now runs on; the seed-noise surprise
(N5) set the n=3 standard that made N8 interpretable; the budget surprise
(N9) retroactively flagged N7's curve as part training-speed.

## 4. Discoveries to date (the durable list)

1. **Reachability is a config-level property, not an architectural
   limit.** `endpoint_window` cannot do global aggregation; `all_past`
   can. Neither says anything deep about set attention per se.
2. **Final-token-only supervision interacts with routing topology.**
   Token attention tolerates sparse supervision (direct gradient routes
   to all positions); set attention with sparse fibers does not. Prefix
   supervision repairs it.
3. **Routing bandwidth is a capacity dimension.** Global counting needs
   many weak evidence paths; top-k selection discards them (monotone
   curve, genuine deficit at matched budget), and also trains slower.
4. **Blur allocation is a memory/quality knob with an interior
   optimum.** b75 (75% coarse heads) beats both finer and coarser
   allocations on quality while reducing VRAM — coarse atoms are better
   evidence units for counting AND shrink the O(L²) score tensor.
5. **In dense-score mode, top-k is memory-neutral.** Sparse routing only
   saves memory if scoring itself is sparse; otherwise full routing
   dominates it outright.
6. **Budget confounds are real and asymmetric.** Token saturates by 2000
   updates at L2048; the set row needs ~2×. Any set-vs-token comparison
   at fixed short budget risks calling undertraining an architecture
   deficit.
7. **The set row is less seed-stable than token** (sd 0.037 vs 0.007 at
   L2048/4000upd) even when its mean matches token's regime. Origin
   unknown — top candidate is routing/pooling optimization noise, not
   data.
8. **The current Pareto claim (defensible phrasing):** b75 full routing
   delivers mean matched-regime quality (within ~0.033 of token) at
   −12.5% VRAM (L1024) and −21% VRAM (L2048), with higher optimization
   variance in the set row.

## 5. Open threads (status)

- **L4096 frontier**: Stage A admission in flight; Stage B gated.
- **Seed-variance origin**: unexplained; candidates are router/pooling
  optimization noise vs pooling set-state variance. No probe yet.
- **Pooling isolation** (mean vs soft-trimmed-Boltzmann vs oracle):
  deprioritized — the gap it was meant to explain shrank to ~0.03.
- **Sparse scoring** (making scoring itself sparse/chunked so bandwidth
  and memory decouple): not implemented; the real lever if L4096 shows
  dense scores dominating memory.
- **Highway / hybrid token path**: deferred by explicit directive;
  must never be used to rescue set-only claims.
- **Learnable pooling alpha**: banned until the past instability is
  understood (fixed alpha only).
- **Budget at L4096**: 4000 upd worked at L2048; L4096 may need more —
  check curves before any conclusion.

## 6. Retrospective: why top-k was never swept before the matrix

Recorded 2026-07-27 (user question). Four compounding reasons:

1. **Frozen default, not a design choice.** `router_topk: 16` sits in
   the earliest set-only configs (`configs/set_only/wikitext2_*.yaml`)
   and was inherited unchanged into every registered matrix row. The
   matrix swept blur × L × B × seeds; router parameters were part of the
   architecture definition, not the design space.
2. **The matrix's tasks couldn't see it.** WikiText LM perplexity and
   MQAR are dominated by local structure; k=16 routed sets per query is
   ample for next-token prediction. No matrix metric produced any signal
   that would motivate a top-k sweep. The blindness was a task-coverage
   gap: nothing in the suite stressed global aggregation until LCA was
   built for exactly that purpose.
3. **It was also a feasibility constraint.** The default score mode was
   `candidate_gather`, where router memory grows with k via the gathered
   `[B,H,T,C,d_phi]` tensor — large k with all_past is the exact
   configuration that OOM-censored Node 2. Small k looked like prudent
   engineering. Only the dense-score mode (adopted under OOM pressure)
   made full routing affordable — and revealed that k matters.
4. **No predictive theory existed — and still doesn't, quite.**
   `docs/theory/multiresolution_formal_model.md` bounds routing entropy
   (`H ≤ log min(c_t,k)`) but explicitly notes top-k "does not, by
   itself, bound the matrix rank by k". There is no capacity theorem
   mapping k to aggregation error for this architecture. The entropy
   bound is suggestive in hindsight (k=16 caps each query at 4 bits of
   routing choice), but connecting it to global-aggregation sample
   efficiency required an empirical task that separates bandwidth from
   reachability — which is what LCA provided. A formal capacity/bandwidth
   result is a genuine open theory item, not something we overlooked in
   the literature of this repo.
