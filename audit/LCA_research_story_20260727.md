# LCA Research Story: From the Paper's Results to the L4096 Frontier

Status document, created 2026-07-27, updated 2026-07-29 (unreported
Codex-era bridge section I.3 added; Stage A results, Stage B seed0
results, plan motivations). Purpose: keep the full
experimental pathway traceable — every question posed, every answer
obtained, how the nodes connect, what remains open. Authoritative numbers
live in `audit/LCA_calibration_20260718.md`; this document is the
narrative map and continues the paper's story from where its Results
section ended.

# Part I — What the paper established, and why everything after it exists

Source: `out/final_paper_bundle/overleaf_ready/example_paper.tex`,
Results section (and its Future Work), produced in the prior Codex
session. All numbers below are the paper's own.

## I.1 What the paper's Results showed

**The exact-dense multiresolution set-dictionary matrix** (WikiText-2,
D=384, 6 layers, 8 heads, 10 epochs, seeds 0-4, strict endpoint
scanning; set rows split heads between fine (w,s)=(2,1) and coarse
(4,2) atoms: b0 all-fine ... b100 all-coarse):

1. **Interior blur optimum.** All-fine is rarely the best set row: b25
   is the lowest-PPL set row in 9/10 token-comparable islands and the
   best overall row in 8/10 (e.g. L2048/B4: b25 916.4±27.1 PPL @ 18117
   MiB vs token 942.8±15.6 @ 18633 MiB; L4096/B3: b25 864.6±21.7 @
   35365 vs token 909.3±28.8 @ 37955). A mean quality–memory frontier
   result, with overlapping 5-seed intervals — not a theorem.
2. **The blur curve.** b50/b75/b100 trade quality for memory in the
   expected direction; b100 is the low-memory/high-PPL boundary
   condition. Consistent with the exact-dense memory theorem's
   coefficient story: blur shrinks the set–set score coefficients, the
   backend stays dense.
3. **The capacity boundary.** At L4096/B4 the registered token, b0, and
   b25 cells have NO endpoint-valid rows (censored feasibility
   observations), while b50 trains (893.2±39.3 @ 41387 MiB). Heavier
   blur extends the feasible region when score memory becomes the
   limiting resource.
4. **Fine/coarse groups learn different routing regimes** (group
   diagnostics), and the memory frontier is carried by the group whose
   removal most damages the model (mechanism sweeps).
5. **MQAR is null.** The registered MRP-3 synthetic associative-recall
   matrix (18 rows, L2048/B4, 3 seeds) completed with every row near
   chance; frozen b25 query accuracy 0.000247 vs the 0.90 support
   threshold. Descriptively: token best loss, b25 best set loss/VRAM
   compromise; but NO recall-specialization mechanism claim is allowed.
6. **The SKA legacy line converges worse** (30-epoch check: the gap is
   not a short-training artifact), and the capacity ablations rule out
   the simplest gap explanations (d_phi, set-state width, set-processing
   depth, anchoring signal, candidate fiber width).
7. **The naive hybrid bridge fails as configured.** The paper's own
   hybrid token–set pilot (shared token stream, sparse local-band,
   L512): best pattern 2393.4 PPL vs 742.4 sparse-token control.
   Token-layer bridging alone did not rescue the set stack.

**The paper's stated claim boundary:** no subquadratic scaling, no
universal token superiority, no token-equivalence of b0, no
associative-recall mechanism. Contribution: a causal set-mediated
architecture, a mean quality–memory frontier in several islands,
feasibility extension at the memory boundary, and diagnostics of when
topology/pooling/interface help or fail.

## I.2 The blind spots the paper could not see — and itself flagged

Every set row in the paper's matrix ran the same frozen routing
configuration: `endpoint_window` candidate fiber, `router_topk=16`,
`router.score_mode=candidate_gather`, final-token LM loss. Two
consequences, one recognized and one not:

1. **Recognized (paper's own Future Work):** "Set states should also be
   evaluated directly as compressed memories, not only through
   next-token perplexity... memory and retrieval continuation tasks...
   can test whether a set state preserves the right information under
   compression." Perplexity is dominated by local context; a global
   aggregation failure would be invisible in it. The only non-LM probe
   in the paper (MQAR) returned null at chance — so after the paper,
   there was NO evidence about what set states actually retain of
   globally distributed information.
2. **Not recognized (discovered later):** the frozen routing config is
   precisely a configuration that cannot aggregate globally — as the
   LCA line would prove. `endpoint_window` gives each query ~2
   candidate atoms; topk16 caps routing bandwidth; final-token loss
   gives one supervised position whose reachable set under that fiber
   is a tiny local window. The paper's frontier was measured entirely
   inside this local regime.

**The motivating question for everything below** (MRP-lca-cmp, branched
2026-07-18): does the set-dictionary pathway retain and aggregate
evidence distributed across the whole context — and if it fails, where
exactly is the loss: reachability, routing bandwidth, supervision,
pooling, or the architecture itself? And if it can be made to succeed,
does it preserve the paper's memory frontier while doing so? The LCA
(long-context aggregation) task was built as the minimal decisive
probe: count markers distributed over L∈{1024,2048,4096} positions,
bucketed answer, synthetic, seed-deterministic, matched token baseline.

## I.3 The unreported bridge (2026-07-08 → 2026-07-18)

The 2026-07-08 "freeze" was a *results* freeze (the 255-row paper5
bundle, `audit/SD_dense_paper5_final_20260708.md`); the Codex session
kept working for ten more days, and the paper `.tex` kept absorbing
results through ~07-15. This section reports that period from the
session rollout and the audit trail — including the results that never
made it into the paper — because the LCA line is its direct
continuation, not a new departure.

**Work absorbed into the paper during the bridge (for completeness):**

1. **Short-B3 bridge extension** (`sd_grid_seeded_v1` namespace,
   2026-07-10 → 07-13): the missing L512/B3 and L1024/B3 islands,
   60 cells, seeds 0-4, token + b0..b100, split blue/lizmark in
   accelerated co-resident mode. Both islands 30/30 endpoint-valid. b25
   was the best mean-PPL set row at BOTH islands (L512/B3:
   943.7±26.4 @ 3272.8 MiB vs token 1048.0±30.0 @ 3195.0; L1024/B3:
   969.2±41.9 @ 6250.3 vs token 988.0±23.8 @ 6207.8) — supporting the
   interior-optimum claim and adding the batch-sensitivity nuance the
   paper now states. Caveat recorded: co-resident runs give valid
   PPL/per-process peak but not exclusive-capacity wall-time.
   Evidence: `audit/SD_short_b3_bridge_20260713.md`.
2. **MRP-3 MQAR primary completion** (2026-07-09 → 07-15): after a
   calibration incident (runner evaluated only at endpoint; patched to
   the registered 500-update cadence —
   `audit/incident_mrp3_calibration_eval_cadence_20260708.md`) the LR
   was frozen at 1e-3/12500 updates. The 18-row primary matrix
   (interrupted by a blue shutdown at 8/18, resumed 07-13) completed
   null: all rows at chance (b25 0.0002474 vs the 0.90 gate; 2.304M
   queries/row), MRP-4 NOT_TRIGGERED. Descriptively: token best loss
   (8.3424), b25 best set-row loss (8.4891) and cheapest-but-one VRAM.
   Reported in the paper as `tab:mqar-prototype-comparison`.
   Evidence: `audit/MRP_3_mqar_mechanism.md`.

**Results that were never reported in the paper:**

3. **L4096/B4/Dkv512 MQAR capacity preflight** (2026-07-09, Lizmark,
   one update, frozen LR/budget, 6 rows): peak VRAM descends
   monotonically with blur — token 34239.0, b0 39922.4, b25 30342.1,
   b50 23539.8, b75 18919.1, b100 12880.6 MiB; no NaN/OOM. Directional
   support for the feasibility-extension claim (b25 saves ~3.9 GiB vs
   token; b100 ~21.4 GiB) but a different task/config than the WT2
   L4096/B4 censored boundary, and one-update feasibility only — it
   cannot rescue the paper's b25-at-L4096/B4 hole. This is the direct
   ancestor of our L4096 admission probe: same question (does blur
   keep buying memory at L4096?), now asked on the task that matters.
   Evidence: `audit/server_copy_provenance_20260709.md`, artifacts
   `out/mqar_capacity_preflight_L4096_B4_lr0p001_u12500/`.
4. **MRP-3 B4 common-batch preflight** (2026-07-09, minor): token
   9365.9 / b0 10951.7 / b25 8826.3 / b50 7396.5 / b75 6050.4 / b100
   3969.1 MiB — same monotone memory story at L2048. Feasibility only.
5. **MRP-2 natural AR-hit: retrained but NEVER evaluated.** No
   compatible checkpoints existed anywhere, so the registered 12-cell
   retrain (token/b0/b25/b100, L2048/B4, seeds 0-2) ran 2026-07-08 →
   ~07-10 and completed 12/12 finals (after two failed launches: HF
   cache mount, then a zero-head b0/b100 encoding bug). **The
   registered `evaluate_ar_hits.py` pass was never launched — no hit
   rates exist.** The paper's "next mechanism test" language still
   stands on nothing; MRP-5 (tokenizer-matched WT2/PG-19) remains
   blocked behind it. Evidence: `audit/MRP_2_natural_ar_hits.md`,
   `audit/phase_sd_status.md` §"MRP-2 Natural AR-Hit Launch State".

**The pivot (2026-07-16 → 07-18):** the MQAR null was read as
"token-precision recall is the wrong probe for a compressive memory —
test aggregation instead". MRP-lca-cmp was planned; grad-accumulation
and eval-microbatching were added with batching-preservation tests; a
false-start branch (`mrp-lca-cmp` from `origin/main`) was corrected to
`mrp-lca-cmp-sd` before any scientific run
(`audit/incident_branch_host_context_20260717.md`); both host checkouts
were repaired to clean `mrp-lca-cmp-sd@2ded5d1`. On 07-18 the
registered 36-row LCA calibration completed — Gate 1 PASS (token
learns), Gate 2 FAIL (all 30 set rows at chance) — which is Node 1 of
Part II. The pending debts at the moment this chat started were:
MRP-2 AR-hit evaluation (checkpoints ready), MRP-5 (blocked), PG-19
transfer table (missing), and the Gate-2 follow-up decision (executed
as Nodes 2-3 below).

**Minor provenance flags from the cross-check:** the audit names only
two of the three MRP-2 launch logs (the middle `175220` launch appears
only in the rollout); the short-B3 launch date differs by local-vs-UTC
labeling (07-10 local = 07-11 UTC); no numeric disagreements were found
between the rollout, the audits, and the artifact TSVs.

# Part II — The pathway, node by node

### Node 1 — Calibration and the Gate-2 failure (2026-07-18/19)
Registered calibration matrix (36 rows: token + b0..b100 × L{1024,2048}
× seeds 0-2, `endpoint_window` fiber, final-token loss, 2000 updates).
**Result: token 0.77 at L1024; every set row at chance (Gate 2 FAIL).**
Diagnosis: with `endpoint_window`, each query routes only to the ~2 set
atoms adjacent to its position — distant markers are topologically
unreachable from the supervised final token, and 6 layers of local
diffusion cannot bridge L=1024. The paper's frozen fiber is incapable
of global aggregation by construction. Question refined: reachability
first.

### Node 2 — Fiber probe: all_past is OOM-censored in candidate_gather
(2026-07-19). Switching `candidate_fiber=all_past` fixes reachability in
principle, but the candidate-gather router materializes
`[B,H,T,C,d_phi]` — OOM on both hosts, including microbatch retry.
Recorded as an implementation boundary, not a scientific result. (This
is exactly the routing-memory target the paper's Future Work named.)

### Node 3 — Router-dense probe: reachability alone is not sufficient
(2026-07-20). `all_past` + `router.score_mode=dense` (scores `q@k^T` as
`[B,H,T,M]`, masking invalid entries) fits (3216 MiB,
`candidate_count_max=1023`) — but does not learn (val_acc 0.498 vs
token 0.77). **Topology fixed; learning still absent.**

### Node 4 — Mechanistic probes: localizing the mismatch (2026-07-20)
Three cheap probes at b25/L1024:
- P1 full routing (`router_topk=1023`): **0.827** — top-k=16 sparse
  selection was a bottleneck.
- P2 prefix supervision (every position predicts its prefix count):
  **0.787** (token 0.903) — final-token-only supervision was an
  independent bottleneck; token attention tolerates it (direct gradient
  routes to all positions), sparse-fiber set attention does not.
- P3 oracle count token: **0.985** — routing/readout can solve the task
  when the count is represented; residual deficit is pooling/set-state
  learning.
**Verdict: set-path/task mismatch (supervision sparsity + routing
sparsity + pooling), NOT an inherent architectural disadvantage.**

### Node 5 — Combined probe and the seed-noise lesson (2026-07-20)
Full routing + prefix supervision composed: 0.936 seed0, apparently
above token (0.903). The prefix3 3-seed mini-calibration showed this
was seed noise: token 0.9443±0.0317, b25-full 0.8956±0.0407, b25-topk16
0.8171±0.0114. Standing lesson, applied ever since: **no single-seed
conclusions; set rows carry higher seed variance than token.**

### Node 6 — Blur frontier sweep: the first Pareto point (2026-07-24)
Blur families {b25,b50,b75,b100} × seeds 0-2, all_past + dense +
full routing + prefix, L1024. **b75: 0.9233±0.0157 @ 2346.7 MiB vs token
0.9443±0.0317 @ 2680.6 MiB — within seed noise at −12.5% VRAM.**
Quality non-monotone in blur (b75 > b50 ≈ b25 > b100): coarse atoms
both shrink the O(L^2) score tensor and are better evidence units for
counting; all-coarse loses detail. Note the inversion vs the paper's LM
matrix: there b25 was the optimum; on global aggregation b75 is —
different tasks weight resolution vs compression differently.

### Node 7 — Top-k bandwidth sweep (2026-07-24)
b75/L1024, topk={16..1023} × seeds 0-2 (21 rows): quality **monotone, no
plateau** (0.784 → 0.923). Simultaneously: **VRAM flat across top-k**
(2347-2408 MiB) — in dense-score mode the `[B,H,T,M]` tensor is
materialized regardless of k, so top-k sparsity is not a memory lever
in this router mode. Claim flagged as budget-confounded (see Node 9).

### Node 8 — L2048 pilot: the gap that wasn't (2026-07-24/26)
Seed 0 @2000 updates: token 0.944 @ 9124 MiB, b75full 0.816 @ 7201 MiB
— VRAM edge holds (−21%) but apparent quality gap (−0.128). Seeds 1-2:
b75full 0.8553±0.0483 — high seed variance, the undertraining
signature.

### Node 9 — Budget probe and confirmation: the L2048 frontier
(2026-07-26/27). Controlled pairs at 4000 updates:
- token: 0.9438 (saturated already at 2000).
- b75full seed0: 0.8158 → **0.9353** (+0.12 — the gap was mostly budget).
- b75full 3-seed @4000: **0.9078±0.0368 @ −21.1% VRAM** — Pareto claim
  holds on average; seed1 (0.866) keeps it from being a per-seed win.
- b75topk256 seed0: 0.7571 → 0.8099 (+0.053 from budget) but still
  −0.125 below full at matched budget: **the sparse-routing deficit is
  both training speed AND genuine bandwidth.** Full routing dominates
  topk256 at L2048 since top-k buys no VRAM in dense-score mode.

### Node 10 — L4096: admission and Stage B (2026-07-27/28)
**Stage A (admission/memory only, 30 updates, native B4):** token
33745.8 MiB vs b75full 24910.2 MiB — **−26.2%**, and the edge grows with
L (−12.5% L1024, −21.1% L2048, −26.2% L4096). Neither row fits Blue's
24 GB (Blue total 24564 MiB; b75 exceeds it by ~346 MiB before safety
margin). L4096 rows are Lizmark-only.
**Stage B seed0 (DONE 2026-07-28):** token 0.9407 @ 33745.8 MiB; b75full
**0.8382** @ 24915.9 MiB. Launcher/result timestamps give 42 min for token
and 4 h 55 min for b75full, so the practical throughput was about 7x slower
for b75 on this host. These are external launcher timestamps, not
framework-internal timers; the training loop currently logs update-indexed
loss curves but no per-update wall-clock column. Token is L-insensitive as
before (0.9443/0.9407/0.9407 at L1024/2048/4096). b75full drops vs its L2048
seed0 (0.9353 → 0.8382), with its train-loss curve still descending at
update 4000 (1000-update means: 0.5208 → 0.3591 → 0.2864 → 0.2486).
**Budget extension (2026-07-29, b75full8k seed0): endpoint 0.7570/
val_loss 0.896 at 8000 updates** while train loss kept descending
(0.2486 → 0.1907 over updates 4000-8000, replicating the 4000-upd curve
exactly through update 4000). This was initially recorded as an
overfitting verdict — and RETRACTED two days later (below).
**Trajectory probe verdict (2026-07-30/31, l4096tj seed0, periodic
eval every 500 upd; commits 9ab8104+25481f3): NOT overfitting —
endpoint oscillation.** The b75 val trajectory oscillates with
amplitude ~0.15 across the whole run (0.650, 0.694, 0.609, 0.872,
0.765, 0.728, 0.854, 0.838, 0.741, 0.749, 0.788, 0.928, 0.932, 0.863,
0.932, 0.927; val_loss range 0.155-1.64). Token oscillates too
(0.825-0.971). Val N=2048 (binomial sd ~0.011), so the swings are real
model behavior under constant lr=1e-4, not eval noise. The stageb8k
endpoint 0.7570 was a TROUGH sample; the trajectory endpoint 0.9269 is
a near-peak sample — neither is a reliable estimator. Consistency
checks: eval @4000 bitwise-matches the stageb 4000-upd endpoint
(0.838196) for b75 and token (0.940659); the two b75 8k train curves
are bitwise-identical through update 5000 and diverge at 5001, the
epoch-2 boundary (20000/4 = 5000 upd/epoch), via the reshuffle
permutation RNG draw — periodic-eval runs do not reproduce
endpoint-only runs past an epoch boundary. Endpoints: token @8000 =
0.9603/0.0934 @ 33745.8 MiB (still improving; no-overfit control
confirmed); b75full @8000 = 0.9269/0.1628 @ 24915.9 MiB. b75 peaks:
0.9319 @7500, 0.9318 @6500, 0.9282 @6000. Revised L4096 picture:
matched-regime gap ~0.03-0.04 (b75 best 0.932 vs token best 0.971) at
−26.2% VRAM — consistent with L1024/L2048, NOT a breakdown at scale.
The binding issue at L4096 is estimator instability (endpoint-only
validation is unreliable; both rows oscillate, the set row more so),
plus epoch-2 data-order sensitivity — not generalization. Operator
work returns to "motivated, not urgent"; the top recipe lever is now
an lr-schedule probe and best-of-trajectory reporting; the 40k
data-scale probe is deprioritized (its overfitting premise is gone);
L4096 seeds 1-2 WITH periodic eval are now justified.

# Part III — Why Stage B: the theoretical and empirical motivation

**Theoretical.** The paper's exact-dense memory analysis says blur
reduces score-memory coefficients but keeps a dense backend; token
attention materializes ~H·L² score elements while the b75 set router
materializes ~Σ_g H_g·L·M_g with M_g = L·s_g/w_g — for b75 (2 fine
heads at M≈L, 6 coarse at M≈L/2) the coefficient is ~5/8 of token's.
As L grows, score memory dominates total memory, so the b75/token peak
ratio should fall toward that asymptotic coefficient ratio — the
measured −12.5% → −21.1% → −26.2% trend is this prediction
materializing. L4096 is the first island where the asymptotic regime is
the dominant term, and the first where token is hardware-excluded from
the 24 GB host. It is also exactly the paper's censored boundary
(L4096/B4 token had no valid row in the LM matrix) — the regime where
set attention must win if the frontier story is true.

**Empirical.** Quality parity (within noise) is established at L1024
and, after the budget correction, at L2048 — but under a config the
paper never ran (all_past + dense scores + full routing + prefix
supervision). Nothing guarantees parity transfers to L4096: the
aggregation task gets harder with L (same markers, more dilution), the
set row's higher seed variance could widen, and token's quality is
L-insensitive so far (0.9443 / 0.9407 / 0.9407 at L1024/2048/4096).
Stage B tests the conjunction that makes the frontier claim meaningful:
**b75 holds quality parity at the L where its memory advantage is
largest and token approaches hardware exclusion.**

# Part IV — Experimentation plan, with step-wise motivations

1. **Stage A — admission (DONE).** Motivation: never spend training
   compute on an island where the memory premise is untested. Cost: 30
   updates. Answer: asymmetry real (−26.2%), Lizmark-only.
2. **Stage B seed0 (DONE).** Motivation: quality-parity transfer +
   practical throughput measurement before committing a matrix. Answer:
   token 0.9407 (42 min by launcher/result timestamps); b75full 0.8382
   (4.9 h by the same source) with its train loss still descending at
   update 4000 — interpretation blocked on budget, exactly as at L2048.
   These are external wall-clock measurements, not framework-internal timers.
2b. **L4096 budget extension (DONE — initial verdict overfitting,
   RETRACTED by 2c.1).** b75full seed0 at 8000 updates (endpoint-only):
   val_acc 0.7570, val_loss 0.896 while train loss kept descending. Now
   known to be a trough sample of the validation oscillation, not a
   generalization trend — see 2c.1.
2c. **Generalization diagnostics.** (1) periodic-eval support +
   trajectory probe: DONE 2026-07-30/31 (commits 9ab8104+25481f3,
   l4096tj b75full+token seed0, 8000 upd, eval every 500). Outcome:
   NO overfitting — the val trajectory oscillates ±0.15 for both rows;
   b75 reaches 0.932 best / 0.9269 endpoint vs token 0.971/0.960 at
   −26.2% VRAM; endpoint-only validation is an unreliable estimator at
   L4096; epoch-2 data-order sensitivity identified (runs bitwise-
   identical through update 5000, diverging at the reshuffle). (2)
   data-scale probe (40k): DEPRIORITIZED — its overfitting premise is
   gone. (3) regularization probe: reframed — the open recipe lever is
   the lr schedule (constant lr=1e-4 underlies the oscillation), plus
   best-of-trajectory / last-k-mean reporting instead of endpoint.
3. **Stage B seeds 1-2 (DONE 2026-08-01, all-Lizmark, host-consistent).**
   b75full + token seeds 1-2, L4096/prefix/B4, 8000 upd, eval every 500.
   3-seed L4096 statistics: b75 endpoints 0.9077+-0.0191, best-of-
   trajectory 0.9222+-0.0085; token endpoints 0.9344+-0.0276, best
   0.9702+-0.0010. Token's reachable ceiling is seed-deterministic
   (~0.97 every seed; its endpoint sd 0.028 is pure oscillation phase);
   b75's ceiling is 0.916-0.932. Gap ~0.027 at endpoints, ~0.048 at
   best-of-trajectory, −26.2% VRAM. Endpoint-only validation at L4096
   is phase luck for BOTH families (token seed1: 0.9054 endpoint vs
   0.9702 best).
4. **Sum-routing probe (motivated, NOT approved).** P3 + the bandwidth
   curve imply the aggregation operator matters: counting is a sum, but
   softmax routing computes a normalized average. A sum/sigmoid-gated
   readout variant (small router change, still set-only) at L1024 tests
   the linear-accumulator hypothesis for a few GPU-hours. If it matches
   full softmax routing at lower bandwidth, the operator diagnosis is
   confirmed mechanistically.
5. **Sparse/chunked scoring (motivated, NOT approved).** The only real
   memory-bandwidth decoupling lever: top-k is memory-neutral in
   dense-score mode, so sparsity must move into scoring itself. Needed
   when dense scores become the binding memory term beyond L4096.
6. **Hybrid linear-sparse branch (deferred by directive).** The user's
   linear-sparse intuition is supported by our evidence (see discoveries
   3, 5, 8) — but the paper's own hybrid pilot failed naively (2393 vs
   743 PPL), so any hybrid must be motivated by the aggregation-operator
   diagnosis (step 4), not by bridging alone; explicit new branch, own
   memory accounting, ablations highway off / pointwise / linear
   channel / sparse retrieval; never used to rescue set-only claims.
7. **Seed-variance origin probe (open).** The set row's higher seed
   variance is unexplained; candidates: router/pooling optimization
   noise vs set-state variance. No probe designed yet.

# Part V — Discoveries to date (the durable list)

1. **Reachability is a config-level property, not an architectural
   limit.** `endpoint_window` cannot do global aggregation; `all_past`
   can. The paper's frozen fiber was the binding constraint.
2. **Final-token-only supervision interacts with routing topology.**
   Token attention tolerates sparse supervision; sparse-fiber set
   attention does not. Prefix supervision repairs it.
3. **Routing bandwidth is a capacity dimension.** Global counting needs
   many weak evidence paths; top-k selection discards them (monotone
   curve, genuine deficit at matched budget) and also trains slower.
4. **Blur allocation has an interior, task-dependent optimum.** b75 on
   global aggregation vs b25 on LM perplexity — resolution and
   compression trade differently across task classes; coarse atoms are
   better counting units AND shrink the O(L^2) score tensor.
5. **In dense-score mode, top-k is memory-neutral.** Sparse routing
   only saves memory if scoring itself is sparse; otherwise full
   routing dominates it outright.
6. **Budget confounds are real, asymmetric — and budget effects are
   estimable only with periodic validation.** Token saturates by 2000
   updates at L2048; the set row needs ~2x. Fixed short budgets risk
   calling undertraining an architecture deficit. The L4096 trajectory
   probe adds the mirror-image lesson: endpoint-only validation at a
   long budget risks calling an oscillation trough overfitting.
7. **The set row is less seed-stable than token — with a host-mixing
   caveat.** The L2048/4000upd numbers (sd 0.037 vs token 0.007) come
   from a host-mixed set: b75 seed 0 ran on Lizmark (0.9353), seeds 1-2
   on Blue (0.8660, 0.9221), and a Blue seed0 rerun ends at 0.8779 —
   0.057 below the Lizmark row at identical config/seed/budget.
   Cross-host numerics diverge (same-host replication is bitwise), so
   L2048 seed variance is confounded with host; the L4096 3-seed set is
   all-Lizmark and clean.
8. **The memory edge grows with L as the coefficient story predicts**
   (−12.5% / −21.1% / −26.2% at L1024/2048/4096), and token becomes
   hardware-excluded first. The frontier value of set attention
   increases exactly where dense token attention stops fitting.
9. **Both rows oscillate under constant lr at L2048 AND L4096; the
   troughs track the data sequence; dropout amplifies.** Endpoint-only
   validation is an unreliable estimator: the b75 val trajectory swings
   ~0.145 at L2048 and ~0.15 at L4096; token also swings (0.825-0.971
   at L4096). Val N=2048, so the swings are real model behavior, not
   eval noise. Cosine decay neither damps the oscillation nor changes
   the endpoint (lr rejected as cause). Rows sharing seed 0's batch
   order (const, cosine, nodrop) all dip in the 1500-2000 region, while
   a seed-3 row with a different order dips elsewhere — the troughs are
   data-sequence driven, not update-indexed dynamics. Dropout=0 does
   not remove the oscillation but raises its floor (0.735 -> 0.819),
   its mean (+0.070), and its ceiling (0.880 -> 0.9412 at L2048), and
   cuts peak VRAM by 25% (7201 -> 5388 MiB). At L4096, two same-seed
   8000-update runs are bitwise-identical through update 5000 and
   diverge at the epoch-2 reshuffle, landing on different oscillation
   phases (0.7570 vs 0.9269 endpoints). There is no monotone validation
   degradation with budget. Addendum 2026-08-04: the dropout=0 ceiling
   lift is confirmed seed-stable at L2048 (3 seeds, all-Blue, with
   dropout-free token controls — see Part VI); the L2048 memory edge
   shrinks to -7.5% under dropout=0 because token VRAM drops more
   (-36% vs b75's -25%), so the decisive memory comparison moves to
   L4096.
10. **The current Pareto claim (defensible phrasing):** b75 full routing
   delivers matched-regime quality at −12.5% VRAM (L1024: 0.9233±0.0157
   vs 0.9443±0.0317), −21.1% VRAM (L2048: 0.9078±0.0368 vs 0.9438 —
   host-mixed, see discovery 7), and −26.2% VRAM (L4096, 3 seeds,
   host-consistent: endpoints 0.9077±0.0191 vs 0.9344±0.0276,
   best-of-trajectory 0.9222±0.0085 vs 0.9702±0.0010). The L4096 gap is
   ~0.03 at endpoints and ~0.05 at best-of-trajectory — wider than at
   L2048 under the more reliable estimator. Token's reachable ceiling is
   seed-deterministic (~0.97); b75's is 0.916-0.932.

# Part VI — Open threads (status)

- **Stage B seed0 b75full + seeds 1-2**: COMPLETE (2026-08-01). L4096
  3-seed set, all-Lizmark, with periodic eval: b75 endpoints
  0.9077+-0.0191 / best 0.9222+-0.0085; token endpoints 0.9344+-0.0276 /
  best 0.9702+-0.0010. Token ceiling seed-deterministic; b75 ceiling
  0.916-0.932. No further L4096 seed rows pending.
- **Generalization diagnostics**: trajectory probe DONE (no
  overfitting — oscillation). Data-scale (40k) DEPRIORITIZED (premise
  gone). lr-schedule probe DONE 2026-07-31 (l2048lr, Blue): cosine
  decay does NOT damp the oscillation or change the endpoint at L2048
  — constant lr rejected as the cause. Oscillation-origin probes DONE
  2026-08-03 (l2048osc, Blue): trough positions track the data sequence
  (seed0-order rows all dip at updates 1500-2000; seed3 with a
  different order dips elsewhere) — update-indexed dynamics rejected;
  dropout does not create the oscillation but amplifies it, and
  dropout=0 lifts the b75 trajectory mean by +0.070, the ceiling from
  0.880 to 0.9412, and cuts peak VRAM from 7201 to 5388 MiB (-25%).
  b75-nodrop best 0.9412 ≈ token reference 0.9438 (host-inconsistent).
  Confirmation wave DONE 2026-08-04 (l2048nd, all-Blue, 3 seeds,
  dropout-free token controls): b75nodrop endpoints 0.9090+-0.0195 /
  best 0.9358+-0.0067 @ 5387.6 MiB; tokennodrop endpoints
  0.9298+-0.0406 / best 0.9617+-0.0067 @ 5827.7 MiB. The ceiling lift
  is CONFIRMED and seed-stable (all seeds best 0.928-0.941); the L2048
  gap narrows to ~0.021 endpoints / ~0.026 best. DROPOUT=0 IS THE
  DEFAULT SET RECIPE on this task family. CAVEAT: the L2048 memory
  edge shrinks to -7.5% under dropout=0 (token VRAM also drops, -36%)
  — activation memory dominates at L2048; the L4096 dropout=0
  re-measurement is the decisive frontier row.
- **Host-consistency rule (new)**: cross-host same-seed runs diverge
  (~0.06 at L2048); same-host replication is bitwise. All future
  multi-seed LCA rows are pinned to one host per island. The L2048
  3-seed set is host-mixed (caveat in discovery 7); the L4096 3-seed
  set is all-Lizmark.
- **Endpoint-only validation measurement gap**: CLOSED 2026-07-30 —
  the runner now supports `training.eval_every` with an evalcurve
  sidecar (commit 9ab8104, smoke-verified). All future long-budget LCA
  rows should use it.
- **Seed-variance origin**: unexplained; no probe yet. The L4096
  oscillation and epoch-2 data-order sensitivity are likely related
  variance sources.
- **Pooling isolation** (mean vs soft-trimmed-Boltzmann vs oracle):
  deprioritized — the gap it was meant to explain shrank to ~0.03.
- **Sum-routing / sparse scoring / hybrid branch**: motivated
  (Part IV 4-6), awaiting user approval.
- **Learnable pooling alpha**: banned until the past instability is
  understood (fixed alpha only).
- **Legacy debts surfaced by the bridge audit (I.3), outside the LCA
  line but tracked**: MRP-2 natural AR-hit evaluation (12/12
  checkpoints ready since ~2026-07-10, `evaluate_ar_hits.py` never
  launched — no hit rates exist); MRP-5 (tokenizer-matched WT2/PG-19)
  blocked behind MRP-2; PG-19 transfer table missing from the paper.

# Part VII — Retrospective: why top-k was never swept before the matrix

Recorded 2026-07-27 (user question). Four compounding reasons:

1. **Frozen default, not a design choice.** `router_topk: 16` sits in
   the earliest set-only configs and was inherited into every registered
   matrix row; the matrix swept blur × L × B × seeds, never router
   parameters.
2. **The matrix's tasks couldn't see it.** WikiText PPL and MQAR are
   dominated by local structure; k=16 is ample there. A task-coverage
   gap: nothing stressed global aggregation until LCA was built for
   exactly that purpose — as the paper's own Future Work prescribed.
3. **It doubled as a feasibility constraint.** In `candidate_gather`
   mode router memory grows with k (`[B,H,T,C,d_phi]`); large-k all_past
   is exactly what OOM-censored Node 2. Only the dense-score mode
   (adopted under OOM pressure) made full routing affordable — and
   revealed that k matters.
4. **No predictive theory existed — and still doesn't, quite.**
   `docs/theory/multiresolution_formal_model.md` bounds routing entropy
   (`H <= log min(c_t,k)`) but explicitly notes top-k "does not, by
   itself, bound the matrix rank by k". No result maps k to aggregation
   error for this architecture. A formal bandwidth/capacity theorem is
   a genuine open theory item, now with an empirical curve to explain.
