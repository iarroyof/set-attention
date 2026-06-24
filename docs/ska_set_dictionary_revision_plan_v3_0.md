# SKA — Causal Set-Dictionary Revision Plan (v3.0 LOCKED)

Status: locked for the **set-dictionary branch only**. Supersedes decision **D1 (R1 direct
embedding residual)** of `docs/ska_pat_feedback_revision_plan_v2_6_locked.md` (v2.7) for work
on this branch. All other v2.7 locked decisions (Option-1 strict-past, T1 tail policy, landmark
backend, config stack, LR-norm + anchor topology references, provenance discipline) remain in
force and are NOT changed here.

Prerequisite is **experimental, not a git merge**. The candidate-gather (Redundancy-1) code already
lives in `paper/final-results-bundle` (default `router_score_mode`); A9 validated it (DONE) but showed
**no short-context VRAM win at `L=512`** — expected, because at small `M` the router tensor is not the
peak driver (see §1). Its remaining R1 confirmation is (a) a per-seed dense-vs-gather `allclose`
exactness test and (b) a long-context (`L=2048/8192`) VRAM re-measure. Commit candidate-gather first as
`a9/candidate-gather-router` off `origin/paper/final-results-bundle`, then create
**`set-dictionary/anchor-span`** off that tip. No merge to `main` while the paper is in progress.

---

## 0. Required context files — read in this order (onboarding)

1. `set_attention_agent_onboarding.md` — vendor-agnostic onboarding instrument (roles, env, trackers).
2. `docs/ska_pat_feedback_revision_plan_v2_6_locked.md` — prior locked plan (still in force except D1).
3. `docs/revision_source_of_truth_definitions.md` — code-backed definitions/values (M, candidate fiber, landmark indices, pooling alpha, router min_temp).
4. **This file** — supersedes D1 for the branch; defines the set-dictionary architecture, losses, variants, DoD.
5. `Context For Revision Agent after NeurIPS2026 LLM feedback.md` — execution env, blue-demon workflow, causality finding.
6. `configs/hyperparameters.md` — public config contract; extend it with the new keys in §6.
7. `audit/phase_sd_status.md` — **new** master tracker for this branch (create on first session; same protocol as `audit/phase_a_status.md`).

Do not fill project-specific values from memory; always read them from files 2–3 and 6.

---

## 1. Motivation (one paragraph)

The reported experiments show matched token-attention baselines beat Set Attention (SKA) on
WikiText-2 PPL at every tested operating point; SKA only wins on long-context VRAM. The audit
attributes the near-token VRAM overhead to carrying **both** a token stream and a set stream plus
routing tensors (Redundancy 2), and the hybrid pilot shows repeated set compression overfilters
(Redundancy 3). This branch reframes the set states as a **causal learned dictionary** (representer-
theorem intuition: token-predictive states are *spanned* from a sparse set basis through causal
routing weights), removes the parallel token tower, and adds a direct learning signal that tells the
basis which predictive information to preserve under compression.

A9 (candidate-gather, R1) confirms the attack order: at `L=512` it produced **no** VRAM win, which is
evidence that the short-context peak is dominated by the dual token+set stream this branch removes, not
by the router tensor. R1's win is long-context; R2 (this branch) is the short-context lever.

---

## 2. Locked framing correction (must hold in code, theory, and paper)

The task remains **token-level causal language modeling**: the model predicts
`p(x_{t+1} | x_{<=t})`. What changes is the *representation path*, not the prediction granularity.

> Set states are learned **causal dictionary atoms** from which the token-level predictive state is
> generated via sparse candidate routing. The objective stays token-level next-token CE.

Do **not** describe this as "set-level prediction replacing token-level causality." The correct phrase
is **set-mediated token-level causal prediction**. (Naming: use "causal dictionary atoms / set basis /
landmark memory," not "support vectors" — there is no margin/KKT object. SVM is intuition only.)

---

## 3. Locked architectural decisions for this branch

### D1′ (supersedes D1/R1): single-stream thin-anchor + multi-head span

There is **no parallel token tower** in the forward prediction path. The final token state is

```text
f_t = A_t + span_t            (output_residual_mode = "anchor_span")
span_t = W_o * sum_{m in C_t} alpha_{t,m} * z_m^(K)     # = existing router output (routed_repr)
A_t    = thin anchor (token identity carrier)
```

- `span_t` is exactly the current multi-head router output (`routed_repr` at
  `src/models/set_only/set_only_lm.py:579`; `W_o` = `set_output_proj`). This is the fair analogue of
  multi-head attention's weighted value sum — expressivity lives here.
- `A_t` is the thin anchor: **`emb(x_t) + pos_t` only**. No `token_mlp`, no cross-layer update, no
  attention. It is strictly weaker than a token transformer's residual stream (which accumulates
  contextual updates across layers), so it cannot encode context and cannot be a silent advantage.
  Its sole job is to give the decode token-resolution degrees of freedom so the span has a
  position-specific signal to correct (this is what prevents the reconstruction floor from making the
  anchoring loss inert).
- For `C_t = 0` (empty strict-past fiber), `span_t = 0`, so `f_t = A_t`. Do not special-case warmup
  with a different formula.

Rationale for fairness (must be auditable, see §5): expressivity = multi-head span (analogue of MHA);
identity = thin anchor (≤ token-attention residual). The anchor is a handicap relative to token
attention, not a freebie.

### D-causal: collapse the two causality flags into one source of truth

The standalone `causal` constructor arg (which gates the set-attention causal mask at
`set_only_lm.py:554`) is **removed as a user-facing knob**. Causal masking of set self-attention is
**derived**:

```python
self.causal = (self.set_causality_mode == "strict_past")
```

- `set_causality_mode` is the single source of truth (`strict_past` | `noncausal`).
- Drop `causal` from the config schema. If a legacy config still passes `causal`, emit a deprecation
  warning and let `set_causality_mode` win (never let `causal=False` silently disable masking under
  `strict_past`). The set-attention causal mask is endpoint-monotonic already
  (`set_positions = arange(m)` with strictly increasing starts/endpoints, `banks.py:193`), so deriving
  the flag closes the leakage footgun by construction.

### D-anchor-loss: predictive reconstruction target (teacher-free, inference-dropped)

Add an auxiliary anchoring loss that forces the span to reconstruct a **causally contextualized,
predictive** target:

```text
L_anchor = (1/L) * sum_t || LN(span_t) - sg(LN(h_t*)) ||_2^2
h_t*     = hidden state of a SHALLOW CAUSAL token pre-encoder (1–2 layers); sg = stop-gradient on the
           target; LN on both sides.
```

- **CRITICAL — the pre-encoder MUST be trained to be predictive (this was the SD-6 failure).** Give it
  its own causal LM head and an auxiliary next-token CE loss `L_CE_pre`; the span then distills from the
  detached, *trained* hidden state. If the pre-encoder receives no gradient — detached target **and** not
  in the forward path **and** no own loss, as in the first SD-6 build — then `h_t*` is a fixed RANDOM
  projection and the anchor loss is inert: `recon_error_norm` parks at √2 ≈ 1.414 (cosine ≈ 0) and the
  rescue is never actually tested. `detach_target=true` is correct **only because** `L_CE_pre`
  independently trains the teacher; without `L_CE_pre`, detaching freezes the teacher at init.
  - Optional fast sanity variant (parameter-free predictive target): `h_t* = detach(emb(x_{t+1}))`, the
    next-token embedding. If `recon_error_norm` drops here but not with the pre-encoder, the pre-encoder
    is still untrained.
- The pre-encoder (and its LM head) is a **training-time auxiliary**: excluded from the inference model
  and from inference param/VRAM/FLOP accounting. It is the lightweight, in-model alternative to a full
  external teacher.
- Do **not** anchor to `token_mlp(emb+pos)` or to `emb+pos`: those targets measure pooling
  invertibility, not predictive usefulness, and were rejected for unfair token-attention alignment.
- **External teacher + KL logit distillation is explicitly deferred to Future Work** (cost). Leave a
  disabled, documented stub (`anchor.teacher.enabled = false`) so the path is reserved but inert.
- Causality requirement: `h_t*` must come from an **autoregressive** (causal) pre-encoder. A
  bidirectional/MLM target leaks future through the regression target even though routing is
  strict-past; `sg` does not protect against this (it controls gradient, not forward visibility).

Primary objective stays CE. Total loss:

```text
L = L_CE + lambda_pre * L_CE_pre + lambda_h * L_anchor + lambda_div * L_div
```

with `lambda_pre ~ 1.0` (the pre-encoder is a small shallow LM; it should train freely). **Anchor
validity guard (DoD): an anchoring run is interpretable only if `recon_error_norm` falls meaningfully
below the √2 ≈ 1.414 random-vector baseline (target: < ~1.2 with a downward trend). A flat
`recon_error_norm ≈ 1.4` means the target is non-predictive — treat the run as confounded, not as
evidence of a capacity limit.**

### D-div (optional): anti-collapse on the set Gram

Reuse `src/models/set_only/losses.py::set_diversity_loss` (do not force orthogonality; penalize
off-diagonal Gram excess / use spectral-entropy regularization). Default `lambda_div` small; this is a
guard against dictionary-atom collapse, monitored via the existing set-Gram spectral-entropy and
top-eigenvalue diagnostics.

### D-multivec (deferred; OFF by default): multi-vector basis

`r` is **not** the head count. The routing head count is `H_r = num_heads` (default 8) and is
**unchanged** by this knob. `r` is a *separate* axis: the number of value sub-vectors each set atom
contributes **per head**. `r = 1` is exactly the current/tested architecture (one value per atom per
head); `r = 2` gives each atom a 2-dim value subspace per head, raising per-token degrees of freedom to
~`H_r * min(d_head, r * C̄)`. It is orthogonal to compression topology — `r=2` does NOT reproduce the
near-2 `(w,s)=(4,2)` topology (which differs by `C̄ = w/s`). Default `r = 1` (`enabled=false`), keeping
the model fully aligned with the tested 8-head configs. Introduce `r = 2` **only** as a floor test if
the step-2 reconstruction error plateaus high, isolating whether candidate-count rank is the binding
constraint.

---

### D-fiber (default `endpoint_window`): candidate fiber width

The span's richness is bounded primarily by the **candidate count `|C_t|`, not by `r`** (the per-token
span ranges over `|C_t| · r` vectors per head; `|C_t| ≈ w/s ≈ 2` is the binding term). Fiber width is
therefore the real expressivity dial:

- `endpoint_window` (**default**): `C_t = {m : t-w < e_m <= t}`. Keeps the compression/locality story
  and the candidate-count theorem (`rank(Π_t) <= min{H_r, C_t}`); first runs use this for
  comparability with the already-run baselines (§7).
- `all_past`: `C_t = {m : e_m <= t}` — `t` spans over all sealed past atoms (the full causal
  dictionary). Strictly causal, much richer span; cost moves toward `O(L·M̄)` and the clean
  candidate-count theory weakens.
- `window_plus_landmarks`: `endpoint_window` plus a few always-valid global atoms per token — cheap
  middle ground that breaks strict locality without abandoning it.

Locked default `endpoint_window`. `all_past` / `window_plus_landmarks` are deferred levers, tried only
if S1/S2 show the span is the binding limit (reconstruction error floors high AND the PPL gap persists
at `C_t ≈ 2`). This is the principled alternative to raising `r`.

## 4. Causality correctness requirements (DoD gate, blocks everything)

1. **Flag collapse** (D-causal) merged and unit-tested: `strict_past` ⟹ set-attention masked.
2. **Numerical leakage probe** (new test, e.g. `tests/test_set_dictionary_causality.py`): for a fixed
   batch, perturb tokens at positions `> t`; assert logits at all positions `<= t` are bitwise/`allclose`
   unchanged, for `output_residual_mode="anchor_span"` and for the anchor pre-encoder path.
3. **Anchor target causality**: assert the pre-encoder applies a causal mask (reuse the same probe).

No variant may launch until §4 passes and an audit note records it.

---

## 5. Fairness audit (DoD gate, prevents silent advantage)

Mirror the v2.7 "matched backend control" discipline. Before reporting any PPL win:

1. **Parameter parity**: report inference param count vs the matched token baseline at the anchor
   reference; the pre-encoder params are excluded from the inference count and reported separately.
2. **Inference VRAM/FLOPs**: measured with the pre-encoder removed; this is the only legitimate
   efficiency claim. Report train-VRAM (with pre-encoder) and inference-VRAM separately.
3. **Span-ablation collapse test**: at eval, zero `span_t` (`f_t = A_t`). PPL **must** rise sharply
   toward the embedding/unigram regime. If PPL barely moves, the anchor is doing the prediction (a
   bypass) → **fail**; the run is not a valid set-mediated result.
4. **Anchor ≤ token residual**: assert in code/review that `A_t` is `emb+pos` with no MLP, no
   attention, no cross-layer update.

---

## 6. Config schema additions (extend `configs/hyperparameters.md` + `src/config/*`)

| Key | Type | Default | Range / notes |
|---|---|---|---|
| `output_residual_mode` | str | (existing) | add value `anchor_span` (= thin anchor + span); keep `direct`/`empty_only`/`none` |
| `causal` | — | **removed** | derive from `set_causality_mode`; legacy → deprecation warning |
| `anchor.enabled` | bool | false | master switch for D-anchor-loss |
| `anchor.target` | str | `pre_encoder` | only `pre_encoder` active; `teacher` reserved/disabled |
| `anchor.pre_encoder_layers` | int | 2 | 1–2; causal; training-only; dropped at inference |
| `anchor.lambda_h` | float | 0.1 | >= 0 |
| `anchor.lambda_pre` | float | 1.0 | >= 0; weight of the pre-encoder's own next-token CE (`L_CE_pre`). **Must be > 0 when `anchor.enabled` — `0` leaves the teacher untrained (SD-6 confound)** |
| `anchor.pre_encoder_head` | bool | true | the pre-encoder has its own causal LM head trained by `L_CE_pre`; required for a predictive target |
| `anchor.detach_target` | bool | true | stop-gradient on `h_t*` (valid only because `L_CE_pre` trains the teacher) |
| `anchor.norm` | str | `layernorm` | LN both sides before MSE |
| `anchor.teacher.enabled` | bool | false | **deferred**; must stay false this branch |
| `set_diversity.lambda_div` | float | 0.0 | >= 0; reuses existing loss |
| `multivector_basis.enabled` | bool | false | floor-test knob; `r` is sub-values per atom per head, NOT head count |
| `multivector_basis.r` | int | 1 | 1–4 when enabled |
| `candidate_fiber` | str | `endpoint_window` | `endpoint_window` \| `all_past` \| `window_plus_landmarks` (D-fiber); default keeps comparability |

Resolved values must appear in `get_resolved_metadata()` (`set_only_lm.py:408`) so provenance captures
them, and in the CSV fingerprint so runs do not collide.

---

## 7. Experiment ladder — reuse already-run baselines for free, reliable comparison

Shared budget (fixed, identical to the matched-control budget so existing baselines apply):
`WikiText-2, L=512, batch=16, D=384, d_ff=1536, 6 layers, 8 heads, learned router, router_topk=16,
tau_r=1.0, tau_pool=0.1, lr=1e-4, 10 epochs, 3 seeds`, dense backend first; extend to sparse/landmark
only after dense shows signal. GPU split per onboarding (dense→GPU0; sparse+linear→GPU1).

The matched **dense token baseline is topology-independent at this budget** (token attention has no
`w,s`): `val/ppl = 781.1` (multi-seed, already run). So the set-dictionary model is comparable to it at
any window/stride without rerunning a baseline. To also compare against an **already-run SET model at
the same compression**, the ladder runs at two topologies that already have multi-seed SKA artifacts:

| Topology | M | L/M | C̄≈w/s | Already-run refs (no rerun) |
|---|---|---|---|---|
| T-headline `(16,8)` | 63 | ~8 | 2 | token dense 781.1; old SKA dense 1422.8 (matched-headline) / 1486.8 (long-ctx) |
| T-near2 `(4,2)` | 255 | ~2 | 2 | token dense 781.1; old set dense empty_only 1273.6 (A7) |

Both sit at `C̄ ≈ 2`. The **routing head count stays `H_r = num_heads = 8`** throughout (unchanged from
every tested config); `r` stays at `1` (= tested architecture) and is deferred (D-multivec). The locked
anchor reference `(16,4), M=125, C̄≈4` is an OPTIONAL third topology for the more-forgiving candidate
regime; it needs a cheap fresh dense baseline since no multi-seed SKA `s=4` matched control exists.

Staged ladder (dense backend, `r=1`, 3 seeds), per topology:

| Step | Config | Compared against | Gate |
|---|---|---|---|
| Ref | existing `direct` artifacts (reused, **no rerun**) + token baseline 781.1 | — | — |
| S1 | `output_residual_mode=anchor_span`, `anchor.enabled=false`, CE only | reused `direct` + 781.1 | **adopt `anchor_span` only if** it betters old `direct` AND moves closer to 781.1 |
| S2 | S1 + `anchor.enabled=true` (shallow causal pre-encoder **trained via `L_CE_pre`, `lambda_pre=1.0`, `pre_encoder_head=true`**, `lambda_h=0.1`, `detach_target=true`) | S1 | does predictive anchoring help? (only valid if `recon_error_norm` falls below √2) |

Follow-ups on the S2 winner only: `lambda_h=1.0`, then `set_diversity.lambda_div>0`. Introduce
multivector `r=2` **only** if S2 reconstruction error floors high (D-multivec). Do not run the full
cross-product.

### S1 outcome (SD-5, DONE) and SD-6 pre-registered decider

**S1 result (null, clean).** Dense `r=1`, CE only: `(4,2)` PPL `1297.9 ± 10.2` vs old `empty_only` ref
`1273.6` (+24, modestly worse); `(16,8)` PPL `1510.9 ± 82.8` vs ref `1422.8` (neutral, within noise);
token baseline `781.1`. **Span ablation = +41k–46k PPL** (above uniform ~33k): prediction is carried
**entirely** by the `C̄≈2` set span with **zero token bypass** (fairness impeccable), so the model is
genuinely set-mediated but overfiltering-bound. The thin anchor is **inert under CE-only** (it should
earn its place only at S2). **S1 fails the adoption gate** → do NOT adopt `anchor_span` as a standalone
win; `direct`/`empty_only` remain the paper defaults.

**SD-6 = S2 is a pre-registered anchoring-rescue test, not automatic continuation.** Launch S2
(`anchor.enabled=true`, shallow causal pre-encoder, `lambda_h=0.1`) and classify by the
`anchor/recon_error_norm` trajectory together with `val/ppl` vs S1, per topology:

- **Branch A — signal-limited (continue):** `recon_error_norm` decreases materially across epochs
  (trends below ~`0.5` with negative last-3-epoch slope) **AND** `val/ppl` improves over S1 toward
  `781.1` beyond the combined 3-seed 95% CI. → proceed to **SD-7** (`lambda_h=1.0`, then `lambda_div>0`).
- **Branch B — capacity-limited (pivot):** `recon_error_norm` floors high (stays above ~`0.5`, last-3-
  epoch slope ≈ 0) **AND** `val/ppl` stays within S1's 3-seed CI. → **stop the `lambda_h` sweep**; pivot
  to **SD-8** = D-fiber `all_past` (CE-only first), **not** `r=2`. Widening the candidate fiber attacks
  the `C̄≈2` rank ceiling that S1 implicates as the binding constraint; `r=2` is only a secondary
  floor test after `all_past`.

Do not launch `lambda_h=1.0` or any multivector/fiber follow-up before the decider classifies S2.
Honest end-state: if S2 (Branch B, **with a trained target** — see below) **and** `all_past` both null
with a high recon floor, the defensible conclusion is that the compression bottleneck is fundamental at
this scale — report it as a negative result, not an execution failure.

> **⚠️ SD-6 confound + SD-6.5 (2026-06-17).** The first S2 run (SD-6) shipped a pre-encoder with **no
> `L_CE_pre`** and a detached target, so the teacher stayed at random init: `recon_error_norm` parked at
> √2 ≈ 1.4 (cosine ≈ 0), the anchoring was inert, and the **Branch-B "capacity-limited" verdict it
> produced is void** — the decider above never ran on a valid signal. S1 (CE-only) and SD-1…4 are
> unaffected (the anchor path only exists when `anchor.enabled=true`), so **only S2 reruns.** **SD-6.5**
> = build D-anchor-loss as now specified (`pre_encoder_head=true`, `lambda_pre=1.0`, keep
> `detach_target=true`), rerun the S2 `(16,8)`+`(4,2)` ladder, and apply the **anchor validity guard**
> (`recon_error_norm` must fall below ~`1.2` or the run is confounded, not capacity-limited) **before**
> reading the Branch A/B decider. SD-7/SD-8 stay blocked behind a valid SD-6.5.

**Metrics per run**: `val/ppl`, inference VRAM, train VRAM, time/epoch, normalized reconstruction
error `||LN(span_t)-LN(h_t*)|| / ||LN(h_t*)||`, routing entropy, router top-1, pooling `n_eff`,
gradient ratios `rho_p, rho_a, rho_pa`, set-Gram spectral entropy, span-ablation Δppl (§5.3).

**Decision gate (DoD)**: a variant is a positive result iff §4 + §5 pass AND `val/ppl` improves over
S1 by a margin exceeding the combined 3-seed CI at its comparison topology AND reconstruction error
decreases. Record nulls explicitly (do not discard). The recon-error-vs-ppl relationship across S1→S2
(and any SD-8 fiber/multivector follow-up) is the headline diagnostic: it separates "insufficient
learning signal" (anchoring closes the gap, Branch A) from "irreducible bottleneck / capacity"
(recon floors high → wider fiber, Branch B) from "routing collapse" (entropy/top-1).

---

## 7b. SD-9 — multi-resolution (mixed-blur) frontier test

A within-family multi-scale variant: at one depth the 8 heads are split into a **fine group**
`(w,s)=(2,1)` (L/M≈1, near-token, detail-preserving) and a **coarse/blurred group** `(4,2)` (L/M≈2),
pooled and routed in parallel, then concatenated. `%blur` = the coarse-head fraction. Cheap first
implementation: two parallel set streams with `H_fine`/`H_coarse` heads, concatenated before the head
(per-head-group banks inside one block are the cleaner long-term form). CE-only, anchor disabled,
`endpoint_window` fiber, 3 seeds. **Backend differs by scale** (each uses its feasible backend, as the
project always did): dense exact at short, landmark at long.

- **Short context** — L=512 on **blue-demon**, **dense exact backend, batch=16**: mixed-25 (6 fine +
  2 coarse), fine `(2,1)`/coarse `(4,2)`; plus the two uniform extremes all-fine `(2,1)` and all-coarse
  `(4,2)` under the same contract.
- **Long context** — **L=8192 on lizmark** (`iarroyof@192.168.241.205`, 48 GiB), matching the *verified*
  latest long-context experiment (A8.3 `set_linear_landmark`, `audit/A8_3_l8192_linear_followup.md`):
  **landmark backend, `landmark_coverage=0.25`, batch=1, lr 1e-4, 10 epochs**. mixed-65 (3 fine +
  5 coarse) using the same `(2,1)/(4,2)` fine/coarse ratios; plus the two uniform extremes, all with the
  landmark backend. Run **concurrently** with the short arm. (L=2048 is the blue-demon regime and is NOT
  the lizmark arm.)
- **Baseline / question (set-vs-set, NOT token attention):** does the mix sit *below* the line joining
  all-fine and all-coarse on the **PPL–peak-VRAM** plane (a Pareto win)? That is the only claim SD-9
  tests; a win is attributable to multi-resolution mixing within the set family, not to beating token
  attention (the fine heads already approach token attention — see §"murky fairness" discussion).
- **Feasibility / backend (DoD):** long context MUST use the landmark backend — dense O(M²) is
  infeasible at L=8192 (M≈4095), which is exactly why A8 used landmark on lizmark. Smoke first on
  lizmark; monitor peak VRAM; do not silently fall back like SD-8. **Verify lizmark credentials** (NOT
  in `../blue-demon.txt`, which is blue-demon only — same user `iarroyof`; confirm the password) and
  sync repo + docker image per the A8 lizmark pattern (`scripts/run_a8_l8192_linear_followup_lizmark.sh`,
  `scripts/run_a8_largeL_smoke_lizmark.sh`).
- **Expectation (pre-registered):** a modest frontier improvement over uniform, not parity with token
  attention, with the usual memory erosion from the fine heads. Record nulls explicitly. SD-9 informs
  whether (b) `contextualize-before-pool` is worth opening; it does not by itself resolve the
  pooling-stage ceiling.

**SD-9 RESULT (DONE 2026-06-20).** Registered verdict = not Pareto vs interpolation, BUT mixed
Pareto-**dominates the all-fine endpoint** on both PPL and VRAM at both contexts. Short: all-fine
912.9/13933 → mixed-25 862.1/13790 (super-additive; span-abl Δ rises, coarse heads carry more used
prediction). Long L=8192: all-fine 1033.1/27928 → mixed-65 1009.0/20193 (−24 PPL **and** −28% VRAM).
Multi-scale hypothesis supported; the long-context coarse-head win de-risks SD-10.

## 7c. SD-10 — causal latent-dictionary that re-reads (the strong form of option b)

Goal: replace the one-shot lossy pool with a learned per-layer **read cross-attention**, make the latent
basis the residual-highway object, and **generate** token states by decoding the refined basis. Targets
the diagnosed pooling-stage ceiling directly. Framing is **compressed long-range memory** (M ≪ L), not a
short-context replacement for attention — keep that discipline or the result is uninterpretable (see §2,
and the "murky fairness / rediscover token attention" discussion).

**STAGING (per user 2026-06-20).** The FULL redesign here — learned/un-materialized latents, dropping the
identity residual, generating token states ONLY as atom combinations — has open, possibly circuit-opening
questions (seeding; whether raw `E[x]` as KV/query/identity opens undesirable shortcuts) and is
**deferred (SD-11, conditional).** The verified-safe ENTRY is **§7e SD-10a**: add per-layer re-read as a
*minimal additive change on the current winner* (pool stays the seed, route unchanged, the verified-weak
`anchor_span` identity unchanged), isolating the one causal hypothesis (re-read vs pool-once) without the
murky parts. Open SD-11 only if SD-10a is positive.

Forward (causal; reuses strict-past endpoints `eₘ` for both masks):
```
Hₜ = E[xₜ] + P[t]                       cheap input features (NOT a refined token stream)
Z⁰ = seed latents, M ≪ L, endpoints eₘ
per layer k:                            latents RE-READ the input every layer
  Z' = Z + ReadXAttn(Q=LN Z, KV=H ; mask latent m ← token j iff j ≤ eₘ)
  Z''= Z' + SelfAttn(LN Z' ; causal over eₘ)
  Z^{k+1} = Z'' + FFN(LN Z'')
hₜ = DecodeXAttn(Q=q(Hₜ), KV=Z^K ; mask token t ← latent m iff eₘ ≤ t) + Hₜ
yₜ = LMhead(hₜ)
```

Module spec / reuse:
- `ReadXAttn` = new cross-attention sub-layer added to the set block (latent queries, token keys/values),
  replacing `bank.pool`. The set block becomes a Perceiver/ISAB block. Causal read mask from `eₘ`.
- `SelfAttn` over latents = the existing `SetAttentionBlock` self-attention (causal over `eₘ`).
- `DecodeXAttn` = the existing candidate-masked router, reframed as token-query ← latent decode (mask
  `eₘ ≤ t`); largely unchanged.
- Seed latents: start with a coarse pool of `E[x]` (input-conditioned) or learned latents + a cheap
  read at layer 0. `Hₜ` thin-identity residual stays (circuit closed; `anchor_span`-style).

DoD obligations:
- **Causal-composition proof + numerical probe:** `hₜ ← latents{eₘ≤t} ← tokens{j≤eₘ≤t}` ⇒ depends only on
  `x_{≤t}`. Reuse the SD-2 future-perturbation probe on the read and decode cross-attentions.
- **Cost/efficiency:** O(L·M·K); the claim is a quality–memory win at **long context** (M ≪ L). Evaluate
  on lizmark L=8192 (landmark-class), against the A8.3 long-context refs and the SD-9 long-context rows,
  and on compressed-memory/recall tasks (needle, associative recall, long-doc continuation), NOT only
  short-context PPL.
- **Fairness:** report inference params/VRAM; at M≈L it approaches token attention — state it; the
  contribution is the compressed-memory regime, not short-context parity.
- Gate: draft only for now; launch decision after SD-9 write-up and explicit user go.

## 7d. SD-9.5 — mechanism probes on the current winner (cheap, do FIRST)

Pre-registered probes on the SD-9 mixed model (verified current implementation). Load SD-9 checkpoints if
saved; else retrain the 3 mixed seeds with the eval instrumentation (reuse SD-9 configs).

- **Per-head-group span-ablation:** extend the existing span-ablation hook to zero the FINE-group routed
  contribution and the COARSE-group contribution *separately*. Report ΔPPL per group, overall + stratified.
  Prediction: coarse-group ablation hurts long-range/global tokens more; fine hurts local.
- **Effective-range probe:** per group report routing reach — mean `|t − center(eₘ)|` weighted by routing
  prob — plus routing entropy/top-1. Prediction: coarse reach ≫ fine.
- **Token-type stratified loss (proxy):** split val loss by (i) position bucket (early vs late context),
  (ii) target rarity (frequent vs rare). Test whether the coarse-head gain concentrates where aggregated
  long-range context matters.
- **Scale-L sweep (lizmark, landmark, batch 1, smoke first):** mixed-65 + all-fine + all-coarse at
  L ∈ {16384, 32768}. Report PPL + peak VRAM; **test whether mixed's VRAM advantage over all-fine grows
  with L** (quantitative compressed-memory claim). seed 0 first, extend if budget allows.
- Output `audit/SD_9_5_probes.md` (3 attributions + scale-L table); sizes the SD-10a opportunity, feeds the
  write-up.

## 7e. SD-10a — minimal re-read ablation (one isolated test, gated entry to SD-10/§7c)

The circuit-safe test of the SD-10 hypothesis: add per-layer **ReadXAttn** to the set block on top of the
*verified winner*, changing nothing else.

- **Change:** in each set-attention layer, set states cross-attend to the token source `H = h₀` with the
  causal read mask (latent m ← token j iff `j ≤ eₘ`), *before* the existing set self-attention. Seed stays
  `Z⁰ = pool(h₀)` (NOT learned latents); route unchanged; `anchor_span` identity `f = h₀ + r` unchanged.
  Flag `set_reread.enabled` (default false). Reuse `eₘ` for the mask.
- **Safety (addresses the user concern):** the murky parts (learned/un-materialized latents, dropping the
  identity, never materializing tokens) are NOT touched. `h₀` is the verified-weak thin anchor (SD-5
  span-ablation), so reusing it as read source/identity is an input interface, not a predictive bypass.
- **DoD:** extend the SD-2 future-perturbation probe + nonzero-gradient check to `ReadXAttn`; run ONE
  comparison — winner (pool-once) vs +re-read — at the SD-9 short mixed `(2,1)/(4,2)` config, 3 seeds,
  dense, CE-only. **Verdict:** does re-read lower PPL / raise the set path's carried prediction vs
  pool-once? Yes → open SD-11 (full §7c redesign); No → pooling ceiling not fixed by re-read; stop, write up.

## 8. Tracking, provenance, and engineering patterns

- Create `audit/phase_sd_status.md` on first session; one row per task (D-causal, §4 gate, §5 gate,
  V0–V4), with status/PID/ETA/log path/audit pointer, following the `audit/phase_a_status.md` protocol.
- New sweeps follow the established pattern (onboarding §8): `scripts/run_sd_<sweep>.sh`,
  `scripts/summarize_sd_<sweep>.py` (must call the `scan_logs()` word-boundary nan/inf detector,
  check `strict_past`, verify finite values, write TSV + manifest JSON + audit markdown), and the
  two-bat launch/sync convention. Keep all set-dictionary configs under `configs/set_dictionary/` so
  they never confound the candidate-gather provenance.
- Update the tracker on every pending→running and running→done transition; never end a session without
  recording launches/results; on any DoD failure write `audit/incident_sd_<task>_<YYYYMMDD>.md`.

## 9. Paper / writing consequences (Phase B, gated on artifacts)

- Edit the dependence-set lemma: `f_t = A_t + span_t` depends only on tokens `<= t` (both paths causal).
- Add the dictionary/representer framing in Model Overview and a "causal dictionary atoms" definition;
  remove any "support vector" wording from formal statements.
- Report the fairness audit (§5) alongside any PPL claim, like the matched-control table.
- External teacher + KL distillation goes in Future Work as deferred (cost), with the disabled stub noted.

## 10. Open questions / deferred (resolve with user before V1)

- Q1 RESOLVED — staged: S1 is CE-only; the pre-encoder + anchoring loss is introduced at S2.
  `pre_encoder_layers` depth (default 2) decided at S2, not now.
- Q2 RESOLVED — reuse existing `direct` artifacts as the reference (no V0 rerun); adopt the new
  `anchor_span` mode only if S1 betters old `direct` and moves closer to the 781.1 baseline.
- Q3 RESOLVED — `r=1` (= tested 8-head architecture) for the whole ladder; `r=2` deferred floor test,
  not in the main ladder. `r` is sub-values per atom per head, NOT the head count.
- Q4 RESOLVED — branch `set-dictionary/anchor-span` off the tip of `paper/final-results-bundle` (after
  the `a9/candidate-gather-router` commit lands); tracker `audit/phase_sd_status.md`.
- Q5 External teacher: confirmed deferred to Future Work for this branch.

All §10 questions are resolved; the dev-agent prompt is `docs/set_dictionary_dev_agent_prompt.md`.
