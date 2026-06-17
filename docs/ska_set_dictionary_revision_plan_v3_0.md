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
h_t*     = hidden state of a SHALLOW CAUSAL token pre-encoder (1–2 layers) used ONLY to produce the
           target; sg = stop-gradient; LN on both sides.
```

- The pre-encoder is a **training-time auxiliary**: it is excluded from the inference model and from
  inference param/VRAM/FLOP accounting. It is the lightweight, in-model alternative to a full external
  teacher.
- Do **not** anchor to `token_mlp(emb+pos)` or to `emb+pos`: those targets measure pooling
  invertibility, not predictive usefulness, and were rejected for unfair token-attention alignment.
- **External teacher + KL logit distillation is explicitly deferred to Future Work** (cost). Leave a
  disabled, documented stub (`anchor.teacher.enabled = false`) so the path is reserved but inert.
- Causality requirement: `h_t*` must come from an **autoregressive** (causal) pre-encoder. A
  bidirectional/MLM target leaks future through the regression target even though routing is
  strict-past; `sg` does not protect against this (it controls gradient, not forward visibility).

Primary objective stays CE. Total loss:

```text
L = L_CE + lambda_h * L_anchor + lambda_div * L_div
```

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
| `anchor.detach_target` | bool | true | stop-gradient on `h_t*` |
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
| S2 | S1 + `anchor.enabled=true` (shallow causal pre-encoder, `lambda_h=0.1`) | S1 | does predictive anchoring help? |

Follow-ups on the S2 winner only: `lambda_h=1.0`, then `set_diversity.lambda_div>0`. Introduce
multivector `r=2` **only** if S2 reconstruction error floors high (D-multivec). Do not run the full
cross-product.

**Metrics per run**: `val/ppl`, inference VRAM, train VRAM, time/epoch, normalized reconstruction
error `||LN(span_t)-LN(h_t*)|| / ||LN(h_t*)||`, routing entropy, router top-1, pooling `n_eff`,
gradient ratios `rho_p, rho_a, rho_pa`, set-Gram spectral entropy, span-ablation Δppl (§5.3).

**Decision gate (DoD)**: a variant is a positive result iff §4 + §5 pass AND `val/ppl` improves over
V0 by a margin exceeding 3-seed CI at its comparison topology AND reconstruction error decreases. Record
nulls explicitly (do not discard). The recon-error-vs-ppl relationship across V0–V4 is the headline
diagnostic: it separates "insufficient learning signal" (anchoring closes the gap) from "irreducible
bottleneck" (recon floors high, multivector needed) from "routing collapse" (entropy/top-1).

---

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
