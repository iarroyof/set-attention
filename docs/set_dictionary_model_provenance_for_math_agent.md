# Set-Dictionary model — code-grounded spec for the math agent

**Purpose.** Hand a math agent everything needed to (re)derive, correct, add, or delete the formal
statements/proofs in the paper so they match the model that the reported runs *actually* execute. Every
object below cites its implementation. Notation matches the harmonized paper
(`out/final_paper_bundle/overleaf_ready/example_paper.tex`, §model-overview and
§sec:future-set-dictionary): anchor `h_t^{(0)}`, routed context `r_t`, span projection `W_o`, processed
atom `s_m^{(K)}`, routing weight `α_{t,m}`, group index `ν`, head sets `𝓤_ν`.

The two things that changed versus the older paper text and MUST drive the math: (i) the prediction head
runs in **`anchor_span`** mode (thin identity + routed span, token MLP off), and (ii) routing heads are
partitioned into **multi-resolution groups** (fine + coarse), each with its own bank/pool/set-stack/router.

**Current evidence boundary (2026-06-30).** The active comparison is exact dense set dictionary versus
exact dense token. Coverage-scaled landmark results are archived reference observations and are not an
active linear-efficiency result. Do not use the landmark-era cross-family or blur-vs-L headline as an
established paper claim. Current empirical scope and pending confirmation are defined by
`docs/sd_dense_paper5_matrix.md`; its regular blur set is `{b0,b25,b50,b75,b100}`.

---

## 1. The model the runs execute (code provenance)

Current dense-grid set runs use `configs/set_dictionary/sd9_multiresolution.yaml`; matched token controls
use `configs/paper_lr_norm/baseline_dense_exact.yaml`. Historical SD-9.x long-context rows used
`baseline_linear_landmark.yaml`, but those rows are outside the active matched comparison.

**Anchor (thin identity carrier).** `h_t^{(0)} = e(x_t) + p_t`, embedding + position, **no token MLP**.
- `_thin_anchor()` — `set_only_lm.py:800`. In `anchor_span` the token MLP is disabled
  (`model.token_mlp.enabled=false`, config line 62–63), so `token_states` is not added to the readout.

**Per-group set construction (multi-resolution).** Heads `{1,…,H_r}` are split into disjoint groups `ν`
with head sets `𝓤_ν` (config `model.multiresolution.groups`, lines 25–35; here fine `𝓤_f` and coarse
`𝓤_c`). Each group `ν` owns an independent stream built in `_build_multiresolution_streams()`
(`set_only_lm.py:603–719`), with `stream_dim = set_state_dim · |𝓤_ν| / H_r`. Per group:
- **Bank** `𝓗_{L,w_ν,s_ν}`: causal windowed sets `S_m^{(ν)} = {j : a_m^{(ν)} ≤ j ≤ e_m^{(ν)}}` with window
  `w_ν`, stride `s_ν` (fine `(2,1)`, coarse `(4,2)`), `M_ν ≈ L/s_ν` sets — `src/models/set_only/banks.py`.
- **Pool** `s_m^{(0,ν)} = 𝓟_ν(h^{(0)})` = `soft_trimmed_boltzmann` pooling over the set's member states
  (config `pooling`, lines 37–41; τ=0.1, q=0.85). Weights `ω_{j,m}^{(ν)}` are a trimmed Boltzmann over
  member tokens.
- **Set stack** `s_m^{(K,ν)}`: `K` **pre-LN** set-attention blocks (causal over set endpoints `≤ e_m`) + set
  FFN — `src/models/set_only/ska_block.py` (`z = z + drop(backend(LN z)); z = z + drop(mlp(LN z))`).
  Geometry bias on (config `geometry`, lines 51–54). Hashed-count content features (config
  `feature_mode: hashed_counts`, `num_bins:128`, lines 46–50).

**Routing (per head, per group).** Candidate fiber `𝓒_t^{(ν)} = {m : t − w_ν < e_m^{(ν)} ≤ t}`
(endpoint-window, strict-past, `candidate_fiber: endpoint_window`). For head `u ∈ 𝓤_ν`:
`π_{t,m}^{(u)} ∝ exp(g̃_{t,m}^{(u)} / τ_r^eff)`, `r_t^{(u)} = Σ_{m∈𝓒_t^{(ν)}} π_{t,m}^{(u)} s̃_m^{(u)}`,
where `s̃_m^{(u)}` is head-`u`'s slice of `s_m^{(K,ν)}` — `src/models/set_only/router.py`
(`router_topk:16`, `min_temp:0.5`, `score_mode: candidate_gather`, config lines 55–61). Empty fiber → `r_t^{(u)}=0`.

**Merge + span projection.** Concatenate routed heads across all groups and project:
`r_t = Concat_{u=1}^{H_r} r_t^{(u)}`, `span_t = W_o r_t` — `set_output_proj`, applied at
`set_only_lm.py:1153` (multi-res path) / `:1343` (single-stream path). **`W_o` is the identity whenever
set-state width = D** (true in every reported run: `set_state_dim = d_model = 384`).

**Readout (`anchor_span`).** `f_t = h_t^{(0)} + span_t` — `set_only_lm.py:1174` (and `:1361`).
Then `y_t = W_lm f_t`, softmax, token-level CE. The paper's `eq:set-dictionary-span`
(`f_t = h_t^{(0)} + W_o Σ α_{t,m} s_m^{(K)}`) is the single-head abstraction of this.

**Span ablation diagnostic.** Setting `r_t := 0` at eval — `set_only_lm.py:1157` / `:1346`
(`routed_repr = torch.zeros_like(routed_repr)`). This is the operator behind the empirical
"prediction is span-carried" claim (Δppl ≈ 50k–63k across the grid).

**Backend (active).** Every current matched cell uses exact dense attention. Per group, the set-attention
score memory scales as `|𝓤_ν| M_ν²`, with `M_ν≈L/s_ν`; increasing blur lowers the constant by assigning
more heads to the larger-stride coarse bank, but does not change the quadratic complexity class.

**Historical backend note.** Coverage-scaled landmark used
`k=round(0.25 M)`, hence materialized `M×k` and `k×M` blocks are still `O(M²)`. It is not an active
efficiency result. A fixed-k design is archived and deferred in
`docs/archive/deferred/sd_linear_matrix_plan.md`.

---

## 2. What is empirically established (so proofs target the right regime)

From the completed exact-dense set grid (`scripts/sd_grid_status.py`; mean val PPL / peak VRAM MiB):

- **Span carries prediction** (no token bypass): span-ablation Δppl ≈ 50k–63k at every `(L, blur)`.
- **Multi-resolution Pareto-dominates uniform** at fixed `L`: the best mixed blur beats both all-fine
  `(2,1)` and all-coarse `(4,2)` on PPL, and beats all-fine on VRAM too.
- **b25 is the lowest-PPL row through L=2048** and uses less VRAM than
  all-fine. At `L=4096,B=4`, b0 and b25 produced repeated legacy OOMs on 49
  GiB; b50 is the lowest-PPL feasible row. This is an observed
  feasibility-constrained result, not proof that the unconstrained optimum
  shifts to b50. The old OOM launchers lack external-process telemetry.
- **VRAM decreases monotonically in blur** at every `L` (coarse heads = fewer/larger-stride sets).
- **Dense set-vs-token is now measured.** At B16/L512, token dominates the lowest-PPL set row on both
  means. At B4/L512 and L1024, the best-PPL b25 set row trades small extra VRAM for a lower/equal mean PPL,
  with overlapping intervals. At B4/L2048, b25 has nearly identical mean PPL and about 536 MiB lower peak
  VRAM. At B4/L4096, token produced 3/3 legacy OOMs while b50 and coarser set
  rows complete. The defensible headline is observed dense-memory feasibility,
  not a consistent PPL win or a retrospectively certified exclusive capacity
  limit. See `audit/SD_dense_matched_results.md`.

These are the claims the formal results should support, bound, or explain. All active islands are
both-O(M²), exact-dense comparisons.

---

## 3. Theorem audit (status to act on) — from §main-theory / Appendix theory

Marked inline in the paper with `\todo[inline]{SD-audit ...}`:

| Statement | Status | Required change |
|---|---|---|
| Routing-entropy bound | **HOLDS per group** | Restate per group `ν` over fiber `𝓒_t^{(ν)}`; entropy bound applies within each `𝓤_ν`. |
| Multihead-capacity (rank ceiling) | **NEEDS UPDATE** | Ceiling becomes `Σ_ν min{|𝓤_ν|, C_t^{(ν)}}` (sum of per-group caps), not `min{H_r, C_t}`. |
| Pooling-transport | **HOLDS per group** | Per-group pooling map `𝓟_ν`; transport statement is per bank topology `(w_ν,s_ν)`. |
| Feasible-region / mechanistic bound | **VERIFY/RESTATE** | (i) `𝓟,Φ_set,𝓡` are per-group; `r_t` is a concatenation → discrepancies `ε_pool,ε_set,ε_route` become group-indexed and summed. (ii) `anchor_span` readout consumes `h_t^{(0)}+r_t` with `h_t^{(0)}` the thin (MLP-free) anchor → restate readout operator `𝓡_out` against this weakened identity term. |
| Tier-C feasible region | **UPDATE** | Recompose from the per-group quantities above. |

**Open formal targets worth stating (new):**
1. A **capacity/expressivity** statement explaining why an interior blur optimum can exist (trade-off
   between fine-resolution rank and coarse-resolution reach under a fixed `H_r`). Do not yet claim its
   location increases with L.
2. A **memory** statement for exact dense set attention:
   `Σ_ν |𝓤_ν|·M_ν²`, with `M_ν≈L/s_ν`. This explains monotone VRAM-in-blur and the L4096 OOM boundary as
   constant-factor/atom-count effects inside an O(L²) family.
3. The **span-only identifiability** claim: since `h_t^{(0)}` is non-contextual and the token MLP is off,
   all context must factor through `span_t = W_o r_t` — formalize what function classes are reachable.

---

## 4. Symbols already defined in the paper (reuse — do NOT reintroduce)

`h_t^{(0)}` anchor; `e(·),p_t` embedding/position; `r_t, r_t^{(u)}` routed context/head; `span_t`; `W_o`
set-output proj; `W_lm` LM head; `s_m^{(0)}/s_m^{(K)}` pooled/processed atom; `α_{t,m}/π_{t,m}^{(u)}`
routing weights; `𝓒_t` fiber; `𝓗_{L,w,s}` bank; `ν` group index; `𝓤_ν` head set; `(w_ν,s_ν)` window/stride;
`H_r` routing heads; `τ_r^eff` routing temperature; `D` model width. Free symbols still available if needed:
`μ, ξ, ζ, χ`. Taken (avoid): `a_t` (gradient diagnostic), `β` (used), `κ_t, η_min`.
