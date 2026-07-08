# Genuinely-linear (fixed-k) sweep — plan + repurpose analysis (2026-06-25)

Status: DEFERRED / NOT APPROVED. Do not implement or launch this matrix while the exact-dense matched
comparison is active. Reopen only after an explicit user request. Nothing in this file overrides
`docs/sd_dense_matched_comparison_plan.md` or `audit/phase_sd_status.md`.

**Goal.** Substantiate a *genuine* sub-quadratic claim (the `0.25·M²` coverage runs cannot), by running the
landmark backend with a **fixed** landmark budget `k` (independent of M) → attention cost `O(M·k)`. Because
the rest of the model is already linear in L — bank build `O(L·w/s)`, local pooling, and routing over a
**bounded windowed fiber** `|𝓒_t|≈w/s` — fixing `k` makes the **whole anchor-set-dictionary LM genuinely
linear in sequence length L.**

Vehicle (decided): the **proven landmark code path with fixed `num_landmarks`** — NOT nystrom (history of
process-kills), NOT linformer (fixed seq-projection, awkward for variable-M sets). The set factory now
exposes it (`set_only_lm.py`: `num_landmarks=backend_params.get("num_landmarks")`, takes precedence over
coverage). So a linear cell = `backend=landmark` + `backend_params.num_landmarks=K` (no coverage effect).

## 1. Repurpose verdict — can existing landmark@0.25 runs be reused as fixed-k points?

**No, not as fixed-k points.** Every existing landmark run used `coverage=0.25` ⇒ `k = round(0.25·M)` which
**scales with M** (and with the group stride): fine group `k≈0.25·L`, coarse group `k≈0.125·L`. So each run
is a point on the line `k=0.25·M`, never a fixed-k value across L. They land on a target fixed-k curve only
at the single L where `0.25·M(L)=K` — and those L's weren't run on landmark (L=512 was dense). So the
fixed-k linear runs are **essentially all new**.

**But reuse them as the *reference* curve (no re-run):** the coverage-0.25 blur curve at L∈{2048,4096,8192}
is the **high-fidelity (generous-k, quadratic) reference**. Plot fixed-k (genuinely linear) against it; the
gap = *quality cost of true linearity*. That's the scientifically useful reuse, and it costs zero compute.

## 2. Required infra change BEFORE launching linear (cell-id disambiguation)

Coverage-landmark and fixed-k-landmark BOTH have `model.backend="landmark"` in metadata → identical cell_id
→ the grid's done-detection would conflate/duplicate them. Fix before launch:
- `sd_grid_status.py`: when `model.backend=="landmark"`, append a tag from metadata —
  `landmark_k{num_landmarks}` if `backend_params.num_landmarks` is set, else `landmark_c{landmark_coverage}`.
  cell_id becomes e.g. `set|landmark_k64|8192|f3c5|0` vs `set|landmark_c0.25|8192|f3c5|0`.
- `run_sd_grid.sh`: add a manifest backend token `landmark_fixed` whose param column = K; emit
  `model.backend=landmark model.backend_params.num_landmarks=K`, and compute the matching `landmark_k{K}`
  cell_id. (Small, mirrors the existing exact/landmark/local_band cases.)

## 3. Proposed matrix (focused — not a full cross-product)

All `anchor_span`, token-MLP off, endpoint_window, CE-only, 3 seeds (1 seed at the extreme L). Smoke first.

**(S) Smoke / feasibility** — confirm the never-run `landmark(fixed-k) × multiresolution × anchor_span` trains
and measure quality+VRAM: `k=64`, blur `b62`, L∈{2048, 8192}, 1 seed. Gate the rest on this.

**(A) Linear scaling curve (the headline efficiency figure)** — fixed `k=64`, two blurs `b0` (fine ref) and
`b62` (long-L winner), L∈{2048, 4096, 8192, 16384, 32768}, 3 seeds (1 seed at 32768). Shows ~flat per-token
memory and PPL-vs-L → genuine linear scaling reaching lengths the coverage runs OOM on. **lizmark** (cheap now).

**(B) Quality-vs-budget** — at L∈{2048, 8192}, sweep `k∈{32,64,128}` × blur `{b0,b25,b62,b100}`, 3 seeds.
Characterizes the compression/quality knee and whether the blur optimum is stable under aggressive `k`.
**blue** (L=2048) + **lizmark** (L=8192).

**(C) Matched linear token baseline** — fixed-k landmark token at the (A) lengths, `k=64`, 3 seeds, so the
linear set-vs-token comparison is apples-to-apples (both genuinely O(M·k)). Reuse none (coverage tokens don't
match). Small.

## 4. What's already done vs missing
- **Done / reuse as reference (no compute):** landmark@0.25 blur curve L∈{2048,4096,8192} (set) + matched
  landmark@0.25 token L∈{2048,4096,8192}. These are the quadratic high-fidelity reference.
- **Missing (all new, this plan):** every fixed-k cell in (S),(A),(B),(C).

## 5. Launch discipline if explicitly reactivated
Only after a new explicit approval, reuse the duplication-proof driver (`run_sd_grid.sh`) after the §2 edit:
add the `landmark_fixed` rows to a
SEPARATE manifest section gated by `INCLUDE_LINEAR=1` (so the dense auto-run is unaffected), dry-run, smoke
(S), then enable (A)/(B)/(C). The watcher can carry it once dense drains. lizmark owns L≥8192; blue ≤4096.

## 6. Open decisions for the user
- `k` set: `{32,64,128}` (proposed) — or include `16`/`256` for a wider knee?
- Blurs in (B): `{b0,b25,b62,b100}` (proposed) vs full 6.
- Whether the linear scaling curve (A) is the paper's efficiency headline, with dense+blur as the mechanism
  core and coverage-0.25 as the high-fidelity reference (recommended framing).
