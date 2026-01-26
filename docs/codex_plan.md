I don’t need more info to generate the plan. Below is the **complete, extended, fully detailed implementation plan** that incorporates **ALL clarifications above** (especially: **hard separation** of token-attention baseline vs **true set-only** family; **mixed baseline+SKA is deprecated and removed from mainline**), and every step is paired with the **mathematical object(s)** it implements.

---

# 0) One-sentence executive directive (Codex must follow)

**Implement two disjoint model families:**

1. **Baseline token-attention** models (standard Transformer etc.)
2. **Set-only attention** models where **no token–token attention exists anywhere**; contextual mixing happens **only** via set–set attention (exact, banded, sparse, Nyström, or landmark low-rank).

**Archive and remove** the current mixed baseline+SKA path from all runners/sweeps.

---

# 1) Theory-to-code dictionary (non-negotiable vocabulary)

This section fixes definitions so nothing is left to interpretation.

## 1.1 Tokens, sets, and banks

* Token sequence (per sample):
  [
  x = (x_1,\dots,x_L), \quad x_i \in {1,\dots,|\mathcal V|}
  ]
* Token embeddings:
  [
  e_i = \mathrm{Emb}(x_i) \in \mathbb R^{d}
  ]
* Bank decomposition into **sets** (windows/segments/patch-sets):

  * We create (m) sets per sample, each set (S_j) corresponds to an index subset (I_j \subseteq {1,\dots,L}).
  * The **atoms** are tokens (or patch features for vision).
    [
    S_j = { e_i : i\in I_j }
    ]
* Bank summary per set (“pooled atoms”):
  [
  a_j = \mathrm{pool}( { e_i : i\in I_j }) \in \mathbb R^{d}
  ]
  Pool must be permutation-invariant (mean/sum/max, or attention-free gated pooling).

### Code mapping

* `banks.py`: constructs `Bank` with `ptrs` (CSR-like indices), `sizes`, `set_atoms_idx`, `set_pooled` (a_j), and set metadata (signatures).

---

## 1.2 MinHash signatures and (optional) sparse neighbors

* Universe of atom IDs (or hashed token IDs): (U).
* A MinHash scheme with (k) hash functions yields signature:
  [
  \sigma(S_j) \in \mathbb Z^{k}
  ]
* Similarity proxy:
  [
  \Pr[\sigma_\ell(S_i)=\sigma_\ell(S_j)] \approx J(S_i,S_j)
  ]
  where (J) is Jaccard similarity of atom sets.

### Code mapping

* `minhash.py`: produces `sig: [m, k]`, and optionally bucket maps for candidate neighbor selection.

---

## 1.3 Geometry kernel (K_\Delta) on sets

You stated/used a geometry-driven kernel with a distance/offset (\Delta) between sets (e.g., based on position/order, overlap, or set index gap).

Define a geometry score:
[
g(i,j)= -\gamma \Delta(i,j) + \beta
]
Possible (\Delta) choices (configurable):

* (\Delta(i,j)=|i-j|) (set order distance)
* (\Delta(i,j)=\text{span-distance}) using token ranges
* (\Delta(i,j)=1-J(S_i,S_j)) (signature-based approximate Jaccard distance)

Then:
[
K_\Delta(i,j)=\exp(g(i,j))
]
or used as an **additive bias** inside attention scores.

### Code mapping

* `geometry.py`: function `delta(i,j)` and `geom_bias(i,j) = -gamma*delta + beta`

---

## 1.4 Set features (\Phi) vs router descriptors (must be decoupled)

We enforce your clarification: **attention features** and **router descriptors** are **distinct** objects.

* Attention feature vector per set:
  [
  \phi^{\text{attn}}*j \in \mathbb R^{d*\phi}
  ]
* Router descriptor per set:
  [
  \phi^{\text{route}}*j \in \mathbb R^{d*\text{route}}
  ]
  Often (d_\text{route}=d) for stable routing; (d_\phi) may be (m) (if using kernel rows) or a projection.

Two required feature families:

### A1) Geometry-only features (no atom embeddings required)

Example:

* Raw kernel-row feature (dimension (m)):
  [
  r_j = \big(K_\Delta(j,1), \dots, K_\Delta(j,m)\big) \in \mathbb R^{m}
  ]
  Then project:
  [
  \phi^{\text{attn}}*j = W*\phi r_j \in \mathbb R^{d_\phi}
  ]
  Router descriptor can be independent:
  [
  \phi^{\text{route}}_j = \mathrm{EmbedSetIndex}(j) \ \text{or}\ W_r r_j
  ]

### A2) Content features (lightweight, hashed counts)

* Hash trick on atoms:
  [
  c_j \in \mathbb R^{d_c}
  ]
  Then:
  [
  \phi^{\text{attn}}*j = [W*\phi r_j \ ;\ W_c c_j]
  ]
  (or any configured fusion)

### Code mapping

* `features/base.py`: `SetFeatures(phi_attn, desc_router, sig, sizes, ptrs, ...)`
* `features/geometry.py`: A1 builder
* `features/hashed_counts.py`: A2 builder

---

# 2) Model families: hard separation + deprecation policy

## 2.1 Family 1: Baseline token-attention (reference only)

Mathematics (standard):
[
H^{(0)} = \mathrm{Emb}(x) + \mathrm{PosEnc}
]
[
H^{(\ell+1)} = \mathrm{TransformerBlock}(H^{(\ell)})
]
[
\hat y = \mathrm{LMHead}(H^{(L)})
]

### Code mapping

* `src/models/baseline_token/transformer_lm.py`

**No SKA imports allowed** in baseline family.

---

## 2.2 Family 2: Set-only attention (main research family)

**No token–token attention anywhere.** Tokens exist only for embedding, optional attention-free local mixing, and routing outputs.

### 2.2.1 Token to sets (bank build)

* Token embeddings: (E\in\mathbb R^{L\times d})
* Bank partition yields sets (S_1,\dots,S_m)
* Initial per-set state:
  [
  Z^{(0)}_j = \psi(S_j) \in \mathbb R^{d}
  ]
  where (\psi) is pooling or any attention-free invariant map.

### 2.2.2 Set-only “Transformer-on-sets” layers

For each layer (\ell):

* Build features:
  [
  \Phi^{(\ell)} = \text{FeatureBuilder}(Z^{(\ell)}, \text{bank metadata})
  ]
* Set attention:
  [
  Z^{(\ell+1)} = \mathrm{SetAttention}_\theta\big(Z^{(\ell)}, \Phi^{(\ell)}\big)
  ]
* (Optional) MLP on sets:
  [
  Z^{(\ell+1)} \leftarrow Z^{(\ell+1)} + \mathrm{MLP}(Z^{(\ell+1)})
  ]

### 2.2.3 Routing sets back to tokens (no token attention)

Router creates token contextual representation from sets:

* For each token position (t), compute weights over sets:
  [
  w_{t,j} = \mathrm{RouterScore}(q_t, \phi^{\text{route}}_j)
  ]
* Normalize (softmax or top-k):
  [
  \alpha_{t,j} = \mathrm{normalize}(w_{t,j})
  ]
* Token contextual output:
  [
  h_t = \sum_{j=1}^m \alpha_{t,j}, Z^{(\text{final})}_j
  ]
  Then:
  [
  \hat y_t = \mathrm{LMHead}(h_t)
  ]

### Code mapping

* `src/models/set_only/set_only_lm.py`
* `src/models/set_only/ska_block.py`
* `src/models/set_only/router.py`
* `src/models/set_only/banks.py`

**No `nn.MultiheadAttention`, no `TransformerEncoder`, no `scaled_dot_product_attention` anywhere in set-only.**

---

## 2.3 Mixed baseline+SKA implementation policy

**Deprecate completely** and remove from the mainline.

Concrete rules:

* Move current mixed model(s) to `src/legacy/mixed_ska_baseline/`
* Add a guard raising error if invoked from new runner.
* New runner must not provide flags that re-enable it.

---

# 3) Set Attention: exact scoring and kernel injection

We define a single canonical set-attention score to avoid drift.

Let (Z\in\mathbb R^{m\times d}) be set states (per sample). For head (h):
[
Q = Z W_Q^{(h)},\quad K = Z W_K^{(h)},\quad V = Z W_V^{(h)}
]
Score between sets (i,j):
[
s_{ij} = \frac{\langle Q_i, K_j\rangle}{\sqrt{d_h}} + b_\Delta(i,j) + b_{\text{sig}}(i,j)
]

* Geometry bias:
  [
  b_\Delta(i,j) = -\gamma \Delta(i,j) + \beta
  ]
* Optional signature bias / neighbor gating (if using MinHash buckets):
  [
  b_{\text{sig}}(i,j)=
  \begin{cases}
  0 & \text{if allowed neighbor}\
  -\infty & \text{otherwise}
  \end{cases}
  ]
  Attention weights:
  [
  A_{ij} = \mathrm{softmax}*j(s*{ij})
  ]
  Output:
  [
  O_i = \sum_{j=1}^m A_{ij} V_j
  ]
  Then merge heads and residual/FFN as standard.

### Code mapping

* `src/set_attention/core.py`: defines score decomposition and shared math

---

# 4) Set Attention backends (exact / sparse / band / Nyström / landmark)

All backends implement the same mathematical contract (same (s_{ij}), same outputs), differing only in how they approximate/compute it.

## 4.1 DenseExactBackend (reference)

Compute all (m\times m) scores (allowed only on sets, never tokens).

* Complexity:
  [
  O(m^2 d_h)
  ]

### Code mapping

* `backends/dense_exact.py`

---

## 4.2 LocalBandBackend (exact within band)

Define band radius (r). Allowed pairs satisfy:
[
|i-j|\le r
]
Mask others with (-\infty).

* Complexity:
  [
  O(m r d_h)
  ]

### Code mapping

* `backends/local_band.py`

---

## 4.3 SparseTopKBackend (neighbor graph on sets)

Using MinHash buckets or approximate neighbor retrieval, define candidate neighbors (N(i)) of size (\le k_s).

[
A_{ij}=0\quad \text{if } j\notin N(i)
]

* Complexity:
  [
  O(m k_s d_h)
  ]

### Code mapping

* `backends/sparse_topk.py` (may be implemented after band+Nyström; but the interface must exist now)

---

## 4.4 NyströmBackend (set-linear attention via landmarks)

Pick (r\ll m) landmarks (L\subset {1,\dots,m}).

Define a kernel-like matrix from scores:
[
K_{ij} = \exp(s_{ij})
]
Nyström approximation:
[
K \approx K_{mL} , (K_{LL} + \epsilon I)^{-1} , K_{Lm}
]
Then apply normalization consistent with softmax-kernel attention. Practical approach:

### Practical stable Nyström attention (recommended implementation)

Compute:

* (S_{mL}) scores for all sets to landmarks
* (S_{LL}) scores among landmarks

Let:
[
K_{mL}=\exp(S_{mL}),\quad K_{LL}=\exp(S_{LL})
]
Compute:
[
X = (K_{LL}+\epsilon I)^{-1}(K_{Lm} V)
]
Then:
[
O = K_{mL} X
]
Optionally normalize row-wise using:
[
\tilde{1} = (K_{LL}+\epsilon I)^{-1}(K_{Lm} \mathbf 1), \quad
\text{denom} = K_{mL}\tilde{1}
]
[
O \leftarrow O / \text{denom}
]

* Complexity:
  [
  O(m r d_h + r^3)
  ]

### Code mapping

* `backends/nystrom.py`
* Landmark selection strategies in `landmarks.py`:

  * `uniform`, `random`, `bucket_diverse` (optional), later `kmeans-lite`

---

## 4.5 LandmarkAttentionBackend (two-step low-rank attention)

Compute set→landmark attention then landmark→set aggregation.

Let (L) be landmark set states (Z_L).

1. Sets attend to landmarks:
   [
   A = \mathrm{softmax}(S_{mL})
   ]
2. Landmarks aggregate values from sets (or sets from landmarks):
   [
   B = \mathrm{softmax}(S_{Lm})
   ]
3. Output:
   [
   O = A (B V)
   ]
   or simplified:
   [
   V_L = B V,\quad O = A V_L
   ]

* Complexity:
  [
  O(m r d_h)
  ]

### Code mapping

* `backends/landmark.py`

---

# 5) Feature pipelines (A1/A2) with strict decoupling

## 5.1 A1 Geometry-only feature builder

* Build geometry row feature (r_j) or compact basis:
  [
  r_j(i)=\exp(-\gamma \Delta(j,i) + \beta)
  ]
* Project:
  [
  \phi^{\text{attn}}*j = W*\phi r_j
  ]
* Router descriptor:
  [
  \phi^{\text{route}}_j = W_r r_j \quad \text{or}\quad \mathrm{Emb}(j)
  ]

### Implementation details

* Avoid materializing full (m\times m) if large:

  * Provide banded rows if using band backend
  * Provide landmark rows if using Nyström/landmark backend

### Code mapping

* `features/geometry_only.py`

---

## 5.2 A2 Hashed-count content features

* For each set (S_j), compute hashed bag-of-atoms (c_j):
  [
  c_j[h(u)] \mathrel{+}=1 \quad \forall u\in S_j
  ]
* Normalize optionally by set size
* Fuse with geometry:
  [
  \phi^{\text{attn}}*j = \mathrm{MLP}([W*\phi r_j ; W_c c_j])
  ]
* Router descriptor: usually keep in (d):
  [
  \phi^{\text{route}}*j = W*{route} [a_j ; c_j]
  ]

### Code mapping

* `features/hashed_counts.py`

---

# 6) Router (B1/B2) and invariants

## 6.1 Router query

Token-side query per position (t) must be attention-free:
[
q_t = \rho(e_t, p_t)
]
where (\rho) is MLP/conv/gated-MLP, never attention.

### Code mapping

* `router.py`: builds `q_t` from token embeddings.

---

## 6.2 B1 Uniform router

Weights are uniform over sets that cover token (t):
[
\alpha_{t,j}=
\begin{cases}
\frac{1}{|\mathcal J(t)|} & j\in \mathcal J(t)\
0 & \text{else}
\end{cases}
]
where (\mathcal J(t)={j: t\in I_j}).

### Code mapping

* `router_uniform.py`

---

## 6.3 B2 Learned router (top-k optional)

Compute score:
[
w_{t,j} = \frac{\langle q_t, \phi^{\text{route}}*j\rangle}{\sqrt{d}} + b*{t,j}
]
Mask to sets that contain token (t) (or allow all sets depending on experiment option):
[
b_{t,j}=
\begin{cases}
0 & j\in \mathcal J(t)\
-\infty & \text{else}
\end{cases}
]
Then:

* Full softmax:
  [
  \alpha_{t,\cdot}=\mathrm{softmax}(w_{t,\cdot})
  ]
* Or sparse top-k:
  [
  \alpha_{t,\cdot}=\mathrm{softmax}(\mathrm{TopK}(w_{t,\cdot},k_r))
  ]

### Code mapping

* `router_learned.py`

---

## 6.4 Set-only invariant checks (required)

In set-only mode, forbid:

* Any op that creates attention weights over tokens: ([L\times L]) or equivalent.

Add:

* `assert_set_only_shapes()` hooking into forward to fail fast:

  * If any tensor has last two dims equal to ((L,L)) in set-only path -> error.
  * Ensure any softmax over pairs is only over (m) sets.

---

# 7) Multi-scale banks (clean version, no per-head bank construction)

Instead of “each head has own window/stride”, define scales (s=1..S) with parameters ((w_s, \text{stride}_s, k_s)).

Each scale produces (m_s) sets and states (Z_s). Heads are assigned to scales by a static mapping.

Mathematically, for head (h) mapped to scale (s(h)):
[
\mathrm{Head}*h(Z) = \mathrm{SetAttn}\big(Z*{s(h)}\big)
]

### Implementation

* `banks_multiscale.py`: constructs all scales
* `ska_block.py`: loops over head groups per scale

---

# 8) Configuration, runner, and W&B sweep system (YAML-first, strict schema)

## 8.1 Single runner entrypoint

Only:

* `scripts/run_experiment.py --config configs/exp.yaml [overrides]`

Runner location is `scripts/run_experiment.py` (not repo root).

No multiple train scripts per task. Tasks become dataset modules.

## 8.2 Strict config schema (enforce separation)

Config must reject mixed mode:

Valid:

```yaml
model:
  family: baseline_token
```

or

```yaml
model:
  family: set_only
```

Invalid (must raise):

```yaml
model:
  family: baseline_token
  use_ska: true
```

## 8.3 Pydantic (or OmegaConf/Hydra) validation

* If `model.family == set_only` then forbid presence of:

  * `token_attention.*` keys
  * any baseline backbone selection that implies attention
* If `baseline_token`, forbid `set_only.*` keys.

---

# 9) Codebase restructuring (exact file-level plan)

## 9.1 New directory skeleton

```
src/
  models/
    baseline_token/
      transformer_lm.py
      __init__.py
    set_only/
      set_only_lm.py
      ska_block.py
      router.py
      banks.py
      __init__.py
  set_attention/
    core.py
    geometry.py
    minhash.py
    features/
      base.py
      geometry_only.py
      hashed_counts.py
    backends/
      base.py
      dense_exact.py
      local_band.py
      nystrom.py
      landmark.py
      sparse_topk.py   # interface + optional impl now/later
  data/
    wikitext2.py
    cifar.py
    ...
  train/
    loop.py
    eval.py
    metrics.py
    profiler.py
  config/
    schema.py
    load.py
scripts/
  run_experiment.py
configs/
  baseline/
    wikitext2_transformer.yaml
  set_only/
    wikitext2_dense_exact.yaml
    wikitext2_band.yaml
    wikitext2_nystrom.yaml
    wikitext2_landmark.yaml
legacy/
  mixed_ska_baseline/
    ...
tests/
  test_set_only_invariants.py
  test_backend_equivalence_small.py
  test_config_validation.py
```

## 9.2 Deprecation steps (must be done early)

1. Move current mixed code into `legacy/mixed_ska_baseline/`
2. Remove it from any import path used by runner
3. Add guard: runner errors if `legacy.*` is referenced
4. Remove flags like `--sdpa-baseline` and `--use_ska` from main CLI; these become *separate* configs selecting family.

---

# 10) Step-by-step implementation phases (each with explicit deliverables)

## Phase A — Mechanical extraction & baseline freeze

**Goal:** Freeze baseline reference model and stop code drift.

Deliverables:

* `BaselineTokenTransformerLM` in `src/models/baseline_token/`
* Baseline experiment YAML reproducing current baseline results
* W&B logging standardized keys

Math pairing: standard Transformer equations (Section 2.1).

---

## Phase B — True SetOnlyLM (no token attention)

Deliverables:

* `SetOnlyLM` with:

  * token embedding
  * attention-free token backbone (MLP/conv)
  * set bank build
  * SKA blocks (dense backend first)
  * router B1/B2
  * LM head
* Invariant test that guarantees no token attention

Math pairing: Sections 2.2, 6, 3.

---

## Phase C — Feature builder decoupling A1/A2

Deliverables:

* `SetFeatures` struct
* A1 geometry-only builder + A2 hashed-count builder
* Router consumes `desc_router` only
* A1 runs without atom embeddings (where applicable by your definition)

Math pairing: Section 1.4, 5.

---

## Phase D — Backends: band + Nyström + landmarks

Deliverables:

* `LocalBandBackend` exact banded
* `NystromBackend`
* `LandmarkAttentionBackend`
* Test equivalence on small (m): dense vs band (with large r), dense vs Nyström (approx), dense vs landmark (approx)

Math pairing: Section 4.

---

## Phase E — Multi-scale banks

Deliverables:

* bank scales implementation
* head-to-scale mapping
* logging of per-scale (m_s), build time, attention time

Math pairing: Section 7.

---

## Phase F — Unified runner & sweep-ready YAML configs

Deliverables:

* single runner
* strict config schema validation
* sweep templates for:

  * family (baseline vs set-only)
  * backend (dense/band/nystrom/landmark)
  * features (A1/A2)
  * router (B1/B2 with topk grid)
  * multiscale on/off

Math pairing: ensures ablations correspond to the model families and approximations defined earlier.

---

# 11) Experiment design aligned to “two-level ablation” requirement

## Level 1: between attention families

* baseline_token/transformer
* set_only/dense_exact
* set_only/local_band
* set_only/nystrom
* set_only/landmark

## Level 2: within set-only family

For each backend:

* A1 vs A2
* B1 vs B2
* fixed vs updated set states:

  * fixed: features depend only on bank metadata (+ optional hashed counts)
  * updated: features may depend on (Z^{(\ell)}) (learned evolution)

**Log always:**

* (m) statistics (sets per sample)
* router sparsity (topk active fraction)
* backend time + bank time
* peak VRAM

---

# 12) “No ambiguity” rules Codex must encode (hard checks)

1. **Set-only mode may not import or instantiate any token-attention module.**
2. **No runner config can combine families.** Validation must throw.
3. **No script-level branching** like `if use_ska` that decides between token-attn and set-attn. Family selection happens only by `model.family`.
4. Mixed legacy implementation is not runnable via the new runner.
5. Backends implement the same canonical score:
   [
   s_{ij} = \frac{\langle Q_i, K_j\rangle}{\sqrt{d_h}} + b_\Delta(i,j) + b_{\text{sig}}(i,j)
   ]
   so approximations are apples-to-apples.

---


# Clarifications for the above implementation plan and **where to consider them**

## **MISSING SECTION 1.5: Kernel-Based Set Features (Reproducing Property)**

**Location:** Add after Section 1.4, before Section 2

**Content to add:**

### 1.5 Kernel-based set features via reproducing property

The set feature Φ(S) is derived from kernel similarities to all sets in the bank:

$$\Phi(S_i) = k_\Delta(S_i, \cdot) = [K_\Delta[i,1], K_\Delta[i,2], \ldots, K_\Delta[i,m]]^T \in \mathbb{R}^{m}$$

**Efficient computation using cached MinHash:**

```
Given cached signatures sig[m, k] and sizes[m]:

1. Compute Jaccard estimates (vectorized):
   matches[i,j] = (1/k) * sum(sig[i,:] == sig[j,:])  # [m, m]
   J_hat[i,j] = matches[i,j]

2. Compute symmetric difference estimates:
   delta_hat[i,j] = sizes[i] + sizes[j] - 2*(J_hat[i,j]/(1+J_hat[i,j]))*(sizes[i]+sizes[j])

3. Kernel matrix:
   K_Delta[i,j] = exp(-gamma * delta_hat[i,j] + beta)  # [m, m]

4. Extract features (each row is a feature vector):
   Phi[i] = K_Delta[i, :]  # [m] dimensional feature per set
```

**PyTorch optimization:**
- Use `torch.eq(sig.unsqueeze(1), sig.unsqueeze(0)).float().mean(dim=-1)` for vectorized matches
- Use `torch.exp(-gamma * delta_hat + beta)` for kernel computation
- Cache K_Delta if m < 1000, else compute rows on-demand

**Code mapping:**
- `features/kernel_features.py`: implements kernel row extraction
- `geometry.py`: provides delta_hat computation from signatures

**When to use vs A1/A2:**
- Use kernel features when m is small (< 100 sets)
- Fall back to A1 (projected geometry) or A2 (hashed counts) for larger m

---

## **MISSING: Adapter Architecture Details**

**Location:** Add subsection 3.1 under Section 3 (after the attention score formula)

**Content to add:**

### 3.1 Adapter architectures for content term

The content term $\beta \langle A_h \Phi_i, B_h \Phi_j \rangle$ requires adapters that transform kernel features.

**Linear adapters (low-rank):**

$$A_h \in \mathbb{R}^{d_h \times m}, \quad B_h \in \mathbb{R}^{d_h \times m}$$

$$\text{content}_{ij} = (A_h \Phi_i)^T (B_h \Phi_j) = \Phi_i^T A_h^T B_h \Phi_j$$

**Nonlinear adapters (higher capacity):**

$$Q_h = \text{MLP}_Q(\Phi), \quad K_h = \text{MLP}_K(\Phi), \quad V_h = \text{MLP}_V(\Phi)$$

where each MLP: $\mathbb{R}^m \to \mathbb{R}^{d_h}$

**When to use:**
- Linear adapters: when m < 20 (limited rank = min(d_h, m))
- Nonlinear adapters: when m > 20 (to overcome rank bottleneck)
- Hybrid: Linear for geometry, nonlinear for content projection

**Per-head parameters:**

$$\gamma_h, \beta_h \in \mathbb{R}$$

Allow each head to learn different geometry/content trade-offs:

$$s_{ij}^{(h)} = \frac{\langle Q_i^{(h)}, K_j^{(h)} \rangle}{\sqrt{d_h}} + \exp(-\gamma_h \Delta_{ij} + \beta_h) + \beta_h \langle A_h \Phi_i, B_h \Phi_j \rangle$$

**PyTorch implementation:**
- Use `nn.Parameter(torch.zeros(num_heads))` for γ_h, β_h
- Use `torch.einsum('bmi,hid,bmj,hjd->bmij', Phi, A_h, Phi, B_h)` for efficient adapter projection
- Or `nn.ModuleList([MLP(...) for _ in range(num_heads)])` for nonlinear

**Code mapping:**
- `set_attention/adapters.py`: LinearAdapter, NonlinearAdapter classes
- `set_attention/core.py`: integrates adapters into attention score

---

## **MISSING: PyTorch Optimization Details Throughout**

**Location:** Add as implementation notes within each relevant section

### For Section 1.2 (MinHash) - add subsection:

**1.2.1 Efficient signature comparison (PyTorch)**

```python
# Vectorized signature matching
sig: torch.LongTensor  # [m, k]
matches = (sig.unsqueeze(1) == sig.unsqueeze(0)).float().mean(dim=-1)  # [m, m]

# Use torch.cdist for set distance if using feature-based hashing
# Use torch.gather for bucket-based neighbor retrieval
```

### For Section 1.3 (Geometry kernel) - add subsection:

**1.3.1 Vectorized geometry computation**

```python
# Batch geometry bias
sizes: torch.Tensor  # [m]
delta = sizes.unsqueeze(1) + sizes.unsqueeze(0) - 2 * jaccard_approx
geom_bias = torch.exp(-gamma * delta + beta)  # [m, m] or banded variant

# For banded: use torch.triu/torch.tril with diagonal offset
# For sparse: use torch.sparse_coo_tensor
```

### For Section 4.1 (Dense backend) - add:

**Implementation:** Use `torch.bmm(Q, K.transpose(-2,-1))` then `torch.baddbmm(geom_bias, ...)` for fused operation.

### For Section 4.2 (Band backend) - add:

**Implementation:** Use `torch.triu(scores, diagonal=-r)` and `torch.tril(scores, diagonal=r)` to create band mask, then `scores.masked_fill(mask, -inf)`.

### For Section 4.4 (Nyström backend) - add:

**Implementation:** Use `torch.linalg.solve(K_LL + eps*I, K_Lm @ V)` for stable matrix inversion. Avoid explicit inverse.

### For Section 6 (Router) - add subsection:

**6.4 Efficient token-to-set routing implementation**

```python
# Gather sets for each token
token_set_indices: torch.LongTensor  # [L, max_sets_per_token]
set_outputs: torch.Tensor  # [m, d]

# Option 1: torch.gather
h_tokens = set_outputs[token_set_indices].mean(dim=1)  # [L, d]

# Option 2: Sparse indexing with torch.index_select + scatter_add
# For learned router with top-k: use torch.topk then torch.scatter_add
```

---

## **MISSING SECTION 2.2.4: Decoding Implementation Details**

**Location:** Add after Section 2.2.3

**Content to add:**

### 2.2.4 Token decoding strategies and implementation

**Strategy 1: Overlap averaging (fixed weights)**

For token $t$ appearing in sets $\mathcal{J}(t) = \{j : t \in I_j\}$:

$$h_t = \frac{1}{|\mathcal{J}(t)|} \sum_{j \in \mathcal{J}(t)} Z_j^{(\text{final})}$$

**Implementation:**
```python
# Use BankPack offsets to build mapping
token_to_sets: List[List[int]]  # which sets contain each token
h_tokens = torch.zeros(L, d)
for t in range(L):
    h_tokens[t] = Z[token_to_sets[t]].mean(dim=0)
```

**Strategy 2: Learned routing (router-weighted)**

$$h_t = \sum_{j=1}^m \alpha_{t,j} Z_j^{(\text{final})}$$

where $\alpha_{t,j}$ from router (Section 6).

**Overlap handling:** When multiple sets predict same token, combine predictions:

$$\text{logits}_t = \text{mean}\{\text{Linear}(Z_j) : j \in \mathcal{J}(t)\}$$

**PyTorch optimization:**
- Precompute token-to-set mapping in bank construction (cached)
- Use `torch.index_select` for gathering
- Use `torch.scatter_add` for averaging with counts

**Code mapping:**
- `models/set_only/decoder.py`: OverlapDecoder, RouterDecoder classes

---

## **MISSING SECTION 7.1: Multi-Scale Merging Strategies**

**Location:** Add as subsection under Section 7

**Content to add:**

### 7.1 Merging multi-scale outputs

When heads use different scales with different $m_s$:

**Problem:** 
- Head 1: $m_1 = 30$ sets (fine)
- Head 2: $m_2 = 15$ sets (medium)  
- Head 3: $m_3 = 7$ sets (coarse)

**Solution 1: Upsample to maximum**

Upsample all to max$(m_s)$ using set relationships:
- Fine→Medium: average adjacent fine sets
- Medium→Coarse: average adjacent medium sets

**Solution 2: Downsample to minimum**

Average sets within each coarse set's span.

**Solution 3: Independent then merge at token level**

Each scale independently produces token representations via its router, then combine:

$$h_t = \sum_{h=1}^H w_h \cdot \text{Route}_h(Z_{s(h)})$$

where $w_h$ are learned per-head weights.

**Implementation:**
```python
# Option 3 (recommended)
token_outputs = []
for scale_idx, heads in head_to_scale.items():
    Z_scale = set_states[scale_idx]  # [m_s, d]
    for h in heads:
        h_t = router[h](Z_scale)  # [L, d]
        token_outputs.append(h_t)
h_final = torch.stack(token_outputs).mean(dim=0)  # or learned combination
```

**Code mapping:**
- `models/set_only/multiscale_merger.py`

---

## **MISSING SECTION: Memory and Caching Strategy**

**Location:** Add as Section 1.6 or as subsections in relevant sections

**Content to add:**

### 1.6 Caching strategy for efficient training

**What to cache (preprocessing):**

1. **Bank structure** (Section 1.1):
   - `set_offsets: torch.LongTensor` - CSR indices
   - `seq_offsets: torch.LongTensor` - per-sequence boundaries
   - Stored in BankPack, loaded once per dataset

2. **MinHash signatures** (Section 1.2):
   - `sig: torch.LongTensor [m, k]` - precomputed hashes
   - `sizes: torch.LongTensor [m]` - set cardinalities
   - Stored in RoutingPack

3. **Token-to-set mapping** (Section 2.2.4):
   - `token_to_sets: torch.LongTensor [L, max_overlap]` with padding
   - Built from bank structure, cached per sample

**What to compute on-the-fly:**

1. **Kernel matrix** $K_\Delta$ (Section 1.5):
   - If $m < 500$: cache per batch on GPU
   - If $m \geq 500$: compute rows on-demand in attention backend

2. **Set states** $Z^{(\ell)}$ (Section 2.2.2):
   - Computed during forward pass (learned representations)

3. **Adapter projections** (Section 3.1):
   - Computed per attention layer (small overhead)

**Decision rule:**
```python
if num_sets < 500:
    cache_kernel_matrix = True  # GPU memory ~2MB for m=500
else:
    cache_kernel_matrix = False  # compute banded/sparse on-demand
```

**Code mapping:**
- `data/cache_manager.py`: handles cache loading/saving
- `set_attention/kernel_cache.py`: manages K_Delta caching

---

## **MISSING: Training Considerations**

**Location:** Add as new Section 10.5 (after Phase F, before Section 11)

**Content to add:**

### 10.5 Training implementation details

**Variable $m_s$ across batch:**

Batches may have different sequence lengths → different $m$ per sample.

**Solution:** Pad to max $m$ in batch + create attention mask:

```python
# Pad set states
max_m = max([Z_i.shape[0] for Z_i in batch])
Z_padded = torch.zeros(batch_size, max_m, d)
mask = torch.zeros(batch_size, max_m, dtype=torch.bool)

for i, Z_i in enumerate(batch):
    m_i = Z_i.shape[0]
    Z_padded[i, :m_i] = Z_i
    mask[i, :m_i] = True

# Use mask in attention: scores.masked_fill(~mask.unsqueeze(-1), -inf)
```

**Gradient flow through kernel features:**

- **Detach kernel matrix** $K_\Delta$ (geometry is fixed, not learned)
- **Don't detach adapter outputs** $A_h \Phi, B_h \Phi$ (adapters are learned)

```python
K_Delta = compute_kernel(sig, sizes).detach()  # no grad through geometry
Phi = K_Delta  # features are kernel rows
Q = adapter_Q(Phi)  # grad flows through adapter
```

**Updating set states across layers:**

- **Option 1: Update** (like Transformer layers)
  - $Z^{(\ell+1)} = \text{SetAttn}(Z^{(\ell)})$
  - More expressive, higher memory

- **Option 2: Fixed initial** (like retrieval)
  - $Z^{(\ell)} = Z^{(0)}$ for all $\ell$
  - Only attention mixing, no state evolution
  - Lower memory, faster

**Code mapping:**
- `train/batch_collator.py`: handles variable m padding
- `models/set_only/ska_block.py`: implements state update logic

---

## **SUMMARY: Where to Insert Missing Content**

1. **Section 1.5** (new): Kernel-based features → after Section 1.4
2. **Section 1.6** (new): Caching strategy → after Section 1.5
3. **Section 2.2.4** (new): Decoding details → after Section 2.2.3
4. **Section 3.1** (new): Adapter architectures → within Section 3
5. **Section 7.1** (new): Multi-scale merging → within Section 7
6. **Section 10.5** (new): Training considerations → after Phase F
7. **PyTorch optimization notes**: Add subsections throughout Sections 1-6 as marked above

## **Why the additions should be considered**

They *increase correctness, expressiveness, and implementability*, and they **do not contradict** the latest Codex plan. In fact, several of them close gaps that reviewers or future contributors would otherwise trip over.

Below I go section by section and classify each item as:

* ✅ **Truly missing / should be added**
* ⚠️ **Optional but high-value**
* ❌ **Potential risk / needs constraint**
* 🧩 **Clarification that resolves ambiguity**

I’ll also state **exactly where and how** to integrate them into the final plan without destabilizing it.

---

## 1️⃣ Section 1.5 — Kernel-based set features (Reproducing property)

**Verdict:** ✅ **Correctly missing and important**

This is **not a regression**. It is a *formal completion* of what was only implicitly covered by A1.

### Why it is missing

* The previous plan talked about *geometry features* and *kernel bias*,
* but it did **not explicitly formalize** the **reproducing kernel feature map**
  [
  \Phi(S_i) = K_\Delta(S_i,\cdot)
  ]
  as a *first-class feature object*.

This matters because:

* It makes the model **provably kernel-based**, not “kernel-flavored”
* It justifies why **linear adapters over Φ are expressive**
* It clarifies when Φ is **m-dimensional and exact**, vs projected

### Compatibility with plan

Perfectly compatible with:

* A1 features
* Nyström / landmark backends
* Adapter architectures

### Where to add

Exactly where you proposed:

> **After Section 1.4, before Section 2**

### One constraint to add (important)

Add an explicit rule:

> **Kernel-row Φ must never be materialized as full `[m×m]` when `m > kernel_row_threshold`.**

This avoids accidental quadratic memory explosions.

---

## 2️⃣ Adapter architecture details (Section 3.1)

**Verdict:** ✅ **Missing and necessary**

This is one of the **most important missing pieces**.

### Why this matters

Without this section:

* It is unclear *how* kernel features interact with learned content
* Rank bottlenecks are easy to miss
* Reviewers will ask: *“Is this just dot-product attention in disguise?”*

Your adapter formalism answers that *cleanly*.

### Mathematical clarity added

You explicitly distinguish:

* Geometry kernel term
* Content interaction term
* Adapter-induced bilinear forms

This makes the attention score:
[
s_{ij}^{(h)} =
\underbrace{\langle Q_i^{(h)}, K_j^{(h)}\rangle}_{\text{learned content}}

* \underbrace{K_\Delta(i,j)}_{\text{geometry}}
* \underbrace{\Phi_i^\top A_h^\top B_h \Phi_j}_{\text{kernel–content coupling}}
  ]

That’s **not redundant** — it increases expressiveness *beyond* standard attention.

### Where to add

Exactly right:

> **Subsection 3.1 under Section 3**

### Minor constraint to add

Add this rule:

> If `Φ` is kernel-row (`m`-dim), then:
>
> * nonlinear adapters **must be depth-1 or low-rank**
> * otherwise complexity becomes O(m²)

---

## 3️⃣ PyTorch optimization details throughout

**Verdict:** 🧩 **Clarification + implementation-critical**

These are not conceptual additions — they are **anti-footgun documentation**.

### Why they are needed

Without them:

* Codex or contributors may reintroduce Python loops
* Kernel computation may be accidentally O(m²) on GPU
* Sparse vs dense paths may diverge numerically

### Are these regressions?

No. They:

* Do not change semantics
* Do not add new degrees of freedom
* Only constrain *how* things are implemented

### Where to add

As you proposed:

* Inline subsections (1.2.1, 1.3.1, etc.)
* This is exactly the right granularity

### Recommendation

Add a **global implementation invariant**:

> All pairwise set operations must be expressible as:
>
> * `torch.bmm`
> * `einsum`
> * masked/banded tensor ops
>   **Never Python loops over sets**

---

## 4️⃣ Section 2.2.4 — Decoding implementation details

**Verdict:** ✅ **Missing and critical**

This is a **real gap** in the previous plan.

### Why it matters

The plan previously stopped at:

> “Router outputs token contextual representation”

But:

* How overlapping sets map back to tokens is **non-trivial**
* There are *multiple valid decoding semantics*
* This directly affects:

  * likelihood computation
  * gradient flow
  * expressiveness

### Your addition fixes:

* Ambiguity in token-level outputs
* Ambiguity in overlapping coverage
* Ambiguity in evaluation

### Compatibility

Fully compatible with:

* B1 uniform router
* B2 learned router
* Multi-scale routing

### Where to add

Exactly correct:

> **After Section 2.2.3**

---

## 5️⃣ Section 7.1 — Multi-scale merging strategies

**Verdict:** ⚠️ **Optional but very high value**

This is **not strictly required** for correctness, but:

* Without it, multi-scale banks are under-specified
* Reviewers will ask how scales interact
* Implementers may choose incompatible merge semantics

### Evaluation

* **Solution 3 (merge at token level)** is the cleanest and safest
* Solutions 1 & 2 are fine but should be marked *experimental*

### Recommendation

Add Section 7.1 **but explicitly mark**:

> *Option 3 is the default and only required path for v1.*

This avoids combinatorial explosion.

---

## 6️⃣ Memory and caching strategy (Section 1.6)

**Verdict:** ✅ **Missing and very important**

This is **not a regression** — it is essential for:

* GPU memory safety
* Training speed
* Reproducibility

### Why it matters

Your model has:

* Static structures (banks, signatures)
* Dynamic states (Z layers)
* Semi-static kernels (KΔ)

Without an explicit caching policy:

* Codex may recompute signatures per step
* Kernel rows may be recomputed wastefully
* GPU memory may explode silently

### Strong recommendation

Add **one explicit rule**:

> Cached objects must be **keyed by (dataset_id, sequence_id, scale_id)**
> Never by batch index.

This prevents subtle data leakage bugs.

---

## 7️⃣ Training considerations (Section 10.5)

**Verdict:** ✅ **Missing and necessary**

This is *not* implementation fluff — it affects correctness.

### Especially important points

* Variable `m` per sample
* Padding semantics for set attention
* Gradient detachment rules for geometry

Without this:

* Backends may silently diverge
* Geometry kernel might accidentally become learnable
* Memory scaling becomes unpredictable

### One clarification to add

Explicitly state:

> Geometry kernels **must never receive gradients**,
> but **adapter projections always do**.

You already imply this — just make it explicit as a rule.

---

## 8️⃣ Any regressions or incompatibilities?

**None.**
But there are **two risks to guard against**, which you should explicitly constrain:

### ⚠️ Risk 1 — Quadratic kernel feature misuse

Mitigation:

* Hard threshold for kernel-row Φ
* Automatic fallback to Nyström / landmarks

### ⚠️ Risk 2 — Adapter over-parameterization

Mitigation:

* Linear adapters default
* Nonlinear adapters gated by `m` and `d_h`

---

## Final verdict

### These additions are:

* ✅ **Not regressions**
* ✅ **Not inconsistencies**
* ✅ **Not incompatible with the plan**
* ✅ **Actually necessary to make the plan reviewer-proof and Codex-safe**

They:

* Close theoretical gaps (RKHS, adapters)
* Close implementation gaps (decoding, caching, batching)
* Prevent silent inefficiencies
* Reduce ambiguity to near-zero

## How to solve the above risks and where to add the constraints

## **SOLUTION FOR RISK 1: Quadratic Kernel Feature Misuse**

### Problem Analysis

When using kernel-row features $\Phi(S_i) = K_\Delta[i, :] \in \mathbb{R}^m$:

- **Memory**: $O(m^2)$ for full kernel matrix
- **Computation**: $O(m^2)$ to compute all pairwise distances
- **Acceptable when**: $m < 500$ (≈1MB for fp32)
- **Prohibitive when**: $m > 1000$ (≈4MB+ per sample)

### Mitigation Strategy

**Add automatic fallback hierarchy:**

```python
if m <= 200: # Do not hardcode these, keep it configurable to safe default properly 
             # propagated throug dependencies. This is only an example.
    # Safe: use full kernel-row features
    use_kernel_features = True
    backend = "dense_exact"
elif 200 < m <= 500:
    # Marginal: use kernel with Nyström approximation
    use_kernel_features = True
    backend = "nystrom"
    num_landmarks = min(100, m // 5)
elif m > 500:
    # Unsafe: fallback to projected geometry (A1) or hashed counts (A2)
    use_kernel_features = False
    feature_mode = "geometry_projected"  # or "hashed_counts"
```

### Nyström-based Kernel Approximation

Instead of full $K_\Delta \in \mathbb{R}^{m \times m}$, approximate with $r$ landmarks ($r \ll m$):

$$\Phi_{\text{approx}}(S_i) = K_{i,L} (K_{LL} + \epsilon I)^{-1/2} \in \mathbb{R}^r$$

where:
- $K_{i,L}$: kernel from set $i$ to $r$ landmarks
- $K_{LL}$: kernel among landmarks
- Dimension reduced: $m \to r$ (typically $r = 50$-$100$)

**Implementation:**
```python
# In features/kernel_features.py
if m > kernel_threshold:
    # Select landmarks (uniform or k-means)
    landmark_idx = select_landmarks(m, num_landmarks=100)
    K_mL = compute_kernel(sig, sizes, row_idx=all, col_idx=landmark_idx)  # [m, r]
    K_LL = compute_kernel(sig, sizes, row_idx=landmark_idx, col_idx=landmark_idx)  # [r, r]
    
    # Approximate features
    L_inv_sqrt = torch.linalg.cholesky(K_LL + 1e-6 * torch.eye(r)).inverse().T
    Phi = K_mL @ L_inv_sqrt  # [m, r] instead of [m, m]
```

---

## **SOLUTION FOR RISK 2: Adapter Over-Parameterization**

### Problem Analysis

**Linear adapters**: $A_h \in \mathbb{R}^{d_h \times m}$

Content term rank: $\text{rank}(\langle A_h \Phi_i, B_h \Phi_j \rangle) \leq \min(d_h, m)$

**When this is a problem:**
- If $m = 7$ and $d_h = 64$: rank $\leq 7$ (severely bottlenecked!)
- The content term can only capture 7-dimensional signal

**When this is acceptable:**
- If $m = 100$ and $d_h = 64$: rank $\leq 64$ (full capacity)

**Nonlinear adapters**: $\text{MLP}: \mathbb{R}^m \to \mathbb{R}^{d_h}$

- No rank bottleneck
- More parameters: $(m \cdot h + h \cdot d_h)$ vs $(m \cdot d_h)$
- More capacity but slower and risk overfitting

### Mitigation Strategy

**Decision rule:**

```python
def select_adapter_type(m, d_h, num_heads):
    """
    Returns: 'linear', 'nonlinear', or 'hybrid'
    """
    effective_rank = min(m, d_h)
    
    # Rule 1: If rank is severely limited, need nonlinear
    if effective_rank < 20:
        return 'nonlinear'  # e.g., m=7, need MLP for expressiveness
    
    # Rule 2: If rank is reasonable, linear is sufficient
    elif effective_rank >= 32:
        return 'linear'  # e.g., m=100, linear adapter has full capacity
    
    # Rule 3: Hybrid for borderline cases
    else:  # 20 <= effective_rank < 32
        return 'hybrid'  # Some heads linear, some nonlinear

def create_adapters(adapter_type, m, d_h, num_heads):
    if adapter_type == 'linear':
        # A_h: [num_heads, d_h, m], B_h: [num_heads, d_h, m]
        A = nn.Parameter(torch.randn(num_heads, d_h, m) * 0.01)
        B = nn.Parameter(torch.randn(num_heads, d_h, m) * 0.01)
        return LinearAdapter(A, B)
    
    elif adapter_type == 'nonlinear':
        # MLP per head: m -> hidden -> d_h
        hidden_dim = max(m * 2, 64)
        mlps_A = nn.ModuleList([
            nn.Sequential(
                nn.Linear(m, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d_h)
            ) for _ in range(num_heads)
        ])
        mlps_B = nn.ModuleList([...])  # same structure
        return NonlinearAdapter(mlps_A, mlps_B)
    
    elif adapter_type == 'hybrid':
        # Alternate heads between linear and nonlinear
        adapters = []
        for h in range(num_heads):
            if h % 2 == 0:
                adapters.append(LinearAdapter(...))
            else:
                adapters.append(NonlinearAdapter(...))
        return HybridAdapter(adapters)
```

### Additional Constraint: Parameter Budget

Even with nonlinear adapters, limit total parameters:

```python
# Compute parameter counts
linear_params = 2 * num_heads * d_h * m
nonlinear_params = num_heads * (m * hidden + hidden * d_h) * 2  # for A and B

# Constraint: adapters should not exceed X% of total model params
max_adapter_fraction = 0.15  # 15% of model
total_model_params = compute_total_params(model)

if nonlinear_params > max_adapter_fraction * total_model_params:
    # Fall back to linear or reduce hidden_dim
    hidden_dim = min(hidden_dim, (max_adapter_fraction * total_model_params) // (2 * num_heads * m))
```

---

## **WHERE TO ADD IN THE PLAN**

### **Addition 1: Section 1.5 (Kernel Features) - Add Subsection 1.5.1**

**Location**: Within Section 1.5, after the PyTorch optimization notes

**Title**: 1.5.1 Automatic fallback for large m

**Content**:
```
### 1.5.1 Automatic fallback for large m

**Thresholds:**
- m ≤ 200: Use full kernel-row features Φ ∈ ℝ^m
- 200 < m ≤ 500: Use Nyström-approximated kernel features Φ ∈ ℝ^r (r ≈ 100)
- m > 500: Fallback to geometry_projected (A1) or hashed_counts (A2)

**Nyström kernel approximation:**
[include math from above]

**Implementation guard:**
```python
assert not (use_kernel_features and m > 500), \
    f"Kernel features forbidden for m={m} > 500. Use --feature-mode geometry_projected"
```

**Config validation:**
- If `feature_mode: kernel` and m > 500 → raise ConfigError
- Auto-select: If m > 500, override to `geometry_projected` with warning

**Code mapping:**
- `features/kernel_features.py`: implements threshold checks
- `features/kernel_nystrom.py`: Nyström approximation
- `config/validators.py`: enforces m thresholds
```

---

### **Addition 2: Section 3.1 (Adapters) - Add Subsection 3.1.1**

**Location**: Within Section 3.1, after nonlinear adapter description

**Title**: 3.1.1 Automatic adapter selection by rank

**Content**:
```
### 3.1.1 Automatic adapter selection by rank

**Problem:** Linear adapters have rank ≤ min(d_h, m). When m is small, content term is bottlenecked.

**Decision rule:**
```python
effective_rank = min(m, d_h)
if effective_rank < 20:
    adapter_type = 'nonlinear'  # MLP: m → hidden → d_h
elif effective_rank >= 32:
    adapter_type = 'linear'     # A_h, B_h matrices
else:
    adapter_type = 'hybrid'     # Mix per head
```

**Parameter budget constraint:**
- Adapters limited to 15% of total model parameters
- If nonlinear adapters exceed budget → reduce hidden_dim or use linear

**Config options:**
```yaml
model:
  set_only:
    adapter_type: auto  # or 'linear', 'nonlinear', 'hybrid'
    adapter_budget_fraction: 0.15
    adapter_hidden_multiplier: 2  # hidden_dim = m * multiplier
```

**Implementation guard:**
```python
if adapter_type == 'linear' and effective_rank < 20:
    warnings.warn(f"Linear adapter has low rank {effective_rank}. Consider nonlinear.")
```

**Code mapping:**
- `set_attention/adapters.py`: LinearAdapter, NonlinearAdapter, HybridAdapter
- `set_attention/adapter_factory.py`: select_adapter_type(), create_adapters()
- `config/validators.py`: validates adapter choices against m and d_h
```

---

### **Addition 3: Section 8.3 (Config Validation) - Add Constraint Rules**

**Location**: Within Section 8.3, after the Pydantic validation rules

**Content**:
```
**Additional validation rules:**

1. **Kernel feature constraint:**
   ```python
   if config.features.mode == 'kernel' and config.bank.max_sets > 500:
       raise ValueError(f"Kernel features require m ≤ 500, got {config.bank.max_sets}")
   ```

2. **Adapter rank warning:**
   ```python
   effective_rank = min(config.model.d_model // config.model.num_heads, config.bank.max_sets)
   if config.adapters.type == 'linear' and effective_rank < 20:
       logger.warning(f"Linear adapters have low rank {effective_rank}. Auto-switching to nonlinear.")
       config.adapters.type = 'nonlinear'
   ```

3. **Auto-correction (with logging):**
   - If m > 500 and kernel features requested → switch to geometry_projected
   - If m < 20 and linear adapters requested → switch to nonlinear
   - Log all auto-corrections clearly
```

---

### **Addition 4: Section 12 (No Ambiguity Rules) - Add Rules 6 & 7**

**Location**: Append to Section 12 list

**Content**:
```
6. **Kernel features are forbidden when m > 500.** Config validation must enforce fallback to A1/A2.

7. **Linear adapters with effective rank < 20 must trigger a warning** or auto-switch to nonlinear (configurable).
```

---

### **Addition 5: New Section 1.7 - Computational Complexity Table**

**Location**: Add after Section 1.6 (Caching Strategy)

**Title**: 1.7 Complexity and threshold summary

**Content**:
```
### 1.7 Complexity and threshold summary

| Component | Memory | Compute | Threshold | Fallback |
|-----------|--------|---------|-----------|----------|
| Kernel features (full) | O(m²) | O(m²) | m ≤ 200 | Nyström |
| Kernel features (Nyström) | O(mr) | O(mr) | m ≤ 500 | A1/A2 |
| Geometry projected (A1) | O(md) | O(md) | Always safe | N/A |
| Hashed counts (A2) | O(d_c m) | O(|S|d_c) | Always safe | N/A |
| Linear adapters | O(Hd_h m) | O(Hd_h m) | rank ≥ 20 | Nonlinear |
| Nonlinear adapters | O(Hmh + Hhd_h) | O(Hmh + Hhd_h) | rank < 20 | N/A |

**Where:**
- m: number of sets
- r: number of landmarks (50-100)
- d: model dimension
- d_h: head dimension
- H: number of heads
- h: hidden dimension in MLP

**Decision boundaries:**
- Use kernel features: m ≤ 200 (full), m ≤ 500 (Nyström)
- Use nonlinear adapters: effective_rank = min(m, d_h) < 20
- Use hybrid adapters: 20 ≤ effective_rank < 32
```

---

## **SUMMARY: Implementation Checklist**

To properly guard against these risks, you need:

1. ✅ **Add threshold checks** in `features/kernel_features.py`
2. ✅ **Implement Nyström approximation** in `features/kernel_nystrom.py`
3. ✅ **Add adapter factory** with automatic selection in `set_attention/adapter_factory.py`
4. ✅ **Add config validators** in `config/validators.py`
5. ✅ **Update config schema** to include `adapter_type: auto` and thresholds
6. ✅ **Add unit tests** for threshold enforcement in `tests/test_threshold_guards.py`

**All changes integrate cleanly into the existing plan structure. No rewrites needed—just additions at the marked locations.**

## Two Extra Micro-Guards for the Risks

The above mitigation strategy is solid, well-targeted, and *exactly* the right way to close the two real risks**.
However, with a careful classification of *what’s excellent*, *what needs a small tightening*, and *two extra micro-guards* I strongly recommend adding so this becomes essentially bullet-proof.

---

## Overall verdict

Your proposed solutions are:

* ✅ **Conceptually correct**
* ✅ **Consistent with the latest Codex plan**
* ✅ **Non-regressive**
* ✅ **Implementation-realistic**
* ✅ **Reviewer-defensible**
* ✅ **Codex-safe (low ambiguity)**

They do **not** introduce incompatibilities or hidden inefficiencies. Instead, they **formalize implicit assumptions** that *must* be explicit in a production-grade research codebase.

What you’ve done here is exactly what strong systems papers do *between* v1 and v2: turn informal heuristics into **hard constraints + automatic fallbacks**.

---

# Evaluation by risk

---

## 🔴 Risk 1: Quadratic kernel feature misuse

### Verdict: ✅ **Mitigation is correct and sufficient**

This is one of the most important sections you added, and your solution is *better than what many published systems do*.

### Why your solution works

You correctly identify the *real* failure modes:

* Memory blow-up at (O(m^2))
* Silent GPU OOMs
* Accidental quadratic compute hidden inside “feature extraction”

Your **three-tier fallback hierarchy** is the right abstraction:

| m range       | Feature             | Backend   | Status     |
| ------------- | ------------------- | --------- | ---------- |
| m ≤ 200       | Full kernel rows    | dense     | Safe       |
| 200 < m ≤ 500 | Nyström kernel rows | nystrom   | Controlled |
| m > 500       | A1 / A2             | projected | Safe       |

This is **not arbitrary**—the thresholds are defensible both computationally and empirically.

### Nyström feature formulation

This is particularly good:

[
\Phi_{\text{approx}}(S_i) = K_{i,L}(K_{LL}+\epsilon I)^{-1/2}
]

Why this matters:

* It preserves the **RKHS geometry**
* It gives you a *proper low-rank feature map*
* It cleanly separates *feature construction* from *attention backend*

Many people incorrectly mix Nyström *attention* with Nyström *features*. You didn’t.

### One small tightening I recommend

Add **one more invariant** (very small, but important):

> **Kernel-row features must never be used with `local_band` backend unless explicitly forced.**

Reason:

* Band + kernel-row features double-encode locality
* This can collapse effective rank and hurt expressiveness

Add this guard in config validation:

```python
if features.mode == "kernel" and backend == "local_band":
    warnings.warn("Kernel features + band backend may be redundant; consider geometry_projected.")
```

This is a *warning*, not an error.

---

## 🔴 Risk 2: Adapter over-parameterization / rank bottleneck

### Verdict: ✅ **Excellent mitigation; this is exactly right**

Your analysis of **effective rank** is spot-on and mathematically correct:

[
\text{rank} \le \min(m, d_h)
]

And your decision logic is clean, explainable, and automatable.

### What you did *very well*

1. You identified the *true failure regime*:

   * Small m + linear adapters = crippled content term
2. You avoided overreaction:

   * Linear adapters are still preferred when rank is sufficient
3. You provided:

   * Automatic selection
   * Hybrid fallback
   * Parameter budget constraint

This is **very strong design**.

### Parameter budget constraint: especially good

This part is unusually mature for research code:

```python
max_adapter_fraction = 0.15
```

Why this matters:

* Prevents adapters from dominating model capacity
* Keeps comparisons fair vs baseline
* Avoids “hidden parameter inflation”

Reviewers *love* seeing this.

### One micro-refinement I recommend

Add **head-group consistency**:

> If using hybrid adapters, ensure *at least one nonlinear head per scale* (in multiscale mode).

Reason:

* Otherwise a coarse scale could end up with only linear heads and low rank

Implementation guard:

```python
if adapter_type == "hybrid" and multiscale:
    assert any(h.adapter == "nonlinear" for h in heads_per_scale[s])
```

This keeps expressiveness balanced.

---

# Evaluation of placement in the plan

Your proposed insert locations are **exactly right**.

### Summary check

| Addition                               | Status                 |
| -------------------------------------- | ---------------------- |
| Section 1.5.1 (kernel fallback)        | ✅ Perfect placement    |
| Section 3.1.1 (adapter auto-selection) | ✅ Perfect              |
| Section 8.3 (config validation rules)  | ✅ Necessary            |
| Section 12 rules 6–7                   | ✅ Good hard invariants |
| Section 1.7 (complexity table)         | ✅ Very good idea       |

The **complexity table** is especially useful—it acts as a *design checksum* for future contributors.

---

# Are there any regressions or incompatibilities introduced?

**No.**
But there are **two subtle things to watch**, both easy to fix.

---

## ⚠️ Subtle point 1: Auto-correction vs reproducibility

Auto-switching adapters or feature modes is great—but:

> **Auto-corrections must be logged and stored in W&B config.**

Add this rule explicitly:

```python
wandb.config.update({"_auto_corrected": True, "adapter_type_final": adapter_type})
```

Otherwise, sweep results can become ambiguous.

---

## ⚠️ Subtle point 2: Thresholds must be overridable (expert mode)

Add:

```yaml
features:
  kernel:
    allow_unsafe: false
```

This allows:

* Stress testing
* Ablation experiments
* Controlled failures

But default remains safe.

---

# Final assessment

### Is this mitigation strategy sufficient?

**Yes — and more than sufficient.**

### Does it:

* eliminate the two major risks? → **Yes**
* preserve expressiveness? → **Yes**
* avoid silent inefficiencies? → **Yes**
* integrate cleanly into the existing plan? → **Yes**
* reduce ambiguity for Codex and humans? → **Yes**

What you have now is not just a mitigation strategy — it’s a **design contract**.

---

## Recommendation

I strongly recommend that in the **final Codex plan**, you:

* Treat these additions as **non-optional invariants**
* Mark them explicitly as **“Safety & Expressiveness Guards”**
* Reference them in both:

  * config validation
  * test coverage

## FINAL CONSIDERATIONS

## **Final details**

### 1. **Section 7.2 Specific Placement**
- Above I mentioned the guard but didn't specify creating a new subsection 7.2
- I added the exact structure: "7.2 Adapter distribution across scales"

### 2. **Test Coverage Specification (Section 9.2 expansion)**
I mentioned tests conceptually, but I added **specific test files**:
```
tests/
  test_threshold_guards.py
  test_adapter_selection.py
  test_config_validation.py
  test_unsafe_overrides.py
  test_multiscale_balance.py  ← This one specifically
```

### 3. **Complexity Budget Logging Function**
The actual `log_complexity_budget()` function with wandb metrics (for the selected attention mechanism (through heads and layers) and for the rest of the model separately):
```python
metrics = {
    'complexity/kernel_memory_mb': ...,
    'complexity/adapter_params': ...,
    'complexity/effective_rank': ...,
    'complexity/<other_relevant_memory_footprint_metrics>':
}
```

### 4. **Documentation Strategy Split**
- Separating "For Paper Submission" vs "For Internal/Public Repo"
- Suggesting `docs/SAFETY_GUARDS.md`

### 5. **The `_is_ablation_run` Flag Concept**
```python
if not config._is_ablation_run:
    raise ValueError("allow_unsafe only permitted with --ablation flag")
```

### 6. **Explicit YAML Structure for Unsafe Override**
```yaml
features:
  kernel:
    allow_unsafe: false
    unsafe_max_m: null  # explicit upper bound
```

### 7. **Integration Checklist with Section Numbers**
The numbered checklist mapping each guard to specific sections (1.5.1, 1.7, 3.1.1, etc.)

---

## **What I Just Elaborated/Reformatted (Was Already in Your Doc)**

- ✓ Overall verdict
- ✓ Extra Guard 1 (kernel+band warning) - **concept was yours**
- ✓ Extra Guard 2 (multiscale adapter balance) - **concept was yours**
- ✓ Subtle Point 1 (auto-correction logging) - **concept was yours**
- ✓ Subtle Point 2 (expert override) - **concept was yours**
- ✓ Why the guards matter (your doc had this)
- ✓ Code examples for guards (I added more detail, but basic concept was there)

---

## **Summary**

1. Section 7.2 placement specification
2. Test file names/structure
3. Complexity logging function
4. `_is_ablation_run` mechanism
5. Documentation strategy split
6. Complete YAML schema for unsafe mode
7. Numbered integration checklist
