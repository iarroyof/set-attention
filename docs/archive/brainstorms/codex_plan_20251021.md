Here’s a **Codex Plan** that keeps the **correct project structure + code snippets** from our latest project version and **including details** (theory, learning behavior, guarantees, acceptance criteria, profiling targets, ablations, etc.) from the earlier write-up.

---

# AUSA Implementation Plan (LM, Diffusion, ViT) — **Merged Spec**

## Architecture updates (Jan 2026)

These updates supersede earlier assumptions about per-run tokenization and on-the-fly bank/routing construction.

- Artifact cache system added (`src/set_attention/data/artifact_cache.py`, `src/set_attention/data/ska_artifacts.py`) with fingerprinted meta for tokens, banks, and routing. Training scripts support `--cache-mode none|tokens|full`, `--cache-only`, and cache guards (e.g., adapter-rank for full routing). Cache builders: `scripts/cache_tokens.py`, `scripts/cache_ska_artifacts.py`. Sweep runners can `--precache`.
- HuggingFace cache rooting unified via `ensure_hf_cache`; env-first behavior using `HF_HOME`, `HF_DATASETS_CACHE`, and `HF_HUB_CACHE`, avoiding legacy per-script cache paths.
- Data loading standardized on DataLoader pipelines with deterministic eval seeding, `worker_init_fn`, and per-task `--num-workers` controls. Full-cache runs force `num_workers=0` to avoid worker churn.
- Sweep robustness upgrades: GPU idle gate + min-free-GB checks, post-run GPU checks, OOM/exitcode rows in CSVs, and sequential execution defaults.
- Artifact generation expanded: `scripts/repro_runs.py` aggregates metrics alongside benchmarks; `scripts/make_stage_artifacts.py` emits Stage A/B tables/plots when inputs are available.
- AUSA tokenizer caching is persistent for Seq2Seq and reused across runs to keep token IDs stable.
- Token caches are tensor-only (no string pairs); Seq2Seq banks now build from token IDs, and refs are derived on-the-fly to avoid host RAM blowups.
- Stage A/B runners and cache scripts expose paper hyperparameters explicitly; cache fingerprints include tokenizer type + seq/window/stride/minhash/router, and full-cache reuse is guarded.

## 0) Core idea (one paragraph)

We **freeze atom discovery** inside training steps. A tokenizer builds an **Active Universe** (U^*) (atoms). We keep a **Universe Pool** with per-atom features (\phi(u)) and MinHash signatures. Each sequence (or image/patch grid) is represented by a **bank of subsets** (multiple sets per sequence). The model runs **set-level attention** (tiny set×set per example block), then **routes tokens/patches** to those set outputs with light adapters—no “sets per token” replication. Optional **learned set prototypes** act as a global set basis.

---

## 1) Repo layout (current modules / new scripts)

```
src/set_attention/
  tokenizers/
    active_tokenizer.py
  training/
    text_utils.py            # encode_sentence, ids_to_tokens, TokenSetStore, text_batch_iterator
  sets/
    banked.py                # BankedSetBatch(values, set_offsets, seq_offsets)
    bank_builders.py         # build_windowed_bank_from_texts, build_windowed_bank_from_ids
    bank_cache.py            # SetBankCache (Sig/Size cache; per-batch Φ pooling)
    bank_utils.py            # gather_bank_batch helper
    atom_adapter.py          # low-rank adapter over atom embeddings
  kernels/
    sketches.py              # MinHasher.sketch_vec (vectorized)
  heads/
    banked_attention.py      # SetBankAttention (multi-head Δ-RBF + content)
    token_router.py          # TokenSetRouter (token→set gating)
  utils/
    profiling.py             # wall time, CPU%, VRAM, steps/s

scripts/
  train_seq2seq_text.py          # tokenizer-aware SKA
  train_seq2seq_text_banked.py   # banked set attention (reference)
  train_toy_seq2seq.py           # toy seq2seq (shared helpers)
  train_toy_lm_banked.py         # draft banked LM
  train_toy_diffusion_banked.py  # draft banked diffusion
  train_tiny_vit_banked.py       # draft banked ViT
```

---

## 2) Canonical data structures

### 2.1 Banked sets (two-level CSR)

```python
@dataclass
class BankedSetBatch:
    values: torch.LongTensor      # (NNZ,) concatenated atom IDs
    set_offsets: torch.LongTensor # (Nsets+1,) CSR over values
    seq_offsets: torch.LongTensor # (B+1,)   CSR over sets

    def to(self, device): ...
# Sets for sequence b: indices [seq_offsets[b], seq_offsets[b+1])
# Atoms for set j: values[ set_offsets[j] : set_offsets[j+1] ]
```

### 2.2 Set bank cache (what to cache vs. recompute)

* **Cache once (no grad):** `sig_sets (Nsets,k)`, `size_sets (Nsets,)`, `values`, `set_offsets`, `seq_offsets`.
* **Recompute per batch (with grad):** pooled set features `Φ_sets` via **differentiable** CSR sum from the **current** atom features.

```python
phi_cur = adapter(atom_emb.weight)              # (V, D_atom), differentiable
Phi = cache.pool_phi_for_sets(set_idx, phi_cur) # uses CSR index_add_, differentiable
```

### 2.3 Token set store (seq2seq SKA)

`training/text_utils.py::TokenSetStore` wraps **precomputed token sets + signatures** per sequence; `.gather(batch_idx)` returns concatenated values/offsets/sigs for the batch.

---

## 3) Set-level attention and token routing

### 3.1 SetBankAttention (multi-head)

* Jaccard via MinHash: `J = mean(eq(Sig_q[:,None,:], Sig_k[None,:,:]))`.
* Symmetric difference (closed form): $\Delta = |A| + |B| - 2|A\cap B|$, with $|A\cap B| \approx \frac{J}{1+J}(|A|+|B|)$.
* Δ-RBF: $exp(-\gamma * Δ)$.
* Optional content term: $(A Φ_q) @ (B Φ_k)^T$.
* Softmax **per sequence block**.
* Values: $V_s([Φ_k, ψ(|K|)]) → (nk,H,Dh)$; head output per query-set: $Z_{sets} = W @ V_s$.

### 3.2 TokenSetRouter (no set replication)

* Gates per token over **its sequence’s** query-set bank:
  `G = softmax( tokens @ W_gate · descriptors^T )` with optional **top-k**.
* Mix per-set head outputs to token level: `Σ_j G[ℓ,j] · Z_sets[j,:,:] → merge heads → linear out`.

---

## 4) Low-rank atom feature adapter

**AtomFeatureAdapter:** $\tilde{\phi}(u)=\phi(u)+\alpha W_2 W_1 \phi(u)$.

**Rule:** **Always** call the adapter in the forward pass and **pool Φ per batch**, not offline; otherwise the adapter won’t receive gradients.

---

## 5) Reference training pattern (seq2seq, banked)

1. Train/load tokenizer → build banks via `build_windowed_bank_from_texts`.
2. `SetBankCache.precompute` **(Sig/Size only)**.
3. Per batch:

   * Token embedding/backbone.
   * `gather_bank_batch` → set indices/pointers for blockwise attention.
   * `phi_cur = adapter(Embedding.weight)`; `Φ_q/k = pool_phi_for_sets(...)`.
   * Gather `Sig_q/k`, `Size_q/k`.
   * `SetBankAttention` → `TokenSetRouter`.
   * Linear head + CE loss.

---

## 6) Language modeling (LM generation)

* Use `train_toy_lm_banked.py` as prototype.
* IDs ↔ text for logs; banks via `build_windowed_bank_from_ids`.
* Metrics: loss, perplexity; optional BLEU on toy.
* CLI: `--window/--stride`, `--adapter-rank`, `--router-topk`, `--minhash-k`.

---

## 7) Diffusion

### 7.1 Text diffusion (`train_toy_diffusion_banked.py`)

* Denoiser backbone with AUSA blocks; collect banks from token IDs.
* Loss: ε-prediction (or v-prediction).
* Eval: MMD/1-NN/qualitative samples.

### 7.2 Image diffusion (UNet/DiT) — future

* Patches → IDs (e.g., VQ codebook).
* Banks from windows/superpixels; AUSA in attention blocks.
* Eval: FID/sFID/Inception.

---

## 8) ViT (classification / SSL)

* Patch tokens as usual; banks from windows or proposals.
* AUSA blocks mix **region-level** set outputs into patch tokens (or CLS).
* Loss: CE (supervised) or DINO/MoCo (SSL).

---

## 9) **Theory addendum & guarantees** (missing details restored)

### 9.1 Embedding of unseen sets (stability)

For any two sets $(S,T\subseteq U^*)$,
\[
\Phi(S)-\Phi(T)=\sum_{u\in S\triangle T}\sigma(u),\phi(u),\quad
|\Phi(S)-\Phi(T)|\le M,|S\triangle T| ;(\text{if }|\phi(u)|\le M).
\]
Thus $S\mapsto\Phi(S)$ is **Lipschitz** in symmetric-difference; unseen sets close to seen ones map nearby.

### 9.2 PSD and characteristic properties

* The **Δ-RBF** kernel $k_\Delta(S,T)=\exp(-\gamma |S\triangle T|)$ is **positive definite** on finite sets when $\gamma>0$ (via Schoenberg: negative-type metric $|S\triangle T|\Rightarrow$ Gaussian of negative type is PD).
* Adding a content inner product $\langle A\Phi(S),B\Phi(T)\rangle$ preserves PD when $A,B$ are linear.
* With sufficiently rich $\phi$ (e.g., universal RKHS features) the combined kernel is **characteristic**; Nyström/RFF approximations introduce an $\varepsilon$ distortion controlled by rank/features.

### 9.3 Learning “attention sets”

* **Observed banks:** learn scoring params $\theta={\gamma,\beta,\tau,A,B}$, value MLP, and token routing. Attention weights $W_\theta$ are **fully learned** (as in dot attention) but over set geometry.
* **Prototype bank (optional):** learn Bernoulli membership vectors $p_j\in[0,1]^{|U^*|}$.
  $\Phi(P_j)=\Phi_Z^\top p_j,;\mathbb{E}[\Delta(S,p_j)]=|S|+|p_j|*1-2\sum*{u\in S}p_{j,u}$.
  Differentiable; regularize with $\ell_1$ or entropy.

---

## 10) Engineering invariants (must-keep)

* **No AUT in step time.** Tokenizer growth only at epoch boundaries (if at all).
* **Cache Sig/Size only.** **Never** cache Φ with grads across batches.
* **Vectorize** MinHash (`sketch_vec`) and pooling (CSR `index_add_`).
* **Blockwise ops** per sequence via `seq_offsets`—no cross-sequence matmuls.
* **Keep tensors on one device**; avoid CPU loops in the hot path.

---

## 11) Acceptance criteria & profiling targets

* **Tiny overfits** (seq2seq, LM, diffusion, ViT) in < N steps; loss ↓ monotonically.
* **GPU util** ≥ 60% at medium batch size; **VRAM** stable with longer sequences (banks small).
* **Throughput:** banked AUSA faster than dense token×token attention for long sequences (report steps/s).
* **Ablations improve understanding:** router top-k, adapter rank, MinHash-k, bank size.

---

## 12) CLI & flags (consistency)

* `--window`, `--stride` (bank construction)
* `--adapter-rank` (0 = frozen; >0 = trainable)
* `--router-topk` (0 = dense; >0 = sparse per token)
* `--minhash-k` (default 128; 32/64 for speed ablations)
* `--tokenizer PATH` (log **Loading** vs **Training**)
* `--prof` (enable profiling summary)

---

## 13) Tests (unit & integration)

* **Bank slicing:** `gather_bank_batch` shapes; last batch handling.
* **Pooling grads:** `pool_phi_for_sets` propagates grads to adapter (gradcheck on toy).
* **MinHash vs Jaccard:** equality fraction ≈ true Jaccard on small sets.
* **Router top-k:** indices stable; renormalized softmax correct.
* **End-to-end tiny tasks:** copy/reverse seq2seq (CE ↓), toy LM (PPL ↓), toy diffusion (ε-MSE ↓), tiny ViT (acc ↑ vs random).

---

## 14) Failure modes & fixes

* **“Backward through graph twice”:** you cached Φ with grads → **fix**: pool Φ per batch inside forward.
* **Low GPU util:** Python loops or per-step tokenization → **fix**: precompute Sig/Size/banks; keep on device; vectorize pooling.
* **Router collapse:** add entropy regularizer on gates; enable `--router-topk`.
* **OOB vocab indices:** ensure `<pad>, <s>, </s>, <unk>` are present and embedding sized to `stoi`.
* **Poor Δ accuracy:** increase `--minhash-k` (start at 128; drop only after validation).

---

## 15) Immediate TODOs (next agent)

* Fix `train_toy_lm_banked.py` vocab alignment (`stoi` ⇔ embedding size).
* Finalize `train_toy_diffusion_banked.py` (splits, logging).
* Polish `train_tiny_vit_banked.py` (accuracy logging, bank coverage stats).
* Factor repeated bank logic into `bank_utils.py` if needed.
* Add the tests in §13; wire `--prof` to `utils/profiling.py`.

---

## 16) Reference snippets (kept)

### 16.1 Pooling with grads

```python
def pool_phi_for_sets(values, set_offsets, set_idx, phi_cur):
    start = set_offsets[set_idx]
    end   = set_offsets[set_idx + 1]
    counts = end - start
    row_ids = torch.repeat_interleave(
        torch.arange(set_idx.numel(), device=phi_cur.device), counts
    )
    atom_ids = torch.cat([values[s:e] for s, e in zip(start.tolist(), end.tolist())])
    D = phi_cur.size(1)
    Phi = torch.zeros(set_idx.numel(), D, device=phi_cur.device, dtype=phi_cur.dtype)
    Phi.index_add_(0, row_ids, phi_cur[atom_ids])
    return Phi
```

### 16.2 Set scores

```python
def set_scores(Phi_q, Sig_q, Size_q, Phi_k, Sig_k, Size_k, A, B, gamma, beta, tau):
    J = (Sig_q[:, None, :] == Sig_k[None, :, :]).float().mean(-1)
    SA, SB = Size_q[:, None].float(), Size_k[None, :].float()
    inter = (J / (1 + J + 1e-8)) * (SA + SB)
    Delta = (SA + SB - 2 * inter).clamp_min(0.0)
    S = torch.exp(-gamma * Delta)
    if beta != 0.0 and A is not None and B is not None:
        S = S + (A(Phi_q) @ B(Phi_k).T)
    return S / tau
```

---

## 17) One-line handoff

**Use the existing bank builders/cache/router from `train_seq2seq_text_banked.py` to finish LM, Diffusion, and ViT trainers.** Cache Sig/Size only; **pool Φ per batch with the adapter** so it truly trains; run blockwise set×set attention and token routing; profile and ablate (`top-k`, adapter rank, minhash-k, bank size) on tiny tasks before scaling.
---

# Current Status Snapshot (2025-10-21)

- Unified all banked trainers on the shared UniversePool + SetFeatureCache flow (scripts/train_seq2seq_text_banked.py, 	rain_toy_lm_banked.py, 	rain_toy_diffusion_banked.py, 	rain_tiny_vit_banked.py).
- Added src/set_attention/universe/ module providing reusable pooling/gather helpers (UniversePool, SetFeatureCache.gather_padded).
- Added regression coverage 	ests/test_set_feature_cache.py (MinHash precondition + differentiable pooling) — passes on the 	orch conda env.
- Smoke commands (CPU-friendly):

The repo is at commit Refactor banked trainers around UniversePool cache (main branch).

# Modularization Priorities (Next Steps)

1. **Bank Construction API** — create a central helper (e.g. set_attention.training.bank_registry) that returns (SetFeatureCache, stats) for text, id, and patch banks, eliminating manual MinHash wiring in scripts.
2. **Trainer Harness** — factor shared training loops into reusable utilities with callbacks for model build / forward / metrics; scripts should only register components.
3. **Dataset Adapters** — move ad-hoc dataset prep into set_attention.data.* modules with typed configs (text corpora, CIFAR patches, toy diffusion sequences).
4. **Model Factories** — expose set_attention.models.ausa_* modules that assemble backbones + banked heads, so scripts simply configure them.
5. **Configuration Layer** — adopt YAML/dataclass experiment configs parsed by scripts, making publication experiments reproducible without copying argparse blocks.
6. **Testing & CI** — extend unit tests to cover router + set-attention integration, ensure fast CPU coverage, and spec out CI instructions.

## Additional information

- Centralize the universe/bank wiring in a reusable helper, e.g. set_attention.training.bank_registry that exposes build_text_bank, build_id_bank, build_patch_bank, so scripts like scripts/train_seq2seq_text_banked.py:183 and scripts/train_toy_diffusion_banked.py:129 stop duplicating the MinHasher/SetFeatureCache ceremony.
- Factor a common trainer harness (set_attention.training.runner) with callbacks for model build, loss, and metrics; current loops in scripts/train_toy_lm_banked.py:134, train_toy_diffusion_banked.py:151, and train_seq2seq_text_banked.py:236 share the same skeleton (profiling, adapter wiring, cache gathers).
- Extract dataset adapters into a dedicated module (set_attention.data.*) so text/vision/diffusion corpora can be declared declaratively; today each script assembles text pairs or CIFAR subsets ad hoc.
- Move the “build backbone + banked head” logic into model factories under set_attention.models, giving you ausa_lm, ausa_seq2seq, ausa_diffusion, ausa_vit modules that scripts simply import and configure.
- Adopt a config layer (YAML + OmegaConf or simple dataclasses) to define experiments; scripts would parse a config path, freeing you to add new experiments without copying argparse blocks (e.g. the long option lists at scripts/train_seq2seq_text_banked.py:75 and scripts/train_tiny_vit_banked.py:106).
- Expand unit coverage for SetFeatureCache integration—add tests that mirror the gating path (e.g. run TokenSetRouter with a tiny cache) so future refactors don’t regress the gather logic.
- Document the new architecture in docs/ (a quick “how to add a new banked script” guide) once the modules exist, keeping publication-facing experiments reproducible.

---

# Current Status Snapshot (2025-10-21)

- Unified all banked trainers on the shared `UniversePool` + `SetFeatureCache` flow (`scripts/train_seq2seq_text_banked.py`, `train_toy_lm_banked.py`, `train_toy_diffusion_banked.py`, `train_tiny_vit_banked.py`).
- Added `src/set_attention/universe/` module providing reusable pooling/gather helpers (`UniversePool`, `SetFeatureCache.gather_padded`).
- Added regression coverage `tests/test_set_feature_cache.py` (MinHash precondition + differentiable pooling) — passes on the `torch` conda env.
- Smoke commands (CPU-friendly):
  * `python scripts/train_seq2seq_text_banked.py --demo --epochs 1 --batch 4 --device cpu`
  * `python scripts/train_toy_lm_banked.py --epochs 1 --batch 32 --device cpu --window 4 --stride 2 --minhash-k 16`
  * `python scripts/train_toy_diffusion_banked.py --epochs 1 --steps 10 --device cpu --window 4 --stride 2 --minhash-k 16 --config configs/diffusion_toy.yaml`
  * `python scripts/train_tiny_vit_banked.py --epochs 1 --batch 64 --device cpu --patch 4 --limit 128 --window 4 --stride 2 --minhash-k 16`

The repo is at commit `Refactor banked trainers around UniversePool cache` (main branch).

# Modularization Priorities (Next Steps)

1. **Bank Construction API** — create a central helper (e.g. `set_attention.training.bank_registry`) that returns `(SetFeatureCache, stats)` for text, id, and patch banks, eliminating manual MinHash wiring in scripts.
2. **Trainer Harness** — factor shared training loops into reusable utilities with callbacks for model build / forward / metrics; scripts should only register components.
3. **Dataset Adapters** — move ad-hoc dataset prep into `set_attention.data.*` modules with typed configs (text corpora, CIFAR patches, toy diffusion sequences).
4. **Model Factories** — expose `set_attention.models.ausa_*` modules that assemble backbones + banked heads, so scripts simply configure them.
5. **Configuration Layer** — adopt YAML/dataclass experiment configs parsed by scripts, making publication experiments reproducible without copying argparse blocks.
6. **Testing & CI** — extend unit tests to cover router + set-attention integration, ensure fast CPU coverage, and spec out CI instructions.
7. **Documentation** — add a "How to add a banked experiment" guide plus MinHash/adapter tuning notes for the paper supplement.
---

# Current Status Snapshot (2025-10-21, Detailed)

## Repository Structure (key paths)
```
.
├── docs/
│   ├── codex_plan.md
│   ├── codex_plan_20251021.md
│   └── codex_plan_diffucion_vision.md
├── scripts/
│   ├── train_seq2seq_text_banked.py
│   ├── train_seq2seq_text.py
│   ├── train_toy_lm_banked.py
│   ├── train_toy_diffusion_banked.py
│   ├── train_tiny_vit_banked.py
│   └── (toy + baseline trainers)
├── src/set_attention/
│   ├── universe/
│   │   ├── __init__.py
│   │   ├── pool.py
│   │   └── feature_cache.py
│   ├── sets/
│   │   ├── banked.py
│   │   ├── bank_builders.py
│   │   ├── bank_cache.py
│   │   ├── bank_utils.py
│   │   └── atom_adapter.py
│   ├── heads/
│   │   ├── banked_attention.py
│   │   └── token_router.py
│   ├── kernels/sketches.py
│   ├── training/text_utils.py
│   └── utils/profiling.py
├── tests/
│   └── test_set_feature_cache.py
└── configs/
    └── diffusion_toy.yaml
```

## Key Modules & Snippets

```python
# src/set_attention/universe/pool.py
class UniversePool:
    def __init__(self, U_ids, phi_bank=None, mh_sigs=None, metadata=None):
        self.U_ids = U_ids.clone().long()
        self.phi_bank = phi_bank.clone() if phi_bank is not None else None
        self.mh_sigs = mh_sigs.clone() if mh_sigs is not None else None
        self.metadata = metadata.copy() if metadata else {}
        self._build_lookup()  # ensures id→position searchsorted helpers
```

```python
# src/set_attention/universe/feature_cache.py
def gather_padded(self, batch_indices: torch.Tensor, phi_cur: torch.Tensor):
    if self.seq_offsets is None or self.sig_sets is None:
        raise RuntimeError("seq_offsets and MinHash signatures required")
    set_idx, ptrs = self.gather_sequence_sets(batch_indices.to(self.seq_offsets.device))
    ...  # builds padded Φ/σ/|S|/mask tensors per sequence, using compute_phi_for_indices
    return Phi_pad, Sig_pad, Size_pad, mask
```

```python
# src/set_attention/heads/banked_attention.py
class SetBankAttention(nn.Module):
    def forward(self, phi_q, sig_q, size_q, mask_q, phi_k, sig_k, size_k, mask_k):
        jacc = (sig_q.unsqueeze(2) == sig_k.unsqueeze(1)).float().mean(-1)
        delta = ...  # symmetric difference via MinHash
        scores = torch.exp(-self.gamma * delta)
        ...
        return Z  # (batch, n_q, num_heads, head_dim)
```

```python
# src/set_attention/heads/token_router.py
class TokenSetRouter(nn.Module):
    def forward(self, tokens, set_values, phi_sets, mask):
        gates = self._compute_gates(tokens, phi_sets, mask)
        mixed = torch.einsum('bqh,bqhd->bhd', gates, set_values)
        return self.out_proj(mixed.reshape(tokens.size(0), tokens.size(1), -1))
```

```python
# scripts/train_seq2seq_text_banked.py (excerpt)
universe = UniversePool(torch.arange(V, device=device), metadata={"tokenizer": args.tokenizer or "inline"})
cache_src = SetFeatureCache(universe, src_bank.values, src_bank.set_offsets, src_bank.seq_offsets, minhash=train_mh).to(device)
...
Phi_q, Sig_q, Size_q, mask_q = cache_tgt.gather_padded(batch_idx_tensor, phi_dynamic)
Phi_k, Sig_k, Size_k, mask_k = cache_src.gather_padded(batch_idx_tensor, phi_dynamic)
Z_sets = set_attn(Phi_q, Sig_q, Size_q, mask_q, Phi_k, Sig_k, Size_k, mask_k)
```

```python
# tests/test_set_feature_cache.py
universe = UniversePool(torch.tensor([10, 20, 30, 40]))
cache = SetFeatureCache(universe, values, set_offsets, seq_offsets, minhash=MinHasher(k=8))
phi_cur = torch.tensor([[1,0],[0,1],[1,1],[2,0]], dtype=torch.float32)
Phi, Sig, Size, mask = cache.gather_padded(torch.tensor([0, 1]), phi_cur)
assert torch.allclose(Phi[0, 0], torch.tensor([1.0, 1.0]))
```

## Where to Hook New Experiments
- **Shared universe helpers:** `src/set_attention/universe/` (import `UniversePool`, `SetFeatureCache`).
- **Bank construction:** `src/set_attention/sets/bank_builders.py` (windowed banks for text/ID/patch sequences).
- **Adapters & routing:** `src/set_attention/sets/atom_adapter.py`, `src/set_attention/heads/token_router.py`.
- **Reference training loop:** `scripts/train_seq2seq_text_banked.py` shows end-to-end usage (argparse → data → SetFeatureCache → SetBankAttention → TokenSetRouter → metrics).
- **Evaluation utilities:** `set_attention.experiments.nlp_eval`, `set_attention.utils.profiling` for consistent metrics and profiling output.

Use the commands listed in the snapshot to verify environments; all paths above are relative to the repo root (`set-attention/`).
