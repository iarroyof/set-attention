Here’s a **self-contained plan** that includes the right context from this thread, so an agent can immediately continue the work, upgrade SKA to an **optimized, Flash-comparable** implementation, and add trainers for **LM / Diffusion / ViT**. If the agent hits missing files or names or information, it should ask for the specific snippet BEFORE START CODING. A new implementation should be completely generated 

---

# Codex Plan — Active-Universe Set Attention (AUSA)

## Architecture updates (Jan 2026)

These updates supersede earlier assumptions about per-run tokenization and on-the-fly bank/routing construction.

- Artifact cache system added (`src/set_attention/data/artifact_cache.py`, `src/set_attention/data/ska_artifacts.py`) with fingerprinted meta for tokens, banks, and routing. Training scripts support `--cache-mode none|tokens|full`, `--cache-only`, and cache guards (e.g., adapter-rank for full routing). Cache builders: `scripts/cache_tokens.py`, `scripts/cache_ska_artifacts.py`. Sweep runners can `--precache`.
- HuggingFace cache rooting unified via `ensure_hf_cache`; env-first behavior using `HF_HOME`, `HF_DATASETS_CACHE`, and `HF_HUB_CACHE`, avoiding legacy per-script cache paths.
- Data loading standardized on DataLoader pipelines with deterministic eval seeding, `worker_init_fn`, and per-task `--num-workers` controls. Full-cache runs force `num_workers=0` to avoid worker churn.
- Sweep robustness upgrades: GPU idle gate + min-free-GB checks, post-run GPU checks, OOM/exitcode rows in CSVs, and sequential execution defaults.
- Artifact generation expanded: `scripts/repro_runs.py` aggregates metrics alongside benchmarks; `scripts/make_stage_artifacts.py` emits Stage A/B tables/plots when inputs are available.
- AUSA tokenizer caching is persistent for Seq2Seq and reused across runs to keep token IDs stable.

**Goal:** Implement and evaluate **optimized** Set-Kernel Attention (SKA) that’s fair to compare with dot-product attention (SDPA/Flash), across **language modeling**, **diffusion (text & images)**, and **ViT**. Keep the **banked set** representation (multiple sets per sequence), **no rediscovery in-step**, and **differentiable per-batch pooling** so adapters truly train.

---

## 0) Project context (what exists & what we’ve adopted)

### Core concepts (adopted)

* **Active Universe (U*)** built by a tokenizer; training never does per-step discovery.
* **Banks of sets per sequence** (two-level ragged CSR: `values`, `set_offsets`, `seq_offsets`).
* **MinHash signatures** + **cardinalities** cached **once**; **Φ (pooled atom features)** is **pooled per batch** from current atom features (embedding + optional low-rank adapter).
* **SetBankAttention**: Δ-RBF from symmetric difference **plus** optional content term `(A Φ_q)(B Φ_k)^T`. Softmax is per sequence block.
* **TokenSetRouter**: token→set routing (optional top-k) to mix set outputs back to tokens.
* **Adapter**: low-rank residual over atom embedding; must receive gradients ⇒ Φ pooled inside forward.

### Why we’re upgrading

* Current SKA is **vectorized Python/Torch** (research-grade). Dot attention baselines use **highly optimized** kernels (SDPA/Flash). For fair comparison, we’ll:

  * Keep a **Naïve vs Naïve** table (PyTorch vs PyTorch, fp32).
  * Produce an **Optimized vs Optimized** table: **SKA (Triton/KeOps + online softmax + bf16/fp16)** vs **SDPA/Flash**.

---

## 1) Repository scaffold (current & expected)

```
src/set_attention/
  tokenizers/active_tokenizer.py
  training/text_utils.py          # encode_sentence, ids_to_tokens, TokenSetStore, text_batch_iterator
  sets/
    banked.py                     # BankedSetBatch(values, set_offsets, seq_offsets)
    bank_builders.py              # build_windowed_bank_from_texts / _from_ids
    bank_cache.py                 # SetBankCache (Sig/Size cached; Φ pooled per batch)
    bank_utils.py                 # gather_bank_batch helper
    atom_adapter.py               # low-rank adapter over atom embeddings
  kernels/
    sketches.py                   # MinHasher.sketch_vec (vectorized)
    delta_rbf_fast.py             # (add) Δ/Jaccard helpers
    triton/
      ska_sig_delta_tile.py       # (add) Triton: equality→Δ→exp tile
      ska_online_softmax.py       # (add) Triton: online softmax + V contraction
    keops/
      ska_lazy.py                 # (add) KeOps fallback
  heads/
    banked_attention.py           # SetBankAttention (multi-head Δ-RBF + content)
    token_router.py               # TokenSetRouter (token→set gating)
  utils/profiling.py

scripts/
  train_seq2seq_text.py
  train_seq2seq_text_banked.py
  train_toy_seq2seq.py
  train_toy_lm_banked.py
  train_toy_diffusion_banked.py
  train_tiny_vit_banked.py

tests/
  test_banked_sets.py             # (exists)
  test_ska_kernels.py             # (add)
  test_pooling_grads.py           # (add)
```

> If a file above is missing, Codex should create it with the API detailed below.

---

## 2) Non-negotiable invariants (keep these)

* **Do not** cache Φ across batches with grads; **pool per batch** from `adapter(Embedding.weight)`.
* **Precompute & cache on device:** `Sig_sets`, `Size_sets`, CSR pointers.
* **No per-token set replication.** Compute set×set once per sequence; tokens route via gates.
* **All hot paths** on GPU; no Python loops in forward/backward.
* **Precision:** allow `bf16/fp16` for optimized path; keep MinHash in integer types (`uint16/uint32`).

---

## 3) APIs Codex must honor

### 3.1 BankedSetBatch

```python
@dataclass
class BankedSetBatch:
    values: torch.LongTensor        # (NNZ,)
    set_offsets: torch.LongTensor   # (Nsets+1,)
    seq_offsets: torch.LongTensor   # (B+1,)
    def to(self, device): ...
```

### 3.2 SetBankCache

```python
class SetBankCache:
    # cached (no grad): values, set_offsets, seq_offsets, sig_sets(Nsets,k), size_sets(Nsets,)
    def pool_phi_for_sets(self, set_idx: torch.LongTensor, phi_cur: torch.Tensor) -> torch.Tensor:
        """CSR index_add_ sum-pool: returns Φ[set_idx] (|J|, D). Differentiable."""
```

### 3.3 SetBankAttention (module boundary remains)

```python
Z_sets, q_ptrs = set_bank_attn(
    Phi_q, Sig_q, Size_q,    # (nq,D), (nq,k), (nq,)
    Phi_k, Sig_k, Size_k,    # (nk,D), (nk,k), (nk,)
    ptrs_q, ptrs_k,          # block pointers for per-sequence tiles
    params                   # gamma, beta, tau, A, B, Vs
)
```

### 3.4 TokenSetRouter

```python
Y = token_router(
    X_tokens,  # (B,L,d_model)
    Z_sets,    # ragged per sequence: (Σ nq, H, Dh)
    q_ptrs,    # block pointers
    topk=K
)
```

---

## 4) Optimization roadmap (deliverables & order)

### Tier 1 — Triton kernels (primary deliverable)

1. **`kernels/triton/ska_sig_delta_tile.py`**

* **Goal:** Compute a tile of `exp(-γΔ)` without materializing `(nq×nk×k)` equality tensors.
* **Inputs:** `Sig_q (nq,k:uint16/32)`, `Sig_k (nk,k:uint16/32)`, `Size_q (nq)`, `Size_k (nk)`, `gamma`.
* **Tiling:** `(BQ × BK)` over `(nq, nk)`, stream `k` in chunks, accumulate equality count.
* **Δ:** `J = eq/k`; `inter = (J/(1+J)) * (|A|+|B|)`; `Δ = |A|+|B| - 2*inter`; `S = exp(-γΔ)`.
* **Dtype:** integer comparisons in registers; `bf16/fp16` for `S` if safe.

2. **`kernels/triton/ska_online_softmax.py`**

* **Goal:** Flash-style streaming `Out = softmax(S) @ V` **without** storing `S` or `W`.
* **State per query row:** `m` (running max), `l` (normalizer), `Out` (accumulator).
* **Loop over K tiles:** compute tile scores (from 1), add content term if enabled, update online softmax state, contract with `V_tile`.
* **Output:** `Out / l`.
* **Multi-head:** broadcast `V_tile` over heads (or treat head in block dimension).

3. **Content term fusion (optional in v1)**

* Precompute `Qc = A(Phi_q)`, `Kc = B(Phi_k)` once per step.
* Add `Qc @ Kc^T` to the tile’s score **before** online softmax update.

4. **Router segmented top-k (optional v2)**

* Triton kernel for per-sequence block segmented top-k indices; fallback to dense softmax OK for v1.

### Tier 1.5 — KeOps fallback (fast to ship)

* `kernels/keops/ska_lazy.py`: express Δ-RBF (+ content) with KeOps LazyTensor, compute `logsumexp` and matmul lazily. Use when Triton isn’t available (e.g., ROCm).

### Tier 2 — Bitset Δ (exact, when bank universe is small)

* For small per-bank universes (≤4096 atoms), pack sets into 64-bit words; Δ via XOR + `popcount` (CUDA `__popcll`). This can replace MinHash in that regime (evaluation feature).

### Tier 3 — System polish

* AMP (bf16/fp16), channels-last where relevant, pinned memory, CUDA graphs for steady-state training, DDP support, `torch.compile` where it fuses.

---

## 5) Integration targets (models/trainers)

### 5.1 Language Modeling (`scripts/train_toy_lm_banked.py`)

* Decoder-only Transformer + **AUSA blocks** (replace some self-attn).
* Build banks via `build_windowed_bank_from_ids`.
* **Per-batch**: `phi_cur = adapter(Emb.weight)` → `pool_phi_for_sets` → Triton SKA → Router.
* CLI flags: `--window --stride --adapter-rank --router-topk --minhash-k --sdpa-baseline`.
* Metrics: CE loss, perplexity (tiny overfit + small corpus run).

### 5.2 Diffusion (text: `train_toy_diffusion_banked.py`; image: `train_diffusion_image.py`)

* Add **AUSA blocks** in UNet/DiT attention positions.
* Text: banks from ID sequences. Image: VQ/VQGAN patch IDs, banks from windows/superpixels.
* Loss: ε/v-prediction; Eval: MMD/1-NN (text), FID/sFID/Inception (image).

### 5.3 ViT (`train_tiny_vit_banked.py`)

* Patch tokens; build banks from patch IDs; AUSA block mixes region-level outputs to tokens (or CLS).
* Metric: accuracy on small dataset; throughput vs dense attention.

---

## 6) Benchmark protocol (publishable)

Report **both**:

* **Naïve vs Naïve:**

  * Dot: plain PyTorch matmuls + softmax (fp32).
  * SKA: current vectorized PyTorch (fp32).

* **Optimized vs Optimized:**

  * Dot: **SDPA/Flash** (bf16/fp16).
  * SKA: **Triton online-softmax kernel** (bf16/fp16).

**Log:**

* HW/versions (GPU, CUDA, PyTorch, Triton/KeOps).
* Shapes: `B, Lq, Lk, h, d`, bank stats `S̄_q, S̄_k`, signature `k`.
* Wall-clock fwd+bwd (median N=100, warmup 20), **cuda.synchronize()**.
* Max VRAM (nvml), GPU util %, tokens/s or examples/s.
* Asymptotics: L from 512→8k; bank sizes 8→64; k from 32→256.
* Correctness: Δ via MinHash vs **exact bitset Δ** (small universe) ⇒ error histograms.
* Quality sanity: small LM task parity (PPL within tolerance).

---

## 7) Tests (must pass)

* **Pooling grads:** Autograd gradcheck on a toy batch: adapter params receive gradients through `pool_phi_for_sets`.
* **MinHash vectorized:** `sketch_vec` equality-fraction ≈ true Jaccard on small synthetic sets (tolerance configurable).
* **Bank slicing:** `gather_bank_batch` shapes/ptrs correct; last batch OK.
* **Kernel parity:** Triton vs Python reference on small tiles (numerical tolerance).
* **Tiny overfits:** seq2seq copy/reverse, LM toy, diffusion toy, ViT toy.

---

## 8) Deliverables & milestones

1. **KeOps fallback path** (days): `ska_lazy.py` + wiring flag `--ska-backend {python,keops}`.
2. **Triton kernels v1**: `sig_delta_tile` + `online_softmax` (Δ-RBF only), integrated into `banked_attention.py`.
3. **Optimized SKA → LM toy** end-to-end; benchmarks vs SDPA/Flash.
4. **Content term fusion** in Triton; ablation on β.
5. **Diffusion & ViT** AUSA blocks + toy runs.
6. **Bitset Δ** (exact) kernel (optional showcase).
7. **Final benchmark suite** + ablation plots + doc.

---

## 9) Flags & config (consistent across trainers)

* `--ska-backend {python,triton,keops}`
* `--precision {fp32,fp16,bf16}`
* `--window --stride`
* `--adapter-rank` (0 = freeze)
* `--router-topk` (0 = dense)
* `--minhash-k` (default 128)
* `--tokenizer PATH` (log **[Tokenizer] Loading** vs **Training**)
* `--prof` (enable profiling log)

---

## 10) Reference snippets (to reuse)

### 10.1 Pool Φ with grads (CSR)

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

### 10.2 Set scores (Python reference; kernels must match numerics)

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

## 11) Workflow & PR checklist

* **Branching:** `feat/triton-ska`, `feat/keops-ska`, `feat/lm-banked`, `feat/diffusion-banked`, `feat/vit-banked`.
* **Small, testable PRs**: kernel + unit test; then integration + benchmark notebook/scripts.
* **CI hooks (if available):** run `tests/test_*` on GPU runner; otherwise provide a lightweight CPU ref check and mark GPU tests as optional.
* **Docs:** update a `docs/ausa_kernels.md` with kernel assumptions, tiling sizes, numerics, dtype choices.

---

## 12) What *not* to change

* Don’t re-introduce **sets per token**.
* Don’t train tokenizer or grow U* **inside** forward/backward.
* Don’t cache Φ with grads across batches (causes “backward through graph twice”).

---

## 13) Ready-to-start tasks (for Codex)

* [ ] Create `kernels/keops/ska_lazy.py` and integrate a `--ska-backend keops` switch in `heads/banked_attention.py`.
* [ ] Create `kernels/triton/ska_sig_delta_tile.py` and `ska_online_softmax.py`; add a minimal wrapper `ska_triton.py`.
* [ ] Wire Triton backend via `--ska-backend triton`; preserve Python backend as `python`.
* [ ] Add `--precision` and set autocast; keep signatures in integer tensors.
* [ ] Refactor `banked_attention.py` to accept **precomputed** `Qc,Kc` if β!=0.
* [ ] Ensure `SetBankCache.pool_phi_for_sets` is used in all trainers; remove any lingering per-step tokenization.
* [ ] Add tests in `tests/test_ska_kernels.py` and `tests/test_pooling_grads.py`.
* [ ] Implement Naïve vs Naïve & Optimized vs Optimized benchmark scripts (LM toy), log CSV.
* [ ] Integrate AUSA blocks into `train_toy_diffusion_banked.py` and `train_tiny_vit_banked.py`; run tiny overfits.

---

### One-line summary for the agent

> Implement Triton (and KeOps fallback) **fused, streaming SKA kernels** with online softmax; keep Φ **pooled per batch** so adapters train; integrate into banked attention + token router across LM/Diffusion/ViT; benchmark **Naïve vs Naïve** and **Optimized vs Optimized** against SDPA/Flash with consistent shapes/precision; add unit tests and tiny overfits.

If you want, I can also draft skeletons for the Triton kernels (tile loops, indexing, masks) in a follow-up.
