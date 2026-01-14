# Ablation Plan

Here’s a tight, ready-to-run ablation plan that uses **proper datasets**, your new **--benchmark** path, and full-training runs where needed. It’s organized by task (LM, Diffusion, ViT), with **commands, grids, metrics, and outputs** so you can drop results into JMLR/ICML tables.
Bellow are the ideas to run ablations, but we need to log metrics (only metrics and relevant attention matrices, not models or other things that may increase memory usage or disk space in useless ways) to W&B.

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

---

# 0) One-time prep

* Make sure your env is GPU-ready (bf16/fp16 works), and your repo is on the branch with:

  * `--ska-backend {python,keops,triton}` and `--precision {fp32,fp16,bf16}`
  * `--benchmark`, `--bench-warmup`, `--bench-iters`
* Optional (helps stability/repro):

  ```python
  import torch, os, random
  torch.manual_seed(1337); random.seed(1337); os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
  torch.backends.cuda.matmul.allow_tf32 = True           # for SDPA baseline
  torch.set_float32_matmul_precision("high")             # PyTorch 2.x
  ```

---

# 1) Language Modeling (token LM)

## Datasets

* **WikiText-103** (main), **WikiText-2** (quick dev). Use HF datasets or your adapters.
* Optional: **OpenWebText-subset** for scale check.

## Benchmark (Naïve vs Optimized, fixed shapes)

Use your trainer’s `--benchmark` mode (no full training). Run **dot (SDPA)** and **SKA** variants.

**Dev sanity (WikiText-2)**

```bash
# SDPA / Flash baseline
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend python --precision fp32 --sdpa-baseline \
  --window 32 --stride 16 --minhash-k 128 --router-topk 0 --adapter-rank 0 \
  --bench-warmup 20 --bench-iters 100

# SKA Naïve (vectorized python)
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend python --precision fp32 \
  --window 32 --stride 16 --minhash-k 128 --router-topk 0 --adapter-rank 0 \
  --bench-warmup 20 --bench-iters 100

# SKA Optimized (KeOps)
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend keops --precision bf16 \
  --window 32 --stride 16 --minhash-k 128 --router-topk 0 --adapter-rank 0 \
  --bench-warmup 20 --bench-iters 100

# SKA Optimized (Triton) — once kernels are filled
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend triton --precision bf16 \
  --window 32 --stride 16 --minhash-k 128 --router-topk 0 --adapter-rank 0 \
  --bench-warmup 20 --bench-iters 100
```

**Ablation grid (record CSV)**
Run SKA with:

* `--minhash-k ∈ {32, 64, 128, 256}`
* `--router-topk ∈ {0, 4, 8}`
* `--adapter-rank ∈ {0, 32, 64}`
* `--window×--stride ∈ {(32,16), (64,32)}`
* `--ska-backend ∈ {python, keops, triton}` (dot baseline once with `--sdpa-baseline`)

Outputs from `--benchmark` should already include steps/s, wall-time, VRAM. If not, enable CSV in your bench hook; otherwise also run:

```bash
python scripts/bench_ska.py --backend python --precision fp32 --csv out/bench_lm_naive.csv
python scripts/bench_ska.py --backend triton --precision bf16 --csv out/bench_lm_opt.csv
```

## Full training (metric parity)

* Goal: **PPL parity** vs SDPA at similar compute.

```bash
# Baseline SDPA
python scripts/train_toy_lm_banked.py --dataset wikitext103 --epochs 3 \
  --sdpa-baseline --precision bf16 --window 64 --stride 32

# SKA (Optimized)
python scripts/train_toy_lm_banked.py --dataset wikitext103 --epochs 3 \
  --ska-backend triton --precision bf16 \
  --window 64 --stride 32 --minhash-k 128 --router-topk 4 --adapter-rank 32
```

Log: train/val **loss**, **perplexity**, and **tokens/s**.

---

# 2) Diffusion

## Datasets

* **Text**: WikiText-2 sentences (toy denoising), or **Yahoo Answers** short texts (HF).
* **Images**: **CIFAR-10** (main), optional **ImageNet-100** or **Tiny-ImageNet** (license friendly).

## Benchmark (UNet/DiT attention blocks)

```bash
# Image diffusion, CIFAR-10
python scripts/train_toy_diffusion_banked.py --dataset cifar10 --benchmark \
  --sdpa-baseline --precision bf16

python scripts/train_toy_diffusion_banked.py --dataset cifar10 --benchmark \
  --ska-backend triton --precision bf16 \
  --bank windows --window 8 --stride 8 --minhash-k 128 --router-topk 0
```

Collect **imgs/s**, **VRAM**, **GPU%**.

## Full training (short runs)

```bash
# Text denoising
python scripts/train_toy_diffusion_banked.py --dataset wikitext2 --epochs 5 \
  --ska-backend triton --precision bf16 \
  --window 32 --stride 16 --minhash-k 128 --router-topk 0

# Image: CIFAR-10
python scripts/train_toy_diffusion_banked.py --dataset cifar10 --epochs 100 \
  --ska-backend triton --precision bf16 \
  --bank windows --window 8 --stride 8 --minhash-k 128 --router-topk 0
```

Metrics: **ε-MSE** (or v-MSE) vs baseline; for images add **FID** (ckpt every N epochs). Throughput: imgs/s.

---

# 3) ViT classification / SSL

## Datasets

* **CIFAR-10/100** (quick dev), **ImageNet-100** (main), optionally **ImageNet-1k** (subset if time).

## Benchmark

```bash
python scripts/train_tiny_vit_banked.py --dataset cifar10 --benchmark \
  --sdpa-baseline --precision bf16

python scripts/train_tiny_vit_banked.py --dataset cifar10 --benchmark \
  --ska-backend triton --precision bf16 \
  --bank windows --window 14 --stride 14 --minhash-k 128 --router-topk 0
```

## Full training (short)

```bash
python scripts/train_tiny_vit_banked.py --dataset imagenet100 --epochs 90 \
  --ska-backend triton --precision bf16 \
  --bank windows --window 14 --stride 14 --minhash-k 128 --router-topk 4
```

Report **top-1 accuracy**, **imgs/s**, **VRAM**.

---

# 4) Required “Naïve vs Naïve / Optimized vs Optimized” tables

Produce 2 tables per task:

**(A) Algorithmic (Naïve vs Naïve)**

* Dot: plain PyTorch attention (`matmul + softmax`, fp32, **no SDPA**)
* SKA: `--ska-backend python`, fp32
* Report: fwd+bwd **ms/step**, **VRAM**, shapes (`B,L,h,d` or bank sizes), and a small quality sanity (PPL / ε-MSE / acc on tiny run).

**(B) System (Optimized vs Optimized)**

* Dot: **SDPA/Flash**, bf16/fp16
* SKA: **triton** backend, bf16/fp16
* Same reporting, plus task metric parity on the short training runs.

> Keep your **bench CSVs** in `out/` (already supported by `bench_ska.py`); add `--csv out/bench_<task>_<backend>.csv` if not present.

---

# 5) Core ablations (run SKA on each)

* **MinHash length**: `--minhash-k ∈ {32, 64, 128, 256}` (speed-accuracy tradeoff)
* **Router sparsity**: `--router-topk ∈ {0, 4, 8}` (0 = dense)
* **Adapter capacity**: `--adapter-rank ∈ {0, 32, 64}`
* **Bank granularity**: `--window×--stride ∈ {(32,16), (64,32)} text; (8,8), (16,16) images`
* **Content term**: with/without `(AΦ_q)(BΦ_k)^T` (toggle `--beta 0/1` or your flag)
* **Precision**: `fp32` vs `bf16` (expect big wins with bf16 on optimized paths)

Use **WikiText-2**, **CIFAR-10**, **CIFAR-10** for ablations (fast), then confirm a subset on **WikiText-103** / **ImageNet-100**.

---

# 6) Correctness & error checks

* **MinHash vs exact Δ** on tiny universes: add a run with `--ska-backend python --delta-reference bitset` (if you implemented bitset) or compare MinHash Δ to exact on small synthetic banks. Emit error histograms.
* **Backend parity**: for a fixed random batch,  compare outputs of `python` vs `keops` vs `triton` (tolerances: `atol=1e-3, rtol=1e-3` in bf16) and ensure grad norms don’t explode.

---

# 7) What to attach to the paper

* **Throughput vs length** curves:

  * LM: vary `L` (512→8k), fixed bank size; plot **steps/s** (SDPA vs SKA-Triton).
* **Memory vs length** curves: SDPA grows with `L^2`; SKA should track **bank size** (flat w.r.t. token length if banks are fixed).
* **Ablation bar plots** for MinHash-k, router-topk, adapter rank.
* **Quality parity** on short runs (PPL / FID / top-1), with compute budgets.

---

# 8) Quick “starter” command block (everything in one go)

```bash
# 1) LM dev (WikiText-2), Naïve vs Naïve
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend python --precision fp32 --window 32 --stride 16 --minhash-k 128 \
  --router-topk 0 --adapter-rank 0 --bench-warmup 20 --bench-iters 100 \
  --csv out/lm_wt2_naive_ska.csv

python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend python --precision fp32 --sdpa-baseline \
  --window 32 --stride 16 --bench-warmup 20 --bench-iters 100 \
  --csv out/lm_wt2_naive_sdpa.csv

# 2) LM dev Optimized vs Optimized (KeOps now, Triton when ready)
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend keops --precision bf16 --window 64 --stride 32 --minhash-k 128 \
  --router-topk 4 --adapter-rank 32 --bench-warmup 20 --bench-iters 100 \
  --csv out/lm_wt2_opt_ska_keops.csv

python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend python --precision bf16 --sdpa-baseline \
  --window 64 --stride 32 --bench-warmup 20 --bench-iters 100 \
  --csv out/lm_wt2_opt_sdpa.csv

# 3) CIFAR-10 diffusion (benchmark only)
python scripts/train_toy_diffusion_banked.py --dataset cifar10 --benchmark \
  --ska-backend keops --precision bf16 --bank windows --window 8 --stride 8 \
  --minhash-k 128 --router-topk 0 --csv out/diff_cifar10_ska.csv

python scripts/train_toy_diffusion_banked.py --dataset cifar10 --benchmark \
  --sdpa-baseline --precision bf16 --csv out/diff_cifar10_sdpa.csv

# 4) ViT (CIFAR-10)
python scripts/train_tiny_vit_banked.py --dataset cifar10 --benchmark \
  --ska-backend keops --precision bf16 --bank windows --window 14 --stride 14 \
  --minhash-k 128 --router-topk 0 --csv out/vit_cifar10_ska.csv

python scripts/train_tiny_vit_banked.py --dataset cifar10 --benchmark \
  --sdpa-baseline --precision bf16 --csv out/vit_cifar10_sdpa.csv
```

---

## Notes / gotchas

* Triton is available, but start with **KeOps** + **bf16**; keep all tensors on GPU.
* Ensure your **dataset adapters** don’t tokenize per batch (you already fixed this).
* For **tokenizer reuse**, always pass `--tokenizer <dir>`; log `[Tokenizer] Loading...`.
* Keep BANK stats in logs: avg sets/seq, avg set size, MinHash-k.

---

Generate **W&B sweep YAMLs** for the ablation grids above (minhash-k, router-topk, adapter rank, precision, backend) so we can launch grid runs with a single command.


## Memory and Hardware considerations
 
Let’s lock it down. Below is a **memory-aware playbook** so the ablations won’t blow VRAM and you still get apples-to-apples speed numbers. It assumes RTX 4090 (24 GB) and/or RTX6000 Ada (48 GB); Triton is unavailable → but use **KeOps** as optional fallback if Triton is not available.

---

## What already helps (and should stay on)

* **No per-token set replication.** One set×set per sequence; tokens just route.
* **Cache only Sig/Size; pool Φ per batch.** Prevents autograd graph reuse/OOM.
* **Blockwise (per-sequence) attention.** Avoids giant cross-batch matmuls.
* **Benchmark mode** with fixed shapes and warmups.
* **bf16/fp16** support (reduce activations), MinHash kept in **uint16/uint32**.

---

## Memory knobs (turn these first)

1. **Bank size (dominant)**

   * Text: `--window, --stride` → fewer sets per sequence → smaller set×set.
   * Vision: patch banks via `--window, --stride` (e.g., 8 or 14) — smaller windows = fewer sets.

2. **Signature length**

   * `--minhash-k {32,64,128}`. Start with **64** for ablations, bump to 128 only for final quality.

3. **Router sparsity**

   * `--router-topk {0,4,8}`. Using **4 or 8** cuts value aggregation memory/compute.

4. **Precision**

   * Use `--precision bf16` whenever possible. Keep MinHash as `uint16/uint32`.

5. **Batch size / grad accumulation**

   * Prefer **smaller B** + `--grad-accum` (if you have it) over enlarging B.

6. **Disable content term initially**

   * `--beta 0` removes `(AΦ_q)(BΦ_k)^T` matmul until you need it.

7. **KeOps backend first** (Windows-friendly if needed)

   * `--ska-backend keops` streams ops lazily; often much lower peak RAM than the pure Python path.

---

## Safe profiles you can copy-paste

### A) **Tight VRAM** (~12–16 GB, RTX4090)

**LM (WikiText-2), Naïve vs Optimized**

```bash
# Naïve (algorithmic apples-to-apples)
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend python --precision fp32 --window 32 --stride 16 \
  --minhash-k 64 --router-topk 4 --adapter-rank 0 \
  --bench-warmup 20 --bench-iters 100

# Optimized (system apples-to-apples)
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend keops --precision bf16 --window 32 --stride 16 \
  --minhash-k 64 --router-topk 4 --adapter-rank 0 \
  --bench-warmup 20 --bench-iters 100
```

**ViT (CIFAR-10)**

```bash
python scripts/train_tiny_vit_banked.py --dataset cifar10 --benchmark \
  --ska-backend keops --precision bf16 --bank windows --window 14 --stride 14 \
  --minhash-k 64 --router-topk 4
```

**Diffusion (CIFAR-10)**

```bash
python scripts/train_toy_diffusion_banked.py --dataset cifar10 --benchmark \
  --ska-backend keops --precision bf16 --bank windows --window 8 --stride 8 \
  --minhash-k 64 --router-topk 0
```

### B) **Moderate VRAM** (~24 GB, 4090 sweet spot)

* Same as A) but increase `--minhash-k 128` and optionally `--router-topk 8`.
* For full training, keep `--beta 0` until the end; enable content term only in the final run.

### C) **Roomy VRAM** (48 GB, RTX6000 Ada)

* Raise `--window 64 --stride 32` (text) and `--minhash-k 128`.
* You can turn on `--beta 1` for the content term.
* If on Linux with Triton installed: `--ska-backend triton --precision bf16`.

---

## If you hit OOM anyway (checklist)

* Drop `--minhash-k` (128 → 64 → 32).
* Increase stride / reduce window (fewer sets per sequence).
* Set `--router-topk 4` (or 0 for dense but small windows).
* Ensure **content term off**: `--beta 0`.
* For full training: use **grad accumulation** and **gradient checkpointing** (if enabled in your backbone).
* Verify nothing builds **padded banks**; go through **flattened CSR** path (`gather_flat`).

---

## Implementation switches that reduce peak memory (keep them on)

* **KeOps** backend for streaming (until Triton kernels are finished).
* **bf16** autocast for activations; keep integers for signatures.
* **No cached Φ** across batches (pool with `index_add_` inside forward).
* **Blockwise online softmax** (once the Triton path is live) to avoid storing scores.

---

## What to record for fair, memory-aware comparisons

* Shapes: `B, L, H, d`, **avg sets/seq**, **avg set size**, `k`.
* **Peak VRAM** (`torch.cuda.max_memory_allocated()` and your `profiling.py`).
* **Steps/s**, **GPU util %**.
* Task metric (PPL / acc / ε-MSE or FID on short runs).
* Note **backend** (`python`, `keops`, `triton`) and **precision**.

---

## Sanity formulas (so you can predict fit)

Rough peak (ignoring model MLPs), per sequence block:

* **Scores (SKA Δ-RBF only, streamed):** ~ `O(nq_tile × nk_tile)` live, **not** `nq×nk×k`.
* **Signatures:** `Nsets × k × (2 bytes if uint16)`.
* **Φ pooling:** `Nsets × D_atom × bytes(precision)`.
  Keep `Nsets` small by controlling `--window/--stride`; keep `k` modest (64–128).

---

## TL;DR

* You **don’t** need to downgrade dot attention; instead run **KeOps+bf16** now (Windows-friendly if needed), and Triton later on Linux.
* Use the **tight VRAM** profile to start, then scale `k` and window size.
* Record VRAM/steps/s + task metric, and you’ll have reviewer-safe tables without memory surprises.

Also add a `--memory-safe` preset that automatically picks conservative `window/stride/minhash-k/router-topk` from your GPU’s free VRAM and warns before launching a run.

At the end of the tasks, report what's finished and what's pending from all context and plans available for this workspace.
