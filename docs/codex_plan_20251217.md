Awesome — here’s a **Codex-ready plan** that reframes the work as **theory + algorithm + reference implementation** (pure **Python/PyTorch**), and narrows experimental scope to **text-only** for an **ACL** submission. It strips any KeOps/Triton requirements and centers on **Naïve vs Naïve**, **scaling curves**, and **memory asymptotics**. Everything below is actionable for Codex to implement next.

---

## Architecture updates (Jan 2026)

These updates supersede earlier assumptions about per-run tokenization and on-the-fly bank/routing construction.

- Artifact cache system added (`src/set_attention/data/artifact_cache.py`, `src/set_attention/data/ska_artifacts.py`) with fingerprinted meta for tokens, banks, and routing. Training scripts support `--cache-mode none|tokens|full`, `--cache-only`, and cache guards (e.g., adapter-rank for full routing). Cache builders: `scripts/cache_tokens.py`, `scripts/cache_ska_artifacts.py`. Sweep runners can `--precache`.
- HuggingFace cache rooting unified via `ensure_hf_cache`; env-first behavior using `HF_HOME`, `HF_DATASETS_CACHE`, and `HF_HUB_CACHE`, avoiding legacy per-script cache paths.
- Data loading standardized on DataLoader pipelines with deterministic eval seeding, `worker_init_fn`, and per-task `--num-workers` controls. Full-cache runs force `num_workers=0` to avoid worker churn.
- Sweep robustness upgrades: GPU idle gate + min-free-GB checks, post-run GPU checks, OOM/exitcode rows in CSVs, and sequential execution defaults.
- Artifact generation expanded: `scripts/repro_runs.py` aggregates metrics alongside benchmarks; `scripts/make_stage_artifacts.py` emits Stage A/B tables/plots when inputs are available.
- AUSA tokenizer caching is persistent for Seq2Seq and reused across runs to keep token IDs stable.

# Codex Plan — AUSA (Set-Kernel Attention) for ACL (Text-Only, PyTorch)

**Positioning**: We publish as **theory + algorithm + reference implementation**. Experiments emphasize **algorithmic comparisons** (no custom GPU kernels), **scaling behavior**, and **memory asymptotics**, with clean methodology and reproducibility.
**Backend**: **PyTorch** only (Python).
**Tasks**:

* Language Modeling: **WikiText-2** (dev), then **10% / 25% / 50%** of **WikiText-103** by **token budget**.
* Text Denoising (Diffusion-like toy): **WikiText-2 sentences**.

We explicitly **do not claim system-level speedups** vs SDPA/Flash; we show **method-level behavior** and **scaling** under **equally naïve** conditions.

---

## 0) Invariants to preserve (already adopted)

* **Active Universe** tokenizer trained **offline**; training does **not** discover atoms in-step.
* Inputs are **banks of subsets per sequence** (two-level CSR).
* Cache **signatures and sizes** once; pool **Φ** per batch (differentiable).
* **No per-token set replication**; set×set computed per sequence; tokens route through a light **Token→Set** adapter/gate.
* All hot loops are **vectorized PyTorch**; **no** KeOps/Triton.

---

## 1) Datasets (text-only) & splits

### 1.1 LM

* **WikiText-2** (dev & ablations, fast).
* **WikiText-103**: create **token-budget subsets** of **10%**, **25%**, **50%**.

  * **Sampling policy (implement)**:

    * Build document-level token counts.
    * **Stratify by length deciles**; sample proportionally until budget is met.
    * Preserve **unigram frequency histogram** within ±ε (log-bins) vs full corpus.
    * Log **bank stats** on sampled data: avg sets/seq, avg atoms/set under the chosen `--window/--stride`.

### 1.2 Text denoising (toy diffusion)

* **WikiText-2** sentences: input x = sentence tokens, add synthetic corruption (mask/drop/permute) and denoise with a tiny UNet/Transformer denoiser that includes **AUSA blocks**.

> Codex: add a `scripts/data_make_subsets.py` to materialize WikiText-103 subsets by token budget using the above policy (writes indices or HF filter files). Make it deterministic.

---

## 2) Fair “Algorithmic” comparisons (Naïve vs Naïve)

We must **disable optimized attention** for the dot baseline and match dtype, shapes, and timing windows.

### 2.1 Dot (naïve) toggle

* Add `--dot-naive` flag that **forces** explicit PyTorch ops:

  ```python
  # in attention baseline
  scores = (Q @ K.transpose(-1, -2)) * (1.0 / math.sqrt(d))
  W = torch.softmax(scores, dim=-1)
  out = W @ V
  ```
* Ensure **no** SDPA/Flash path is invoked. Also set:

  ```python
  import torch
  torch.backends.cuda.enable_flash_sdp(False)
  torch.backends.cuda.enable_mem_efficient_sdp(False)
  torch.backends.cuda.enable_math_sdp(True)  # if you use SDPA elsewhere, keep this off for naive
  ```

### 2.2 SKA (naïve)

* Use `--ska-backend python` (existing vectorized implementation).
* Match **dtype** (`fp32`) and shapes with dot-naïve.

### 2.3 Benchmarking harness (already scaffolded)

* Use `--benchmark`, `--bench-warmup`, `--bench-iters`.
* **Exclude** dataset build, tokenizer training, and vocab streaming from timed region.
* Log **CSV** with:

  * `backend`, `dtype`, `B`, `L`, `H`, `D`, `window`, `stride`, `minhash_k`, `router_topk`
  * `avg_sets_per_seq`, `avg_atoms_per_set`
  * `elapsed_ms`, `steps_per_s`, `max_vram_mb`
  * **Workload-normalized throughput** fields:

    * `scores_computed`:

      * dot: `B * H * Lq * Lk`
      * ska: `sum_over_sequences(|J_q| * |J_k|) * H`
    * `throughput_per_1e6_scores = scores_computed / elapsed_ms / 1e6`

> Codex: extend `bench_ska.py` and `scripts/run_benchmarks.py` to add the **dot-naïve** pair and to compute + store normalized throughput.

---

## 3) Experiments (text-only)

### 3.1 Language Modeling (ACL main)

**Ablations (WikiText-2):**

* Compare **dot-naïve (fp32)** vs **ska-python (fp32)** under identical model configs.
* Sweep SKA knobs:

  * `--minhash-k ∈ {32, 64, 128, 256}`
  * `--router-topk ∈ {0, 4, 8}`
  * `--window×--stride ∈ {(32, 16), (64, 32)}`
  * `--adapter-rank ∈ {0, 32, 64}`
  * `--beta ∈ {0, 1}` (content term off/on)
* Outputs: **throughput**, **VRAM**, **normalized throughput**, **PPL on dev** (small training run, e.g., 1–3 epochs).

**Scaling (WikiText-103 subsets):**

* Use **10% / 25% / 50%** token budgets.
* Fixed model; run **short epochs** (or fixed update budget) to record **throughput vs length**, **VRAM vs length**, and **dev PPL**.
* Plot **steps/s** vs **sequence length** and **VRAM** vs **length** for both methods.

**One confirmatory run (reference):**

* Run **25%** subset with **fixed budget of updates** (e.g., 20k steps) and report **final dev/test PPL** for dot-naïve vs ska-python.
* Make no system claims — the goal is **parity** in quality under reference implementation.

**Commands (examples):**

```bash
# Algorithmic: dot-naïve (fp32), wikitext-2
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --dot-naive --precision fp32 \
  --batch 8 --seq-len 256 --seq-stride 256 \
  --bench-warmup 20 --bench-iters 100 \
  --benchmark-csv out/lm_wt2_dot_naive.csv

# Algorithmic: ska-python (fp32), ablation example
python scripts/train_toy_lm_banked.py --dataset wikitext2 --benchmark \
  --ska-backend python --precision fp32 \
  --window 64 --stride 32 --minhash-k 128 --router-topk 4 --adapter-rank 32 --beta 0 \
  --batch 8 --seq-len 256 --seq-stride 256 \
  --bench-warmup 20 --bench-iters 100 \
  --benchmark-csv out/lm_wt2_ska_python.csv

# Small training (perplexity), 1-3 epochs
python scripts/train_toy_lm_banked.py --dataset wikitext2 --epochs 2 \
  --dot-naive --precision fp32

python scripts/train_toy_lm_banked.py --dataset wikitext2 --epochs 2 \
  --ska-backend python --precision fp32 \
  --window 64 --stride 32 --minhash-k 128 --router-topk 4
```

### 3.2 Text Denoising (toy diffusion-like)

* Data: **WikiText-2 sentences**.
* Model: small denoiser with **AUSA block** in the bottleneck or attention layers.
* Objective: ε-loss on synthetic noise; evaluate **reconstruction MSE** and optionally **token-level accuracy**.
* Compare **dot-naïve** vs **ska-python** at same shapes; ablate `minhash-k`, `window/stride`, `router-topk`.
* Focus on **throughput**, **VRAM**, and **reconstruction quality** under naïve conditions.

**Commands (examples):**

```bash
# Benchmark (naïve vs naïve), text denoising
python scripts/train_toy_diffusion_banked.py --dataset wikitext2 --benchmark \
  --dot-naive --precision fp32 \
  --bench-warmup 20 --bench-iters 100 \
  --benchmark-csv out/diff_text_dot_naive.csv

python scripts/train_toy_diffusion_banked.py --dataset wikitext2 --benchmark \
  --ska-backend python --precision fp32 \
  --window 64 --stride 32 --minhash-k 128 --router-topk 0 \
  --bench-warmup 20 --bench-iters 100 \
  --benchmark-csv out/diff_text_ska_python.csv

# Short training for reconstruction metric
python scripts/train_toy_diffusion_banked.py --dataset wikitext2 --epochs 3 \
  --dot-naïve --precision fp32

python scripts/train_toy_diffusion_banked.py --dataset wikitext2 --epochs 3 \
  --ska-backend python --precision fp32 \
  --window 64 --stride 32 --minhash-k 128
```

---

## 4) Memory-aware configs (so runs don’t OOM)

* Start with `--minhash-k 64`, `--router-topk 0/4`, `--window 32 --stride 16` for WT2.
* For WT103 subsets, keep `--window 64 --stride 32` and **batch 8**; reduce if needed.
* Always log: `avg_sets_per_seq`, `avg_atoms_per_set`, `max_vram_mb`.
* Use `bf16/fp16` **only** for descriptive diagnostics if you like; **Tables A** must remain in **fp32** for pure algorithmic parity.

---

## 5) Reporting (what Codex must autosave)

### 5.1 Tables A (Algorithmic, naïve vs naïve, fp32)

* **LM (WT2)**:

  * dot-naïve vs ska-python; report **steps/s**, **VRAM**, **normalized throughput**, and a **small PPL** run.
* **LM (WT103 subsets)**: 10/25/50% scaling rows: same metrics + dev PPL on a short run.
* **Text denoising (WT2)**: dot-naïve vs ska-python; **steps/s**, **VRAM**, **recon MSE**.

### 5.2 Figures

* **Throughput vs length**: WT2/WT103; both methods, same shapes.
* **VRAM vs length**: show AUSA’s **bank-driven** memory scaling.
* **Ablation plots**: `minhash-k`, `router-topk`, `window/stride`.

### 5.3 Appendix logs

* Sampling policy for WT103 subsets (token budget, length-stratified sampling).
* Bank statistics per split (distribution plots).
* Normalized throughput definition & formulas.

> Codex: extend `bench_ska.py` / `run_benchmarks.py` to aggregate CSVs into `out/report_tables/` and dump `report.json` with metadata (hw, versions, seeds, flags).

---

## 6) Code tasks for Codex (checklist)

* [ ] **Add `--dot-naive` flag** and implement explicit matmul+softmax attention. Disable SDPA for this code path.
* [ ] **Enforce fp32** for **Tables A** runs (`--precision fp32`).
* [ ] **Extend CSV logging** (bench + training) with normalized throughput & bank stats.
* [ ] **Implement dataset slicer**: `scripts/data_make_subsets.py` for WT103 (10/25/50% token budgets) with length stratification + frequency preservation (log hist divergence).
* [ ] **Wire benchmark suites** in `scripts/run_benchmarks.py` for:

  * LM WT2 (naïve vs naïve)
  * LM WT103 subsets (naïve vs naïve)
  * Text denoising WT2 (naïve vs naïve)
* [ ] **Add plotting notebook** (`notebooks/acl_plots.ipynb`) that reads CSVs and produces:

  * Tables A
  * Throughput vs length
  * VRAM vs length
  * Ablation bar charts
* [ ] **Repro harness**: record seeds, PyTorch/CUDA versions, GPU name, and SDP flags into the CSV headers or a sidecar `run_meta.json`.

---

## 7) Claims discipline (what to print in paper)

* We claim **theoretical correctness**, **learnability**, and **algorithmic scaling** of AUSA vs dot attention.
* We do **not** claim system-level wins; all results are **naïve vs naïve** under matched PyTorch implementations.
* We provide **reference code** and **subset sampling methodology** (so others can reproduce).

---

## 8) One-command smoke tests (what Codex should run before PR)

```bash
# WT2: naïve vs naïve (LM)
python scripts/run_benchmarks.py --tasks lm --dataset wikitext2 --output-dir out/bench_naive

# WT103 subsets: generate and run
python scripts/data_make_subsets.py --dataset wikitext103 --budgets 0.10 0.25 0.50 --out .cache/wt103_slices
python scripts/run_benchmarks.py --tasks lm --dataset wikitext103 --slices .cache/wt103_slices --output-dir out/bench_naive_wt103

# Text denoising (WT2)
python scripts/run_benchmarks.py --tasks denoise --dataset wikitext2 --output-dir out/bench_denoise

# Tiny training (LM) to report PPL parity on WT2
python scripts/train_toy_lm_banked.py --dataset wikitext2 --epochs 2 --dot-naive --precision fp32
python scripts/train_toy_lm_banked.py --dataset wikitext2 --epochs 2 --ska-backend python --precision fp32 --window 64 --stride 32 --minhash-k 128 --router-topk 4
```

---

**Hand-off note:** If Codex encounters missing files (e.g., `data_make_subsets.py`), it should **create them** with the specified behavior. The **goal** is a clean, PyTorch-only reference implementation with rigorous **naïve vs naïve** comparisons, **scaling curves**, and **memory asymptotics** suitable for ACL.
