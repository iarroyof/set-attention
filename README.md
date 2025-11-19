Set Attention — Patchable Kernel-Based Attention for PyTorch
===========================================================

Overview
--------
- Drop-in attention heads to patch PyTorch Transformers/Diffusion with kernel-based attention.
- Supports standard dot-product, cosine, and RBF similarities; includes Δ-RBF set kernels utilities.
- One-liner patching to replace `nn.MultiheadAttention` blocks across an existing model.
- W&B-ready hooks for quick experiment logging (optional dependency).

Install
-------
- Editable install: `pip install -e .` (optionally `.[wandb]`)

Datasets
--------
- All scripts accept `--data-root` (default `~/.cache/set-attention`) so downloads are reused across experiments.
- Hugging Face assets default to `~/.cache/set-attention/hf_datasets`; override via `--hf-cache-dir`.
- Language modeling script now supports `--dataset {wikitext2,wikitext103}` with configurable `--seq-len`/`--seq-stride`.
- Vision scripts download CIFAR-10 into `data_root/vision/cifar10`.

Quickstart
----------
1) Replace attention in a Transformer encoder layer:

```
from torch import nn
from set_attention.patch import replace_multihead_attn
from set_attention.heads.kernel_attention import KernelMultiheadAttention

layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
replace_multihead_attn(layer, sim="rbf", rbf_gamma=0.5)

x = torch.randn(8, 128, 256)
out = layer(x)
```

2) Run demo script:
```
python scripts/demo_patch.py
```

3) Train toy experiments (1k samples) and compare attention types:

- Transformer classification (dot vs cosine vs RBF attention)
```
python scripts/train_toy_transformer.py --attn dot
python scripts/train_toy_transformer.py --attn cosine
python scripts/train_toy_transformer.py --attn rbf
python scripts/train_toy_transformer.py --attn intersect
python scripts/train_toy_transformer.py --attn ska            # alias to RBF
python scripts/train_toy_transformer.py --attn ska_intersect  # alias to intersect
```

- Diffusion on continuous sequences (toy DDPM, dot vs RBF)
```
python scripts/train_toy_diffusion.py --attn dot
python scripts/train_toy_diffusion.py --attn rbf
python scripts/train_toy_diffusion.py --attn intersect
python scripts/train_toy_diffusion.py --attn ska
python scripts/train_toy_diffusion.py --attn ska_intersect
```

Set `WANDB_PROJECT` to enable logging.

Sweeps
------
Run simple sweeps over attention types and seeds:

```
python scripts/run_sweep.py --which transformer
python scripts/run_sweep.py --which diffusion
python scripts/run_sweep.py --which vit
```

Configs
-------
- configs/transformer_toy.yaml
- configs/diffusion_toy.yaml

You can pass `--config` paths to scripts and override with CLI args (e.g., `--d-model`, `--nhead`, `--gamma`).

Profiling
---------
- Add `--profile` to print per-epoch wall time and peak VRAM. W&B can also capture system metrics if enabled.
- For repeatable timing, most scripts now accept `--benchmark` (with `--bench-warmup`/`--bench-iters`) to run fixed-shape fwd/bwd loops and report throughput.

Benchmarks
----------
- Microbenchmark fused SKA kernels with `python scripts/bench_ska.py --backends python --precision fp32 --seqs 16 --sets-q 32 --sets-k 32 --steps 20`.
- Compare full trainers via `--ska-backend {python,triton,keops}` and `--precision {fp32,fp16,bf16}` plus `--benchmark`. Example:
```
python scripts/train_seq2seq_text_banked.py --demo --demo-samples 64 --batch 32 --ska-backend python --benchmark
python scripts/train_toy_diffusion_banked.py --device cuda --ska-backend python --benchmark
python scripts/train_tiny_vit_banked.py --data-mode synthetic --demo-samples 128 --ska-backend python --benchmark
```
These modes fix the batch, run warmup iterations, then log tokens/images per second and elapsed wall time for Naïve vs Optimized comparisons.

Weights & Biases Logging
------------------------
- Install the optional extra: `pip install -e .[wandb]`.
- Set your project/entity (defaults to `ska-naive-ablation`):  
  `export WANDB_PROJECT=ska-naive-ablation`  
  `export WANDB_ENTITY=your-team`
- Append `--wandb` (plus optional `--wandb-run-name`, `--wandb-tags`) to any trainer or benchmark command to record loss/accuracy/BLEU, throughput, and VRAM statistics in W&B.

More tasks
----------
- Seq2Seq (toy reverse+shift): `python scripts/train_toy_seq2seq.py --attn rbf`
- Character LM (next-token): `python scripts/train_toy_lm.py --attn intersect`
- True SKA options are available in all scripts via `--attn ska_true`.

Future image benchmarks (suggested)
-----------------------------------
- Patch-based classification (e.g., CIFAR10/MNIST) with tiny ViT vs SKA variants.
- Object-centric set tasks: treat detected patches as sets; compare attention types on classification and reconstruction.
- For fairness: match heads/params, sweep seeds, report runtime/VRAM, and log W&B runs.

Tiny ViT on CIFAR10 (requires torchvision)
-----------------------------------------
```
python scripts/train_tiny_vit_cifar.py --attn dot --epochs 1 --profile
python scripts/train_tiny_vit_cifar.py --attn rbf --epochs 1 --profile
python scripts/train_tiny_vit_cifar.py --attn intersect --epochs 1 --profile
python scripts/train_tiny_vit_cifar.py --attn ska_true --epochs 1 --profile
```

Smoke tests
-----------
Run quick 1‑epoch checks to ensure functionality after changes:
```
python scripts/train_toy_transformer.py --attn dot --epochs 1
python scripts/train_toy_transformer.py --attn rbf --epochs 1
python scripts/train_toy_transformer.py --attn intersect --epochs 1
python scripts/train_toy_transformer.py --attn ska_true --epochs 1
python scripts/train_toy_diffusion.py --attn dot --epochs 1
python scripts/train_toy_diffusion.py --attn intersect --epochs 1
python scripts/train_toy_diffusion.py --attn ska_true --epochs 1
python scripts/train_toy_seq2seq.py --attn rbf --epochs 1 --profile
python scripts/train_toy_lm.py --attn intersect --epochs 1 --profile
```

Design
------
- `KernelMultiheadAttention` mirrors the API of `nn.MultiheadAttention` (supports `batch_first`).
- Similarity choices: `dot`, `cosine`, `rbf` (Gaussian in feature space). Temperature scaling included.
- Δ-RBF + MinHash sketches are provided for set-level experiments and evaluation.

W&B Logging (optional)
----------------------
- Enable by installing `wandb` extra and setting `WANDB_PROJECT`.
- See `scripts/demo_patch.py` for a minimal logging example.

Repo Layout
-----------
- `src/set_attention/` — package with heads, kernels, and patch utilities.
- `scripts/` — demo and utility scripts.
- `tests/` — minimal shape/correctness tests.

License
-------
- Add your preferred license here. (Not included by default.)
