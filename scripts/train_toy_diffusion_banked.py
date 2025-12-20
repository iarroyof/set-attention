import argparse
import copy
import csv
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.experiments.data_toy import ToyDiffConfig, make_toy_continuous_sequences
from set_attention.experiments.diffusion_core import SimpleDDPM
from set_attention.experiments.models import PositionalEncoding, timestep_embedding
from set_attention.eval.mmd_simple import mmd2_unbiased_from_feats
from set_attention.utils.metrics import chamfer_l2, one_nn_two_sample
from set_attention.utils.profiling import profiler
from set_attention.utils.sample_logging import select_sample_indices
from set_attention.utils.wandb import init_wandb
from set_attention.experiments.nlp_eval import corpus_bleu
from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher
from set_attention.data.wikitext import load_wikitext_lines, tokenize_lines, chunk_tokens
from common.repro import set_seed


TEXT_SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]


def _build_text_vocab(tokens: List[str]):
    stoi = {tok: idx for idx, tok in enumerate(TEXT_SPECIAL_TOKENS)}
    itos = list(TEXT_SPECIAL_TOKENS)
    for tok in tokens:
        if tok in stoi:
            continue
        idx = len(stoi)
        stoi[tok] = idx
        itos.append(tok)
    return stoi, itos


def _encode_token_chunks(chunks: List[List[str]], stoi: dict) -> List[torch.Tensor]:
    unk_id = stoi.get("<unk>", 0)
    encoded: List[torch.Tensor] = []
    for chunk in chunks:
        ids = torch.tensor([stoi.get(tok, unk_id) for tok in chunk], dtype=torch.long)
        encoded.append(ids)
    return encoded


def _embed_token_chunks(id_chunks: List[torch.Tensor], embeddings: torch.Tensor) -> torch.Tensor:
    if not id_chunks:
        return torch.empty(0, 0, embeddings.size(1), dtype=embeddings.dtype)
    stacked = [embeddings.index_select(0, ids) for ids in id_chunks]
    return torch.stack(stacked, dim=0)


def decode_embeddings_to_ids(
    batch_vectors: torch.Tensor, embedding_table: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    """Map continuous vectors back to token ids via cosine similarity."""
    B, L, D = batch_vectors.shape
    flat = batch_vectors.reshape(-1, D)
    flat = F.normalize(flat, dim=-1)
    emb = F.normalize(embedding_table, dim=-1)
    best_scores = torch.full((flat.size(0),), float("-inf"), device=flat.device)
    best_ids = torch.zeros(flat.size(0), dtype=torch.long, device=flat.device)
    for start in range(0, emb.size(0), chunk_size):
        chunk = emb[start : start + chunk_size]
        scores = flat @ chunk.T
        max_vals, max_idx = scores.max(dim=1)
        better = max_vals > best_scores
        best_scores = torch.where(better, max_vals, best_scores)
        best_ids = torch.where(better, start + max_idx, best_ids)
    return best_ids.view(B, L)


def prepare_text_diffusion_data(
    dataset: str,
    cache_dir: Path,
    seq_len: int,
    stride: int,
    train_line_limit: Optional[int],
    val_line_limit: Optional[int],
    train_seq_limit: Optional[int],
    val_seq_limit: Optional[int],
    embed_dim: int,
    embed_seed: int,
):
    train_lines = load_wikitext_lines(dataset, "train", cache_dir, train_line_limit)
    val_lines = load_wikitext_lines(dataset, "validation", cache_dir, val_line_limit)
    train_tokens = tokenize_lines(train_lines)
    if not train_tokens:
        raise RuntimeError("No tokens available in training split for text diffusion data.")
    stride = max(1, stride)
    train_chunks = chunk_tokens(train_tokens, seq_len, stride)
    if train_seq_limit is not None:
        train_chunks = train_chunks[: max(0, int(train_seq_limit))]
    val_tokens = tokenize_lines(val_lines)
    val_chunks = chunk_tokens(val_tokens, seq_len, stride)
    if val_seq_limit is not None:
        val_chunks = val_chunks[: max(0, int(val_seq_limit))]
    if not train_chunks:
        raise RuntimeError("No sequences produced for training; check --text-seq-len/stride settings.")
    if not val_chunks:
        raise RuntimeError("No sequences produced for validation; check --text-seq-len/stride settings.")
    stoi, itos = _build_text_vocab(train_tokens)
    train_ids = _encode_token_chunks(train_chunks, stoi)
    val_ids = _encode_token_chunks(val_chunks, stoi)
    if not train_ids or not val_ids:
        raise RuntimeError("Failed to encode text diffusion sequences.")
    vocab_size = len(stoi)
    gen = torch.Generator().manual_seed(int(embed_seed))
    embeddings = torch.randn(vocab_size, embed_dim, generator=gen, dtype=torch.float32)
    train_data = _embed_token_chunks(train_ids, embeddings)
    val_data = _embed_token_chunks(val_ids, embeddings)
    return {
        "train_data": train_data,
        "val_data": val_data,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_ids_tensor": torch.stack(train_ids),
        "val_ids_tensor": torch.stack(val_ids),
        "vocab_size": vocab_size,
        "embeddings": embeddings,
        "itos": itos,
    }


def _append_benchmark_row(csv_path: str, row: dict) -> None:
    if not csv_path:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            existing = reader.fieldnames or []
            existing_rows = list(reader)
        if not existing:
            fieldnames = list(row.keys())
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return
        extras = [col for col in row if col not in existing]
        if extras:
            new_fields = existing + [col for col in extras if col not in existing]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=new_fields)
                writer.writeheader()
                for prev in existing_rows:
                    writer.writerow({col: prev.get(col, "") for col in new_fields})
            existing = new_fields
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=existing)
            writer.writerow({col: row.get(col, "") for col in existing})
    else:
        fieldnames = list(row.keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)


def _summarize_ska_batch(q_ptrs: torch.Tensor, size_tensor: torch.Tensor, num_heads: int):
    if q_ptrs.numel() <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    counts = (q_ptrs[1:] - q_ptrs[:-1]).to(torch.float32)
    avg_sets = float(counts.mean().item()) if counts.numel() > 0 else 0.0
    avg_atoms = float(size_tensor.to(torch.float32).mean().item()) if size_tensor.numel() > 0 else 0.0
    scores_per_batch = float((counts * counts).sum().item() * max(1, num_heads))
    min_sets = float(counts.min().item()) if counts.numel() > 0 else 0.0
    max_sets = float(counts.max().item()) if counts.numel() > 0 else 0.0
    return avg_sets, avg_atoms, scores_per_batch, min_sets, max_sets


def _configure_dot_naive(dot_naive: bool) -> None:
    if not dot_naive:
        return
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    print("[SDP] dot-naive enabled: flash/mem-efficient SDP disabled; using math backend.")


def _system_info():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    torch_version = torch.__version__
    cuda_version = torch.version.cuda or "cpu"
    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_sha = "unknown"
    return {
        "device": device,
        "gpu_name": gpu_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "git_sha": git_sha,
    }


def summarize_tensor(tensor: torch.Tensor, max_items: int = 16) -> str:
    flat = tensor.detach().cpu().flatten()
    head = flat[:max_items].tolist()
    preview = ", ".join(f"{v:.3f}" for v in head)
    if flat.numel() > max_items:
        preview += ", ..."
    shape = "x".join(str(dim) for dim in tensor.shape)
    return f"{shape}: [{preview}]"


def make_id_sequence(x: torch.Tensor) -> torch.Tensor:
    # x: (L, D)
    mean = x.mean(dim=1)
    sign = (mean > 0).long()
    base = torch.arange(x.size(0), dtype=torch.long) * 2
    return base + sign


class BankedDenoiser(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int,
        layers: int,
        router_topk: int,
        ska_backend: str = "python",
        precision: str = "fp32",
        use_ska: bool = True,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.use_ska = use_ska
        self.router = None
        self.set_attn = None
        if use_ska:
            self.router = TokenSetRouter(d_model=d_model, num_heads=nhead, topk=router_topk)
            self.set_attn = SetBankAttention(
                d_model=d_model,
                num_heads=nhead,
                tau=1.0,
                gamma=0.3,
                beta=1.0,
                score_mode="delta_plus_dot",
                eta=1.0,
                backend=ska_backend,
                precision=precision,
            )
        self.proj_out = nn.Linear(d_model, in_dim)
        self._bank = None

    def set_current_bank(self, Phi, Sig, Size, q_ptrs):
        if self.use_ska:
            self._bank = (Phi, Sig, Size, q_ptrs)

    def forward(self, x_t: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x_t)
        h = self.pos_enc(h)
        if t_embed.dim() == 2:
            t_embed = t_embed.unsqueeze(1)
        h = h + t_embed
        h = self.enc(h)
        if not self.use_ska:
            return self.proj_out(h)
        assert self._bank is not None, "bank not set"
        Phi, Sig, Size, q_ptrs = self._bank
        Z, q_ptrs = self.set_attn(Phi, Sig, Size, q_ptrs, Phi, Sig, Size, q_ptrs)
        routed = self.router(h, Z, Phi, q_ptrs)
        return self.proj_out(routed)


def run_diffusion_benchmark(
    args,
    model,
    ddpm,
    adapter,
    atom_emb,
    phi_snapshot,
    cache,
    train_data,
    optimizer,
    device,
    wandb_run,
    benchmark_csv,
    seed,
    rep,
    run_uid,
):
    bench_batch = min(args.batch, train_data.size(0))
    if bench_batch == 0:
        print("[benchmark] no training data available.")
        return
    batch_idx = torch.arange(bench_batch, dtype=torch.long, device=device)
    xb = train_data[:bench_batch].to(device)

    stats_snapshot: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    if args.sdpa_baseline:
        backend_label = "sdpa"

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = ddpm.loss(model, xb, lambda t, dim: timestep_embedding(t, args.d_model), args.d_model)
            loss.backward()
            optimizer.step()

    else:
        backend_label = f"{args.ska_backend}/{args.precision}"

        def step():
            nonlocal stats_snapshot
            optimizer.zero_grad(set_to_none=True)
            phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
            Phi, Sig, Size, q_ptrs = cache.gather_flat(batch_idx, phi_dynamic)
            model.set_current_bank(Phi, Sig, Size, q_ptrs)
            loss = ddpm.loss(model, xb, lambda t, dim: timestep_embedding(t, args.d_model), args.d_model)
            loss.backward()
            optimizer.step()
            stats_snapshot = (q_ptrs.detach().cpu(), Size.detach().cpu())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.bench_warmup):
        step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.bench_iters):
        step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    max_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    sequences = bench_batch * args.bench_iters
    throughput = sequences / elapsed if elapsed > 0 else 0.0
    info = _system_info()
    if args.sdpa_baseline:
        seq_len = xb.size(1)
        scores_total = float(bench_batch * seq_len * seq_len * max(1, args.nhead) * args.bench_iters)
        avg_sets = avg_atoms = min_sets = max_sets = 0.0
    else:
        avg_sets = avg_atoms = min_sets = max_sets = 0.0
        scores_total = 0.0
        if stats_snapshot is not None:
            q_ptrs_snap, size_snap = stats_snapshot
            avg_sets, avg_atoms, scores_per_batch, min_sets, max_sets = _summarize_ska_batch(
                q_ptrs_snap, size_snap, args.nhead
            )
            scores_total = scores_per_batch * args.bench_iters
    scores_per_s = scores_total / elapsed if elapsed > 0 else 0.0
    scores_per_1e6 = scores_per_s / 1e6 if elapsed > 0 else 0.0
    print(f"[benchmark][diffusion] backend={backend_label} seq/s={throughput:.1f} elapsed={elapsed:.3f}s")
    if wandb_run.enabled:
        wandb_run.log(
            {
                "benchmark/sequences_per_s": throughput,
                "benchmark/elapsed_s": elapsed,
                "benchmark/avg_sets_per_seq": avg_sets,
                "benchmark/avg_atoms_per_set": avg_atoms,
                "benchmark/scores_total": scores_total,
                "benchmark/scores_per_s": scores_per_s,
                "benchmark/scores_per_1e6": scores_per_1e6,
                "benchmark/min_sets_per_seq": min_sets,
                "benchmark/max_sets_per_seq": max_sets,
                "benchmark/max_vram_mb": max_vram_mb,
                "benchmark/seed": seed,
                "benchmark/rep": rep,
            }
        )
    text_mode = args.data_mode == "text"
    stride_text = args.text_stride if args.text_stride > 0 else args.text_seq_len
    config_label = args.config
    if text_mode:
        config_label = f"text::{args.text_dataset}@{args.text_seq_len}/{stride_text}"
    _append_benchmark_row(
        benchmark_csv,
        {
            "script": "train_toy_diffusion_banked",
            "task": "textdiff" if text_mode else "diffusion",
            "config": config_label,
            "dataset_id": config_label,
            "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
            "precision": args.precision,
            "window": args.window,
            "stride": args.stride,
            "minhash_k": args.minhash_k,
            "router_topk": args.router_topk,
            "batch": args.batch,
            "steps": args.steps,
            "bench_warmup": args.bench_warmup,
            "bench_iters": args.bench_iters,
            "seed": seed,
            "rep": rep,
            "run_uid": run_uid,
            "device": info["device"],
            "gpu_name": info["gpu_name"],
            "torch_version": info["torch_version"],
            "cuda_version": info["cuda_version"],
            "git_sha": info["git_sha"],
            "data_mode": args.data_mode,
            "text_dataset": args.text_dataset if text_mode else "",
            "text_seq_len": args.text_seq_len if text_mode else "",
            "text_stride": stride_text if text_mode else "",
            "text_train_limit": args.text_train_limit if text_mode else "",
            "text_val_limit": args.text_val_limit if text_mode else "",
            "sequences_per_s": throughput,
            "elapsed_s": elapsed,
            "avg_sets_per_seq": avg_sets,
            "avg_atoms_per_set": avg_atoms,
            "scores_total": scores_total,
            "scores_per_s": scores_per_s,
            "scores_per_1e6": scores_per_1e6,
            "min_sets_per_seq": min_sets,
            "max_sets_per_seq": max_sets,
            "max_vram_mb": max_vram_mb,
        },
    )


def tensor_batch_iterator(
    data: torch.Tensor, batch_size: int, shuffle: bool, generator: Optional[torch.Generator] = None
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    if data.numel() == 0:
        return
    if shuffle:
        perm = torch.randperm(data.size(0), generator=generator)
        indices = perm
    else:
        indices = torch.arange(data.size(0), dtype=torch.long)
    for start in range(0, data.size(0), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield batch_idx.clone(), data[batch_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--minhash-k", type=int, default=32)
    ap.add_argument("--adapter-rank", type=int, default=0)
    ap.add_argument("--router-topk", type=int, default=0)
    ap.add_argument("--profile", "--prof", action="store_true", dest="profile")
    ap.add_argument("--config", type=str, default="configs/diffusion_toy.yaml")
    ap.add_argument("--ska-backend", choices=["python", "triton", "keops"], default="python")
    ap.add_argument(
        "--dot-naive",
        action="store_true",
        help="Force dot-product attention modules to use naive (math) implementation.",
    )
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--samples", type=int, default=None, help="Override number of synthetic sequences.")
    ap.add_argument("--data-seq-len", type=int, default=None, help="Override synthetic sequence length.")
    ap.add_argument("--data-dim", type=int, default=None, help="Override synthetic feature dimensionality.")
    ap.add_argument("--data-batch-size", type=int, default=None, help="Override synthetic batch size.")
    ap.add_argument("--data-val-frac", type=float, default=None, help="Override validation fraction.")
    ap.add_argument("--data-modes", type=int, default=None, help="Override number of mixture modes.")
    ap.add_argument("--data-seed", type=int, default=None, help="Override synthetic data seed.")
    ap.add_argument("--data-mode", choices=["continuous", "synthetic", "text"], default="continuous")
    ap.add_argument("--text-dataset", choices=["wikitext2", "wikitext103"], default="wikitext2")
    ap.add_argument(
        "--text-cache-dir",
        type=str,
        default="~/.cache/set-attention/hf_datasets",
        help="Cache directory for HuggingFace Wikitext data.",
    )
    ap.add_argument("--text-seq-len", type=int, default=128)
    ap.add_argument("--text-stride", type=int, default=128)
    ap.add_argument("--text-train-line-limit", type=int, default=None)
    ap.add_argument("--text-val-line-limit", type=int, default=None)
    ap.add_argument("--text-train-limit", type=int, default=None, help="Cap number of text sequences for training.")
    ap.add_argument("--text-val-limit", type=int, default=None, help="Cap number of text sequences for validation.")
    ap.add_argument("--text-embed-seed", type=int, default=1337)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="")
    ap.add_argument("--wandb-run-name", type=str, default="")
    ap.add_argument("--wandb-tags", type=str, default="")
    ap.add_argument(
        "--sdpa-baseline",
        action="store_true",
        help="Run without set-attention banks (plain Transformer encoder).",
    )
    ap.add_argument(
        "--precompute-bank",
        action="store_true",
        help="Move banks/caches to the training device (default keeps them on CPU).",
    )
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument(
        "--benchmark-csv",
        type=str,
        default="",
        help="Optional CSV path to log benchmark metrics.",
    )
    ap.add_argument("--sample-count", type=int, default=10, help="Number of validation samples to log.")
    ap.add_argument("--sample-seed", type=int, default=1337, help="Seed for selecting logged samples.")
    ap.add_argument("--seed", type=int, default=2024, help="Base RNG seed when --seeds is not provided.")
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma/space separated list of seeds (overrides --seed).",
    )
    ap.add_argument("--reps", type=int, default=1, help="Number of repetitions per seed.")
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic torch algorithms (may reduce throughput).",
    )
    ap.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Enable benchmark-friendly seeding (disables deterministic algs but seeds RNG).",
    )
    defaults = ap.parse_args([])
    args = ap.parse_args()
    _configure_dot_naive(args.dot_naive)

    seed_values: List[int] = []
    if args.seeds:
        for part in args.seeds.replace(",", " ").split():
            part = part.strip()
            if not part:
                continue
            try:
                seed_values.append(int(part))
            except ValueError:
                continue
    if not seed_values:
        seed_values = [int(args.seed)]
    reps = max(1, int(args.reps))
    multi_run = len(seed_values) * reps > 1
    for seed in seed_values:
        for rep in range(1, reps + 1):
            run_args = copy.deepcopy(args)
            run_uid = f"{int(time.time() * 1e6)}-{os.getpid()}-{seed}-{rep}"
            run_single(run_args, defaults, seed, rep, run_uid, multi_run)
    return


def run_single(args, defaults, seed: int, rep: int, run_uid: str, multi_run: bool):
    torch.backends.cudnn.benchmark = True
    set_seed(seed, deterministic=args.deterministic, benchmark_mode=args.benchmark_mode)
    print(f"[Run] seed={seed} rep={rep} uid={run_uid}")

    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    dataset_label = args.text_dataset if args.data_mode == "text" else args.config
    wandb_config = {
        "script": "train_toy_diffusion_banked",
        "ska_backend": args.ska_backend,
        "precision": args.precision,
        "window": args.window,
        "stride": args.stride,
        "minhash_k": args.minhash_k,
        "router_topk": args.router_topk,
        "adapter_rank": args.adapter_rank,
        "steps": args.steps,
        "batch": args.batch,
        "dataset": dataset_label,
        "sdpa_baseline": args.sdpa_baseline,
        "precompute_bank": args.precompute_bank,
        "sample_count": args.sample_count,
        "seed": seed,
        "rep": rep,
        "run_uid": run_uid,
    }
    run_name = args.wandb_run_name or None
    if run_name and multi_run:
        run_name = f"{run_name}-s{seed}-r{rep}"
    wandb_run = init_wandb(
        args.wandb,
        args.wandb_project or None,
        run_name,
        wandb_config,
        wandb_tags,
    )

    cfg_yaml = {}
    yaml_path = args.config
    if os.path.isfile(args.config):
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as handle:
            cfg_yaml = yaml.safe_load(handle) or {}

    def override_from_cfg(cfg_key: str, arg_name: str):
        if cfg_key in cfg_yaml and getattr(args, arg_name) == getattr(defaults, arg_name):
            setattr(args, arg_name, cfg_yaml[cfg_key])

    override_from_cfg("steps", "steps")
    override_from_cfg("d_model", "d_model")
    override_from_cfg("nhead", "nhead")
    override_from_cfg("layers", "layers")

    text_mode = args.data_mode == "text"
    cfg_seed = int(cfg_yaml.get("seed", 2024))
    data_seed = cfg_seed
    if args.data_seed is not None:
        data_seed = int(args.data_seed)
    data_seed = seed
    torch.manual_seed(data_seed)
    random.seed(data_seed)
    device = torch.device(args.device)

    data_cfg = ToyDiffConfig(
        n_samples=int(cfg_yaml.get("n_samples", 1000)),
        seq_len=int(cfg_yaml.get("seq_len", 16)),
        dim=int(cfg_yaml.get("dim", 8)),
        batch_size=int(cfg_yaml.get("batch_size", 64)),
        val_frac=float(cfg_yaml.get("val_frac", 0.2)),
        seed=data_seed,
        n_modes=int(cfg_yaml.get("n_modes", 4)),
    )
    if args.samples is not None:
        data_cfg.n_samples = int(args.samples)
    if args.data_seq_len is not None:
        data_cfg.seq_len = int(args.data_seq_len)
    if args.data_dim is not None:
        data_cfg.dim = int(args.data_dim)
    batch_override = args.data_batch_size if args.data_batch_size is not None else args.batch
    data_cfg.batch_size = int(batch_override)
    if args.data_val_frac is not None:
        data_cfg.val_frac = float(args.data_val_frac)
    if args.data_modes is not None:
        data_cfg.n_modes = int(args.data_modes)
    data_cfg.seed = data_seed
    if args.data_mode == "synthetic":
        print("[Data] Using synthetic continuous sequences (toy diffusion).")
    text_train_ids: Optional[List[torch.Tensor]] = None
    text_val_ids: Optional[List[torch.Tensor]] = None
    text_train_ids_tensor: Optional[torch.Tensor] = None
    text_val_ids_tensor: Optional[torch.Tensor] = None
    text_vocab_size: Optional[int] = None
    text_embeddings: Optional[torch.Tensor] = None
    text_itos: Optional[List[str]] = None

    if text_mode:
        cache_dir = Path(args.text_cache_dir).expanduser()
        stride = args.text_stride if args.text_stride > 0 else args.text_seq_len
        embed_dim = args.data_dim if args.data_dim is not None else args.d_model
        text_payload = prepare_text_diffusion_data(
            args.text_dataset,
            cache_dir,
            args.text_seq_len,
            stride,
            args.text_train_line_limit,
            args.text_val_line_limit,
            args.text_train_limit,
            args.text_val_limit,
            embed_dim,
            args.text_embed_seed,
        )
        train_data = text_payload["train_data"]
        val_data = text_payload["val_data"]
        text_train_ids = text_payload["train_ids"]
        text_val_ids = text_payload["val_ids"]
        text_train_ids_tensor = text_payload["train_ids_tensor"]
        text_val_ids_tensor = text_payload["val_ids_tensor"]
        text_vocab_size = text_payload["vocab_size"]
        text_embeddings = text_payload["embeddings"]
        text_itos = text_payload["itos"]
        data_cfg.n_samples = train_data.size(0)
        data_cfg.seq_len = train_data.size(1)
        data_cfg.dim = train_data.size(2)
        data_cfg.batch_size = args.batch
    else:
        train_loader, val_loader = make_toy_continuous_sequences(data_cfg)

        def subset_tensor(loader):
            ds = loader.dataset
            if hasattr(ds, "dataset") and hasattr(ds, "indices"):
                base_tensor = ds.dataset.tensors[0]
                return base_tensor[ds.indices].clone()
            if hasattr(ds, "tensors"):
                return ds.tensors[0].clone()
            raise ValueError("Unsupported dataset type for tensor extraction")

        train_data = subset_tensor(train_loader)
        val_data = subset_tensor(val_loader)

    seq_len = train_data.size(1) if train_data.dim() > 1 else 0
    feat_dim = train_data.size(2) if train_data.dim() > 2 else 0
    mode_label = "text" if text_mode else args.data_mode
    print(
        f"[Diffusion-Banked] dataset loaded ({mode_label}) | train sequences {train_data.size(0)} | "
        f"val sequences {val_data.size(0)} | seq len {seq_len} | dim {feat_dim}"
    )
    text_embedding_table = text_embeddings.to(device) if text_mode and text_embeddings is not None else None

    train_cache = val_cache = None
    atom_emb = adapter = phi_snapshot = None
    universe = None
    if not args.sdpa_baseline:
        if text_mode and text_train_ids is not None and text_val_ids is not None:
            train_ids = text_train_ids
            val_ids = text_val_ids
        else:
            train_ids = [make_id_sequence(x) for x in train_data]
            val_ids = [make_id_sequence(x) for x in val_data]
        train_bank = build_windowed_bank_from_ids(train_ids, window=args.window, stride=args.stride)
        val_bank = build_windowed_bank_from_ids(val_ids, window=args.window, stride=args.stride)

        if text_mode and text_vocab_size is not None:
            vocab_size = text_vocab_size
        else:
            vocab_size = (
                int(max(train_bank.values.max(), val_bank.values.max()).item() + 1)
                if train_bank.values.numel() > 0
                else data_cfg.seq_len * 2
            )
        atom_emb = nn.Embedding(vocab_size, args.d_model).to(device)
        if args.adapter_rank > 0:
            adapter = AtomFeatureAdapter(args.d_model, rank=args.adapter_rank).to(device)
            with torch.no_grad():
                phi_snapshot = adapter(atom_emb.weight).detach()
        else:
            phi_snapshot = atom_emb.weight

        universe_ids = torch.arange(vocab_size, dtype=torch.long)
        metadata = {"task": "text_diffusion" if text_mode else "toy_diffusion"}
        if text_mode:
            metadata["dataset"] = args.text_dataset
        universe = UniversePool(universe_ids, metadata=metadata)
        train_minhash = MinHasher(k=args.minhash_k, device=train_bank.values.device)
        val_minhash = MinHasher(k=args.minhash_k, device=val_bank.values.device)
        train_cache = SetFeatureCache(
            universe, train_bank.values, train_bank.set_offsets, train_bank.seq_offsets, minhash=train_minhash
        )
        val_cache = SetFeatureCache(
            universe, val_bank.values, val_bank.set_offsets, val_bank.seq_offsets, minhash=val_minhash
        )
        if args.precompute_bank:
            universe = universe.to(device)
            train_cache = train_cache.to(device)
            val_cache = val_cache.to(device)

    model = BankedDenoiser(
        in_dim=data_cfg.dim,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        router_topk=args.router_topk,
        ska_backend=args.ska_backend,
        precision=args.precision,
        use_ska=not args.sdpa_baseline,
    ).to(device)
    ddpm = SimpleDDPM(T=args.steps, device=device)
    params = list(model.parameters())
    if not args.sdpa_baseline and atom_emb is not None:
        params += list(atom_emb.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)

    if args.benchmark:
        run_diffusion_benchmark(
            args,
            model,
            ddpm,
            adapter,
            atom_emb,
            phi_snapshot,
            train_cache,
            train_data,
            optimizer,
            device,
            wandb_run,
            args.benchmark_csv,
            seed,
            rep,
            run_uid,
        )
        wandb_run.finish()
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        train_gen = torch.Generator().manual_seed(seed + epoch)
        with profiler(args.profile) as prof:
            for batch_idx, xb in tensor_batch_iterator(
                train_data, data_cfg.batch_size, shuffle=True, generator=train_gen
            ):
                xb = xb.to(device)
                if not args.sdpa_baseline:
                    phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                    Phi, Sig, Size, q_ptrs = train_cache.gather_flat(batch_idx.to(device), phi_dynamic)
                    model.set_current_bank(Phi, Sig, Size, q_ptrs)
                loss = ddpm.loss(model, xb, lambda t, dim: timestep_embedding(t, args.d_model), args.d_model)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * xb.size(0)
                count += xb.size(0)
        train_loss = total_loss / max(1, count)

        model.eval()
        sample_indices = select_sample_indices(val_data.size(0), args.sample_count, args.sample_seed + epoch)
        sample_lookup = {idx: order for order, idx in enumerate(sample_indices)}
        captured_samples: dict[int, dict] = {}
        with torch.no_grad():
            mmds, chamfers, nn1s = [], [], []
            text_token_matches = 0.0
            text_token_total = 0.0
            text_refs_all: List[List[str]] = []
            text_hyps_all: List[List[str]] = []
            for batch_idx, xb in tensor_batch_iterator(val_data, data_cfg.batch_size, shuffle=False):
                xb_cpu = xb
                xb = xb_cpu.to(device)
                if not args.sdpa_baseline:
                    phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                    Phi, Sig, Size, q_ptrs = val_cache.gather_flat(batch_idx.to(device), phi_dynamic)
                    model.set_current_bank(Phi, Sig, Size, q_ptrs)
                batch_indices = batch_idx.detach().cpu().tolist()
                need_samples = any(idx in sample_lookup for idx in batch_indices)
                want_generated = need_samples or text_mode
                val_mmd, val_chamfer, val_nn1, gen_batch = eval_suite(
                    ddpm, model, xb, args.d_model, return_generated=want_generated
                )
                mmds.append(val_mmd)
                chamfers.append(val_chamfer)
                nn1s.append(val_nn1)
                ref_token_batch = None
                hyp_token_batch = None
                if text_mode and gen_batch is not None and text_embedding_table is not None:
                    if text_val_ids_tensor is not None:
                        tgt_ids = text_val_ids_tensor[batch_idx].to(device)
                    else:
                        tgt_ids = decode_embeddings_to_ids(xb, text_embedding_table)
                    pred_ids = decode_embeddings_to_ids(gen_batch.to(device), text_embedding_table)
                    text_token_matches += float((pred_ids == tgt_ids).sum().item())
                    text_token_total += float(pred_ids.numel())
                    ref_token_batch = [
                        [text_itos[int(tok)] for tok in seq.tolist()] for seq in tgt_ids.detach().cpu()
                    ]
                    hyp_token_batch = [
                        [text_itos[int(tok)] for tok in seq.tolist()] for seq in pred_ids.detach().cpu()
                    ]
                    text_refs_all.extend(ref_token_batch)
                    text_hyps_all.extend(hyp_token_batch)
                if need_samples and gen_batch is not None:
                    for local_pos, global_idx in enumerate(batch_indices):
                        order = sample_lookup.get(int(global_idx))
                        if order is None or order in captured_samples:
                            continue
                        entry = {"idx": int(global_idx)}
                        if text_mode and ref_token_batch is not None and hyp_token_batch is not None:
                            entry["ref_text"] = " ".join(ref_token_batch[local_pos])
                            entry["gen_text"] = " ".join(hyp_token_batch[local_pos])
                        else:
                            entry["ref_tensor"] = xb_cpu[local_pos].detach().cpu()
                            entry["gen_tensor"] = gen_batch[local_pos].detach().cpu()
                        captured_samples[order] = entry

        def safe_mean(values):
            return float(sum(values) / len(values)) if values else float("nan")

        val_mmd_mean = safe_mean(mmds)
        val_chamfer_mean = safe_mean(chamfers)
        val_nn1_mean = safe_mean(nn1s)
        text_token_acc = (text_token_matches / text_token_total) if text_token_total > 0 else None
        text_bleu = corpus_bleu(text_refs_all, text_hyps_all) if text_refs_all else None

        mode_tag = "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}/{args.precision}"
        msg = (
            f"[Diffusion-Banked][{mode_tag}] epoch {epoch:02d} "
            f"train loss {train_loss:.4f} | val MMD {val_mmd_mean:.4f} | "
            f"Chamfer {val_chamfer_mean:.4f} | 1NN {val_nn1_mean:.3f}"
        )
        if text_mode and text_token_acc is not None:
            msg += f" | token acc {text_token_acc * 100:.2f}% | BLEU {text_bleu or 0.0:.3f}"
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
        print(msg)
        sample_text = None
        if captured_samples:
            blocks = []
            for order, idx in enumerate(sample_indices):
                entry = captured_samples.get(order)
                if entry is None:
                    continue
                idx_display = entry.get("idx", idx)
                if text_mode and "ref_text" in entry:
                    blocks.append(
                        f"[{order}] idx={idx_display}\n"
                        f"TGT {entry['ref_text']}\n"
                        f"GEN {entry['gen_text']}"
                    )
                else:
                    ref_sample = entry["ref_tensor"]
                    gen_sample = entry["gen_tensor"]
                    blocks.append(
                        f"[{order}] idx={idx_display}\n"
                        f"TGT {summarize_tensor(ref_sample)}\n"
                        f"GEN {summarize_tensor(gen_sample)}"
                    )
            if blocks:
                sample_text = "\n\n".join(blocks)
        if wandb_run.enabled:
            wandb_payload = {
                "train/loss": train_loss,
                "val/mmd": val_mmd_mean,
                "val/chamfer": val_chamfer_mean,
                "val/1nn": val_nn1_mean,
            }
            if text_mode and text_token_acc is not None:
                wandb_payload["val/token_acc"] = text_token_acc
                if text_bleu is not None:
                    wandb_payload["val/bleu"] = text_bleu
            if args.profile:
                wandb_payload["train/time_s"] = prof["time_s"]
            if sample_text is not None:
                wandb_payload["samples/generated"] = sample_text
            wandb_run.log(wandb_payload, step=epoch)

    wandb_run.finish()


def eval_suite(
    ddpm: SimpleDDPM,
    model: BankedDenoiser,
    X: torch.Tensor,
    d_model: int,
    return_generated: bool = False,
):
    shape = X.shape
    x_gen = ddpm.sample(model, shape, lambda t, dim: timestep_embedding(t, d_model), d_model)
    x_flat = X.contiguous().view(shape[0], -1)
    g_flat = x_gen.contiguous().view(shape[0], -1)
    if x_flat.shape[1] != g_flat.shape[1]:
        d = min(x_flat.shape[1], g_flat.shape[1])
        x_flat = x_flat[:, :d]
        g_flat = g_flat[:, :d]
    mmd = float(mmd2_unbiased_from_feats(x_flat, g_flat, gamma=0.5).detach().cpu())
    chamfer = chamfer_l2(X, x_gen)
    nn1 = one_nn_two_sample(x_flat, g_flat)
    generated = x_gen.detach().cpu() if return_generated else None
    return mmd, chamfer, nn1, generated


if __name__ == "__main__":
    main()
