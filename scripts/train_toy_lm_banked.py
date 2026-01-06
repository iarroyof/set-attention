import argparse
import copy
import csv
import json
import math
import subprocess
import time
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import multiprocessing as mp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.repro import set_seed

from set_attention.utils.bench_skip import (
    should_skip_dense,
    should_skip_ska,
)
from set_attention.data import configure_hf_cache, resolve_data_root
from set_attention.data.wikitext import (
    chunk_tokens,
    iter_wikitext_lines,
    load_wikitext_hf_dataset,
    load_wikitext_lines,
    tokenize_lines,
)
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher
from set_attention.utils.profiling import profiler
from set_attention.utils.sample_logging import format_text_samples
from set_attention.utils.wandb import init_wandb
from set_attention.experiments.nlp_eval import corpus_bleu
from set_attention.training.text_utils import ids_to_tokens, text_batch_iterator


def make_char_data(n=2000, seq_len=64, vocab=32, seed=3):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, vocab, (n, seq_len), generator=g)
    Y = torch.roll(X, shifts=-1, dims=1)
    return X, Y


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
EXPLICIT_ATTN_IMPLS = ("pytorch_math", "explicit_matmul")


def build_vocab_from_tokens(tokens):
    vocab = list(SPECIAL_TOKENS)
    seen = set(vocab)
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            vocab.append(tok)
    stoi = {tok: idx for idx, tok in enumerate(vocab)}
    itos = {idx: tok for tok, idx in stoi.items()}
    return stoi, itos


def encode_token_sequences(token_sequences, stoi):
    unk_id = stoi.get("<unk>", 0)
    encoded = []
    for seq in token_sequences:
        ids = [stoi.get(tok, unk_id) for tok in seq]
        encoded.append(torch.tensor(ids, dtype=torch.long))
    if not encoded:
        return torch.empty(0, 0, dtype=torch.long)
    return torch.stack(encoded, dim=0)


def tensors_to_text_pairs(X, Y, itos):
    pairs = []
    refs = []
    for src, tgt in zip(X, Y):
        src_tokens = [itos[int(tok)] for tok in src.tolist()]
        tgt_tokens = [itos[int(tok)] for tok in tgt.tolist()]
        pairs.append((" ".join(src_tokens), " ".join(tgt_tokens)))
        refs.append(tgt_tokens)
    return pairs, refs


def _chunk_lines(line_iter: Iterator[str], chunk_size: int):
    batch: List[str] = []
    for line in line_iter:
        batch.append(line)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _unique_tokens_from_lines(lines: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for line in lines:
        for tok in line.split():
            if tok in seen:
                continue
            seen.add(tok)
            ordered.append(tok)
    return ordered


def build_vocab_from_stream(line_iter, num_workers: int = 1, chunk_size: int = 2048):
    vocab = list(SPECIAL_TOKENS)
    stoi = {tok: idx for idx, tok in enumerate(vocab)}
    itos = {idx: tok for tok, idx in stoi.items()}
    workers = max(1, int(num_workers))
    if workers == 1:
        for line in line_iter:
            for tok in line.split():
                if tok in stoi:
                    continue
                idx = len(stoi)
                stoi[tok] = idx
                itos[idx] = tok
        return stoi, itos

    chunk_iter = _chunk_lines(line_iter, chunk_size)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for chunk_tokens in pool.imap(_unique_tokens_from_lines, chunk_iter):
            for tok in chunk_tokens:
                if tok in stoi:
                    continue
                idx = len(stoi)
                stoi[tok] = idx
                itos[idx] = tok
    return stoi, itos


def make_basic_line_encoder(stoi, unk_id: int):
    def encode(line: str) -> List[int]:
        tokens = line.split()
        if not tokens:
            return []
        return [stoi.get(tok, unk_id) for tok in tokens]

    return encode


def make_basic_decoder(itos):
    def decode(ids_tensor: torch.Tensor) -> List[str]:
        return ids_to_tokens(ids_tensor, itos)

    return decode


def load_hf_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'transformers' package is required for --hf-tokenizer-name."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(name)
    additions = {}
    if tokenizer.pad_token is None:
        additions["pad_token"] = "<pad>"
    if tokenizer.bos_token is None:
        additions["bos_token"] = "<s>"
    if tokenizer.eos_token is None:
        additions["eos_token"] = "</s>"
    if additions:
        tokenizer.add_special_tokens(additions)
    return tokenizer


class WikitextStreamingData:
    def __init__(
        self,
        dataset: str,
        cache_dir: Path,
        seq_len: int,
        seq_stride: int,
        line_encoder: Callable[[str], List[int]],
        decode_tokens_fn: Callable[[torch.Tensor], List[str]],
        *,
        line_limit: Optional[int] = None,
        dataset_obj=None,
    ) -> None:
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.seq_len = int(seq_len)
        self.stride = int(seq_stride) if seq_stride > 0 else int(seq_len)
        self.line_limit = line_limit
        self.line_encoder = line_encoder
        self.decode_tokens_fn = decode_tokens_fn
        self._dataset_obj = dataset_obj or load_wikitext_hf_dataset(dataset, self.cache_dir)

    def _line_iter(self, split: str):
        return iter_wikitext_lines(
            self.dataset,
            split,
            self.cache_dir,
            self.line_limit,
            dataset_obj=self._dataset_obj,
        )

    def _chunk_iter(self, split: str):
        buffer: deque[int] = deque()
        for line in self._line_iter(split):
            encoded = self.line_encoder(line)
            if not encoded:
                continue
            buffer.extend(encoded)
            while len(buffer) >= self.seq_len:
                chunk = list(islice(buffer, 0, self.seq_len))
                yield chunk
                for _ in range(self.stride):
                    if not buffer:
                        break
                    buffer.popleft()

    def iter_sequences(self, split: str):
        for idx, chunk in enumerate(self._chunk_iter(split)):
            if not chunk:
                continue
            ids = torch.tensor(chunk, dtype=torch.long)
            yield idx, ids

    def batch_iterator(self, split: str, batch_size: int):
        batch_indices = []
        batch_seqs = []
        for seq_idx, seq in self.iter_sequences(split):
            batch_indices.append(seq_idx)
            batch_seqs.append(seq)
            if len(batch_seqs) == batch_size:
                yield self._build_batch(batch_indices, batch_seqs)
                batch_indices = []
                batch_seqs = []
        if batch_seqs:
            yield self._build_batch(batch_indices, batch_seqs)

    def _build_batch(self, indices, seqs):
        src = torch.stack(seqs, dim=0)
        tgt = torch.roll(src, shifts=-1, dims=1)
        refs = [self.decode_tokens_fn(row) for row in tgt]
        return torch.tensor(indices, dtype=torch.long), src, tgt, refs

    def sequences_for_bank(self, split: str):
        for _, seq in self.iter_sequences(split):
            yield seq

    def benchmark_batch(self, batch_size: int):
        iterator = self.batch_iterator("train", batch_size)
        try:
            return next(iterator)
        except StopIteration:
            return None


def save_vocab_file(path: Path, itos: dict) -> None:
    ordered = [itos[i] for i in range(len(itos))]
    payload = {"itos": ordered}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_vocab_file(path: Path):
    payload = json.loads(path.read_text())
    tokens = payload.get("itos", [])
    stoi = {tok: idx for idx, tok in enumerate(tokens)}
    itos = {idx: tok for idx, tok in enumerate(tokens)}
    return stoi, itos


def load_wikitext_dataset(args, cache_dir):
    line_limit = args.dataset_lines if args.dataset_lines > 0 else None
    subset_indices = args.subset_indices if getattr(args, "subset_indices", None) else None
    train_lines = load_wikitext_lines(args.dataset, "train", cache_dir, line_limit, indices=subset_indices)
    val_lines = load_wikitext_lines(args.dataset, "validation", cache_dir, line_limit)
    train_tokens = tokenize_lines(train_lines)
    val_tokens = tokenize_lines(val_lines)
    train_chunks = chunk_tokens(train_tokens, args.seq_len, args.seq_stride)
    val_chunks = chunk_tokens(val_tokens, args.seq_len, args.seq_stride)
    if not train_chunks:
        raise RuntimeError("No training sequences were generated from Wikitext.")
    stoi, itos = build_vocab_from_tokens(train_tokens)
    train_X = encode_token_sequences(train_chunks, stoi)
    val_X = encode_token_sequences(val_chunks, stoi)
    train_Y = torch.roll(train_X, shifts=-1, dims=1)
    val_Y = torch.roll(val_X, shifts=-1, dims=1)
    return train_X, train_Y, val_X, val_Y, stoi, itos


def _explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    # q, k, v: (B, H, L, Dh)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


class ExplicitMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for explicit attention.")
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, L, self.nhead, self.d_head).transpose(1, 2)  # (B, H, L, Dh)
        k = k.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        ctx = _explicit_attention(q, k, v, self.scale)
        ctx = self.dropout(ctx)
        ctx = ctx.transpose(1, 2).reshape(B, L, self.d_model)
        return self.out_proj(ctx)


class ExplicitTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.self_attn = ExplicitMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(src)
        src = self.norm1(src + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src


class TinyLMBackbone(nn.Module):
    def __init__(self, vocab: int, d_model: int, nhead: int, layers: int, attn_baseline: str = "pytorch"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.attn_baseline = attn_baseline
        self.nhead = nhead
        if attn_baseline == "explicit":
            self.layers = nn.ModuleList(
                [ExplicitTransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.0) for _ in range(layers)]
            )
        else:
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb(x)
        if self.attn_baseline == "explicit":
            out = emb
            for layer in self.layers:
                out = layer(out.float())  # enforce fp32 in explicit baseline
            return out
        return self.enc(emb)


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


def _attn_impl_label(args, sdpa_mode: bool) -> str:
    if sdpa_mode:
        return "dot_explicit" if args.attn_baseline == "explicit" else "dot_builtin"
    return f"ska/{args.ska_backend}"


def _sanity_check_explicit_attention(device: torch.device, d_model: int, nhead: int, tol: float = 1e-5) -> None:
    torch.manual_seed(1337)
    B, L = 2, 4
    d_head = d_model // nhead
    x = torch.randn(B, nhead, L, d_head, device=device)
    explicit = _explicit_attention(x, x, x, 1.0 / math.sqrt(d_head))
    sdp = F.scaled_dot_product_attention(x, x, x)
    max_diff = (explicit - sdp).abs().max().item()
    if max_diff > tol:
        raise RuntimeError(
            f"Explicit attention sanity check failed (max diff {max_diff:.2e} > {tol}). "
            "Explicit baseline no longer matches PyTorch SDP math."
        )


def _configure_dot_naive(dot_naive: bool) -> None:
    if not dot_naive:
        return
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    print("[SDP] dot-naive enabled: flash/mem-efficient SDP disabled; using math backend.")


def run_lm_benchmark(
    args,
    backbone,
    set_attn,
    router,
    head,
    adapter,
    atom_emb,
    phi_snapshot,
    cache,
    optimizer,
    bench_batch,
    max_len,
    device,
    wandb_run,
    benchmark_csv,
    seed,
    rep,
    run_uid,
):
    attn_impl = _attn_impl_label(args, args.sdpa_baseline)
    # Normalize benchmark batch
    batch_idx = src_ids = tgt_ids = None
    bench_batch_size = None
    seq_len = max_len
    if bench_batch is not None:
        if isinstance(bench_batch, (list, tuple)) and len(bench_batch) >= 3:
            batch_idx, src_ids, tgt_ids = bench_batch[:3]
        else:
            raise RuntimeError("Unexpected benchmark batch structure for LM.")
        bench_batch_size = src_ids.size(0)
        seq_len = tgt_ids.size(1)
    # Pre-skip based on VRAM estimate or dry-run
    if args.gpu_vram > 0 and bench_batch_size is not None:
        if args.sdpa_baseline:
            decision = should_skip_dense(bench_batch_size, max_len, args.d_model, args.nhead, args.precision, args.gpu_vram)
        else:
            decision = should_skip_ska(bench_batch_size, max_len, args.window, args.stride, args.nhead, args.precision, args.gpu_vram)
        if decision.skip:
            _append_benchmark_row(
                benchmark_csv,
                {
                    "script": "train_toy_lm_banked",
                    "task": "lm",
                    "dataset": args.dataset or "custom",
                    "dataset_id": args.dataset or "custom",
                    "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
                    "attn_impl": attn_impl,
                    "precision": args.precision,
                    "status": "skipped",
                    "skip_reason": decision.reason,
                    "gpu_vram_gb": args.gpu_vram,
                    "batch": bench_batch_size,
                    "seq_len": seq_len,
                    "seq_stride": args.seq_stride,
                },
            )
            return
    if args.dry_run:
        _append_benchmark_row(
            benchmark_csv,
            {
                "script": "train_toy_lm_banked",
                "task": "lm",
                "dataset": args.dataset or "custom",
                "dataset_id": args.dataset or "custom",
                "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
                "attn_impl": attn_impl,
                "precision": args.precision,
                "status": "dry_run",
                "skip_reason": "dry_run",
                "gpu_vram_gb": args.gpu_vram,
                "batch": bench_batch_size or args.batch,
                "seq_len": seq_len,
                "seq_stride": args.seq_stride,
            },
        )
        return
    if bench_batch is None:
        print("[benchmark] no data available.")
        return
    batch_idx = batch_idx.to(device)
    src_ids = src_ids.to(device)
    tgt_ids = tgt_ids.to(device)

    stats_ptrs = None
    stats_sizes = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def step():
        nonlocal stats_ptrs, stats_sizes
        optimizer.zero_grad(set_to_none=True)
        phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
        Phi, Sig, Size, q_ptrs = cache.gather_flat(batch_idx, phi_dynamic)
        token_states = backbone(src_ids)
        Z, q_ptrs = set_attn(Phi, Sig, Size, q_ptrs, Phi, Sig, Size, q_ptrs)
        routed = router(token_states, Z, Phi, q_ptrs)
        logits = head(routed)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
        loss.backward()
        optimizer.step()
        stats_ptrs = q_ptrs.detach().cpu()
        stats_sizes = Size.detach().cpu()

    try:
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
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        msg = str(exc)
        is_oom = "out of memory" in msg.lower()
        if args.skip_oom and is_oom:
            torch.cuda.empty_cache()
            _append_benchmark_row(
                benchmark_csv,
                {
                    "script": "train_toy_lm_banked",
                    "task": "lm",
                    "dataset": args.dataset or "custom",
                    "dataset_id": args.dataset or "custom",
                    "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
                    "attn_impl": attn_impl,
                    "precision": args.precision,
                    "status": "oom",
                    "skip_reason": msg[:160],
                    "gpu_vram_gb": args.gpu_vram,
                    "batch": bench_batch_size or args.batch,
                    "seq_len": max_len,
                    "seq_stride": args.seq_stride,
                },
            )
            return
        raise
    max_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    tokens_total = bench_batch_size * max_len * args.bench_iters
    throughput = tokens_total / elapsed if elapsed > 0 else 0.0
    info = _system_info()
    if stats_ptrs is not None and stats_sizes is not None:
        avg_sets, avg_atoms, scores_per_batch, min_sets, max_sets = _summarize_ska_batch(
            stats_ptrs, stats_sizes, set_attn.num_heads if set_attn is not None else args.nhead
        )
        scores_total = scores_per_batch * args.bench_iters
    else:
        avg_sets = avg_atoms = min_sets = max_sets = 0.0
        scores_total = float(
            bench_batch_size * max_len * max_len * max(1, args.nhead) * args.bench_iters
        )
    scores_per_s = scores_total / elapsed if elapsed > 0 else 0.0
    throughput_per_m = scores_per_s / 1e6
    scores_per_token = scores_total / tokens_total if tokens_total > 0 else 0.0
    print(
        f"[benchmark][lm] backend={args.ska_backend} ({attn_impl}) precision={args.precision} "
        f"tokens/s={throughput:.1f} elapsed={elapsed:.3f}s"
    )
    if wandb_run.enabled:
        wandb_run.log(
            {
                "benchmark/tokens_per_s": throughput,
                "benchmark/elapsed_s": elapsed,
                "benchmark/batch_tokens": tokens_total,
                "benchmark/avg_sets_per_seq": avg_sets,
                "benchmark/avg_atoms_per_set": avg_atoms,
                "benchmark/scores_total": scores_total,
                "benchmark/scores_per_s": scores_per_s,
                "benchmark/scores_per_1e6": throughput_per_m,
                "benchmark/scores_per_token": scores_per_token,
                "benchmark/min_sets_per_seq": min_sets,
                "benchmark/max_sets_per_seq": max_sets,
                "benchmark/max_vram_mb": max_vram_mb,
                "benchmark/seed": seed,
                "benchmark/rep": rep,
            }
        )
    _append_benchmark_row(
        benchmark_csv,
        {
            "script": "train_toy_lm_banked",
            "task": "lm",
            "dataset": args.dataset or "custom",
            "dataset_id": args.dataset or "custom",
            "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
            "attn_impl": attn_impl,
            "precision": args.precision,
            "attn": args.attn,
            "batch": bench_batch_size or args.batch,
            "seq_len": max_len,
            "seq_stride": args.seq_stride,
            "window": args.window,
            "stride": args.stride,
            "minhash_k": args.minhash_k,
            "router_topk": args.router_topk,
            "adapter_rank": args.adapter_rank,
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
            "tokens_per_s": throughput,
            "elapsed_s": elapsed,
            "tokens_total": tokens_total,
            "avg_sets_per_seq": avg_sets,
            "avg_atoms_per_set": avg_atoms,
            "scores_total": scores_total,
            "scores_per_s": scores_per_s,
            "scores_per_1e6": throughput_per_m,
            "scores_per_token": scores_per_token,
            "min_sets_per_seq": min_sets,
            "max_sets_per_seq": max_sets,
            "max_vram_mb": max_vram_mb,
            "status": "ok",
            "skip_reason": "",
            "gpu_vram_gb": args.gpu_vram,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true"], default="dot")
    ap.add_argument(
        "--attn-baseline",
        choices=["pytorch", "explicit"],
        default="pytorch",
        help="Baseline attention implementation when using --sdpa-baseline.",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--minhash-k", type=int, default=64)
    ap.add_argument("--adapter-rank", type=int, default=0)
    ap.add_argument("--router-topk", type=int, default=0)
    ap.add_argument("--profile", "--prof", action="store_true", dest="profile")
    ap.add_argument("--ska-backend", choices=["python", "triton", "keops"], default="python")
    ap.add_argument(
        "--dot-naive",
        action="store_true",
        help="Force PyTorch attention to run in naive (math) mode for dot baselines.",
    )
    ap.add_argument("--vocab-path", type=str, default="", help="Optional path to persist the streaming vocabulary JSON.")
    ap.add_argument("--vocab-workers", type=int, default=0, help="Worker processes to build streaming vocabularies (0=auto).")
    ap.add_argument("--reuse-vocab", dest="reuse_vocab", action="store_true", help="Reuse/save streaming vocab files to skip re-tokenization.")
    ap.add_argument("--no-reuse-vocab", dest="reuse_vocab", action="store_false", help="Disable vocab caching for streaming datasets.")
    ap.set_defaults(reuse_vocab=True)
    ap.add_argument("--sdpa-baseline", action="store_true", help="Use standard SDPA instead of banked attention.")
    ap.add_argument("--streaming", dest="streaming", action="store_true", help="Stream datasets instead of materializing tensors.")
    ap.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming data loaders.")
    ap.set_defaults(streaming=None)
    ap.add_argument("--precompute-bank", action="store_true", help="Move precomputed banks and caches to the training device.")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument(
        "--benchmark-csv",
        type=str,
        default="",
        help="Optional CSV file to append benchmark metrics.",
    )
    ap.add_argument("--gpu-vram", type=float, default=0.0, help="Approx GPU VRAM in GB for skip estimation (0=disable).")
    ap.add_argument("--skip-oom", action="store_true", help="Skip configs that OOM or exceed estimated VRAM.")
    ap.add_argument("--dry-run", action="store_true", help="Print skip/ok decision and write CSV status without running.")
    ap.add_argument(
        "--metrics-csv",
        type=str,
        default="",
        help="Optional CSV file to append training/validation metrics per epoch.",
    )
    ap.add_argument("--dataset", choices=["", "wikitext2", "wikitext103"], default="")
    ap.add_argument("--dataset-lines", type=int, default=0, help="Limit number of text lines per split (0 = all).")
    ap.add_argument(
        "--subset-path",
        type=str,
        default="",
        help="Optional JSON produced by scripts/data_make_subsets.py containing 'indices' for train split.",
    )
    ap.add_argument(
        "--subset-budget-tokens",
        type=float,
        default=0.0,
        help="Optional token budget fraction (0<frac<=1) or absolute tokens for WT103; requires --subset-path if not zero.",
    )
    ap.add_argument(
        "--hf-tokenizer-name",
        type=str,
        default="",
        help="Optional HuggingFace tokenizer name to replace the default whitespace tokenizer (requires transformers).",
    )
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seq-stride", type=int, default=128)
    ap.add_argument("--data-root", type=str, default="")
    ap.add_argument("--hf-cache-dir", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="")
    ap.add_argument("--wandb-run-name", type=str, default="")
    ap.add_argument("--wandb-tags", type=str, default="")
    ap.add_argument("--sample-count", type=int, default=10, help="Number of validation samples to log per epoch.")
    ap.add_argument("--sample-seed", type=int, default=1337, help="Seed controlling which samples are logged.")
    ap.add_argument("--eval-seed", type=int, default=1337, help="Seed to make validation evaluation deterministic across variants.")
    ap.add_argument("--seed", type=int, default=2024, help="Base RNG seed used when --seeds is not provided.")
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated list of seeds (overrides --seed).",
    )
    ap.add_argument("--reps", type=int, default=1, help="Number of repetitions per seed.")
    args = ap.parse_args()
    _configure_dot_naive(args.dot_naive)
    if args.sdpa_baseline and args.attn_baseline == "explicit":
        _sanity_check_explicit_attention(torch.device(args.device), args.d_model, args.nhead)
    seed_values = []
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
        seed_values = [args.seed]
    reps = max(1, int(args.reps))
    total_runs = len(seed_values) * reps
    for seed in seed_values:
        for rep in range(1, reps + 1):
            run_uid = f"{seed}-{rep}-{int(time.time() * 1e6) & 0xFFFFFFFFFFFF}"
            run_args = copy.deepcopy(args)
            run_single(run_args, seed, rep, run_uid, total_runs > 1)
    return


def run_single(args, seed: int, rep: int, run_uid: str, multi_run: bool):
    set_seed(seed)
    print(f"[Run] seed={seed} rep={rep} uid={run_uid}")

    if args.streaming is None:
        args.streaming = bool(args.dataset == "wikitext103")
    if not args.dataset:
        args.streaming = False
    use_hf_tokenizer = bool(args.hf_tokenizer_name)
    if use_hf_tokenizer and not args.dataset:
        raise ValueError("--hf-tokenizer-name requires --dataset to be set.")
    if use_hf_tokenizer and not args.streaming:
        raise ValueError("--hf-tokenizer-name currently requires --streaming.")
    subset_indices = None
    if args.subset_path:
        subset_payload = json.loads(Path(args.subset_path).read_text())
        subset_indices = subset_payload.get("indices", [])
        if not subset_indices:
            raise ValueError(f"No indices found in subset file: {args.subset_path}")
        args.streaming = False  # subset selection requires non-streaming mode
        args.subset_indices = subset_indices
        if args.subset_budget_tokens and subset_payload.get("target_tokens"):
            print(
                f"[Data] Subset token budget target={subset_payload.get('target_tokens')} "
                f"actual={subset_payload.get('actual_tokens')} "
                f"fraction={subset_payload.get('actual_fraction'):.4f}"
            )
        print(f"[Data] Using subset indices from {args.subset_path} (count={len(subset_indices)})")

    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_config = {
        "script": "train_toy_lm_banked",
        "dataset": args.dataset or "char",
        "seq_len": args.seq_len,
        "seq_stride": args.seq_stride,
        "window": args.window,
        "stride": args.stride,
        "minhash_k": args.minhash_k,
        "router_topk": args.router_topk,
        "adapter_rank": args.adapter_rank,
        "ska_backend": args.ska_backend,
        "precision": args.precision,
        "batch": args.batch,
        "sample_count": args.sample_count,
        "sdpa_baseline": args.sdpa_baseline,
        "streaming": bool(args.streaming),
        "precompute_bank": args.precompute_bank,
        "reuse_vocab": args.reuse_vocab,
        "vocab_path": args.vocab_path or "",
        "vocab_workers": args.vocab_workers,
        "hf_tokenizer_name": args.hf_tokenizer_name or "",
        "seed": seed,
        "rep": rep,
        "run_uid": run_uid,
        "attn_impl": _attn_impl_label(args, args.sdpa_baseline),
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

    data_root = resolve_data_root(args.data_root)
    hf_cache = configure_hf_cache(data_root, args.hf_cache_dir)

    streaming_data: Optional[WikitextStreamingData] = None
    train_text_pairs = []
    val_text_pairs = []
    train_refs = []
    val_refs = []
    train_X = train_Y = val_X = val_Y = None
    vocab_file_path: Optional[Path] = None

    hf_tokenizer = None
    hf_special_ids: Optional[set[int]] = None

    if args.dataset:
        line_limit = args.dataset_lines if args.dataset_lines > 0 else None
        if args.streaming:
            hf_dataset = load_wikitext_hf_dataset(args.dataset, hf_cache)
            if use_hf_tokenizer:
                hf_tokenizer = load_hf_tokenizer(args.hf_tokenizer_name)
                stoi = hf_tokenizer.get_vocab()
                itos = {idx: tok for tok, idx in stoi.items()}
                hf_special_ids = {int(tid) for tid in hf_tokenizer.all_special_ids if tid is not None}

                def hf_line_encoder(text: str) -> List[int]:
                    return list(hf_tokenizer.encode(text, add_special_tokens=False))

                def hf_decode(tokens: torch.Tensor) -> List[str]:
                    ids = tokens.tolist()
                    toks = hf_tokenizer.convert_ids_to_tokens(ids)
                    skip_ids = hf_special_ids or set()
                    return [tok for tok, idx in zip(toks, ids) if idx not in skip_ids]

                line_encoder = hf_line_encoder
                decode_tokens_fn = hf_decode
            else:
                if args.vocab_path:
                    vocab_file_path = Path(args.vocab_path)
                else:
                    vocab_file_path = Path(hf_cache) / f"{args.dataset}_vocab.json"
                if args.vocab_workers > 0:
                    vocab_workers = args.vocab_workers
                else:
                    cpu_count = os.cpu_count() or 1
                    vocab_workers = max(1, min(32, cpu_count))
                if args.reuse_vocab and vocab_file_path.exists():
                    stoi, itos = load_vocab_file(vocab_file_path)
                    print(f"[Data] Reusing streaming vocab at {vocab_file_path}")
                else:
                    print(f"[Data] Building streaming vocab with {vocab_workers} workers")
                    stoi, itos = build_vocab_from_stream(
                        iter_wikitext_lines(
                            args.dataset,
                            "train",
                            hf_cache,
                            line_limit,
                            dataset_obj=hf_dataset,
                        ),
                        num_workers=vocab_workers,
                    )
                    if args.reuse_vocab:
                        save_vocab_file(vocab_file_path, itos)
                        print(f"[Data] Saved streaming vocab to {vocab_file_path}")
                unk_id = stoi.get("<unk>", 0)
                line_encoder = make_basic_line_encoder(stoi, unk_id)
                decode_tokens_fn = make_basic_decoder(itos)
            streaming_data = WikitextStreamingData(
                args.dataset,
                hf_cache,
                args.seq_len,
                args.seq_stride,
                line_encoder,
                decode_tokens_fn,
                line_limit=line_limit,
                dataset_obj=hf_dataset,
            )
            print(
                f"[Data] Streaming {args.dataset} (seq_len={args.seq_len}, stride={args.seq_stride}) from {hf_cache}"
            )
            max_len = args.seq_len
        else:
            train_X, train_Y, val_X, val_Y, stoi, itos = load_wikitext_dataset(args, hf_cache)
            print(f"[Data] Loaded {args.dataset} into cache {hf_cache}")
            train_text_pairs, train_refs = tensors_to_text_pairs(train_X, train_Y, itos)
            val_text_pairs, val_refs = tensors_to_text_pairs(val_X, val_Y, itos)
            max_len = train_X.size(1)
    else:
        X, Y = make_char_data(seq_len=args.seq_len)
        n_val = int(0.2 * X.size(0))
        train_X, val_X = X[:-n_val], X[-n_val:]
        train_Y, val_Y = Y[:-n_val], Y[-n_val:]
        raw_vocab = int(X.max().item() + 1)
        token_list = [f"tok{i}" for i in range(raw_vocab)]
        vocab_tokens = SPECIAL_TOKENS + token_list
        stoi = {tok: idx for idx, tok in enumerate(vocab_tokens)}
        itos = {idx: tok for tok, idx in stoi.items()}
        train_text_pairs, train_refs = tensors_to_text_pairs(train_X, train_Y, itos)
        val_text_pairs, val_refs = tensors_to_text_pairs(val_X, val_Y, itos)
        max_len = train_X.size(1)

    if streaming_data is not None:
        train_text_pairs = []
        val_text_pairs = []
        train_refs = []
        val_refs = []

    vocab_size = len(stoi)
    device = torch.device(args.device)

    if streaming_data is not None:
        benchmark_batch = streaming_data.benchmark_batch(args.batch)
    elif train_X is not None and train_Y is not None:
        bench_size = min(args.batch, train_X.size(0))
        if bench_size > 0:
            batch_idx = torch.arange(bench_size, dtype=torch.long)
            src_ids = train_X[:bench_size].clone()
            tgt_ids = train_Y[:bench_size].clone()
            benchmark_batch = (batch_idx, src_ids, tgt_ids, [])
        else:
            benchmark_batch = None
    else:
        benchmark_batch = None

    if not args.sdpa_baseline:
        if streaming_data is None:
            sequences_train = [row.clone() for row in train_X]
            sequences_val = [row.clone() for row in val_X] if (val_X is not None and val_X.numel() > 0) else []
        else:
            sequences_train = streaming_data.sequences_for_bank("train")
            sequences_val = streaming_data.sequences_for_bank("validation")
        train_bank = build_windowed_bank_from_ids(sequences_train, window=args.window, stride=args.stride)
        val_bank = build_windowed_bank_from_ids(sequences_val, window=args.window, stride=args.stride)

        atom_emb = nn.Embedding(vocab_size, args.d_model).to(device)
        adapter = None
        if args.adapter_rank > 0:
            adapter = AtomFeatureAdapter(args.d_model, rank=args.adapter_rank).to(device)
            with torch.no_grad():
                phi_snapshot = adapter(atom_emb.weight).detach()
        else:
            phi_snapshot = atom_emb.weight

        universe_ids = torch.arange(vocab_size, dtype=torch.long)
        universe = UniversePool(universe_ids, metadata={"task": "toy_lm"})
        train_minhash = MinHasher(k=args.minhash_k, device=train_bank.values.device)
        train_cache = SetFeatureCache(
            universe, train_bank.values, train_bank.set_offsets, train_bank.seq_offsets, minhash=train_minhash
        )
        if val_bank.seq_offsets.numel() > 1:
            val_minhash = MinHasher(k=args.minhash_k, device=val_bank.values.device)
            val_cache = SetFeatureCache(
                universe, val_bank.values, val_bank.set_offsets, val_bank.seq_offsets, minhash=val_minhash
            )
        else:
            val_cache = None
        if args.precompute_bank:
            universe = universe.to(device)
            train_cache = train_cache.to(device)
            if val_cache is not None:
                val_cache = val_cache.to(device)
    else:
        train_cache = val_cache = None
        atom_emb = adapter = None
        phi_snapshot = None

    backbone = TinyLMBackbone(
        vocab_size,
        args.d_model,
        args.nhead,
        args.layers,
        attn_baseline="explicit" if (args.sdpa_baseline and args.attn_baseline == "explicit") else "pytorch",
    ).to(device)
    if not args.sdpa_baseline:
        ska_score_mode = "delta_plus_dot"
        ska_gamma = 0.3
        if args.attn == "dot":
            ska_score_mode = "dot"
            ska_gamma = 0.0
        elif args.attn == "rbf":
            ska_score_mode = "delta_rbf"
        elif args.attn == "intersect":
            ska_score_mode = "intersect_plus_dot"
        set_attn = SetBankAttention(
            d_model=args.d_model,
            num_heads=args.nhead,
            tau=1.0,
            gamma=ska_gamma,
            beta=1.0,
            score_mode=ska_score_mode,
            eta=1.0,
            backend=args.ska_backend,
            precision=args.precision,
        ).to(device)
        router = TokenSetRouter(d_model=args.d_model, num_heads=args.nhead, topk=args.router_topk).to(device)
    else:
        set_attn = None
        router = None
    head = nn.Linear(args.d_model, vocab_size).to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    if set_attn is not None:
        params += list(set_attn.parameters()) + list(router.parameters())
    if atom_emb is not None:
        params += list(atom_emb.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)

    if args.benchmark:
        sys_info = _system_info()
        if args.dry_run:
            dataset_id = args.dataset or "custom"
            bench_size = None
            seq_len = args.seq_len
            if benchmark_batch is not None and isinstance(benchmark_batch, (list, tuple)) and len(benchmark_batch) >= 3:
                _, src_ids, tgt_ids, _ = benchmark_batch[:4]
                bench_size = src_ids.size(0)
                seq_len = tgt_ids.size(1)
            _append_benchmark_row(
                args.benchmark_csv,
                {
                    "script": "train_toy_lm_banked",
                    "task": "lm",
                    "dataset": dataset_id,
                    "dataset_id": dataset_id,
                    "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
                    "attn_impl": _attn_impl_label(args, args.sdpa_baseline),
                    "precision": args.precision,
                    "attn": args.attn,
                    "batch": bench_size or args.batch,
                    "seq_len": seq_len,
                    "seq_stride": args.seq_stride,
                    "window": args.window,
                    "stride": args.stride,
                    "minhash_k": args.minhash_k,
                    "router_topk": args.router_topk,
                    "bench_warmup": args.bench_warmup,
                    "bench_iters": args.bench_iters,
                    "seed": seed,
                    "rep": rep,
                    "run_uid": run_uid,
                    "device": sys_info["device"],
                    "gpu_name": sys_info["gpu_name"],
                    "torch_version": sys_info["torch_version"],
                    "cuda_version": sys_info["cuda_version"],
                    "git_sha": sys_info["git_sha"],
                    "tokens_per_s": 0.0,
                    "elapsed_s": 0.0,
                    "tokens_total": 0.0,
                    "avg_sets_per_seq": 0.0,
                    "avg_atoms_per_set": 0.0,
                    "scores_total": 0.0,
                    "scores_per_s": 0.0,
                    "scores_per_1e6": 0.0,
                    "scores_per_token": 0.0,
                    "min_sets_per_seq": 0.0,
                    "max_sets_per_seq": 0.0,
                    "max_vram_mb": 0.0,
                    "status": "dry_run",
                    "skip_reason": "dry_run",
                    "gpu_vram_gb": args.gpu_vram,
                },
            )
            if wandb_run.enabled:
                wandb_run.finish()
            return
        if args.sdpa_baseline:
            if benchmark_batch is None:
                print("[benchmark] no data available.")
                return
            _, src_ids, tgt_ids, _ = benchmark_batch
            bench_batch = src_ids.size(0)
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            def step():
                optimizer.zero_grad(set_to_none=True)
                token_states = backbone(src_ids)
                logits = head(token_states)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
                loss.backward()
                optimizer.step()

            for _ in range(args.bench_warmup):
                step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.bench_iters):
                step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            max_vram_mb = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0.0
            )
            tokens = bench_batch * tgt_ids.size(1) * args.bench_iters
            throughput = tokens / elapsed if elapsed > 0 else 0.0
            scores_total = float(bench_batch * tgt_ids.size(1) * tgt_ids.size(1) * args.nhead)
            scores_per_s = scores_total / elapsed if elapsed > 0 else 0.0
            throughput_per_m = scores_per_s / 1e6
            attn_impl = "dot_explicit" if args.attn_baseline == "explicit" else "dot_builtin"
            print(
                f"[benchmark][lm] backend=sdpa ({attn_impl}) precision={args.precision} "
                f"tokens/s={throughput:.1f} elapsed={elapsed:.3f}s"
            )
            if wandb_run.enabled:
                wandb_run.log(
                    {
                        "benchmark/tokens_per_s": throughput,
                        "benchmark/elapsed_s": elapsed,
                        "benchmark/batch_tokens": tokens,
                        "benchmark/seed": seed,
                        "benchmark/rep": rep,
                    }
                )
            dataset_id = args.dataset or "custom"
            _append_benchmark_row(
                args.benchmark_csv,
                {
                    "script": "train_toy_lm_banked",
                    "task": "lm",
                    "dataset": dataset_id,
                    "dataset_id": dataset_id,
                    "mode": "sdpa",
                    "attn_impl": attn_impl,
                    "precision": args.precision,
                    "attn": args.attn,
                    "batch": bench_batch,
                    "seq_len": tgt_ids.size(1),
                    "seq_stride": args.seq_stride,
                    "window": args.window,
                    "stride": args.stride,
                    "minhash_k": args.minhash_k,
                    "router_topk": args.router_topk,
                    "adapter_rank": args.adapter_rank,
                    "bench_warmup": args.bench_warmup,
                    "bench_iters": args.bench_iters,
                    "seed": seed,
                    "rep": rep,
                    "run_uid": run_uid,
                    "device": sys_info["device"],
                    "gpu_name": sys_info["gpu_name"],
                    "torch_version": sys_info["torch_version"],
                    "cuda_version": sys_info["cuda_version"],
                    "git_sha": sys_info["git_sha"],
                    "tokens_per_s": throughput,
                    "elapsed_s": elapsed,
                    "tokens_total": tokens,
                    "avg_sets_per_seq": 0.0,
                    "avg_atoms_per_set": 0.0,
                    "scores_total": scores_total,
                    "scores_per_s": scores_per_s,
                    "scores_per_1e6": throughput_per_m,
                    "max_vram_mb": max_vram_mb,
                    "status": "ok",
                    "skip_reason": "",
                    "gpu_vram_gb": args.gpu_vram,
                },
            )
        else:
            run_lm_benchmark(
                args,
                backbone,
                set_attn,
                router,
                head,
                adapter,
                atom_emb,
                phi_snapshot,
                train_cache,
                optimizer,
                benchmark_batch,
                max_len,
                device,
                wandb_run,
                args.benchmark_csv,
                seed,
                rep,
                run_uid,
            )
        wandb_run.finish()
        return

    def run_epoch(train: bool):
        if not train:
            torch.manual_seed(args.eval_seed)
            random.seed(args.eval_seed)
            g = torch.Generator(device=device).manual_seed(args.eval_seed)
        model_list = [m for m in (backbone, router, head, adapter) if m is not None]
        for m in model_list:
            m.train(train)
        refs_epoch, hyps_epoch = [], []
        total_loss = 0.0
        total_tokens = 0
        if streaming_data is not None and args.dataset:
            split = "train" if train else "validation"
            iterator = streaming_data.batch_iterator(split, args.batch)
        else:
            iterator = text_batch_iterator(
                train_text_pairs if train else val_text_pairs,
                stoi,
                stoi,
                train_refs if train else val_refs,
                max_len,
                args.batch,
                shuffle=train,
                generator=(torch.Generator(device=device).manual_seed(args.eval_seed) if not train else None),
                worker_init_fn=(_make_worker_init_fn(args.eval_seed) if not train else None),
            )
        active_cache = train_cache if train else val_cache
        if not args.sdpa_baseline and active_cache is None:
            return 0.0, refs_epoch, hyps_epoch
        for batch_idx, src_ids, tgt_ids, ref_tokens in iterator:
            batch_idx = batch_idx.to(device)
            src_ids = src_ids.to(device, non_blocking=True)
            tgt_ids = tgt_ids.to(device, non_blocking=True)
            tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

            if not args.sdpa_baseline:
                phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                Phi, Sig, Size, q_ptrs = active_cache.gather_flat(batch_idx, phi_dynamic)
                token_states = backbone(src_ids)
                Z, q_ptrs = set_attn(Phi, Sig, Size, q_ptrs, Phi, Sig, Size, q_ptrs)
                routed = router(token_states, Z, Phi, q_ptrs)
            else:
                token_states = backbone(src_ids)
                routed = token_states
            logits = head(routed)

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach()) * tgt_ids.numel()
            total_tokens += tgt_ids.numel()
            preds = logits.argmax(dim=-1)
            refs_epoch.extend(ref_tokens)
            hyps_epoch.extend([ids_to_tokens(row, itos) for row in preds.detach().cpu()])

        return total_loss / max(1, total_tokens), refs_epoch, hyps_epoch

    for epoch in range(1, args.epochs + 1):
        with profiler(args.profile) as prof:
            train_loss, train_refs_epoch, train_hyps_epoch = run_epoch(train=True)
            val_loss, val_refs_epoch, val_hyps_epoch = run_epoch(train=False)

        train_bleu = corpus_bleu(train_refs_epoch, train_hyps_epoch) if train_refs_epoch else 0.0
        val_bleu = corpus_bleu(val_refs_epoch, val_hyps_epoch) if val_refs_epoch else 0.0
        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(val_loss, 20.0))
        attn_impl = "dot_explicit" if (args.sdpa_baseline and args.attn_baseline == "explicit") else "dot_builtin"
        mode_tag = "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}/{args.precision}"
        if args.sdpa_baseline:
            mode_tag = f"{mode_tag}/{attn_impl}"
        msg = (
            f"[LM-Banked][{mode_tag}][{args.attn}] epoch {epoch:02d} "
            f"train loss {train_loss:.4f} ppl {train_ppl:.2f} BLEU {train_bleu:.3f} | "
            f"val loss {val_loss:.4f} ppl {val_ppl:.2f} BLEU {val_bleu:.3f}"
        )
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
        print(msg)
        if wandb_run.enabled:
            payload = {
                "train/loss": train_loss,
                "train/ppl": train_ppl,
                "train/bleu": train_bleu,
                "val/loss": val_loss,
                "val/ppl": val_ppl,
                "val/bleu": val_bleu,
                "impl/attn_impl": attn_impl,
            }
            if args.profile:
                payload["train/time_s"] = prof["time_s"]
                if torch.cuda.is_available():
                    payload["train/peak_vram_mib"] = prof["gpu_peak_mem_mib"]
            sample_text = format_text_samples(
                val_refs_epoch,
                val_hyps_epoch,
                args.sample_count,
                args.sample_seed + epoch,
            )
            if sample_text:
                payload["samples/val_text"] = sample_text
            wandb_run.log(payload, step=epoch)
        if args.metrics_csv:
            _append_benchmark_row(
                args.metrics_csv,
                {
                    "script": "train_toy_lm_banked",
                    "task": "lm",
                    "dataset": args.dataset or "custom",
                    "dataset_id": args.dataset or "custom",
                    "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
                    "attn_impl": attn_impl,
                    "precision": args.precision,
                    "attn": args.attn,
                    "batch": args.batch,
                    "seq_len": args.seq_len,
                    "seq_stride": args.seq_stride,
                    "window": args.window,
                    "stride": args.stride,
                    "minhash_k": args.minhash_k,
                    "router_topk": args.router_topk,
                    "adapter_rank": args.adapter_rank,
                    "seed": seed,
                    "rep": rep,
                    "run_uid": run_uid,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_ppl": train_ppl,
                    "train_bleu": train_bleu,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "val_bleu": val_bleu,
                },
            )

    wandb_run.finish()
if __name__ == "__main__":
    main()
def _make_worker_init_fn(base_seed: int):
    def _init(worker_id: int):
        seed = int(base_seed) + int(worker_id)
        import random

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32 - 1))
        except Exception:
            pass
        import torch

        torch.manual_seed(seed)

    return _init
