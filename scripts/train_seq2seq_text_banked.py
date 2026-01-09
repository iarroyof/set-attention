import argparse
import copy
import csv
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.utils.bench_skip import should_skip_dense, should_skip_ska
from set_attention.utils.repro_workers import make_worker_init_fn
from set_attention.tokenizers.active_tokenizer import ACTIVE_TOKENIZER_TYPE
from set_attention.tokenizers.hf_bpe import HF_BPE_TYPE
from set_attention.tokenizers.hf_unigram import HF_UNIGRAM_TYPE
from set_attention.tokenizers.registry import (
    available_tokenizer_types,
    create_tokenizer,
    load_tokenizer,
    save_tokenizer,
)
from set_attention.tokenizers.cache import (
    default_tokenizer_dir,
    meta_matches,
    save_tokenizer_meta,
    tokenizer_fingerprint,
)
from set_attention.training.seq_loaders import get_seq2seq_datasets
from set_attention.data.hf_cache import ensure_hf_cache
from set_attention.data.artifact_cache import (
    ArtifactSpec,
    artifact_root,
    assert_meta_compatible,
    file_signature,
    fingerprint,
    resolve_hf_root,
    write_meta,
)
from set_attention.data.ska_artifacts import (
    BankPack,
    RoutingPack,
    cache_from_packs,
    load_bank_pack,
    load_routing_pack,
    save_bank_pack,
    save_routing_pack,
)
from set_attention.sets.bank_builders import build_windowed_bank_from_texts
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.training.text_utils import (
    encode_sentence,
    ids_to_tokens,
    text_batch_iterator,
    SPECIAL_TOKENS,
)
from set_attention.utils.sample_logging import format_text_samples
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher
from set_attention.utils.profiling import profiler
from set_attention.utils.wandb import init_wandb
from set_attention.experiments.nlp_eval import corpus_bleu, rouge_l
from common.repro import set_seed


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


def _seq_data_signature(args) -> Optional[dict]:
    if args.dataset:
        val_limit = getattr(args, "val_limit", None)
        if val_limit is None:
            val_limit = args.limit
        return {
            "dataset": args.dataset,
            "limit": args.limit,
            "val_limit": val_limit,
        }
    if args.src and args.tgt:
        return {
            "src": file_signature(Path(args.src)),
            "tgt": file_signature(Path(args.tgt)),
        }
    return None


def _seq_token_spec(args, dataset_id: str, data_sig: Optional[dict], max_len: int) -> dict:
    spec = ArtifactSpec(
        task="seq2seq",
        dataset_id=dataset_id,
        split="train+val",
        subset=data_sig,
        tokenizer={
            "type": "whitespace",
            "special_tokens": list(SPECIAL_TOKENS),
            "max_len": int(max_len),
        },
        sequence={"max_len": int(max_len)},
        model={"d_model": int(args.atom_dim), "heads": int(args.heads), "layers": int(args.layers)},
        routing_depends_on_learned_params=bool(args.adapter_rank > 0),
    )
    return spec.to_dict()


def _seq_bank_spec(args, dataset_id: str, data_sig: Optional[dict], max_len: int, tokens_fp: str) -> dict:
    spec = ArtifactSpec(
        task="seq2seq",
        dataset_id=dataset_id,
        split="train+val",
        subset=data_sig,
        tokenizer={
            "type": args.tokenizer_type,
            "tokenizer_config": _default_tokenizer_config(args.tokenizer_type, max_len),
            "tokens_fp": tokens_fp,
        },
        sequence={"max_len": int(max_len)},
        ska={
            "window": int(args.window),
            "stride": int(args.stride),
            "minhash_k": int(args.minhash_k),
            "router_topk": int(args.router_topk),
            "backend": args.ska_backend,
            "precision": args.precision,
            "attn": args.set_kernel,
        },
        model={"d_model": int(args.atom_dim), "heads": int(args.heads), "layers": int(args.layers)},
        routing_depends_on_learned_params=bool(args.adapter_rank > 0),
    )
    return spec.to_dict()


def _artifact_root_with_override(spec: dict, override_fp: str, hf_root: Path) -> Path:
    if override_fp:
        return hf_root / "artifacts" / spec["task"] / spec["dataset_id"] / override_fp
    return artifact_root(spec, hf_root)


def _encode_pairs_to_ids(pairs: List[Tuple[str, str]], src_stoi: dict, tgt_stoi: dict, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    src_ids = torch.stack([encode_sentence(s, src_stoi, max_len) for (s, _) in pairs], dim=0)
    tgt_ids = torch.stack([encode_sentence(t, tgt_stoi, max_len) for (_, t) in pairs], dim=0)
    return src_ids, tgt_ids


def _save_seq_tokens(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_seq_tokens(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


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


def _configure_dot_naive(dot_naive: bool) -> None:
    if not dot_naive:
        return
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    print("[SDP] dot-naive enabled: flash/mem-efficient SDP disabled; using math backend.")



def _parse_tokenizer_config_arg(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads(value)


def _default_tokenizer_config(kind: str, max_len: int) -> Dict[str, Any]:
    if kind == ACTIVE_TOKENIZER_TYPE:
        return {"seed_lengths": (3, 4, 5), "min_freq": 2, "max_len": max_len}
    if kind in {HF_UNIGRAM_TYPE, HF_BPE_TYPE}:
        return {"vocab_size": 16000, "min_frequency": 2}
    return {}


def _normalize_tokenizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(config)
    if "seed_lengths" in out:
        out["seed_lengths"] = tuple(int(v) for v in out["seed_lengths"])
    if "special_tokens" in out:
        out["special_tokens"] = tuple(str(v) for v in out["special_tokens"])
    for key in ("min_freq", "min_frequency", "vocab_size", "max_len"):
        if key in out:
            out[key] = int(out[key])
    return out


def _explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
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
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        ctx = _explicit_attention(q, k, v, self.scale)
        ctx = self.dropout(ctx)
        return self.out_proj(ctx.transpose(1, 2).reshape(B, L, self.d_model))


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
            f"Explicit attention sanity check failed (max diff {max_diff:.2e} > {tol})."
        )


class TinySeq2SeqBackbone(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, nhead: int, layers: int, attn_baseline: str = "pytorch"):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.use_explicit = attn_baseline == "explicit"
        if self.use_explicit:
            self.enc_layers = nn.ModuleList(
                [ExplicitTransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.0) for _ in range(layers)]
            )
            self.dec_layers = nn.ModuleList([ExplicitTransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.0)])
        else:
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.dec_self = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
                num_layers=1,
            )

    def forward(self, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_emb = self.src_emb(src_ids)
        tgt_emb = self.tgt_emb(tgt_in_ids)
        if self.use_explicit:
            mem = src_emb
            for layer in self.enc_layers:
                mem = layer(mem.float())
            dec = tgt_emb
            for layer in self.dec_layers:
                dec = layer(dec.float())
        else:
            mem = self.enc(src_emb)
            dec = self.dec_self(tgt_emb)
        return dec, mem



def run_seq2seq_benchmark(
    args,
    model,
    set_attn,
    router,
    out_proj,
    adapter,
    atom_emb,
    phi_snapshot,
    cache_src,
    cache_tgt,
    train_pairs,
    train_src_stoi,
    train_tgt_stoi,
    train_ref_tokens,
    max_len,
    opt,
    device,
    wandb_run,
    benchmark_csv,
    seed,
    rep,
    run_uid,
    train_src_ids=None,
    train_tgt_ids=None,
):
    attn_impl = _attn_impl_label(args, args.sdpa_baseline)
    iterator = text_batch_iterator(
        train_pairs,
        train_src_stoi,
        train_tgt_stoi,
        train_ref_tokens,
        max_len,
        args.batch,
        shuffle=False,
        generator=torch.Generator(device=device).manual_seed(args.eval_seed),
        worker_init_fn=make_worker_init_fn(args.eval_seed),
        src_ids=train_src_ids,
        tgt_ids=train_tgt_ids,
    )
    try:
        batch_idx_tensor, src_ids, tgt_ids, _ = next(iterator)
    except StopIteration:
        print("[benchmark] insufficient data for benchmark batch.")
        return
    batch_idx_tensor = batch_idx_tensor.to(device)
    src_ids = src_ids.to(device, non_blocking=True)
    tgt_ids = tgt_ids.to(device, non_blocking=True)
    tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

    stats_ptrs = None
    stats_sizes = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Pre-skip estimate
    if args.gpu_vram > 0:
        decision = should_skip_dense(
            tgt_ids.size(0), max_len, args.atom_dim, args.heads, args.precision, args.gpu_vram
        ) if args.sdpa_baseline else should_skip_ska(
            tgt_ids.size(0), max_len, args.window, args.stride, args.heads, args.precision, args.gpu_vram
        )
        if decision.skip or args.dry_run:
            _append_benchmark_row(
                benchmark_csv,
                {
                    "script": "train_seq2seq_text_banked",
                    "task": "seq2seq",
                    "dataset": args.dataset or "custom",
                    "dataset_id": args.dataset or "custom",
                    "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
                    "attn_impl": attn_impl,
                    "precision": args.precision,
                    "status": "skipped",
                    "skip_reason": decision.reason or ("dry_run" if args.dry_run else ""),
                    "gpu_vram_gb": args.gpu_vram,
                },
            )
            if args.dry_run or decision.skip:
                return

    attn_impl = _attn_impl_label(args, args.sdpa_baseline)
    if args.sdpa_baseline:
        def step():
            opt.zero_grad(set_to_none=True)
            dec_h, _ = model(src_ids, tgt_in)
            logits = out_proj(dec_h)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
            loss.backward()
            opt.step()
        backend_label = f"sdpa/{attn_impl}"
    else:
        def step():
            nonlocal stats_ptrs, stats_sizes
            opt.zero_grad(set_to_none=True)
            phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
            Phi_q, Sig_q, Size_q, q_ptrs = cache_tgt.gather_flat(batch_idx_tensor, phi_dynamic)
            Phi_k, Sig_k, Size_k, k_ptrs = cache_src.gather_flat(batch_idx_tensor, phi_dynamic)
            dec_h, _ = model(src_ids, tgt_in)
            Z_sets, q_ptrs = set_attn(Phi_q, Sig_q, Size_q, q_ptrs, Phi_k, Sig_k, Size_k, k_ptrs)
            tok_out = router(dec_h, Z_sets, Phi_q, q_ptrs)
            logits = out_proj(tok_out)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
            loss.backward()
            opt.step()
            stats_ptrs = q_ptrs.detach().cpu()
            stats_sizes = Size_q.detach().cpu()
        backend_label = attn_impl

    for _ in range(args.bench_warmup):
        step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        for _ in range(args.bench_iters):
            step()
    except torch.cuda.OutOfMemoryError:
        if args.skip_oom:
            torch.cuda.empty_cache()
            _append_benchmark_row(
                benchmark_csv,
                {
                    "script": "train_seq2seq_text_banked",
                    "task": "seq2seq",
                    "dataset": args.dataset or "custom",
                    "dataset_id": args.dataset or "custom",
                    "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
                    "attn_impl": attn_impl,
                    "precision": args.precision,
                    "status": "oom",
                    "skip_reason": "runtime_oom",
                    "gpu_vram_gb": args.gpu_vram,
                },
            )
            return
        raise
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    max_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    tokens = tgt_ids.size(0) * max_len * args.bench_iters
    throughput = tokens / elapsed if elapsed > 0 else 0.0
    info = _system_info()
    if args.sdpa_baseline:
        seq_len = tgt_ids.size(1)
        scores_total = float(seq_len * seq_len * tgt_ids.size(0) * args.heads * args.bench_iters)
        avg_sets = avg_atoms = min_sets = max_sets = 0.0
    else:
        if stats_ptrs is not None and stats_sizes is not None:
            avg_sets, avg_atoms, scores_per_batch, min_sets, max_sets = _summarize_ska_batch(
                stats_ptrs, stats_sizes, args.heads
            )
            scores_total = scores_per_batch * args.bench_iters
        else:
            avg_sets = avg_atoms = scores_total = min_sets = max_sets = 0.0
    scores_per_s = scores_total / elapsed if elapsed > 0 else 0.0
    scores_per_1e6 = scores_per_s / 1e6
    scores_per_token = scores_total / tokens if tokens > 0 else 0.0
    print(
        f"[benchmark][seq2seq] backend={backend_label} "
        f"tokens/s={throughput:.1f} elapsed={elapsed:.3f}s"
    )
    if wandb_run.enabled:
        wandb_run.log(
            {
                "benchmark/tokens_per_s": throughput,
                "benchmark/elapsed_s": elapsed,
                "benchmark/batch_tokens": tokens,
                "benchmark/avg_sets_per_seq": avg_sets,
                "benchmark/avg_atoms_per_set": avg_atoms,
                "benchmark/scores_total": scores_total,
                "benchmark/scores_per_s": scores_per_s,
                "benchmark/scores_per_1e6": scores_per_1e6,
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
            "script": "train_seq2seq_text_banked",
            "task": "seq2seq",
            "dataset": args.dataset or "custom",
            "dataset_id": args.dataset or "custom",
            "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
            "attn_impl": attn_impl,
            "precision": args.precision,
            "set_kernel": args.set_kernel,
            "batch": args.batch,
            "max_len": max_len,
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
            "tokens_total": tokens,
            "avg_sets_per_seq": avg_sets,
            "avg_atoms_per_set": avg_atoms,
            "scores_total": scores_total,
            "scores_per_s": scores_per_s,
            "scores_per_1e6": scores_per_1e6,
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
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=False)
    parser.add_argument("--tgt", required=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-samples", type=int, default=200)
    parser.add_argument("--dataset", choices=["", "wmt16_en_ro", "cnn_dailymail"], default="")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on dataset size; omit to use the full split.",
    )
    tokenizer_choices = tuple(available_tokenizer_types())
    parser.add_argument(
        "--tokenizer-dir",
        "--tokenizer",
        dest="tokenizer_dir",
        default="",
        help="Optional directory to load/save tokenizer state.",
    )
    parser.add_argument(
        "--tokenizer-type",
        choices=tokenizer_choices,
        default=ACTIVE_TOKENIZER_TYPE,
        help="Tokenizer backend to use when training a new tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        default="",
        help="JSON string or path with tokenizer config overrides.",
    )
    parser.add_argument("--atom-dim", type=int, default=128)
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--minhash-k", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--router-topk", type=int, default=0)
    parser.add_argument("--set-kernel", choices=["delta_rbf", "delta_plus_dot", "intersect_norm", "intersect_plus_dot", "dot"], default="delta_plus_dot")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--ska-backend", choices=["python", "triton", "keops"], default="python")
    parser.add_argument(
        "--dot-naive",
        action="store_true",
        help="Force dot-product attention in baseline to use naive (math) mode.",
    )
    parser.add_argument(
        "--attn-baseline",
        choices=["pytorch", "explicit"],
        default="pytorch",
        help="Baseline attention implementation when using --sdpa-baseline.",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--sdpa-baseline", action="store_true", help="Skip set attention/router and run plain Transformer decoder.")
    parser.add_argument(
        "--precompute-bank",
        action="store_true",
        help="Move banks/universe to the training device (default: keep on CPU).",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument(
        "--benchmark-csv",
        type=str,
        default="",
        help="Optional CSV path to log benchmark runs.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="",
        help="Optional CSV path to log training/validation metrics per epoch.",
    )
    parser.add_argument("--gpu-vram", type=float, default=0.0, help="Approx GPU VRAM in GB for skip estimation (0=disable).")
    parser.add_argument("--skip-oom", action="store_true", help="Skip configs that OOM or exceed estimated VRAM.")
    parser.add_argument("--dry-run", action="store_true", help="Print skip/ok decision and write CSV status without running.")
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="",
        help="Shared HF datasets cache root; empty uses HF_HOME/HF_DATASETS_CACHE.",
    )
    parser.add_argument("--cache-mode", choices=["none", "tokens", "full"], default="none")
    parser.add_argument("--artifact-cache-root", type=str, default="")
    parser.add_argument("--artifact-fingerprint", type=str, default="")
    parser.add_argument("--overwrite-cache", action="store_true", help="Overwrite existing token/bank caches.")
    parser.add_argument("--cache-only", action="store_true", help="Build caches and exit without training/benchmarking.")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of validation samples to log.")
    parser.add_argument("--sample-seed", type=int, default=1337, help="Seed for selecting logged samples.")
    parser.add_argument("--eval-seed", type=int, default=1337, help="Seed to make validation evaluation deterministic across variants.")
    parser.add_argument("--seed", type=int, default=2024, help="Base RNG seed used when --seeds is not provided.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma/space separated list of seeds (overrides --seed).",
    )
    parser.add_argument("--reps", type=int, default=1, help="Number of repetitions per seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic torch algorithms (may reduce throughput).",
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Enable benchmark mode in seed helper (disables deterministic algs but seeds RNG).",
    )
    args = parser.parse_args()
    _configure_dot_naive(args.dot_naive)
    if args.sdpa_baseline and args.attn_baseline == "explicit":
        _sanity_check_explicit_attention(torch.device(args.device), args.atom_dim, args.heads)
    if args.benchmark and args.limit is None:
        args.limit = 50000
        print("[benchmark] limiting dataset to 50k pairs for memory safety.")

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
    multi_run = len(seed_values) * reps > 1
    for seed in seed_values:
        for rep in range(1, reps + 1):
            run_args = copy.deepcopy(args)
            run_uid = f"{int(time.time() * 1e6)}-{os.getpid()}-{seed}-{rep}"
            run_single(run_args, seed, rep, run_uid, multi_run)
    return


def run_single(args, seed: int, rep: int, run_uid: str, multi_run: bool):
    torch.backends.cudnn.benchmark = True
    set_seed(seed, deterministic=args.deterministic, benchmark_mode=args.benchmark_mode)
    print(f"[Run] seed={seed} rep={rep} uid={run_uid}")
    cache_dir = ensure_hf_cache(args.hf_cache_dir)
    hf_root = resolve_hf_root(args.artifact_cache_root or None)
    cache_mode = args.cache_mode
    if cache_mode == "full" and args.adapter_rank > 0:
        raise ValueError("--cache-mode full is only supported when --adapter-rank 0.")

    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_config = {
        "script": "train_seq2seq_text_banked",
        "dataset": args.dataset or "custom",
        "tokenizer_type": args.tokenizer_type,
        "ska_backend": args.ska_backend,
        "precision": args.precision,
        "window": args.window,
        "stride": args.stride,
        "minhash_k": args.minhash_k,
        "router_topk": args.router_topk,
        "adapter_rank": args.adapter_rank,
        "set_kernel": args.set_kernel,
        "sdpa_baseline": args.sdpa_baseline,
        "precompute_bank": args.precompute_bank,
        "batch": args.batch,
        "sample_count": args.sample_count,
        "seed": seed,
        "rep": rep,
        "run_uid": run_uid,
        "attn_impl": _attn_impl_label(args, args.sdpa_baseline),
        "hf_cache_dir": args.hf_cache_dir,
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

    train_ds, val_ds = get_seq2seq_datasets(
        dataset=args.dataset,
        limit=args.limit,
        val_limit=args.limit,
        src_path=args.src,
        tgt_path=args.tgt,
        demo=args.demo,
        demo_samples=args.demo_samples,
        max_len=64,
        cache_dir=str(cache_dir),
    )

    train_pairs = train_ds.pairs
    src_texts = [s for (s, _) in train_pairs]
    tgt_texts = [t for (_, t) in train_pairs]
    train_src_stoi = train_ds.src_stoi
    train_tgt_stoi = train_ds.tgt_stoi
    train_src_itos = train_ds.src_itos
    train_tgt_itos = train_ds.tgt_itos
    max_len = train_ds.max_len
    train_ref_tokens = [t.split() for _, t in train_pairs]

    tokenizer_dir = args.tokenizer_dir
    config_overrides = _parse_tokenizer_config_arg(args.tokenizer_config)
    tokenizer_config = _default_tokenizer_config(args.tokenizer_type, max_len)
    if config_overrides:
        tokenizer_config.update(config_overrides)
    tokenizer_config = _normalize_tokenizer_config(tokenizer_config)

    dataset_id = args.dataset or "local"
    fingerprint = tokenizer_fingerprint(
        args.tokenizer_type,
        tokenizer_config,
        dataset_id,
        max_len,
        limit=args.limit,
        src_path=args.src,
        tgt_path=args.tgt,
    )
    if not tokenizer_dir:
        hf_root = Path(os.environ.get("HF_HOME") or str(cache_dir.parent))
        tokenizer_dir = str(default_tokenizer_dir(hf_root, dataset_id, args.tokenizer_type, fingerprint))

    expected_meta = {
        "fingerprint": fingerprint,
        "dataset": dataset_id,
        "tokenizer_type": args.tokenizer_type,
        "tokenizer_config": tokenizer_config,
        "limit": args.limit,
        "max_len": max_len,
        "src_path": args.src,
        "tgt_path": args.tgt,
    }

    if tokenizer_dir and os.path.isdir(tokenizer_dir) and meta_matches(Path(tokenizer_dir), expected_meta):
        print(f"[Tokenizer] Loading {args.tokenizer_type} tokenizer from {tokenizer_dir}")
        load_cfg = config_overrides if config_overrides else None
        tokenizer = load_tokenizer(tokenizer_dir, kind=args.tokenizer_type, config=load_cfg)
    else:
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            print("[Tokenizer] Cache mismatch; retraining to avoid inconsistent token IDs.")
        print(f"[Tokenizer] Training new {args.tokenizer_type} tokenizer on source corpus")
        tokenizer = create_tokenizer(args.tokenizer_type, tokenizer_config)
        tokenizer.fit(src_texts)
        if tokenizer_dir:
            os.makedirs(tokenizer_dir, exist_ok=True)
            save_tokenizer(tokenizer, tokenizer_dir)
            save_tokenizer_meta(Path(tokenizer_dir), expected_meta)
            print(f"[Tokenizer] Saved to {tokenizer_dir}")

    src_bank = tgt_bank = None
    val_src_bank = val_tgt_bank = None

    has_val = val_ds is not None
    if has_val:
        val_pairs = val_ds.pairs
        val_src_texts = [s for (s, _) in val_pairs]
        val_tgt_texts = [t for (_, t) in val_pairs]
        val_ref_tokens = [t.split() for _, t in val_pairs]
    else:
        val_pairs = []
        val_ref_tokens = []

    train_src_ids = train_tgt_ids = None
    val_src_ids = val_tgt_ids = None
    token_spec = None
    if cache_mode in ("tokens", "full"):
        data_sig = _seq_data_signature(args)
        token_spec = _seq_token_spec(args, dataset_id, data_sig, max_len)
        token_root = _artifact_root_with_override(token_spec, args.artifact_fingerprint, hf_root)
        tokens_path = token_root / "tokens.pt"
        if tokens_path.exists() and not args.overwrite_cache:
            assert_meta_compatible(token_root, token_spec)
            payload = _load_seq_tokens(tokens_path)
            train_pairs = payload.get("train_pairs", train_pairs)
            val_pairs = payload.get("val_pairs", val_pairs)
            train_ref_tokens = payload.get("train_ref_tokens", [t.split() for _, t in train_pairs])
            val_ref_tokens = payload.get("val_ref_tokens", [t.split() for _, t in val_pairs])
            train_src_ids = payload.get("train_src_ids")
            train_tgt_ids = payload.get("train_tgt_ids")
            val_src_ids = payload.get("val_src_ids")
            val_tgt_ids = payload.get("val_tgt_ids")
            src_itos_list = payload.get("src_itos") or []
            tgt_itos_list = payload.get("tgt_itos") or []
            if src_itos_list:
                train_src_stoi = {tok: idx for idx, tok in enumerate(src_itos_list)}
                train_src_itos = {idx: tok for idx, tok in enumerate(src_itos_list)}
            if tgt_itos_list:
                train_tgt_stoi = {tok: idx for idx, tok in enumerate(tgt_itos_list)}
                train_tgt_itos = {idx: tok for idx, tok in enumerate(tgt_itos_list)}
            max_len = int(payload.get("max_len", max_len))
            src_texts = [s for (s, _) in train_pairs]
            tgt_texts = [t for (_, t) in train_pairs]
            val_src_texts = [s for (s, _) in val_pairs]
            val_tgt_texts = [t for (_, t) in val_pairs]
            print(f"[Cache] Loaded seq2seq tokens from {tokens_path}")
        else:
            train_src_ids, train_tgt_ids = _encode_pairs_to_ids(train_pairs, train_src_stoi, train_tgt_stoi, max_len)
            if val_pairs:
                val_src_ids, val_tgt_ids = _encode_pairs_to_ids(val_pairs, train_src_stoi, train_tgt_stoi, max_len)
            payload = {
                "train_pairs": train_pairs,
                "val_pairs": val_pairs,
                "train_ref_tokens": train_ref_tokens,
                "val_ref_tokens": val_ref_tokens,
                "train_src_ids": train_src_ids,
                "train_tgt_ids": train_tgt_ids,
                "val_src_ids": val_src_ids,
                "val_tgt_ids": val_tgt_ids,
                "src_itos": [train_src_itos[i] for i in range(len(train_src_itos))],
                "tgt_itos": [train_tgt_itos[i] for i in range(len(train_tgt_itos))],
                "max_len": max_len,
            }
            _save_seq_tokens(tokens_path, payload)
            write_meta(token_root, token_spec)
            print(f"[Cache] Saved seq2seq tokens to {tokens_path}")

    if args.cache_only and cache_mode in ("tokens", "full") and (args.sdpa_baseline or cache_mode == "tokens"):
        print("[Cache] cache-only complete (tokens).")
        wandb_run.finish()
        return

    loaded_bank_cache = False
    universe = None
    cache_src = cache_tgt = None
    val_cache_src = val_cache_tgt = None
    if not args.sdpa_baseline:
        if cache_mode == "full" and token_spec is not None:
            tokens_fp = fingerprint(token_spec)
            data_sig = _seq_data_signature(args)
            bank_spec = _seq_bank_spec(args, dataset_id, data_sig, max_len, tokens_fp)
            bank_root = _artifact_root_with_override(bank_spec, args.artifact_fingerprint, hf_root)
            paths = {
                "src_train": bank_root / "bank_src_train.pt",
                "tgt_train": bank_root / "bank_tgt_train.pt",
                "src_val": bank_root / "bank_src_val.pt",
                "tgt_val": bank_root / "bank_tgt_val.pt",
                "r_src_train": bank_root / "routing_src_train.pt",
                "r_tgt_train": bank_root / "routing_tgt_train.pt",
                "r_src_val": bank_root / "routing_src_val.pt",
                "r_tgt_val": bank_root / "routing_tgt_val.pt",
            }
            if all(p.exists() for p in paths.values()) and not args.overwrite_cache:
                assert_meta_compatible(bank_root, bank_spec)
                src_pack = load_bank_pack(paths["src_train"])
                tgt_pack = load_bank_pack(paths["tgt_train"])
                val_src_pack = load_bank_pack(paths["src_val"])
                val_tgt_pack = load_bank_pack(paths["tgt_val"])
                r_src_train = load_routing_pack(paths["r_src_train"])
                r_tgt_train = load_routing_pack(paths["r_tgt_train"])
                r_src_val = load_routing_pack(paths["r_src_val"])
                r_tgt_val = load_routing_pack(paths["r_tgt_val"])
                if not torch.equal(src_pack.universe_ids, tgt_pack.universe_ids):
                    raise RuntimeError("Cached seq2seq bank packs have mismatched universe ids.")
                universe = UniversePool(
                    src_pack.universe_ids,
                    metadata={
                        "tokenizer_type": args.tokenizer_type,
                        "tokenizer_path": tokenizer_dir or "inline",
                    },
                )
                cache_src = cache_from_packs(universe, src_pack, r_src_train)
                cache_tgt = cache_from_packs(universe, tgt_pack, r_tgt_train)
                if has_val:
                    val_cache_src = cache_from_packs(universe, val_src_pack, r_src_val)
                    val_cache_tgt = cache_from_packs(universe, val_tgt_pack, r_tgt_val)
                loaded_bank_cache = True
                print(f"[Cache] Loaded seq2seq bank+routing from {bank_root}")

        if not loaded_bank_cache:
            src_bank = build_windowed_bank_from_texts(tokenizer, src_texts, window=args.window, stride=args.stride)
            tgt_bank = build_windowed_bank_from_texts(tokenizer, tgt_texts, window=args.window, stride=args.stride)
            if has_val:
                val_src_bank = build_windowed_bank_from_texts(tokenizer, val_src_texts, window=args.window, stride=args.stride)
                val_tgt_bank = build_windowed_bank_from_texts(tokenizer, val_tgt_texts, window=args.window, stride=args.stride)

    device = torch.device(args.device)
    V = tokenizer.vocab_size()

    atom_emb = adapter = phi_snapshot = None

    if not args.sdpa_baseline:
        atom_emb = nn.Embedding(V, args.atom_dim).to(device)
        if args.adapter_rank > 0:
            adapter = AtomFeatureAdapter(args.atom_dim, rank=args.adapter_rank).to(device)
            print("[Adapter] enabled (low-rank adapter trainable; Φ pooled per-batch with autograd)")
            with torch.no_grad():
                phi_snapshot = adapter(atom_emb.weight).detach()
        else:
            for p in atom_emb.parameters():
                p.requires_grad = False
            phi_snapshot = atom_emb.weight
            print("[Adapter] disabled (atom pool frozen; Φ pooled from fixed atom_emb)")

        if not loaded_bank_cache:
            universe_ids = torch.arange(V, dtype=torch.long)
            universe = UniversePool(
                universe_ids,
                metadata={
                    "tokenizer_type": args.tokenizer_type,
                    "tokenizer_path": tokenizer_dir or "inline",
                },
            )
            print(universe.log_summary(prefix="[Universe]"))

            train_minhash = MinHasher(k=args.minhash_k, device=src_bank.values.device)
            cache_src = SetFeatureCache(
                universe, src_bank.values, src_bank.set_offsets, src_bank.seq_offsets, minhash=train_minhash
            )
            cache_tgt = SetFeatureCache(
                universe, tgt_bank.values, tgt_bank.set_offsets, tgt_bank.seq_offsets, minhash=train_minhash
            )

            if has_val and val_src_bank is not None and val_tgt_bank is not None:
                val_minhash = MinHasher(k=args.minhash_k, device=val_src_bank.values.device)
                val_cache_src = SetFeatureCache(
                    universe, val_src_bank.values, val_src_bank.set_offsets, val_src_bank.seq_offsets, minhash=val_minhash
                )
                val_cache_tgt = SetFeatureCache(
                    universe, val_tgt_bank.values, val_tgt_bank.set_offsets, val_tgt_bank.seq_offsets, minhash=val_minhash
                )

            if cache_mode == "full" and token_spec is not None:
                tokens_fp = fingerprint(token_spec)
                data_sig = _seq_data_signature(args)
                bank_spec = _seq_bank_spec(args, dataset_id, data_sig, max_len, tokens_fp)
                bank_root = _artifact_root_with_override(bank_spec, args.artifact_fingerprint, hf_root)
                save_bank_pack(bank_root / "bank_src_train.pt", BankPack.from_banked_batch(src_bank, universe_ids))
                save_bank_pack(bank_root / "bank_tgt_train.pt", BankPack.from_banked_batch(tgt_bank, universe_ids))
                save_routing_pack(
                    bank_root / "routing_src_train.pt",
                    RoutingPack(cache_src.sig_sets.detach().cpu(), cache_src.size_sets.detach().cpu()),
                )
                save_routing_pack(
                    bank_root / "routing_tgt_train.pt",
                    RoutingPack(cache_tgt.sig_sets.detach().cpu(), cache_tgt.size_sets.detach().cpu()),
                )
                if has_val and val_cache_src is not None and val_cache_tgt is not None:
                    save_bank_pack(bank_root / "bank_src_val.pt", BankPack.from_banked_batch(val_src_bank, universe_ids))
                    save_bank_pack(bank_root / "bank_tgt_val.pt", BankPack.from_banked_batch(val_tgt_bank, universe_ids))
                    save_routing_pack(
                        bank_root / "routing_src_val.pt",
                        RoutingPack(val_cache_src.sig_sets.detach().cpu(), val_cache_src.size_sets.detach().cpu()),
                    )
                    save_routing_pack(
                        bank_root / "routing_tgt_val.pt",
                        RoutingPack(val_cache_tgt.sig_sets.detach().cpu(), val_cache_tgt.size_sets.detach().cpu()),
                    )
                write_meta(bank_root, bank_spec)
                print(f"[Cache] Saved seq2seq bank+routing to {bank_root}")
        if args.precompute_bank:
            universe = universe.to(device)
            cache_src = cache_src.to(device)
            cache_tgt = cache_tgt.to(device)
            if val_cache_src is not None:
                val_cache_src = val_cache_src.to(device)
            if val_cache_tgt is not None:
                val_cache_tgt = val_cache_tgt.to(device)
        if args.cache_only and cache_mode == "full":
            print("[Cache] cache-only complete (full).")
            wandb_run.finish()
            return

    model = TinySeq2SeqBackbone(
        src_vocab=len(train_src_stoi),
        tgt_vocab=len(train_tgt_stoi),
        d_model=args.atom_dim,
        nhead=args.heads,
        layers=args.layers,
        attn_baseline="explicit" if (args.sdpa_baseline and args.attn_baseline == "explicit") else "pytorch",
    ).to(device)
    set_attn = router = None
    if not args.sdpa_baseline:
        set_attn = SetBankAttention(
            d_model=args.atom_dim,
            num_heads=args.heads,
            tau=args.tau,
            gamma=0.3,
            beta=args.beta,
            score_mode=args.set_kernel,
            eta=args.eta,
            backend=args.ska_backend,
            precision=args.precision,
        ).to(device)
        router = TokenSetRouter(d_model=args.atom_dim, num_heads=args.heads, topk=args.router_topk).to(device)
    out_proj = nn.Linear(args.atom_dim, len(train_tgt_stoi)).to(device)

    params = list(model.parameters()) + list(out_proj.parameters())
    if not args.sdpa_baseline and set_attn is not None and router is not None:
        params += list(set_attn.parameters()) + list(router.parameters())
        if atom_emb is not None:
            params += list(atom_emb.parameters())
        if adapter is not None:
            params += list(adapter.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)

    if args.benchmark:
        run_seq2seq_benchmark(
            args,
            model,
            set_attn,
            router,
            out_proj,
            adapter,
            atom_emb,
            phi_snapshot,
            cache_src,
            cache_tgt,
            train_pairs,
            train_src_stoi,
            train_tgt_stoi,
            train_ref_tokens,
            max_len,
            opt,
            device,
            wandb_run,
            args.benchmark_csv,
            seed,
            rep,
            run_uid,
            train_src_ids=train_src_ids,
            train_tgt_ids=train_tgt_ids,
        )
        wandb_run.finish()
        return

    attn_impl = _attn_impl_label(args, args.sdpa_baseline)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if set_attn is not None:
            set_attn.train()
        if router is not None:
            router.train()
        if adapter is not None:
            adapter.train()

        train_refs, train_hyps = [], []
        train_loss_total = 0.0
        train_token_total = 0
        try:
            with profiler(True) as prof:
                for batch_idx_tensor, src_ids, tgt_ids, ref_tokens in text_batch_iterator(
                    train_pairs,
                    train_src_stoi,
                    train_tgt_stoi,
                    train_ref_tokens,
                    max_len,
                    args.batch,
                    shuffle=True,
                    src_ids=train_src_ids,
                    tgt_ids=train_tgt_ids,
                ):
                    batch_idx_tensor = batch_idx_tensor.to(device)
                    src_ids = src_ids.to(device, non_blocking=True)
                    tgt_ids = tgt_ids.to(device, non_blocking=True)
                    tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

                    dec_h, _ = model(src_ids, tgt_in)
                    if args.sdpa_baseline:
                        tok_out = dec_h
                    else:
                        phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                        Phi_q, Sig_q, Size_q, q_ptrs = cache_tgt.gather_flat(batch_idx_tensor, phi_dynamic)
                        Phi_k, Sig_k, Size_k, k_ptrs = cache_src.gather_flat(batch_idx_tensor, phi_dynamic)
                        Z_sets, q_ptrs = set_attn(Phi_q, Sig_q, Size_q, q_ptrs, Phi_k, Sig_k, Size_k, k_ptrs)
                        tok_out = router(dec_h, Z_sets, Phi_q, q_ptrs)
                    logits = out_proj(tok_out)

                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    train_loss_total += float(loss.detach()) * tgt_ids.numel()
                    train_token_total += tgt_ids.numel()
                    preds = logits.argmax(dim=-1)
                    train_refs.extend(ref_tokens)
                    train_hyps.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = "out of memory" in str(exc).lower()
            if args.skip_oom and is_oom:
                torch.cuda.empty_cache()
                if args.metrics_csv:
                    _append_benchmark_row(
                        args.metrics_csv,
                        {
                            "script": "train_seq2seq_text_banked",
                            "task": "seq2seq",
                            "dataset": args.dataset or "custom",
                            "dataset_id": args.dataset or "custom",
                            "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
                            "attn_impl": attn_impl,
                            "precision": args.precision,
                            "set_kernel": args.set_kernel,
                            "batch": args.batch,
                            "max_len": max_len,
                            "window": args.window,
                            "stride": args.stride,
                            "minhash_k": args.minhash_k,
                            "router_topk": args.router_topk,
                            "adapter_rank": args.adapter_rank,
                            "epoch": epoch,
                            "status": "oom",
                            "skip_reason": str(exc)[:160],
                        },
                    )
                if wandb_run.enabled:
                    wandb_run.finish()
                return
            raise

        tr_loss = train_loss_total / max(1, train_token_total)
        tr_ppl = math.exp(min(tr_loss, 20.0))
        tr_bleu = corpus_bleu(train_refs, train_hyps)
        tr_rouge = rouge_l(train_refs, train_hyps)
        mode_tag = "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}/{args.precision}"
        if args.sdpa_baseline:
            mode_tag = f"{mode_tag}/{attn_impl}"
        kernel_tag = "sdpa" if args.sdpa_baseline else args.set_kernel
        msg = (
            f"[Seq2Seq-Text-Banked][{mode_tag}][{kernel_tag}] epoch {epoch:02d} "
            f"train loss {tr_loss:.4f} ppl {tr_ppl:.2f} BLEU {tr_bleu:.3f} | ROUGE-L {tr_rouge:.3f} | time {prof['time_s']:.2f}s"
        )
        if prof.get("cpu_pct") is not None:
            msg += f" | CPU {prof['cpu_pct']:.1f}%"
        if torch.cuda.is_available():
            msg += f" | VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
            if prof.get("gpu_util_pct") is not None:
                msg += f" | GPU {prof['gpu_util_pct']:.1f}%"
            if prof.get("gpu_mem_util_pct") is not None:
                msg += f" | GPU-MEM {prof['gpu_mem_util_pct']:.1f}%"
            if prof.get("gpu_active_s") is not None:
                msg += f" | GPU-ACT {prof['gpu_active_s']:.2f}s"
        print(msg)
        if wandb_run.enabled:
            wandb_payload = {
                "train/bleu": tr_bleu,
                "train/rougeL": tr_rouge,
                "train/time_s": prof["time_s"],
                "impl/attn_impl": attn_impl,
            }
            if prof.get("gpu_peak_mem_mib") is not None:
                wandb_payload["train/peak_vram_mib"] = prof["gpu_peak_mem_mib"]
            wandb_run.log(wandb_payload, step=epoch)

        val_refs, val_hyps, val_src = [], [], []
        if has_val:
            model.eval()
            if set_attn is not None:
                set_attn.eval()
            if router is not None:
                router.eval()
            if adapter is not None:
                adapter.eval()
            torch.manual_seed(args.eval_seed)
            random.seed(args.eval_seed)
            val_loss_total = 0.0
            val_token_total = 0
            with torch.no_grad():
                for batch_idx_tensor, src_ids, tgt_ids, ref_tokens in text_batch_iterator(
                    val_pairs,
                    train_src_stoi,
                    train_tgt_stoi,
                    val_ref_tokens,
                    max_len,
                    args.batch,
                    shuffle=False,
                    generator=torch.Generator(device=device).manual_seed(args.eval_seed),
                    worker_init_fn=make_worker_init_fn(args.eval_seed),
                    src_ids=val_src_ids,
                    tgt_ids=val_tgt_ids,
                ):
                    batch_idx_tensor = batch_idx_tensor.to(device)
                    src_ids = src_ids.to(device, non_blocking=True)
                    tgt_ids = tgt_ids.to(device, non_blocking=True)
                    tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

                    dec_h, _ = model(src_ids, tgt_in)
                    if args.sdpa_baseline:
                        tok_out = dec_h
                    else:
                        phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                        Phi_q, Sig_q, Size_q, q_ptrs = val_cache_tgt.gather_flat(batch_idx_tensor, phi_dynamic)
                        Phi_k, Sig_k, Size_k, k_ptrs = val_cache_src.gather_flat(batch_idx_tensor, phi_dynamic)
                        Z_sets, q_ptrs = set_attn(Phi_q, Sig_q, Size_q, q_ptrs, Phi_k, Sig_k, Size_k, k_ptrs)
                        tok_out = router(dec_h, Z_sets, Phi_q, q_ptrs)
                    logits = out_proj(tok_out)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
                    val_loss_total += float(loss.detach()) * tgt_ids.numel()
                    val_token_total += tgt_ids.numel()
                    preds = logits.argmax(dim=-1)
                    val_refs.extend(ref_tokens)
                    val_hyps.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])
                    batch_indices = batch_idx_tensor.detach().cpu().tolist()
                    val_src.extend(val_pairs[i][0] for i in batch_indices)

            v_loss = val_loss_total / max(1, val_token_total)
            v_ppl = math.exp(min(v_loss, 20.0))
            v_bleu = corpus_bleu(val_refs, val_hyps)
            v_rouge = rouge_l(val_refs, val_hyps)
            print(
                f"[Seq2Seq-Text-Banked][{mode_tag}][{kernel_tag}] epoch {epoch:02d} "
                f"VAL loss {v_loss:.4f} ppl {v_ppl:.2f} BLEU {v_bleu:.3f} | ROUGE-L {v_rouge:.3f}"
            )
        else:
            v_loss = v_ppl = v_bleu = v_rouge = None
            val_refs = val_hyps = val_src = []

        if wandb_run.enabled:
            payload = {
                "train/loss": tr_loss,
                "train/ppl": tr_ppl,
                "train/bleu": tr_bleu,
                "train/rougeL": tr_rouge,
            }
            if v_loss is not None:
                payload.update(
                    {
                        "val/loss": v_loss,
                        "val/ppl": v_ppl,
                        "val/bleu": v_bleu,
                        "val/rougeL": v_rouge,
                    }
                )
            payload["impl/attn_impl"] = attn_impl
            sample_text = format_text_samples(
                val_refs,
                val_hyps,
                args.sample_count,
                args.sample_seed + epoch,
                sources=val_src,
            )
            if sample_text:
                payload["samples/val_text"] = sample_text
            wandb_run.log(payload, step=epoch)
        if args.metrics_csv:
            _append_benchmark_row(
                args.metrics_csv,
                {
                    "script": "train_seq2seq_text_banked",
                    "task": "seq2seq",
                    "dataset": args.dataset or "custom",
                    "dataset_id": args.dataset or "custom",
                    "mode": "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}",
                    "attn_impl": attn_impl,
                    "precision": args.precision,
                    "set_kernel": args.set_kernel,
                    "batch": args.batch,
                    "max_len": 64,
                    "window": args.window,
                    "stride": args.stride,
                    "minhash_k": args.minhash_k,
                    "router_topk": args.router_topk,
                    "adapter_rank": args.adapter_rank,
                    "seed": seed,
                    "rep": rep,
                    "run_uid": run_uid,
                    "epoch": epoch,
                    "train_bleu": tr_bleu,
                    "train_rougeL": tr_rouge,
                    "train_loss": tr_loss,
                    "train_ppl": tr_ppl,
                    "val_bleu": v_bleu if v_bleu is not None else "NA",
                    "val_rougeL": v_rouge if v_rouge is not None else "NA",
                    "val_loss": v_loss if v_loss is not None else "NA",
                    "val_ppl": v_ppl if v_ppl is not None else "NA",
                    "status": "ok",
                },
            )

    wandb_run.finish()


if __name__ == "__main__":
    main()
