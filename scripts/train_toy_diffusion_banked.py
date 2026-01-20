import argparse
import copy
import csv
import gc
import json
import os
import random
import subprocess
import time
import math
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from set_attention.utils.bench_skip import should_skip_dense, should_skip_ska
from set_attention.data.hf_cache import ensure_hf_cache
from set_attention.data.artifact_cache import (
    ArtifactSpec,
    artifact_root,
    assert_meta_compatible,
    file_signature,
    fingerprint,
    require_cached_artifacts,
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
from set_attention.experiments.data_toy import ToyDiffConfig, make_toy_continuous_sequences
from set_attention.experiments.diffusion_core import SimpleDDPM
from set_attention.experiments.models import PositionalEncoding, timestep_embedding
from set_attention.experiments.nlp_eval import distinct_n, self_bleu
from set_attention.eval.mmd_simple import mmd2_unbiased_from_feats
from set_attention.utils.metrics import chamfer_l2, one_nn_two_sample
from set_attention.utils.profiling import profiler
from set_attention.utils.sample_logging import select_sample_indices
from set_attention.utils.wandb import init_wandb
from set_attention.utils.grad_stats import component_grad_norms, global_grad_norm, iter_named_parameters
from set_attention.utils.model_config import (
    add_model_args,
    build_model_config,
    model_config_fields,
    model_impl_label,
)
from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher
from set_attention.data.wikitext import load_wikitext_lines, tokenize_lines, chunk_tokens
from set_attention.utils.repro_workers import make_worker_init_fn
from common.repro import set_seed


TEXT_SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
EXPLICIT_ATTN_IMPLS = ("pytorch", "explicit")


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
    train_indices: Optional[List[int]] = None,
):
    train_lines = load_wikitext_lines(dataset, "train", cache_dir, train_line_limit, indices=train_indices)
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


def _text_data_signature(args) -> dict:
    return {
        "dataset": args.text_dataset,
        "train_line_limit": args.text_train_line_limit,
        "val_line_limit": args.text_val_line_limit,
        "train_seq_limit": args.text_train_limit,
        "val_seq_limit": args.text_val_limit,
        "subset": file_signature(Path(args.text_subset_path)) if args.text_subset_path else None,
        "embed_seed": args.text_embed_seed,
        "seq_len": args.text_seq_len,
        "stride": args.text_stride,
    }


def _text_token_spec(args) -> dict:
    spec = ArtifactSpec(
        task="textdiff",
        dataset_id=args.text_dataset,
        split="train+val",
        subset=_text_data_signature(args),
        tokenizer={"type": "whitespace", "special_tokens": TEXT_SPECIAL_TOKENS},
        sequence={"seq_len": int(args.text_seq_len), "stride": int(args.text_stride)},
        ska={
            "window": int(args.window),
            "stride": int(args.stride),
            "minhash_k": int(args.minhash_k),
            "router_topk": int(args.router_topk),
            "backend": args.ska_backend,
            "precision": args.precision,
        },
        model={
            "d_model": int(args.d_model),
            "nhead": int(args.nhead),
            "layers": int(args.layers),
            "precision": args.precision,
        },
        routing_depends_on_learned_params=bool(args.adapter_rank > 0),
    )
    return spec.to_dict()


def _text_bank_spec(args, tokens_fp: str) -> dict:
    spec = ArtifactSpec(
        task="textdiff",
        dataset_id=args.text_dataset,
        split="train+val",
        subset=_text_data_signature(args),
        tokenizer={"type": "whitespace", "special_tokens": TEXT_SPECIAL_TOKENS, "tokens_fp": tokens_fp},
        sequence={"seq_len": int(args.text_seq_len), "stride": int(args.text_stride)},
        ska={
            "window": int(args.window),
            "stride": int(args.stride),
            "minhash_k": int(args.minhash_k),
            "router_topk": int(args.router_topk),
            "backend": args.ska_backend,
            "precision": args.precision,
        },
        model={
            "d_model": int(args.d_model),
            "nhead": int(args.nhead),
            "layers": int(args.layers),
            "precision": args.precision,
        },
        routing_depends_on_learned_params=bool(args.adapter_rank > 0),
    )
    return spec.to_dict()


def _artifact_root_with_override(spec: dict, override_fp: str, hf_root: Path) -> Path:
    if override_fp:
        return hf_root / "artifacts" / spec["task"] / spec["dataset_id"] / override_fp
    return artifact_root(spec, hf_root)


def _save_text_tokens(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_text_tokens(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _append_benchmark_row(csv_path: str, row: dict) -> None:
    if not csv_path:
        return
    if row.get("model_type") != "ska":
        for key in ("window", "stride", "minhash_k", "router_topk"):
            if key in row:
                row[key] = "NA"
    if "bench_warmup" in row:
        task = row.get("task")
        if task in ("lm", "seq2seq"):
            row.setdefault("sequences_per_s", "NA")
            row.setdefault("images_per_s", "NA")
        elif task in ("textdiff", "diffusion"):
            row.setdefault("tokens_per_s", "NA")
            row.setdefault("tokens_total", "NA")
            row.setdefault("images_per_s", "NA")
        elif task == "vit":
            row.setdefault("tokens_per_s", "NA")
            row.setdefault("tokens_total", "NA")
            row.setdefault("sequences_per_s", "NA")

    def _sanitize(value: object) -> object:
        if value is None:
            return "NA"
        if isinstance(value, float) and math.isnan(value):
            return "NA"
        return value

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
                    writer.writerow({col: _sanitize(prev.get(col, "NA")) for col in new_fields})
            existing = new_fields
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=existing)
            writer.writerow({col: _sanitize(row.get(col, "NA")) for col in existing})
    else:
        fieldnames = list(row.keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({col: _sanitize(row.get(col, "NA")) for col in fieldnames})


_WANDB_KEYS_PRINTED: set[tuple[str, str]] = set()


def _log_csv_row_wandb(wandb_run, row: dict, prefix: str, step: int | None = None, summarize: bool = False) -> None:
    if not getattr(wandb_run, "enabled", False):
        return
    if row.get("model_type") != "ska":
        for key in ("window", "stride", "minhash_k", "router_topk"):
            if key in row:
                row[key] = "NA"
    if "bench_warmup" in row:
        task = row.get("task")
        if task in ("lm", "seq2seq"):
            row.setdefault("sequences_per_s", "NA")
            row.setdefault("images_per_s", "NA")
        elif task in ("textdiff", "diffusion"):
            row.setdefault("tokens_per_s", "NA")
            row.setdefault("tokens_total", "NA")
            row.setdefault("images_per_s", "NA")
        elif task == "vit":
            row.setdefault("tokens_per_s", "NA")
            row.setdefault("tokens_total", "NA")
            row.setdefault("sequences_per_s", "NA")
    if os.environ.get("SA_PRINT_WANDB_KEYS") == "1":
        key_id = (row.get("script", "unknown"), prefix)
        if key_id not in _WANDB_KEYS_PRINTED:
            keys = sorted(f"{prefix}/{key}" for key in row.keys())
            print(f"[wandb-keys] {key_id[0]} {prefix}: {', '.join(keys)}")
            _WANDB_KEYS_PRINTED.add(key_id)
    if os.environ.get("SA_LOG_CSV_TO_WANDB") != "1":
        return
    payload: dict[str, object] = {}
    for key, value in row.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = "NA"
        if isinstance(value, (list, tuple, dict)):
            payload[f"{prefix}/{key}"] = json.dumps(value, sort_keys=True)
        else:
            payload[f"{prefix}/{key}"] = value
    wandb_run.log(payload, step=step)
    if summarize:
        for key, value in row.items():
            wandb_run.set_summary(f"{prefix}/{key}", value)


def _prime_wandb_summary(wandb_run, config: dict, extra: dict | None = None) -> None:
    if not getattr(wandb_run, "enabled", False):
        return
    for key, value in config.items():
        wandb_run.set_summary(key, value if value is not None else "NA")
    if extra:
        for key, value in extra.items():
            wandb_run.set_summary(key, value)


def _summarize_wandb_payload(wandb_run, payload: dict) -> None:
    if not getattr(wandb_run, "enabled", False):
        return
    for key, value in payload.items():
        if isinstance(value, (list, tuple, dict)):
            continue
        wandb_run.set_summary(key, value)


def _append_exception_rows(args, seed: int, rep: int, run_uid: str, exc: Exception, wandb_run=None) -> None:
    dataset_id = args.text_dataset if args.data_mode == "text" else args.data_mode
    model_cfg = build_model_config(args, allow_set_kernel=False)
    row = {
        "script": "train_toy_diffusion_banked",
        "task": "textdiff" if args.data_mode == "text" else "diffusion",
        "dataset": dataset_id,
        "dataset_id": dataset_id,
        **model_config_fields(model_cfg),
        "precision": args.precision,
        "batch": args.batch,
        "seq_len": args.text_seq_len if args.data_mode == "text" else args.data_seq_len,
        "seq_stride": args.text_stride if args.data_mode == "text" else "NA",
        "window": args.window,
        "stride": args.stride,
        "minhash_k": args.minhash_k,
        "router_topk": args.router_topk,
        "adapter_rank": args.adapter_rank,
        "cache_fp": getattr(args, "cache_fp", "NA"),
        "seed": seed,
        "rep": rep,
        "run_uid": run_uid,
        "status": "exception",
        "error_type": type(exc).__name__,
        "error_msg": str(exc)[:200],
    }
    if args.metrics_csv:
        _append_benchmark_row(args.metrics_csv, row)
        _log_csv_row_wandb(wandb_run, row, "csv/metrics", summarize=True)
    if args.benchmark_csv:
        _append_benchmark_row(args.benchmark_csv, row)
        _log_csv_row_wandb(wandb_run, row, "csv/benchmark", summarize=True)


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


def _gpu_free_gb(device: Optional[torch.device] = None) -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        if device is not None and device.type == "cuda":
            with torch.cuda.device(device):
                free, total = torch.cuda.mem_get_info()
        else:
            free, total = torch.cuda.mem_get_info()
    except Exception:
        return None
    return free / (1024**3)


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


class BankedDenoiser(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int,
        layers: int,
        router_topk: int,
        ska_backend: str = "python",
        ska_score_mode: str = "delta_plus_dot",
        precision: str = "fp32",
        use_ska: bool = True,
        attn_baseline: str = "pytorch",
        seq_len: int = 128,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.attn_baseline = attn_baseline
        self.use_explicit = attn_baseline == "explicit"
        if self.use_explicit:
            self.enc_layers = nn.ModuleList(
                [ExplicitTransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.0) for _ in range(layers)]
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.use_ska = use_ska
        self.router = None
        self.set_attn = None
        if use_ska:
            ska_gamma = 0.0 if ska_score_mode == "dot" else 0.3
            self.router = TokenSetRouter(d_model=d_model, num_heads=nhead, topk=router_topk)
            self.set_attn = SetBankAttention(
                d_model=d_model,
                num_heads=nhead,
                tau=1.0,
                gamma=ska_gamma,
                beta=1.0 / d_model,
                score_mode=ska_score_mode,
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
        if self.use_explicit:
            for layer in self.enc_layers:
                h = layer(h.float())
        else:
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
    text_mode = args.data_mode == "text"
    stride_text = args.text_stride if args.text_stride > 0 else args.text_seq_len
    config_label = args.config
    dataset_label = args.config
    if text_mode:
        config_label = f"text::{args.text_dataset}@{args.text_seq_len}/{stride_text}"
        dataset_label = args.text_dataset
    model_cfg = getattr(args, "model_config", build_model_config(args, allow_set_kernel=False))
    is_baseline = model_cfg.model_type == "baseline"
    impl_label = model_impl_label(model_cfg)
    cache_fp = getattr(args, "cache_fp", "NA")
    bench_batch = min(args.batch, train_data.size(0))
    if bench_batch == 0:
        print("[benchmark] no training data available.")
        return
    batch_idx = torch.arange(bench_batch, dtype=torch.long, device=device)
    xb = train_data[:bench_batch].to(device)

    stats_snapshot: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    free_gb_at_start = _gpu_free_gb(device)
    if args.gpu_vram > 0:
        # use seq len from data
        seq_len = xb.size(1) if xb.dim() > 1 else args.window
        decision = (
            should_skip_dense(bench_batch, seq_len, args.d_model, args.nhead, args.precision, args.gpu_vram)
            if is_baseline
            else should_skip_ska(
                bench_batch, seq_len, args.window, args.stride, args.nhead, args.precision, args.gpu_vram
            )
        )
        if decision.skip or args.dry_run:
            row = {
                "script": "train_toy_diffusion_banked",
                "task": "textdiff" if text_mode else "diffusion",
                "dataset": dataset_label,
                "config": config_label,
                "dataset_id": config_label,
                **model_config_fields(model_cfg),
                "precision": args.precision,
                "cache_fp": cache_fp,
                "status": "skipped",
                "skip_reason": decision.reason or ("dry_run" if args.dry_run else ""),
                "gpu_vram_gb": args.gpu_vram,
            }
            _append_benchmark_row(benchmark_csv, row)
            _log_csv_row_wandb(wandb_run, row, "csv/benchmark", summarize=True)
            if args.dry_run or decision.skip:
                return
    backend_label = impl_label
    if is_baseline:

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = ddpm.loss(model, xb, lambda t, dim: timestep_embedding(t, args.d_model), args.d_model)
            loss.backward()
            optimizer.step()

    else:
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
    try:
        for _ in range(args.bench_iters):
            step()
    except torch.cuda.OutOfMemoryError:
        if args.skip_oom:
            torch.cuda.empty_cache()
            row = {
                "script": "train_toy_diffusion_banked",
                "task": "textdiff" if text_mode else "diffusion",
                "dataset": dataset_label,
                "config": config_label,
                "dataset_id": config_label,
                **model_config_fields(model_cfg),
                "precision": args.precision,
                "cache_fp": cache_fp,
                "status": "oom",
                "skip_reason": "runtime_oom",
                "gpu_vram_gb": args.gpu_vram,
                "free_gb_at_start": free_gb_at_start if free_gb_at_start is not None else "NA",
                "peak_allocated_mb": (
                    torch.cuda.max_memory_allocated() / (1024**2)
                    if torch.cuda.is_available()
                    else "NA"
                ),
            }
            _append_benchmark_row(benchmark_csv, row)
            _log_csv_row_wandb(wandb_run, row, "csv/benchmark", summarize=True)
            return
        raise
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    max_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    sequences = bench_batch * args.bench_iters
    throughput = sequences / elapsed if elapsed > 0 else 0.0
    info = _system_info()
    if is_baseline:
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
    row = {
        "script": "train_toy_diffusion_banked",
        "task": "textdiff" if text_mode else "diffusion",
        "dataset": dataset_label,
        "config": config_label,
        "dataset_id": config_label,
        **model_config_fields(model_cfg),
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
        "cache_fp": cache_fp,
        "status": "ok",
        "skip_reason": "",
        "gpu_vram_gb": args.gpu_vram,
    }
    _append_benchmark_row(benchmark_csv, row)
    _log_csv_row_wandb(wandb_run, row, "csv/benchmark", summarize=True)


class TensorBatchDataset(Dataset):
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int):
        return idx, self.data[idx]


def _collate_tensor_batch(batch):
    idxs, rows = zip(*batch)
    return torch.tensor(idxs, dtype=torch.long), torch.stack(rows, dim=0)


def build_tensor_dataloader(
    data: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    generator: Optional[torch.Generator] = None,
    worker_init_fn=None,
) -> DataLoader:
    if data.numel() == 0:
        return DataLoader([])
    dataset = TensorBatchDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, int(num_workers)),
        generator=generator,
        worker_init_fn=worker_init_fn,
        collate_fn=_collate_tensor_batch,
        drop_last=False,
        persistent_workers=False,
    )


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap, default_model_type="ska", default_ska_score_mode="delta_plus_dot")
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
    ap.add_argument(
        "--dot-naive",
        action="store_true",
        help="Force dot-product attention modules to use naive (math) implementation.",
    )
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
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
        "--text-subset-path",
        type=str,
        default="",
        help="Optional JSON with 'indices' for text train split (train-only; validation unchanged).",
    )
    ap.add_argument(
        "--text-cache-dir",
        type=str,
        default="",
        help="Cache directory for HuggingFace Wikitext data; empty uses HF_DATASETS_CACHE/HF_HOME.",
    )
    ap.add_argument("--cache-mode", choices=["none", "tokens", "full"], default="none")
    ap.add_argument("--artifact-cache-root", type=str, default="")
    ap.add_argument("--artifact-fingerprint", type=str, default="")
    ap.add_argument("--overwrite-cache", action="store_true", help="Overwrite existing token/bank caches.")
    ap.add_argument("--cache-only", action="store_true", help="Build caches and exit without training/benchmarking.")
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
    ap.add_argument("--gpu-vram", type=float, default=0.0, help="Approx GPU VRAM in GB for skip estimation (0=disable).")
    ap.add_argument("--skip-oom", action="store_true", help="Skip configs that OOM or exceed estimated VRAM.")
    ap.add_argument("--dry-run", action="store_true", help="Print skip/ok decision and write CSV status without running.")
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
    ap.add_argument(
        "--metrics-csv",
        type=str,
        default="",
        help="Optional CSV path to log training/validation metrics per epoch.",
    )
    ap.add_argument("--sample-count", type=int, default=10, help="Number of validation samples to log.")
    ap.add_argument("--sample-seed", type=int, default=1337, help="Seed for selecting logged samples.")
    ap.add_argument(
        "--diversity-max-samples",
        type=int,
        default=256,
        help="Max generated samples to use for self-BLEU/distinct-n (0=all).",
    )
    ap.add_argument("--self-bleu-n", type=int, default=4, help="Max n-gram for self-BLEU.")
    ap.add_argument(
        "--eval-seed",
        type=int,
        default=1337,
        help="Seed to make validation evaluation deterministic across variants.",
    )
    ap.add_argument("--grad-log-interval", type=int, default=1, help="Log grad norms every N steps (0=disable).")
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
    args.model_config = build_model_config(args, allow_set_kernel=False)
    if args.data_mode == "text":
        if args.benchmark:
            if (
                args.text_train_line_limit is None
                and args.text_train_limit is None
                and not args.text_subset_path
            ):
                raise RuntimeError(
                    "--benchmark requires explicit --text-train-limit/--text-train-line-limit or --text-subset-path "
                    "(no silent dataset caps)."
                )
        elif (
            args.text_train_line_limit is not None
            or args.text_val_line_limit is not None
            or args.text_train_limit is not None
            or args.text_val_limit is not None
        ):
            print("[Data] ignoring text limit flags outside benchmark; use --text-subset-path instead.")
            args.text_train_line_limit = None
            args.text_val_line_limit = None
            args.text_train_limit = None
            args.text_val_limit = None
    _configure_dot_naive(args.dot_naive)
    if args.model_config.model_type == "baseline" and args.model_config.baseline_impl == "explicit":
        _sanity_check_explicit_attention(torch.device(args.device), args.d_model, args.nhead)
    if args.cache_mode == "full" and args.num_workers > 0:
        print("[Data] cache-mode=full forces num_workers=0 to avoid duplicating cached tensors.")
        args.num_workers = 0
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
            try:
                run_single(run_args, defaults, seed, rep, run_uid, multi_run)
            except Exception as exc:
                _append_exception_rows(run_args, seed, rep, run_uid, exc)
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    return


def run_single(args, defaults, seed: int, rep: int, run_uid: str, multi_run: bool):
    model_cfg = getattr(args, "model_config", build_model_config(args, allow_set_kernel=False))
    is_baseline = model_cfg.model_type == "baseline"
    impl_label = model_impl_label(model_cfg)
    config_label = args.text_dataset if args.data_mode == "text" else args.data_mode
    hf_cache = ensure_hf_cache(args.text_cache_dir)
    hf_root = resolve_hf_root(args.artifact_cache_root or None)
    cache_mode = args.cache_mode
    cache_fp = "NA"
    if cache_mode == "full" and args.adapter_rank > 0:
        raise ValueError("--cache-mode full is only supported when --adapter-rank 0.")
    if args.precompute_bank and cache_mode != "full":
        raise RuntimeError("--precompute-bank requires --cache-mode full.")
    if args.precompute_bank and args.benchmark:
        raise RuntimeError("--precompute-bank must not be used in benchmark mode.")
    torch.backends.cudnn.benchmark = True
    set_seed(seed, deterministic=args.deterministic, benchmark_mode=args.benchmark_mode)
    print(f"[Run] seed={seed} rep={rep} uid={run_uid}")

    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    dataset_label = args.text_dataset if args.data_mode == "text" else args.config
    seq_len = args.text_seq_len if args.data_mode == "text" else args.data_seq_len
    seq_stride = args.text_stride if args.data_mode == "text" else "NA"
    tokenizer_type = "whitespace" if args.data_mode == "text" else "NA"
    window_val = args.window if not is_baseline else "NA"
    stride_val = args.stride if not is_baseline else "NA"
    minhash_val = args.minhash_k if not is_baseline else "NA"
    router_val = args.router_topk if not is_baseline else "NA"
    wandb_config = {
        "script": "train_toy_diffusion_banked",
        "task": "textdiff" if args.data_mode == "text" else "diffusion",
        "dataset": dataset_label,
        "dataset_id": dataset_label,
        "data_mode": args.data_mode,
        "tokenizer_type": tokenizer_type,
        **model_config_fields(model_cfg),
        "precision": args.precision,
        "seq_len": seq_len,
        "seq_stride": seq_stride,
        "window": window_val,
        "stride": stride_val,
        "minhash_k": minhash_val,
        "router_topk": router_val,
        "adapter_rank": args.adapter_rank,
        "steps": args.steps,
        "batch": args.batch,
        "precompute_bank": args.precompute_bank,
        "streaming": False,
        "reuse_vocab": False,
        "vocab_path": "NA",
        "vocab_workers": 0,
        "hf_tokenizer_name": "NA",
        "hf_cache_dir": args.text_cache_dir or "NA",
        "sample_count": args.sample_count,
        "diversity_max_samples": args.diversity_max_samples,
        "self_bleu_n": args.self_bleu_n,
        "grad_log_interval": args.grad_log_interval,
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
    _prime_wandb_summary(
        wandb_run,
        wandb_config,
        {
            "train/acc": "NA",
            "train/coverage": "NA",
            "train/sets_per_seq": "NA",
            "train/top5": "NA",
            "train/time_s": "NA",
            "train/time_per_epoch_s": "NA",
            "train/peak_vram_mib": "NA",
            "train/grad_norm": "NA",
            "train/grad_norm_ska": "NA",
            "train/grad_norm_baseline_attn": "NA",
            "train/grad_norm_ffn": "NA",
            "val/loss": "NA",
            "val/acc": "NA",
            "val/top5": "NA",
            "val/self_bleu": "NA",
            "val/distinct_1": "NA",
            "val/distinct_2": "NA",
            "val/self_bleu_samples": "NA",
            "samples/val_text": "NA",
            "samples/generated": "NA",
            "samples/val_preview": "NA",
        },
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
    if cache_mode != "none" and not text_mode:
        raise ValueError("--cache-mode is only supported for --data-mode text.")
    cfg_seed = int(cfg_yaml.get("seed", 2024))
    data_seed = cfg_seed
    if args.data_seed is not None:
        data_seed = int(args.data_seed)
    data_seed = seed
    torch.manual_seed(data_seed)
    random.seed(data_seed)
    device = torch.device(args.device)
    free_gb_at_start = _gpu_free_gb(device)

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
    token_spec = None

    if text_mode:
        text_train_indices: Optional[List[int]] = None
        if args.text_subset_path:
            subset_payload = json.loads(Path(args.text_subset_path).read_text())
            text_train_indices = subset_payload.get("indices", [])
            if not text_train_indices:
                raise ValueError(f"No indices found in subset file: {args.text_subset_path}")
            print(f"[Data] Using text subset indices from {args.text_subset_path} (count={len(text_train_indices)})")
        if cache_mode != "none":
            token_spec = _text_token_spec(args)
            token_root = _artifact_root_with_override(token_spec, args.artifact_fingerprint, hf_root)
            tokens_path = token_root / "tokens.pt"
            if tokens_path.exists() and not args.overwrite_cache:
                assert_meta_compatible(token_root, token_spec)
                payload = _load_text_tokens(tokens_path)
                train_data = payload["train_data"]
                val_data = payload["val_data"]
                text_train_ids_tensor = payload["train_ids_tensor"]
                text_val_ids_tensor = payload["val_ids_tensor"]
                text_vocab_size = payload["vocab_size"]
                text_embeddings = payload["embeddings"]
                text_itos = payload["itos"]
                print(f"[Cache] Loaded text diffusion tokens from {tokens_path}")
                cache_fp = token_root.name
            else:
                cache_dir = Path(hf_cache)
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
                    text_train_indices,
                )
                train_data = text_payload["train_data"]
                val_data = text_payload["val_data"]
                text_train_ids_tensor = text_payload["train_ids_tensor"]
                text_val_ids_tensor = text_payload["val_ids_tensor"]
                text_vocab_size = text_payload["vocab_size"]
                text_embeddings = text_payload["embeddings"]
                text_itos = text_payload["itos"]
                payload = {
                    "train_data": train_data,
                    "val_data": val_data,
                    "train_ids_tensor": text_train_ids_tensor,
                    "val_ids_tensor": text_val_ids_tensor,
                    "vocab_size": text_vocab_size,
                    "embeddings": text_embeddings,
                    "itos": text_itos,
                }
                _save_text_tokens(tokens_path, payload)
                write_meta(token_root, token_spec)
                print(f"[Cache] Saved text diffusion tokens to {tokens_path}")
                cache_fp = token_root.name
            text_train_ids = [row.clone() for row in text_train_ids_tensor] if text_train_ids_tensor is not None else None
            text_val_ids = [row.clone() for row in text_val_ids_tensor] if text_val_ids_tensor is not None else None
        else:
            cache_dir = Path(hf_cache)
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
                text_train_indices,
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
        if args.cache_only and cache_mode in ("tokens", "full") and (is_baseline or cache_mode == "tokens"):
            print("[Cache] cache-only complete (tokens).")
            wandb_run.finish()
            return
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
    loaded_bank_cache = False
    atom_emb = adapter = phi_snapshot = None
    universe = None
    if not is_baseline:
        if cache_mode == "full" and text_mode and token_spec is not None:
            tokens_fp = fingerprint(token_spec)
            bank_spec = _text_bank_spec(args, tokens_fp)
            bank_root = _artifact_root_with_override(bank_spec, args.artifact_fingerprint, hf_root)
            bank_train_path = bank_root / "bank_train.pt"
            bank_val_path = bank_root / "bank_val.pt"
            routing_train_path = bank_root / "routing_train.pt"
            routing_val_path = bank_root / "routing_val.pt"
            if not args.precompute_bank:
                if args.overwrite_cache:
                    raise RuntimeError("--overwrite-cache requires --precompute-bank in cache-mode full.")
                require_cached_artifacts(
                    bank_root,
                    [bank_train_path, bank_val_path, routing_train_path, routing_val_path],
                    "textdiff",
                )
            if (
                bank_train_path.exists()
                and bank_val_path.exists()
                and routing_train_path.exists()
                and routing_val_path.exists()
                and not args.overwrite_cache
            ):
                assert_meta_compatible(bank_root, bank_spec)
                train_pack = load_bank_pack(bank_train_path)
                val_pack = load_bank_pack(bank_val_path)
                train_route = load_routing_pack(routing_train_path)
                val_route = load_routing_pack(routing_val_path)
                if not torch.equal(train_pack.universe_ids, val_pack.universe_ids):
                    raise RuntimeError("Cached diffusion bank packs have mismatched universe ids.")
                universe = UniversePool(train_pack.universe_ids, metadata={"task": "text_diffusion" if text_mode else "toy_diffusion"})
                train_cache = cache_from_packs(universe, train_pack, train_route)
                val_cache = cache_from_packs(universe, val_pack, val_route)
                loaded_bank_cache = True
                print(f"[Cache] Loaded diffusion bank+routing from {bank_root}")
                bank_fp = bank_root.name
                print(f"[CacheReuse] bank+routing loaded | fp={bank_fp} | task=textdiff")
                cache_fp = bank_fp

        if not loaded_bank_cache:
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
            if cache_mode == "full" and text_mode and token_spec is not None:
                tokens_fp = fingerprint(token_spec)
                bank_spec = _text_bank_spec(args, tokens_fp)
                bank_root = _artifact_root_with_override(bank_spec, args.artifact_fingerprint, hf_root)
                save_bank_pack(bank_root / "bank_train.pt", BankPack.from_banked_batch(train_bank, universe_ids))
                save_bank_pack(bank_root / "bank_val.pt", BankPack.from_banked_batch(val_bank, universe_ids))
                save_routing_pack(
                    bank_root / "routing_train.pt",
                    RoutingPack(train_cache.sig_sets.detach().cpu(), train_cache.size_sets.detach().cpu()),
                )
                save_routing_pack(
                    bank_root / "routing_val.pt",
                    RoutingPack(val_cache.sig_sets.detach().cpu(), val_cache.size_sets.detach().cpu()),
                )
                write_meta(bank_root, bank_spec)
                print(f"[Cache] Saved diffusion bank+routing to {bank_root}")
                bank_fp = bank_root.name
                print(f"[CacheCreate] bank+routing created | fp={bank_fp} | task=textdiff")
                cache_fp = bank_fp

        if text_mode and text_vocab_size is not None:
            args.cache_fp = cache_fp
            vocab_size = text_vocab_size
        elif train_cache is not None:
            args.cache_fp = cache_fp
            vocab_size = int(train_cache.values.max().item() + 1) if train_cache.values.numel() > 0 else data_cfg.seq_len * 2
        else:
            args.cache_fp = cache_fp
            vocab_size = data_cfg.seq_len * 2
        atom_emb = nn.Embedding(vocab_size, args.d_model).to(device)
        if args.adapter_rank > 0:
            adapter = AtomFeatureAdapter(args.d_model, rank=args.adapter_rank).to(device)
            with torch.no_grad():
                phi_snapshot = adapter(atom_emb.weight).detach()
        else:
            phi_snapshot = atom_emb.weight

        if args.precompute_bank:
            universe = universe.to(device)
            train_cache = train_cache.to(device)
            val_cache = val_cache.to(device)
        if args.cache_only and cache_mode == "full" and text_mode:
            print("[Cache] cache-only complete (full).")
            wandb_run.finish()
            return

    attn_baseline = "explicit" if (is_baseline and model_cfg.baseline_impl == "explicit") else "pytorch"
    model = BankedDenoiser(
        in_dim=data_cfg.dim,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        router_topk=args.router_topk,
        ska_backend=args.ska_backend,
        ska_score_mode=args.ska_score_mode,
        precision=args.precision,
        use_ska=not is_baseline,
        attn_baseline=attn_baseline,
        seq_len=data_cfg.seq_len,
    ).to(device)
    ddpm = SimpleDDPM(T=args.steps, device=device)
    component_named_params = list(
        iter_named_parameters(
            {
                "model": model,
                "adapter": adapter,
                "atom_emb": atom_emb,
            }
        )
    )
    params = list(model.parameters())
    if not is_baseline and atom_emb is not None:
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

    global_step = 0
    grad_comp_sums = {"ska": 0.0, "baseline_attn": 0.0, "ffn": 0.0}
    grad_comp_counts = {"ska": 0, "baseline_attn": 0, "ffn": 0}
    for epoch in range(1, args.epochs + 1):
        grad_norm_sum = 0.0
        grad_norm_count = 0
        grad_comp_sums = {"ska": 0.0, "baseline_attn": 0.0, "ffn": 0.0}
        grad_comp_counts = {"ska": 0, "baseline_attn": 0, "ffn": 0}
        model.train()
        total_loss = 0.0
        count = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_gen = torch.Generator().manual_seed(seed + epoch)
        try:
            with profiler(args.profile) as prof:
                for batch_idx, xb in build_tensor_dataloader(
                    train_data,
                    data_cfg.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    generator=train_gen,
                    worker_init_fn=None,
                ):
                    xb = xb.to(device)
                    if not is_baseline:
                        phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                        Phi, Sig, Size, q_ptrs = train_cache.gather_flat(batch_idx.to(device), phi_dynamic)
                        model.set_current_bank(Phi, Sig, Size, q_ptrs)
                    loss = ddpm.loss(model, xb, lambda t, dim: timestep_embedding(t, args.d_model), args.d_model)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_log_interval > 0:
                        if global_step % args.grad_log_interval == 0:
                            grad_norm = global_grad_norm(params)
                            grad_norm_sum += grad_norm
                            grad_norm_count += 1
                            comp_norms = component_grad_norms(component_named_params)
                            for key, value in comp_norms.items():
                                grad_comp_sums[key] += value
                                grad_comp_counts[key] += 1
                            if wandb_run.enabled:
                                payload = {"train/grad_norm": grad_norm}
                                if "ska" in comp_norms:
                                    payload["train/grad_norm_ska"] = comp_norms["ska"]
                                if "baseline_attn" in comp_norms:
                                    payload["train/grad_norm_baseline_attn"] = comp_norms["baseline_attn"]
                                if "ffn" in comp_norms:
                                    payload["train/grad_norm_ffn"] = comp_norms["ffn"]
                                wandb_run.log(payload, step=global_step)
                    global_step += 1
                    optimizer.step()
                    total_loss += float(loss.detach()) * xb.size(0)
                    count += xb.size(0)
            train_loss = total_loss / max(1, count)
            train_time_s = prof.get("time_s", 0.0)
            peak_vram_mib = (
                float(torch.cuda.max_memory_allocated()) / (1024**2) if torch.cuda.is_available() else None
            )
            train_grad_norm = (grad_norm_sum / grad_norm_count) if grad_norm_count else None
            train_grad_norm_ska = (
                grad_comp_sums["ska"] / grad_comp_counts["ska"] if grad_comp_counts["ska"] else None
            )
            train_grad_norm_baseline = (
                grad_comp_sums["baseline_attn"] / grad_comp_counts["baseline_attn"]
                if grad_comp_counts["baseline_attn"]
                else None
            )
            train_grad_norm_ffn = (
                grad_comp_sums["ffn"] / grad_comp_counts["ffn"] if grad_comp_counts["ffn"] else None
            )

            model.eval()
            sample_indices = select_sample_indices(val_data.size(0), args.sample_count, args.sample_seed + epoch)
            sample_lookup = {idx: order for order, idx in enumerate(sample_indices)}
            captured_samples: dict[int, dict] = {}
            with torch.no_grad():
                torch.manual_seed(args.eval_seed)
                random.seed(args.eval_seed)
                mmds, chamfers, nn1s = [], [], []
                val_loss_total = 0.0
                val_count = 0
                val_generated_tokens: List[List[str]] = []
                val_gen = torch.Generator().manual_seed(args.eval_seed)
                for batch_idx, xb in build_tensor_dataloader(
                    val_data,
                    data_cfg.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    generator=val_gen,
                    worker_init_fn=make_worker_init_fn(args.eval_seed),
                ):
                    xb_cpu = xb
                    xb = xb_cpu.to(device)
                    if not is_baseline:
                        phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                        Phi, Sig, Size, q_ptrs = val_cache.gather_flat(batch_idx.to(device), phi_dynamic)
                        model.set_current_bank(Phi, Sig, Size, q_ptrs)
                    val_loss = ddpm.loss(model, xb, lambda t, dim: timestep_embedding(t, args.d_model), args.d_model)
                    val_loss_total += float(val_loss.detach()) * xb.size(0)
                    val_count += xb.size(0)
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
                        ref_token_batch = [
                            [text_itos[int(tok)] for tok in seq.tolist()] for seq in tgt_ids.detach().cpu()
                        ]
                        hyp_token_batch = [
                            [text_itos[int(tok)] for tok in seq.tolist()] for seq in pred_ids.detach().cpu()
                        ]
                        val_generated_tokens.extend(hyp_token_batch)
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
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = "out of memory" in str(exc).lower()
            if args.skip_oom and is_oom:
                torch.cuda.empty_cache()
                if args.metrics_csv:
                    row = {
                        "script": "train_toy_diffusion_banked",
                        "task": "textdiff" if text_mode else "diffusion",
                        "dataset": dataset_label,
                        "config": config_label,
                        "dataset_id": config_label,
                        **model_config_fields(model_cfg),
                        "precision": args.precision,
                        "batch": data_cfg.batch_size,
                        "window": args.window,
                        "stride": args.stride,
                        "minhash_k": args.minhash_k,
                        "router_topk": args.router_topk,
                        "adapter_rank": args.adapter_rank,
                        "epoch": epoch,
                        "cache_fp": getattr(args, "cache_fp", "NA"),
                        "status": "oom",
                        "skip_reason": str(exc)[:160],
                        "free_gb_at_start": free_gb_at_start if free_gb_at_start is not None else "NA",
                        "peak_allocated_mb": (
                            torch.cuda.max_memory_allocated() / (1024**2)
                            if torch.cuda.is_available()
                            else "NA"
                        ),
                    }
                    _append_benchmark_row(args.metrics_csv, row)
                    _log_csv_row_wandb(wandb_run, row, "csv/metrics", step=epoch, summarize=True)
                if wandb_run.enabled:
                    wandb_run.finish()
                return
            raise

        def safe_mean(values):
            return float(sum(values) / len(values)) if values else float("nan")

        val_mmd_mean = safe_mean(mmds)
        val_chamfer_mean = safe_mean(chamfers)
        val_nn1_mean = safe_mean(nn1s)
        val_loss_mean = val_loss_total / max(1, val_count)
        val_self_bleu = None
        val_distinct_1 = None
        val_distinct_2 = None
        self_bleu_samples = None
        if text_mode and val_generated_tokens:
            diversity_samples = val_generated_tokens
            if args.diversity_max_samples > 0 and len(diversity_samples) > args.diversity_max_samples:
                rng = random.Random(args.eval_seed + epoch)
                indices = rng.sample(range(len(diversity_samples)), args.diversity_max_samples)
                diversity_samples = [diversity_samples[i] for i in indices]
            self_bleu_samples = len(diversity_samples)
            val_self_bleu = self_bleu(diversity_samples, max_n=args.self_bleu_n)
            val_distinct_1 = distinct_n(diversity_samples, n=1)
            val_distinct_2 = distinct_n(diversity_samples, n=2)

        kernel_tag = model_cfg.ska_score_mode if model_cfg.model_type == "ska" else model_cfg.baseline_impl
        mode_tag = impl_label
        msg = (
            f"[Diffusion-Banked][{mode_tag}][{kernel_tag}] epoch {epoch:02d} "
            f"train loss {train_loss:.4f} | val loss {val_loss_mean:.4f} | val MMD {val_mmd_mean:.4f} | "
            f"Chamfer {val_chamfer_mean:.4f} | 1NN {val_nn1_mean:.3f}"
        )
        if val_self_bleu is not None:
            msg += (
                f" | self-BLEU {val_self_bleu:.4f} | distinct-1 {val_distinct_1:.4f}"
                f" | distinct-2 {val_distinct_2:.4f}"
            )
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
                "val/loss": val_loss_mean,
                "val/mmd": val_mmd_mean,
                "val/chamfer": val_chamfer_mean,
                "val/1nn": val_nn1_mean,
                "impl/model_type": model_cfg.model_type,
            }
            if model_cfg.baseline_impl:
                wandb_payload["impl/baseline_impl"] = model_cfg.baseline_impl
            if model_cfg.ska_backend:
                wandb_payload["impl/ska_backend"] = model_cfg.ska_backend
            if model_cfg.ska_score_mode:
                wandb_payload["impl/ska_score_mode"] = model_cfg.ska_score_mode
            wandb_payload["val/self_bleu"] = val_self_bleu if val_self_bleu is not None else "NA"
            wandb_payload["val/distinct_1"] = val_distinct_1 if val_distinct_1 is not None else "NA"
            wandb_payload["val/distinct_2"] = val_distinct_2 if val_distinct_2 is not None else "NA"
            wandb_payload["val/self_bleu_samples"] = (
                self_bleu_samples if self_bleu_samples is not None else "NA"
            )
            wandb_payload["train/time_s"] = train_time_s
            wandb_payload["train/time_per_epoch_s"] = train_time_s
            if peak_vram_mib is not None:
                wandb_payload["train/peak_vram_mib"] = peak_vram_mib
            if train_grad_norm is not None:
                wandb_payload["train/grad_norm_epoch_mean"] = train_grad_norm
            if train_grad_norm_ska is not None:
                wandb_payload["train/grad_norm_ska_epoch_mean"] = train_grad_norm_ska
            if train_grad_norm_baseline is not None:
                wandb_payload["train/grad_norm_baseline_attn_epoch_mean"] = train_grad_norm_baseline
            if train_grad_norm_ffn is not None:
                wandb_payload["train/grad_norm_ffn_epoch_mean"] = train_grad_norm_ffn
            if sample_text is not None:
                wandb_payload["samples/generated"] = sample_text
                if text_mode:
                    wandb_payload["samples/val_text"] = sample_text
            wandb_run.log(wandb_payload, step=epoch)
            if epoch == args.epochs:
                _summarize_wandb_payload(wandb_run, wandb_payload)
        if args.metrics_csv:
            row = {
                "script": "train_toy_diffusion_banked",
                "task": "textdiff" if text_mode else "diffusion",
                "dataset": dataset_label,
                "dataset_id": config_label,
                **model_config_fields(model_cfg),
                "precision": args.precision,
                "window": args.window,
                "stride": args.stride,
                "minhash_k": args.minhash_k,
                "router_topk": args.router_topk,
                "batch": args.batch,
                "steps": args.steps,
                "seed": seed,
                "rep": rep,
                "run_uid": run_uid,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_mean,
                "val_mmd": val_mmd_mean,
                "val_chamfer": val_chamfer_mean,
                "val_1nn": val_nn1_mean,
                "train_grad_norm": train_grad_norm if train_grad_norm is not None else "NA",
                "train_grad_norm_ska": train_grad_norm_ska if train_grad_norm_ska is not None else "NA",
                "train_grad_norm_baseline_attn": (
                    train_grad_norm_baseline if train_grad_norm_baseline is not None else "NA"
                ),
                "train_grad_norm_ffn": train_grad_norm_ffn if train_grad_norm_ffn is not None else "NA",
                "val_self_bleu": val_self_bleu if val_self_bleu is not None else "NA",
                "val_distinct_1": val_distinct_1 if val_distinct_1 is not None else "NA",
                "val_distinct_2": val_distinct_2 if val_distinct_2 is not None else "NA",
                "val_self_bleu_samples": self_bleu_samples if self_bleu_samples is not None else "NA",
                "cache_fp": getattr(args, "cache_fp", "NA"),
                "status": "ok",
                "train_time_s": train_time_s,
                "train_time_per_epoch_s": train_time_s,
                "train_peak_vram_mib": peak_vram_mib if peak_vram_mib is not None else "NA",
            }
            _append_benchmark_row(args.metrics_csv, row)
            _log_csv_row_wandb(
                wandb_run,
                row,
                "csv/metrics",
                step=epoch,
                summarize=(epoch == args.epochs),
            )

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
