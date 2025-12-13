import argparse
import json
import math
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
from set_attention.training.text_utils import (
    ids_to_tokens,
    text_batch_iterator,
)


def make_char_data(n=2000, seq_len=64, vocab=32, seed=3):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, vocab, (n, seq_len), generator=g)
    Y = torch.roll(X, shifts=-1, dims=1)
    return X, Y


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]


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
    train_lines = load_wikitext_lines(args.dataset, "train", cache_dir, line_limit)
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


class TinyLMBackbone(nn.Module):
    def __init__(self, vocab: int, d_model: int, nhead: int, layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(self.emb(x))


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
):
    if bench_batch is None:
        print("[benchmark] no data available.")
        return
    batch_idx, src_ids, tgt_ids, _ = bench_batch
    bench_batch_size = src_ids.size(0)
    batch_idx = batch_idx.to(device)
    src_ids = src_ids.to(device)
    tgt_ids = tgt_ids.to(device)

    def step():
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
    tokens = bench_batch_size * max_len * args.bench_iters
    throughput = tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[benchmark][lm] backend={args.ska_backend} precision={args.precision} "
        f"tokens/s={throughput:.1f} elapsed={elapsed:.3f}s"
    )
    if wandb_run.enabled:
        wandb_run.log(
            {
                "benchmark/tokens_per_s": throughput,
                "benchmark/elapsed_s": elapsed,
                "benchmark/batch_tokens": tokens,
            }
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true"], default="dot")
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
    ap.add_argument("--dataset", choices=["", "wikitext2", "wikitext103"], default="")
    ap.add_argument("--dataset-lines", type=int, default=0, help="Limit number of text lines per split (0 = all).")
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
    args = ap.parse_args()

    if args.streaming is None:
        args.streaming = bool(args.dataset == "wikitext103")
    if not args.dataset:
        args.streaming = False
    use_hf_tokenizer = bool(args.hf_tokenizer_name)
    if use_hf_tokenizer and not args.dataset:
        raise ValueError("--hf-tokenizer-name requires --dataset to be set.")
    if use_hf_tokenizer and not args.streaming:
        raise ValueError("--hf-tokenizer-name currently requires --streaming.")

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
    }
    wandb_run = init_wandb(
        args.wandb,
        args.wandb_project or None,
        args.wandb_run_name or None,
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

    backbone = TinyLMBackbone(vocab_size, args.d_model, args.nhead, args.layers).to(device)
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
        if args.sdpa_baseline:
            if benchmark_batch is None:
                print("[benchmark] no data available.")
                return
            _, src_ids, tgt_ids, _ = benchmark_batch
            bench_batch = src_ids.size(0)
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

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
            tokens = bench_batch * tgt_ids.size(1) * args.bench_iters
            throughput = tokens / elapsed if elapsed > 0 else 0.0
            print(
                f"[benchmark][lm] backend=sdpa precision={args.precision} "
                f"tokens/s={throughput:.1f} elapsed={elapsed:.3f}s"
            )
            if wandb_run.enabled:
                wandb_run.log(
                    {
                        "benchmark/tokens_per_s": throughput,
                        "benchmark/elapsed_s": elapsed,
                        "benchmark/batch_tokens": tokens,
                    }
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
            )
        return

    def run_epoch(train: bool):
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
        mode_tag = "sdpa" if args.sdpa_baseline else f"ska/{args.ska_backend}/{args.precision}"
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

    wandb_run.finish()
if __name__ == "__main__":
    main()
