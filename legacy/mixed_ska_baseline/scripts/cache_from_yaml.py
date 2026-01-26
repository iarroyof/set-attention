#!/usr/bin/env python3
"""Build/verify token + SKA caches for Stage A/B sweeps using sweep YAMLs.

FIXED VERSION: Uses fingerprint-based cache lookup to match runtime behavior.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

try:  # Optional dependency for HF cache checks.
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

try:
    from set_attention.data import resolve_data_root
    from set_attention.data.cache import DEFAULT_DATA_ROOT
    from set_attention.data.artifact_cache import (
        file_signature,
        resolve_hf_root,
        fingerprint,
        ArtifactSpec,
    )
    from set_attention.data.hf_cache import ensure_hf_cache
except Exception as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError("Run with PYTHONPATH=src so set_attention modules are importable.") from exc


DEFAULT_TASKS = ["lm", "seq2seq", "textdiff", "vit"]
WIKITEXT_CONFIGS = {
    "wikitext2": "wikitext-2-raw-v1",
    "wikitext103": "wikitext-103-raw-v1",
}
FULL_REQUIRED = {
    "lm": ["bank_train.pt", "bank_val.pt", "routing_train.pt", "routing_val.pt"],
    "textdiff": ["bank_train.pt", "bank_val.pt", "routing_train.pt", "routing_val.pt"],
    "seq2seq": [
        "bank_src_train.pt",
        "bank_tgt_train.pt",
        "bank_src_val.pt",
        "bank_tgt_val.pt",
        "routing_src_train.pt",
        "routing_tgt_train.pt",
        "routing_src_val.pt",
        "routing_tgt_val.pt",
    ],
}
LEGACY_HF_ROOT = DEFAULT_DATA_ROOT / "hf"

# Special tokens must match training scripts exactly
LM_SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
TEXTDIFF_SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
SEQ2SEQ_SPECIAL_TOKENS = ["<pad>", "<s>", "</s>"]  # Note: no <unk> for seq2seq

# Model defaults per task (must match training scripts)
MODEL_DEFAULTS = {
    "lm": {"d_model": 128, "nhead": 4, "layers": 4},
    "seq2seq": {"d_model": 128, "nhead": 8, "layers": 2},  # atom_dim->d_model, heads->nhead
    "textdiff": {"d_model": 64, "nhead": 2, "layers": 2},
}

# Tokenizer types
WHITESPACE_TOKENIZER_TYPE = "whitespace"


def _default_tokenizer_config(kind: str, max_len: int) -> dict:
    """Match train_seq2seq_text_banked.py tokenizer config."""
    if kind == WHITESPACE_TOKENIZER_TYPE:
        return {"lowercase": True, "min_freq": 2, "max_vocab": 200_000, "max_len": max_len}
    return {}


class CacheJob:
    """Represents a cache build/verify job with all parameters needed for fingerprinting."""

    def __init__(
        self,
        *,
        task: str,
        cache_mode: str,
        dataset: str,
        subset_path: Optional[str],
        precision: str,
        score_mode: Optional[str],
        ska_backend: Optional[str],
        seq_len: Optional[int] = None,
        seq_stride: Optional[int] = None,
        max_len: Optional[int] = None,
        window: Optional[int] = None,
        bank_stride: Optional[int] = None,
        minhash_k: Optional[int] = None,
        router_topk: Optional[int] = None,
        tokenizer: Optional[str] = None,
        sources: Optional[list[str]] = None,
        # Model architecture - defaults set per-task in build_token_spec
        d_model: Optional[int] = None,
        nhead: Optional[int] = None,
        layers: Optional[int] = None,
        dataset_lines: int = 0,
        hf_tokenizer_name: str = "",
        adapter_rank: int = 0,
    ) -> None:
        self.task = task
        self.cache_mode = cache_mode
        self.dataset = dataset
        self.subset_path = subset_path
        self.precision = precision
        self.score_mode = score_mode
        self.ska_backend = ska_backend
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.max_len = max_len
        self.window = window
        self.bank_stride = bank_stride
        self.minhash_k = minhash_k
        self.router_topk = router_topk
        self.tokenizer = tokenizer or WHITESPACE_TOKENIZER_TYPE
        self.sources = sources or []
        self.subset_sig = file_signature(Path(subset_path)) if subset_path else None
        # Model architecture - use task-specific defaults if not provided
        defaults = MODEL_DEFAULTS.get(task, {})
        self.d_model = d_model if d_model is not None else defaults.get("d_model", 128)
        self.nhead = nhead if nhead is not None else defaults.get("nhead", 4)
        self.layers = layers if layers is not None else defaults.get("layers", 4)
        self.dataset_lines = dataset_lines
        self.hf_tokenizer_name = hf_tokenizer_name
        self.adapter_rank = adapter_rank
        # TextDiff-specific defaults
        self.text_train_line_limit = None
        self.text_val_line_limit = None
        self.text_train_limit = None
        self.text_val_limit = None
        self.text_embed_seed = 1337
        # Seq2Seq-specific defaults
        self.limit = None
        self.val_limit = None

    def _build_textdiff_data_signature(self) -> dict:
        """Build subset signature matching _text_data_signature in train_toy_diffusion_banked.py"""
        return {
            "dataset": self.dataset,
            "train_line_limit": self.text_train_line_limit,
            "val_line_limit": self.text_val_line_limit,
            "train_seq_limit": self.text_train_limit,
            "val_seq_limit": self.text_val_limit,
            "subset": self.subset_sig,
            "embed_seed": self.text_embed_seed,
            "seq_len": int(self.seq_len),
            "stride": int(self.seq_stride),
        }

    def _build_seq2seq_data_signature(self) -> dict:
        """Build subset signature matching _seq_data_signature in train_seq2seq_text_banked.py"""
        return {
            "dataset": self.dataset,
            "limit": self.limit,
            "val_limit": self.val_limit,
            "subset": self.subset_sig,
        }

    def key_base(self) -> tuple[Any, ...]:
        return (
            self.task,
            self.dataset,
            self.precision,
            self.ska_backend,
            self.seq_len,
            self.seq_stride,
            self.max_len,
            self.window,
            self.bank_stride,
            self.minhash_k,
            self.router_topk,
            self.tokenizer,
            self.d_model,
            self.nhead,
            self.layers,
            self.adapter_rank,
            self.subset_sig["sha256"] if self.subset_sig else None,
        )

    def build_token_spec(self) -> dict:
        """Build token spec matching training scripts exactly."""
        routing_depends = bool(self.adapter_rank > 0)
        
        if self.task == "lm":
            # Match _lm_token_spec in train_toy_lm_banked.py
            return ArtifactSpec(
                task="lm",
                dataset_id=self.dataset,
                split="train+val",
                subset=self.subset_sig,
                tokenizer={
                    "type": "whitespace",
                    "special_tokens": LM_SPECIAL_TOKENS,
                    "hf_tokenizer_name": self.hf_tokenizer_name,
                },
                sequence={
                    "seq_len": int(self.seq_len),
                    "stride": int(self.seq_stride),
                    "dataset_lines": int(self.dataset_lines),
                },
                ska={
                    "window": int(self.window),
                    "stride": int(self.bank_stride),
                    "minhash_k": int(self.minhash_k),
                    "router_topk": int(self.router_topk),
                    "backend": self.ska_backend,
                    "precision": self.precision,
                },
                model={
                    "d_model": int(self.d_model),
                    "nhead": int(self.nhead),
                    "layers": int(self.layers),
                    "precision": self.precision,
                },
                routing_depends_on_learned_params=routing_depends,
            ).to_dict()
        elif self.task == "textdiff":
            # Match _text_token_spec in train_toy_diffusion_banked.py
            return ArtifactSpec(
                task="textdiff",
                dataset_id=self.dataset,
                split="train+val",
                subset=self._build_textdiff_data_signature(),
                tokenizer={
                    "type": "whitespace",
                    "special_tokens": TEXTDIFF_SPECIAL_TOKENS,
                },
                sequence={
                    "seq_len": int(self.seq_len),
                    "stride": int(self.seq_stride),
                },
                ska={
                    "window": int(self.window),
                    "stride": int(self.bank_stride),
                    "minhash_k": int(self.minhash_k),
                    "router_topk": int(self.router_topk),
                    "backend": self.ska_backend,
                    "precision": self.precision,
                },
                model={
                    "d_model": int(self.d_model),
                    "nhead": int(self.nhead),
                    "layers": int(self.layers),
                    "precision": self.precision,
                },
                routing_depends_on_learned_params=routing_depends,
            ).to_dict()
        elif self.task == "seq2seq":
            # Match _seq_token_spec in train_seq2seq_text_banked.py
            return ArtifactSpec(
                task="seq2seq",
                dataset_id=self.dataset,
                split="train+val",
                subset=self._build_seq2seq_data_signature(),
                tokenizer={
                    "type": self.tokenizer,
                    "special_tokens": SEQ2SEQ_SPECIAL_TOKENS,
                    "max_len": int(self.max_len),
                    "tokenizer_config": _default_tokenizer_config(self.tokenizer, int(self.max_len)),
                },
                sequence={"max_len": int(self.max_len)},
                ska={
                    "window": int(self.window),
                    "stride": int(self.bank_stride),
                    "minhash_k": int(self.minhash_k),
                    "router_topk": int(self.router_topk),
                    "backend": self.ska_backend,
                    "precision": self.precision,
                },
                model={
                    "d_model": int(self.d_model),
                    "heads": int(self.nhead),  # seq2seq uses "heads" not "nhead"
                    "layers": int(self.layers),
                    "precision": self.precision,
                },
                routing_depends_on_learned_params=routing_depends,
            ).to_dict()
        else:
            raise ValueError(f"Unknown task for spec building: {self.task}")

    def build_bank_spec(self, tokens_fp: str) -> dict:
        """Build bank spec matching training scripts exactly."""
        routing_depends = bool(self.adapter_rank > 0)
        
        if self.task == "lm":
            # Match _lm_bank_spec in train_toy_lm_banked.py
            return ArtifactSpec(
                task="lm",
                dataset_id=self.dataset,
                split="train+val",
                subset=self.subset_sig,
                tokenizer={
                    "type": "whitespace",
                    "special_tokens": LM_SPECIAL_TOKENS,
                    "hf_tokenizer_name": self.hf_tokenizer_name,
                    "tokens_fp": tokens_fp,
                },
                sequence={
                    "seq_len": int(self.seq_len),
                    "stride": int(self.seq_stride),
                    "dataset_lines": int(self.dataset_lines),
                },
                ska={
                    "window": int(self.window),
                    "stride": int(self.bank_stride),
                    "minhash_k": int(self.minhash_k),
                    "router_topk": int(self.router_topk),
                    "backend": self.ska_backend,
                    "precision": self.precision,
                },
                model={
                    "d_model": int(self.d_model),
                    "nhead": int(self.nhead),
                    "layers": int(self.layers),
                    "precision": self.precision,
                },
                routing_depends_on_learned_params=routing_depends,
            ).to_dict()
        elif self.task == "textdiff":
            # Match _text_bank_spec in train_toy_diffusion_banked.py
            return ArtifactSpec(
                task="textdiff",
                dataset_id=self.dataset,
                split="train+val",
                subset=self._build_textdiff_data_signature(),
                tokenizer={
                    "type": "whitespace",
                    "special_tokens": TEXTDIFF_SPECIAL_TOKENS,
                    "tokens_fp": tokens_fp,
                },
                sequence={
                    "seq_len": int(self.seq_len),
                    "stride": int(self.seq_stride),
                },
                ska={
                    "window": int(self.window),
                    "stride": int(self.bank_stride),
                    "minhash_k": int(self.minhash_k),
                    "router_topk": int(self.router_topk),
                    "backend": self.ska_backend,
                    "precision": self.precision,
                },
                model={
                    "d_model": int(self.d_model),
                    "nhead": int(self.nhead),
                    "layers": int(self.layers),
                    "precision": self.precision,
                },
                routing_depends_on_learned_params=routing_depends,
            ).to_dict()
        elif self.task == "seq2seq":
            # Match _seq_bank_spec in train_seq2seq_text_banked.py
            # NOTE: Bank spec tokenizer differs from token spec - no special_tokens or max_len
            return ArtifactSpec(
                task="seq2seq",
                dataset_id=self.dataset,
                split="train+val",
                subset=self._build_seq2seq_data_signature(),
                tokenizer={
                    "type": self.tokenizer,
                    "tokenizer_config": _default_tokenizer_config(self.tokenizer, int(self.max_len)),
                    "tokens_fp": tokens_fp,
                },
                sequence={"max_len": int(self.max_len)},
                ska={
                    "window": int(self.window),
                    "stride": int(self.bank_stride),
                    "minhash_k": int(self.minhash_k),
                    "router_topk": int(self.router_topk),
                    "backend": self.ska_backend,
                    "precision": self.precision,
                },
                model={
                    "d_model": int(self.d_model),
                    "heads": int(self.nhead),
                    "layers": int(self.layers),
                    "precision": self.precision,
                },
                routing_depends_on_learned_params=routing_depends,
            ).to_dict()
        else:
            raise ValueError(f"Unknown task for spec building: {self.task}")

    def get_expected_fingerprints(self) -> tuple[str, Optional[str]]:
        """Get expected token and bank fingerprints."""
        token_spec = self.build_token_spec()
        token_fp = fingerprint(token_spec)
        if self.cache_mode == "tokens":
            return token_fp, None
        bank_spec = self.build_bank_spec(token_fp)
        bank_fp = fingerprint(bank_spec)
        return token_fp, bank_fp


def _param_value(params: dict[str, Any], key: str) -> Any:
    raw = params.get(key)
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "value" in raw:
            return raw["value"]
        if "values" in raw:
            return raw["values"]
    return raw


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _length_list(params: dict[str, Any], sweep_key: str, single_key: str) -> list[Any]:
    lengths = _as_list(_param_value(params, sweep_key))
    if lengths:
        return lengths
    return _as_list(_param_value(params, single_key))


def _run(cmd: list[str], dry_run: bool) -> int:
    print("[cache_yaml]", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _normalize_tasks(tasks: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for task in tasks:
        if task == "seq":
            normalized.append("seq2seq")
        else:
            normalized.append(task)
    return normalized


def _add_job(job: CacheJob, jobs: list[CacheJob], seen_any: set[tuple[Any, ...]], seen_full: set[tuple[Any, ...]]) -> None:
    base = job.key_base()
    existing = next((j for j in jobs if j.key_base() == base), None)
    if job.cache_mode == "full":
        if base in seen_full:
            if existing and job.score_mode and existing.score_mode != job.score_mode:
                existing.score_mode = "shared"
            if existing:
                existing.sources = sorted(set(existing.sources + job.sources))
            return
        seen_full.add(base)
        if base in seen_any:
            jobs[:] = [j for j in jobs if j.key_base() != base]
            seen_any.discard(base)
        jobs.append(job)
        seen_any.add(base)
        return
    if base in seen_full or base in seen_any:
        if existing and job.score_mode and existing.score_mode != job.score_mode:
            existing.score_mode = "shared"
        if existing:
            existing.sources = sorted(set(existing.sources + job.sources))
        return
    jobs.append(job)
    seen_any.add(base)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build/verify caches from sweep YAMLs.")
    ap.add_argument("--yaml", nargs="+", required=True, help="Sweep YAML files.")
    ap.add_argument("--tasks", nargs="*", default=None, help="Override tasks (lm/seq2seq/textdiff/vit).")
    ap.add_argument("--cache-mode", choices=["none", "tokens", "full"], default=None)
    ap.add_argument("--artifact-cache-root", type=str, default="")
    ap.add_argument("--overwrite-cache", action="store_true")
    ap.add_argument("--cache-device", type=str, default="cpu", help="Device for cache builds (cpu/cuda).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="Only verify caches; do not build missing.")
    ap.add_argument("--prefetch", action="store_true", help="Run HF dataset prefetch when cache is missing.")
    ap.add_argument("--report-path", type=str, default="", help="Optional CSV report path.")
    args = ap.parse_args()

    yaml_paths = [Path(p) for p in args.yaml]
    for yaml_path in yaml_paths:
        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing YAML: {yaml_path}")

    jobs: list[CacheJob] = []
    seen_full: set[tuple[Any, ...]] = set()
    seen_any: set[tuple[Any, ...]] = set()

    missing_subsets: list[str] = []
    wikitext_needed: set[str] = set()
    wmt16_needed = False
    need_vit = False

    for yaml_path in yaml_paths:
        cfg = yaml.safe_load(yaml_path.read_text())
        params = cfg.get("parameters", {}) if isinstance(cfg, dict) else {}

        yaml_cache_mode = args.cache_mode or _param_value(params, "cache-mode") or "none"
        model_types = _as_list(_param_value(params, "model-type"))
        score_modes = _as_list(_param_value(params, "ska-score-mode"))
        model_dim = _param_value(params, "model-dim")
        model_heads = _param_value(params, "model-heads")
        model_layers = _param_value(params, "model-layers")
        adapter_rank = _param_value(params, "adapter-rank")
        if not score_modes:
            score_modes = [None]
        elif len(score_modes) > 1:
            score_modes = ["shared"]
        ska_backend = _param_value(params, "ska-backend")

        if args.tasks is not None:
            tasks = _normalize_tasks(args.tasks)
        else:
            task_param = _param_value(params, "tasks")
            tasks = _normalize_tasks(_as_list(task_param)) if task_param else DEFAULT_TASKS

        if yaml_cache_mode == "none":
            continue

        need_ska = "ska" in model_types if model_types else True
        cache_mode = yaml_cache_mode
        if cache_mode == "full" and not need_ska:
            cache_mode = "tokens"

        for task in tasks:
            if task == "vit":
                need_vit = True
                continue
            if task == "lm":
                lengths = _length_list(params, "lm-lengths", "lm-seq-len")
                dataset = _param_value(params, "lm-dataset")
                subset = _param_value(params, "lm-subset-path")
                precision = _param_value(params, "lm-precision")
                seq_stride = _param_value(params, "lm-seq-stride")
                window = _param_value(params, "lm-window")
                stride = _param_value(params, "lm-stride")
                minhash_k = _param_value(params, "lm-minhash-k")
                router_topk = _param_value(params, "lm-router-topk")
                if subset and not Path(subset).exists():
                    missing_subsets.append(str(subset))
                if dataset in WIKITEXT_CONFIGS:
                    wikitext_needed.add(dataset)
                for seq_len in lengths:
                    lm_stride = (
                        int(seq_stride)
                        if seq_stride is not None and int(seq_stride) > 0
                        else int(seq_len)
                    )
                    for score_mode in score_modes:
                        job = CacheJob(
                            task="lm",
                            cache_mode=cache_mode,
                            dataset=str(dataset),
                            subset_path=str(subset) if subset else None,
                            precision=str(precision),
                            score_mode=score_mode,
                            ska_backend=str(ska_backend) if ska_backend else None,
                            seq_len=int(seq_len),
                            seq_stride=lm_stride,
                            window=int(window),
                            bank_stride=int(stride),
                            minhash_k=int(minhash_k),
                            router_topk=int(router_topk),
                            sources=[yaml_path.name],
                            # Model defaults for LM
                            d_model=model_dim,
                            nhead=model_heads,
                            layers=model_layers,
                            dataset_lines=0,
                            hf_tokenizer_name="",
                            adapter_rank=int(adapter_rank) if adapter_rank is not None else 0,
                        )
                        _add_job(job, jobs, seen_any, seen_full)
            elif task == "seq2seq":
                lengths = _length_list(params, "seq-lengths", "seq-max-len")
                dataset = _param_value(params, "seq-dataset")
                subset = _param_value(params, "seq-subset-path")
                precision = _param_value(params, "seq-precision")
                tokenizer = _param_value(params, "seq-tokenizer-type")
                window = _param_value(params, "seq-window")
                stride = _param_value(params, "seq-stride")
                minhash_k = _param_value(params, "seq-minhash-k")
                router_topk = _param_value(params, "seq-router-topk")
                if subset and not Path(subset).exists():
                    missing_subsets.append(str(subset))
                wmt16_needed = True
                for max_len in lengths:
                    for score_mode in score_modes:
                        job = CacheJob(
                            task="seq2seq",
                            cache_mode=cache_mode,
                            dataset=str(dataset),
                            subset_path=str(subset) if subset else None,
                            precision=str(precision),
                            score_mode=score_mode,
                            ska_backend=str(ska_backend) if ska_backend else None,
                            max_len=int(max_len),
                            window=int(window),
                            bank_stride=int(stride),
                            minhash_k=int(minhash_k),
                            router_topk=int(router_topk),
                            tokenizer=str(tokenizer),
                            sources=[yaml_path.name],
                            d_model=model_dim,
                            nhead=model_heads,
                            layers=model_layers,
                            adapter_rank=int(adapter_rank) if adapter_rank is not None else 0,
                        )
                        _add_job(job, jobs, seen_any, seen_full)
            elif task == "textdiff":
                lengths = _length_list(params, "textdiff-lengths", "textdiff-seq-len")
                dataset = _param_value(params, "textdiff-dataset")
                subset = _param_value(params, "textdiff-subset-path")
                precision = _param_value(params, "textdiff-precision")
                seq_stride = _param_value(params, "textdiff-stride")
                window = _param_value(params, "textdiff-window")
                bank_stride = _param_value(params, "textdiff-bank-stride")
                minhash_k = _param_value(params, "textdiff-minhash-k")
                router_topk = _param_value(params, "textdiff-router-topk")
                if subset and not Path(subset).exists():
                    missing_subsets.append(str(subset))
                if dataset in WIKITEXT_CONFIGS:
                    wikitext_needed.add(dataset)
                for seq_len in lengths:
                    for score_mode in score_modes:
                        job = CacheJob(
                            task="textdiff",
                            cache_mode=cache_mode,
                            dataset=str(dataset),
                            subset_path=str(subset) if subset else None,
                            precision=str(precision),
                            score_mode=score_mode,
                            ska_backend=str(ska_backend) if ska_backend else None,
                            seq_len=int(seq_len),
                            seq_stride=int(seq_stride),
                            window=int(window),
                            bank_stride=int(bank_stride),
                            minhash_k=int(minhash_k),
                            router_topk=int(router_topk),
                            sources=[yaml_path.name],
                            d_model=model_dim,
                            nhead=model_heads,
                            layers=model_layers,
                            adapter_rank=int(adapter_rank) if adapter_rank is not None else 0,
                        )
                        _add_job(job, jobs, seen_any, seen_full)

    if missing_subsets:
        missing = sorted(set(missing_subsets))
        raise RuntimeError("Missing subset files: " + ", ".join(missing))

    _check_hf_cache(wikitext_needed, wmt16_needed, args.prefetch)
    if need_vit:
        _check_vit_data()

    if not jobs:
        print("[cache_yaml] No cache jobs to run.")
        return 0

    report_rows: list[dict[str, Any]] = []
    hf_root = resolve_hf_root(args.artifact_cache_root or None)
    artifact_roots = [hf_root]
    if LEGACY_HF_ROOT != hf_root and (LEGACY_HF_ROOT / "artifacts").exists():
        artifact_roots.append(LEGACY_HF_ROOT)
        print(f"[cache_yaml] Using legacy artifact root: {LEGACY_HF_ROOT}")

    for job in jobs:
        hit_root = _find_cache_root(job, hf_root)
        status = "HIT" if hit_root else "MISS"

        if status == "MISS" and not args.verify_only:
            cmd = _build_cache_cmd(job, args.artifact_cache_root, args.overwrite_cache, args.cache_device)
            rc = _run(cmd, args.dry_run)
            if rc != 0:
                status = "ERROR"
            else:
                hit_root = _find_cache_root(job, hf_root)
                status = "BUILT" if hit_root else "MISS"

        report_rows.append(_job_report(job, status, hit_root))

    _print_report(report_rows)
    if args.report_path:
        _write_report(Path(args.report_path), report_rows)

    if args.verify_only and any(r["status"] == "MISS" for r in report_rows):
        return 2
    return 0


def _check_hf_cache(wikitext_needed: set[str], wmt16_needed: bool, prefetch: bool) -> None:
    if not wikitext_needed and not wmt16_needed:
        return
    cache_dir = ensure_hf_cache(None)
    missing_wikitext: list[str] = []
    missing_wmt16 = False

    if load_dataset is None:
        print("[cache_yaml] datasets not installed; skipping HF cache checks.")
        return

    for ds in sorted(wikitext_needed):
        config = WIKITEXT_CONFIGS.get(ds)
        if not config:
            continue
        if not _hf_cached("wikitext", config, cache_dir):
            missing_wikitext.append(ds)

    if wmt16_needed and not _hf_cached("wmt16", "ro-en", cache_dir):
        missing_wmt16 = True

    if missing_wikitext and prefetch:
        cmd = [sys.executable, "scripts/prefetch_datasets.py", "--datasets"] + missing_wikitext
        rc = _run(cmd, dry_run=False)
        if rc == 0:
            missing_wikitext = [
                ds for ds in missing_wikitext if not _hf_cached("wikitext", WIKITEXT_CONFIGS[ds], cache_dir)
            ]
        else:
            print("[cache_yaml] Prefetch failed; keeping cache-miss warnings.")

    if missing_wikitext:
        print("[cache_yaml] Missing HF cache for:", ", ".join(missing_wikitext))
    if missing_wmt16:
        print("[cache_yaml] Missing HF cache for: wmt16/ro-en (will download at runtime).")


def _hf_cached(name: str, config: str, cache_dir: Path) -> bool:
    try:
        load_dataset(name, config, cache_dir=str(cache_dir), download_mode="reuse_dataset_if_exists", trust_remote_code=True)
    except Exception:
        # Fallback: coarse on-disk check to avoid false negatives.
        dataset_root = cache_dir / name
        if dataset_root.exists() and any(dataset_root.iterdir()):
            return True
        config_root = dataset_root / config
        if config_root.exists() and any(config_root.iterdir()):
            return True
        return False
    return True


def _check_vit_data() -> None:
    data_root = resolve_data_root(None)
    vision_root = data_root / "vision" / "cifar10"
    if (vision_root / "cifar-10-batches-py").exists():
        return
    if vision_root.exists() and any(vision_root.iterdir()):
        return
    print(f"[cache_yaml] CIFAR-10 data not found under {vision_root}. torchvision will download on first run.")


def _find_cache_root(job: CacheJob, hf_root: Path) -> Optional[Path]:
    """Find cache using fingerprint-based lookup (matches runtime behavior exactly)."""
    required = ["tokens.pt"] if job.cache_mode == "tokens" else FULL_REQUIRED.get(job.task, [])

    try:
        token_fp, bank_fp = job.get_expected_fingerprints()
    except Exception as e:
        print(f"[cache_yaml] WARNING: Could not compute fingerprint for {job.task}: {e}")
        return None

    base = hf_root / "artifacts" / job.task / job.dataset

    if job.cache_mode == "tokens":
        token_dir = base / token_fp
        if token_dir.exists() and all((token_dir / name).exists() for name in required):
            return token_dir
        return None

    # For full mode, check bank directory
    if bank_fp:
        bank_dir = base / bank_fp
        if bank_dir.exists() and all((bank_dir / name).exists() for name in required):
            return bank_dir

    return None


def _meta_matches(job: CacheJob, meta: dict[str, Any]) -> bool:
    """Legacy meta matching - kept for reference but not used."""
    if meta.get("task") != job.task:
        return False
    if meta.get("dataset_id") != job.dataset:
        return False
    subset = meta.get("subset")
    if job.subset_sig:
        if isinstance(subset, dict) and "subset" in subset:
            subset = subset.get("subset")
        if not isinstance(subset, dict):
            return False
        if subset.get("sha256") != job.subset_sig.get("sha256"):
            return False
    seq = meta.get("sequence", {})
    def _norm_int(value: Optional[int]) -> int:
        return -1 if value is None else int(value)
    if job.task in ("lm", "textdiff"):
        if int(seq.get("seq_len", -1)) != _norm_int(job.seq_len):
            return False
        if int(seq.get("stride", -1)) != _norm_int(job.seq_stride):
            return False
    if job.task == "seq2seq":
        if int(seq.get("max_len", -1)) != _norm_int(job.max_len):
            return False
    ska = meta.get("ska", {})
    if int(ska.get("window", -1)) != _norm_int(job.window):
        return False
    if int(ska.get("stride", -1)) != _norm_int(job.bank_stride):
        return False
    if int(ska.get("minhash_k", -1)) != _norm_int(job.minhash_k):
        return False
    if int(ska.get("router_topk", -1)) != _norm_int(job.router_topk):
        return False
    if job.ska_backend and "backend" in ska and ska.get("backend") != job.ska_backend:
        return False
    if job.precision:
        meta_precision = ska.get("precision")
        if meta_precision is None:
            meta_precision = meta.get("model", {}).get("precision")
        if meta_precision is not None and meta_precision != job.precision:
            return False
    if job.task == "seq2seq" and job.tokenizer:
        tokenizer = meta.get("tokenizer", {})
        if tokenizer.get("type") != job.tokenizer:
            return False
    return True


def _build_cache_cmd(job: CacheJob, artifact_cache_root: str, overwrite_cache: bool, cache_device: str) -> list[str]:
    cache_script = "scripts/cache_ska_artifacts.py" if job.cache_mode == "full" else "scripts/cache_tokens.py"
    cmd = [sys.executable, cache_script, "--task", job.task]

    if job.task == "lm":
        cmd += [
            "--lm-dataset",
            job.dataset,
            "--lm-seq-len",
            str(job.seq_len),
            "--lm-seq-stride",
            str(job.seq_stride),
            "--lm-precision",
            job.precision,
            "--lm-window",
            str(job.window),
            "--lm-stride",
            str(job.bank_stride),
            "--lm-minhash-k",
            str(job.minhash_k),
            "--lm-router-topk",
            str(job.router_topk),
        ]
        if job.subset_path:
            cmd += ["--lm-subset-path", job.subset_path]
    elif job.task == "seq2seq":
        cmd += [
            "--seq-dataset",
            job.dataset,
            "--seq-max-len",
            str(job.max_len),
            "--seq-tokenizer-type",
            str(job.tokenizer),
            "--seq-precision",
            job.precision,
            "--seq-window",
            str(job.window),
            "--seq-stride",
            str(job.bank_stride),
            "--seq-minhash-k",
            str(job.minhash_k),
            "--seq-router-topk",
            str(job.router_topk),
        ]
        if job.subset_path:
            cmd += ["--seq-subset-path", job.subset_path]
    elif job.task == "textdiff":
        cmd += [
            "--textdiff-dataset",
            job.dataset,
            "--textdiff-seq-len",
            str(job.seq_len),
            "--textdiff-stride",
            str(job.seq_stride),
            "--textdiff-precision",
            job.precision,
            "--textdiff-window",
            str(job.window),
            "--textdiff-bank-stride",
            str(job.bank_stride),
            "--textdiff-minhash-k",
            str(job.minhash_k),
            "--textdiff-router-topk",
            str(job.router_topk),
        ]
        if job.subset_path:
            cmd += ["--textdiff-subset-path", job.subset_path]

    if job.ska_backend:
        cmd += ["--ska-backend", job.ska_backend]
    if job.cache_mode == "full":
        cmd += ["--model-type", "ska"]
    if cache_device:
        cmd += ["--device", cache_device]
    if artifact_cache_root:
        cmd += ["--artifact-cache-root", artifact_cache_root]
    if overwrite_cache:
        cmd.append("--overwrite-cache")
    return cmd


def _job_report(job: CacheJob, status: str, root: Optional[Path]) -> dict[str, Any]:
    token_fp, bank_fp = job.get_expected_fingerprints()
    return {
        "task": job.task,
        "cache_mode": job.cache_mode,
        "dataset": job.dataset,
        "subset_path": job.subset_path or "",
        "precision": job.precision,
        "score_mode": job.score_mode or "",
        "ska_backend": job.ska_backend or "",
        "seq_len": job.seq_len or "",
        "seq_stride": job.seq_stride or "",
        "max_len": job.max_len or "",
        "window": job.window or "",
        "bank_stride": job.bank_stride or "",
        "minhash_k": job.minhash_k or "",
        "router_topk": job.router_topk or "",
        "tokenizer": job.tokenizer or "",
        "sources": ",".join(sorted(set(job.sources))),
        "status": status,
        "expected_token_fp": token_fp,
        "expected_bank_fp": bank_fp or "",
        "cache_root": str(root) if root else "",
    }


def _print_report(rows: list[dict[str, Any]]) -> None:
    print("[cache_yaml] summary:")
    for row in rows:
        fp_info = f"token_fp={row['expected_token_fp'][:8]}"
        if row['expected_bank_fp']:
            fp_info += f" bank_fp={row['expected_bank_fp'][:8]}"
        print(
            f"  {row['task']} mode={row['cache_mode']} "
            f"len={row['seq_len'] or row['max_len'] or '-'} "
            f"precision={row['precision']} "
            f"score={row['score_mode'] or '-'} "
            f"{fp_info} "
            f"status={row['status']}"
        )


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



if __name__ == "__main__":
    raise SystemExit(main())
