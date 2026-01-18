#!/usr/bin/env python3
"""Build/verify token + SKA caches for Stage A/B sweeps using sweep YAMLs."""
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
    from set_attention.data.artifact_cache import file_signature, resolve_hf_root
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


class CacheJob:
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
        self.tokenizer = tokenizer
        self.sources = sources or []
        self.subset_sig = file_signature(Path(subset_path)) if subset_path else None

    def key_base(self) -> tuple[Any, ...]:
        return (
            self.task,
            self.dataset,
            self.precision,
            self.score_mode,
            self.ska_backend,
            self.seq_len,
            self.seq_stride,
            self.max_len,
            self.window,
            self.bank_stride,
            self.minhash_k,
            self.router_topk,
            self.tokenizer,
            self.subset_sig["sha256"] if self.subset_sig else None,
        )


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
    if job.cache_mode == "full":
        if base in seen_full:
            return
        seen_full.add(base)
        if base in seen_any:
            jobs[:] = [j for j in jobs if j.key_base() != base]
            seen_any.discard(base)
        jobs.append(job)
        seen_any.add(base)
        return
    if base in seen_full or base in seen_any:
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
        if not score_modes:
            score_modes = [None]
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
                            seq_stride=int(seq_stride),
                            window=int(window),
                            bank_stride=int(stride),
                            minhash_k=int(minhash_k),
                            router_topk=int(router_topk),
                            sources=[yaml_path.name],
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

    for job in jobs:
        hit_root = _find_cache_root(job, hf_root)
        status = "HIT" if hit_root else "MISS"

        if status == "MISS" and not args.verify_only:
            cmd = _build_cache_cmd(job, args.artifact_cache_root, args.overwrite_cache)
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
        _run(cmd, dry_run=False)
        missing_wikitext = [ds for ds in missing_wikitext if not _hf_cached("wikitext", WIKITEXT_CONFIGS[ds], cache_dir)]

    if missing_wikitext:
        print("[cache_yaml] Missing HF cache for:", ", ".join(missing_wikitext))
    if missing_wmt16:
        print("[cache_yaml] Missing HF cache for: wmt16/ro-en (will download at runtime).")


def _hf_cached(name: str, config: str, cache_dir: Path) -> bool:
    try:
        load_dataset(name, config, cache_dir=str(cache_dir), download_mode="reuse_dataset_if_exists", local_files_only=True)
    except Exception:
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
    base = hf_root / "artifacts" / job.task / job.dataset
    if not base.exists():
        return None
    required = ["tokens.pt"] if job.cache_mode == "tokens" else FULL_REQUIRED[job.task]
    for fp_dir in base.iterdir():
        meta_path = fp_dir / "meta.json"
        if not fp_dir.is_dir() or not meta_path.exists():
            continue
        try:
            meta = yaml.safe_load(meta_path.read_text())
        except Exception:
            continue
        if not _meta_matches(job, meta):
            continue
        if all((fp_dir / name).exists() for name in required):
            return fp_dir
    return None


def _meta_matches(job: CacheJob, meta: dict[str, Any]) -> bool:
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
    if job.task in ("lm", "textdiff"):
        if int(seq.get("seq_len", -1)) != int(job.seq_len or -1):
            return False
        if int(seq.get("stride", -1)) != int(job.seq_stride or -1):
            return False
    if job.task == "seq2seq":
        if int(seq.get("max_len", -1)) != int(job.max_len or -1):
            return False
    ska = meta.get("ska", {})
    if int(ska.get("window", -1)) != int(job.window or -1):
        return False
    if int(ska.get("stride", -1)) != int(job.bank_stride or -1):
        return False
    if int(ska.get("minhash_k", -1)) != int(job.minhash_k or -1):
        return False
    if int(ska.get("router_topk", -1)) != int(job.router_topk or -1):
        return False
    if job.ska_backend and ska.get("backend") != job.ska_backend:
        return False
    if job.precision and ska.get("precision") != job.precision:
        return False
    if job.score_mode and ska.get("score_mode") != job.score_mode:
        return False
    if job.task == "seq2seq" and job.tokenizer:
        tokenizer = meta.get("tokenizer", {})
        if tokenizer.get("type") != job.tokenizer:
            return False
    return True


def _build_cache_cmd(job: CacheJob, artifact_cache_root: str, overwrite_cache: bool) -> list[str]:
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
    if job.score_mode:
        cmd += ["--ska-score-mode", job.score_mode, "--model-type", "ska"]
    if artifact_cache_root:
        cmd += ["--artifact-cache-root", artifact_cache_root]
    if overwrite_cache:
        cmd.append("--overwrite-cache")
    return cmd


def _job_report(job: CacheJob, status: str, root: Optional[Path]) -> dict[str, Any]:
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
        "cache_root": str(root) if root else "",
    }


def _print_report(rows: list[dict[str, Any]]) -> None:
    print("[cache_yaml] summary:")
    for row in rows:
        print(
            f"  {row['task']} mode={row['cache_mode']} "
            f"len={row['seq_len'] or row['max_len'] or '-'} "
            f"score={row['score_mode'] or '-'} "
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
