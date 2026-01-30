#!/usr/bin/env python3
"""
Prefetch HuggingFace datasets into the shared cache so downstream scripts can run offline.
"""
import argparse
import os
import socket
from pathlib import Path

from datasets import load_dataset
from set_attention.data.hf_cache import ensure_hf_cache


def print_env():
    keys = [
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HF_HUB_CACHE",
    ]
    print("[env] current environment:")
    for k in keys:
        print(f"  {k}={os.environ.get(k)}")


def force_online():
    for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if k in os.environ:
            os.environ.pop(k)
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    print(f"[env] offline flags cleared; HF_HUB_ENABLE_HF_TRANSFER={os.environ['HF_HUB_ENABLE_HF_TRANSFER']} set.")


def dns_check():
    hosts = ["huggingface.co", "cdn-lfs.huggingface.co"]
    for host in hosts:
        try:
            ip = socket.gethostbyname(host)
            print(f"[dns] {host} -> {ip}")
        except Exception as exc:  # pragma: no cover
            print(f"[dns] {host} DNS FAIL: {exc}")


def _set_offline_mode(enable: bool) -> None:
    if enable:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            if k in os.environ:
                os.environ.pop(k)


def _try_cached_first(name: str, cfg: str, cache_dir: Path) -> bool:
    _set_offline_mode(True)
    try:
        load_dataset(name, cfg, cache_dir=str(cache_dir), download_mode="reuse_dataset_if_exists")
        return True
    except Exception:
        return False
    finally:
        _set_offline_mode(False)


def _has_cache_marker(name: str, cfg: str, cache_dir: Path) -> bool:
    dataset_dir = cache_dir / name.replace("/", "___")
    if dataset_dir.exists():
        return True
    pattern = f"*{name.replace('/', '___')}_{cfg}_*.lock"
    return any(cache_dir.glob(pattern))


def main():
    ap = argparse.ArgumentParser(description="Prefetch datasets into HF cache for offline reuse.")
    ap.add_argument("--cache-dir", type=str, default="", help="HF cache root; empty uses HF_HOME/HF_DATASETS_CACHE.")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Only use cached datasets; do not attempt network access.",
    )
    ap.add_argument(
        "--force-online",
        action="store_true",
        help="Force network access even if cache exists.",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext2", "wikitext103", "cnn_dailymail", "wmt14_fr_en", "opus_books_en_fr"],
        choices=["wikitext2", "wikitext103", "cnn_dailymail", "wmt14_fr_en", "opus_books_en_fr"],
        help="Datasets to prefetch.",
    )
    args = ap.parse_args()

    cache_dir = ensure_hf_cache(args.cache_dir)
    print_env()
    dns_check()
    if args.force_online:
        force_online()
    elif args.offline:
        _set_offline_mode(True)
        print("[env] offline mode enabled for cached-only loads.")
    print("[prefetch] using cache:", cache_dir)

    name_map = {
        "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
        "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
        "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0"),
        "wmt14_fr_en": ("wmt/wmt14", "fr-en"),
        "opus_books_en_fr": ("opus_books", "en-fr"),
    }
    for ds in args.datasets:
        name, cfg = name_map[ds]
        print(f"[prefetch] downloading {ds} ({name}:{cfg})...")
        if not args.force_online and _has_cache_marker(name, cfg, cache_dir):
            print(f"[prefetch] cached {ds} (marker)")
            continue
        if not args.force_online and not args.offline:
            if _try_cached_first(name, cfg, cache_dir):
                print(f"[prefetch] cached {ds}")
                continue
        try:
            load_dataset(name, cfg, cache_dir=str(cache_dir), download_mode="reuse_dataset_if_exists")
        except Exception as exc:
            if args.offline:
                raise
            print(f"[prefetch] cache load failed for {ds}: {exc}")
            print("[prefetch] retrying with forced online mode...")
            force_online()
            load_dataset(name, cfg, cache_dir=str(cache_dir), download_mode="reuse_dataset_if_exists")
        print(f"[prefetch] done {ds}")
    print("[prefetch] completed.")


if __name__ == "__main__":
    main()
