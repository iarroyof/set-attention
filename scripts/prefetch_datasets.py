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


def main():
    ap = argparse.ArgumentParser(description="Prefetch datasets into HF cache for offline reuse.")
    ap.add_argument("--cache-dir", type=str, default="", help="HF cache root; empty uses HF_HOME/HF_DATASETS_CACHE.")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext2", "wikitext103"],
        choices=["wikitext2", "wikitext103"],
        help="Datasets to prefetch.",
    )
    args = ap.parse_args()

    cache_dir = ensure_hf_cache(args.cache_dir)
    print_env()
    dns_check()
    force_online()
    print("[prefetch] using cache:", cache_dir)

    name_map = {"wikitext2": "wikitext-2-raw-v1", "wikitext103": "wikitext-103-raw-v1"}
    for ds in args.datasets:
        cfg = name_map[ds]
        print(f"[prefetch] downloading {ds} ({cfg})...")
        load_dataset("wikitext", cfg, cache_dir=str(cache_dir), download_mode="reuse_dataset_if_exists")
        print(f"[prefetch] done {ds}")
    print("[prefetch] completed.")


if __name__ == "__main__":
    main()
