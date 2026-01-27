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
        default=["wikitext2", "wikitext103", "wmt16_en_ro", "wmt16_en_es", "cnn_dailymail"],
        choices=[
            "wikitext2",
            "wikitext103",
            "wmt16_en_ro",
            "wmt16_en_es",
            "cnn_dailymail",
            "wmt14_fr_en",
        ],
        help="Datasets to prefetch.",
    )
    args = ap.parse_args()

    cache_dir = ensure_hf_cache(args.cache_dir)
    print_env()
    dns_check()
    force_online()
    print("[prefetch] using cache:", cache_dir)

    name_map = {
        "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
        "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
        "wmt16_en_ro": ("wmt16", "ro-en"),
        "wmt16_en_es": ("wmt16", "es-en"),
        "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0"),
        "wmt14_fr_en": ("wmt/wmt14", "fr-en"),
    }
    for ds in args.datasets:
        name, cfg = name_map[ds]
        print(f"[prefetch] downloading {ds} ({name}:{cfg})...")
        load_dataset(name, cfg, cache_dir=str(cache_dir), download_mode="reuse_dataset_if_exists")
        print(f"[prefetch] done {ds}")
    print("[prefetch] completed.")


if __name__ == "__main__":
    main()
