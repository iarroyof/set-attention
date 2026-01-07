"""Tokenizer cache helpers to ensure deterministic reuse across runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _file_signature(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    h = hashlib.sha256()
    h.update(str(p).encode("utf-8"))
    h.update(str(stat.st_size).encode("utf-8"))
    h.update(str(int(stat.st_mtime)).encode("utf-8"))
    try:
        with p.open("rb") as handle:
            h.update(handle.read(65536))
    except Exception:
        pass
    return {
        "path": str(p),
        "exists": True,
        "size": int(stat.st_size),
        "mtime": int(stat.st_mtime),
        "hash": h.hexdigest()[:12],
    }


def tokenizer_fingerprint(
    tokenizer_type: str,
    tokenizer_config: Dict[str, Any],
    dataset_id: str,
    max_len: int,
    limit: Optional[int] = None,
    src_path: Optional[str] = None,
    tgt_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "tokenizer_type": tokenizer_type,
        "tokenizer_config": tokenizer_config,
        "dataset_id": dataset_id,
        "max_len": int(max_len),
        "limit": int(limit) if limit is not None else None,
        "src_sig": _file_signature(src_path),
        "tgt_sig": _file_signature(tgt_path),
        "extra": extra or {},
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:12]


def default_tokenizer_dir(hf_home: Path, dataset_id: str, tokenizer_type: str, fingerprint: str) -> Path:
    return hf_home / "tokenizers" / dataset_id / tokenizer_type / fingerprint


def load_tokenizer_meta(tokenizer_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = tokenizer_dir / "tokenizer_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def save_tokenizer_meta(tokenizer_dir: Path, meta: Dict[str, Any]) -> None:
    meta_path = tokenizer_dir / "tokenizer_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def meta_matches(tokenizer_dir: Path, expected: Dict[str, Any]) -> bool:
    existing = load_tokenizer_meta(tokenizer_dir)
    if not existing:
        return False
    return existing.get("fingerprint") == expected.get("fingerprint")


__all__ = [
    "default_tokenizer_dir",
    "meta_matches",
    "save_tokenizer_meta",
    "tokenizer_fingerprint",
]
