"""Artifact cache helpers for token/bank/routing reuse."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Iterable


BANK_FORMAT_VERSION = "1"


@dataclass
class ArtifactSpec:
    task: str
    dataset_id: str
    split: str
    subset: Optional[Dict[str, Any]] = None
    tokenizer: Optional[Dict[str, Any]] = None
    sequence: Optional[Dict[str, Any]] = None
    ska: Optional[Dict[str, Any]] = None
    model: Optional[Dict[str, Any]] = None
    routing_depends_on_learned_params: Optional[bool] = None
    bank_format_version: str = BANK_FORMAT_VERSION
    code_version: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime": int(stat.st_mtime),
        "sha256": h.hexdigest(),
    }


def _normalize_spec(spec: Dict[str, Any], exclude: Iterable[str]) -> Dict[str, Any]:
    drop = set(exclude)
    return {k: v for k, v in spec.items() if k not in drop}


def fingerprint(spec: Dict[str, Any], exclude: Iterable[str] = ("code_version", "created_at")) -> str:
    normalized = _normalize_spec(spec, exclude)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hash_bytes(payload)[:16]


def resolve_hf_root(user_root: Optional[str] = None) -> Path:
    if user_root:
        return Path(user_root).expanduser()
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home).expanduser()
    env_ds = os.environ.get("HF_DATASETS_CACHE")
    if env_ds:
        return Path(env_ds).expanduser().parent
    return Path("~/.hf").expanduser()


def artifact_root(spec: Dict[str, Any], hf_root: Optional[Path] = None) -> Path:
    root = hf_root or resolve_hf_root()
    fp = fingerprint(spec)
    return root / "artifacts" / spec["task"] / spec["dataset_id"] / fp


def write_meta(root: Path, spec: Dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "meta.json"
    payload = dict(spec)
    if payload.get("created_at") is None:
        payload["created_at"] = None
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_meta(root: Path) -> Dict[str, Any]:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {root}")
    return json.loads(meta_path.read_text())


def assert_meta_compatible(root: Path, spec: Dict[str, Any]) -> None:
    existing = read_meta(root)
    want_fp = fingerprint(spec)
    have_fp = fingerprint(existing)
    if want_fp != have_fp:
        raise RuntimeError(
            "Cache mismatch: artifact metadata does not match the requested spec. "
            f"expected fp={want_fp} got fp={have_fp} ({root})"
        )


def require_cached_artifacts(root: Path, paths: Iterable[Path], task: str) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing bank+routing artifacts for {task}. "
            "Run once with --precompute-bank to create them. "
            f"root={root} missing=[{missing_str}]"
        )


__all__ = [
    "ArtifactSpec",
    "BANK_FORMAT_VERSION",
    "artifact_root",
    "assert_meta_compatible",
    "file_signature",
    "fingerprint",
    "read_meta",
    "resolve_hf_root",
    "write_meta",
    "require_cached_artifacts",
]
