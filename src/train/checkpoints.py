from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_SCHEMA = "set_attention_checkpoint_v1"


class CheckpointCompatibilityError(ValueError):
    pass


def stable_config_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_commit() -> str | None:
    configured = os.environ.get("SET_ATTENTION_SOURCE_COMMIT")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_loader_states(
    loaders: Mapping[str, Any] | None,
) -> dict[str, torch.Tensor]:
    states: dict[str, torch.Tensor] = {}
    for name, loader in (loaders or {}).items():
        generator = getattr(loader, "generator", None)
        if isinstance(generator, torch.Generator):
            states[str(name)] = generator.get_state()
    return states


def restore_loader_states(
    states: Mapping[str, torch.Tensor],
    loaders: Mapping[str, Any] | None,
) -> None:
    for name, state in states.items():
        loader = (loaders or {}).get(name)
        generator = getattr(loader, "generator", None)
        if not isinstance(generator, torch.Generator):
            raise CheckpointCompatibilityError(
                f"loader {name!r} has no restorable torch.Generator"
            )
        generator.set_state(state)


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    config_fingerprint: str,
    dataset_provenance: Mapping[str, Any],
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    loaders: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    training = config.get("training", {})
    model_config = config.get("model", {})
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model_state": model.state_dict(),
        "model_config": dict(model_config),
        "model_config_digest": stable_config_digest(model_config),
        "resolved_config": dict(config),
        "config_fingerprint": str(config_fingerprint),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "requested_seed": int(training["seed"]),
        "applied_seed": int(training["applied_seed"]),
        "torch_initial_seed": int(training["torch_initial_seed"]),
        "dataset_provenance": dict(dataset_provenance),
        "dataset_digest": str(dataset_provenance["dataset_digest"]),
        "tokenizer_digest": str(dataset_provenance["tokenizer_digest"]),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng_state(),
        "loader_states": capture_loader_states(loaders),
        "source_commit": source_commit(),
    }


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(payload: Mapping[str, Any], path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    digest = sha256_file(destination)
    _atomic_write_bytes(
        destination.with_suffix(destination.suffix + ".sha256"),
        f"{digest}  {destination.name}\n".encode("ascii"),
    )
    return digest


def read_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint schema: {getattr(payload, 'get', lambda *_: None)('schema')!r}"
        )
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    expected_model_config: Mapping[str, Any] | None = None,
    expected_dataset_digest: str | None = None,
    expected_tokenizer_digest: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    loaders: Mapping[str, Any] | None = None,
    restore_training_state: bool = False,
) -> dict[str, Any]:
    payload = read_checkpoint(path, map_location=map_location)
    if expected_model_config is not None:
        expected = stable_config_digest(expected_model_config)
        if payload.get("model_config_digest") != expected:
            raise CheckpointCompatibilityError("model config digest mismatch")
    if expected_dataset_digest is not None:
        if payload.get("dataset_digest") != expected_dataset_digest:
            raise CheckpointCompatibilityError("dataset digest mismatch")
    if expected_tokenizer_digest is not None:
        if payload.get("tokenizer_digest") != expected_tokenizer_digest:
            raise CheckpointCompatibilityError("tokenizer digest mismatch")

    model.load_state_dict(payload["model_state"], strict=True)
    if restore_training_state:
        if optimizer is None:
            raise CheckpointCompatibilityError(
                "resume requires an optimizer for optimizer-state restoration"
            )
        if payload.get("optimizer_state") is None:
            raise CheckpointCompatibilityError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None:
            if payload.get("scheduler_state") is None:
                raise CheckpointCompatibilityError("checkpoint has no scheduler state")
            scheduler.load_state_dict(payload["scheduler_state"])
        restore_rng_state(payload["rng_state"])
        restore_loader_states(payload.get("loader_states", {}), loaders)
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CheckpointCompatibilityError",
    "build_checkpoint_payload",
    "load_checkpoint",
    "read_checkpoint",
    "save_checkpoint",
    "sha256_file",
    "source_commit",
    "stable_config_digest",
]
