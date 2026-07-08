from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.repro import set_seed  # noqa: E402
from run_experiment import apply_training_seed  # noqa: E402


def test_strict_determinism_sets_fail_closed_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    cfg = {
        "training": {
            "seed": 9,
            "deterministic": True,
            "strict_deterministic": True,
            "benchmark_mode": False,
        }
    }
    apply_training_seed(cfg)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert cfg["training"]["cublas_workspace_config"] == ":4096:8"
    assert cfg["resolved"]["strict_deterministic"] is True


def test_invalid_cublas_workspace_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        set_seed(1, deterministic=True, strict_deterministic=True)


def test_strict_mode_requires_deterministic() -> None:
    with pytest.raises(ValueError, match="requires deterministic"):
        set_seed(1, deterministic=False, strict_deterministic=True)
