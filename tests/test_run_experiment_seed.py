from __future__ import annotations

import torch

from scripts.run_experiment import apply_training_seed


def _initialized_weight(seed: int) -> tuple[torch.Tensor, dict]:
    cfg = {
        "training": {
            "seed": seed,
            "deterministic": True,
            "benchmark_mode": False,
        }
    }
    apply_training_seed(cfg)
    return torch.nn.Linear(8, 4).weight.detach().clone(), cfg


def test_same_seed_reproduces_initial_parameters() -> None:
    first, _ = _initialized_weight(17)
    second, _ = _initialized_weight(17)
    assert torch.equal(first, second)


def test_different_seed_changes_initial_parameters() -> None:
    first, _ = _initialized_weight(17)
    second, _ = _initialized_weight(18)
    assert not torch.equal(first, second)


def test_applied_seed_provenance_is_explicit() -> None:
    _, cfg = _initialized_weight(23)
    assert cfg["training"]["seed_applied"] is True
    assert cfg["training"]["applied_seed"] == 23
    assert cfg["training"]["torch_initial_seed"] == 23
    assert cfg["resolved"]["requested_seed"] == 23
    assert cfg["resolved"]["applied_seed"] == 23


def test_missing_seed_fails_closed() -> None:
    try:
        apply_training_seed({"training": {}})
    except ValueError as exc:
        assert "training.seed is required" in str(exc)
    else:
        raise AssertionError("missing training.seed did not fail closed")
