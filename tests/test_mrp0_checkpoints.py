from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train.checkpoints import (  # noqa: E402
    CheckpointCompatibilityError,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)


def _config() -> dict:
    return {
        "model": {
            "implementation": "baseline_token",
            "vocab_size": 7,
            "d_model": 4,
        },
        "training": {
            "seed": 3,
            "applied_seed": 3,
            "torch_initial_seed": 3,
        },
    }


def _provenance() -> dict:
    return {
        "dataset_digest": "dataset-sha",
        "tokenizer_digest": "tokenizer-sha",
        "vocabulary": ["a", "b"],
    }


def test_checkpoint_round_trip_reproduces_logits_and_is_immutable(
    tmp_path: Path,
) -> None:
    torch.manual_seed(3)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 3),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fixed_input = torch.randn(2, 4)
    expected = model(fixed_input).detach().clone()
    payload = build_checkpoint_payload(
        model=model,
        config=_config(),
        config_fingerprint="config-sha",
        dataset_provenance=_provenance(),
        epoch=2,
        global_step=11,
        optimizer=optimizer,
    )
    path = tmp_path / "model.pt"
    digest = save_checkpoint(payload, path)
    before = path.read_bytes()

    restored = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 3),
    )
    metadata = load_checkpoint(
        path,
        model=restored,
        expected_model_config=_config()["model"],
        expected_dataset_digest="dataset-sha",
        expected_tokenizer_digest="tokenizer-sha",
    )

    assert torch.equal(expected, restored(fixed_input))
    assert metadata["epoch"] == 2
    assert metadata["global_step"] == 11
    assert path.read_bytes() == before
    assert sha256_file(path) == digest
    assert path.with_suffix(".pt.sha256").is_file()


@pytest.mark.parametrize(
    ("dataset_digest", "tokenizer_digest", "message"),
    [
        ("wrong", "tokenizer-sha", "dataset digest mismatch"),
        ("dataset-sha", "wrong", "tokenizer digest mismatch"),
    ],
)
def test_checkpoint_digest_mismatch_fails_closed(
    tmp_path: Path,
    dataset_digest: str,
    tokenizer_digest: str,
    message: str,
) -> None:
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "model.pt"
    save_checkpoint(
        build_checkpoint_payload(
            model=model,
            config=_config(),
            config_fingerprint="config-sha",
            dataset_provenance=_provenance(),
            epoch=1,
            global_step=1,
        ),
        path,
    )
    with pytest.raises(CheckpointCompatibilityError, match=message):
        load_checkpoint(
            path,
            model=torch.nn.Linear(4, 3),
            expected_model_config=_config()["model"],
            expected_dataset_digest=dataset_digest,
            expected_tokenizer_digest=tokenizer_digest,
        )


def test_checkpoint_model_config_mismatch_fails_closed(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "model.pt"
    save_checkpoint(
        build_checkpoint_payload(
            model=model,
            config=_config(),
            config_fingerprint="config-sha",
            dataset_provenance=_provenance(),
            epoch=1,
            global_step=1,
        ),
        path,
    )
    mismatched = dict(_config()["model"])
    mismatched["d_model"] = 8
    with pytest.raises(
        CheckpointCompatibilityError,
        match="model config digest mismatch",
    ):
        load_checkpoint(
            path,
            model=torch.nn.Linear(4, 3),
            expected_model_config=mismatched,
        )


def test_resume_restores_optimizer_rng_and_loader_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(13)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()
    generator = torch.Generator().manual_seed(99)
    loader = DataLoader(
        TensorDataset(torch.arange(8)),
        batch_size=2,
        shuffle=True,
        generator=generator,
    )
    path = tmp_path / "resume.pt"
    save_checkpoint(
        build_checkpoint_payload(
            model=model,
            config=_config(),
            config_fingerprint="config-sha",
            dataset_provenance=_provenance(),
            epoch=3,
            global_step=17,
            optimizer=optimizer,
            loaders={"train": loader},
        ),
        path,
    )
    expected_random = torch.rand(4)
    expected_loader_state = generator.get_state().clone()

    restored = torch.nn.Linear(4, 3)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=9e-2)
    restored_loader = DataLoader(
        TensorDataset(torch.arange(8)),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(1),
    )
    metadata = load_checkpoint(
        path,
        model=restored,
        expected_model_config=_config()["model"],
        optimizer=restored_optimizer,
        loaders={"train": restored_loader},
        restore_training_state=True,
    )
    assert metadata["epoch"] == 3
    assert metadata["global_step"] == 17
    assert torch.equal(torch.rand(4), expected_random)
    assert torch.equal(
        restored_loader.generator.get_state(),
        expected_loader_state,
    )
    assert restored_optimizer.param_groups[0]["lr"] == 1e-3
