from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_mqar import MQARSummarizerError, validate_row  # noqa: E402


def _row(**updates: str) -> dict[str, str]:
    row = {
        "task": "mqar",
        "dataset": "mqar",
        "stage": "mqar_primary_registered",
        "model.implementation": "baseline_token",
        "model.backend": "exact",
        "data.seq_len": "2048",
        "data.num_kv_pairs": "256",
        "data.num_train_examples": "100000",
        "data.num_val_examples": "3000",
        "training.max_updates": "20000",
        "training.seed": "0",
        "config_fingerprint": "cfg",
        "data.dataset_digest": "dataset",
        "data.tokenizer_digest": "tokenizer",
        "val/loss": "0.1",
        "val/accuracy": "0.99",
        "val/valid_tokens": "768000",
        "val/exact_sequence_accuracy": "0.95",
    }
    for name in ("lag_1_32", "lag_33_128", "lag_129_512", "lag_513_1024", "lag_1025_2047"):
        row[f"val/lag/{name}_query_count"] = "1"
        row[f"val/lag/{name}_accuracy"] = "1.0"
    row.update(updates)
    return row


def test_summarizer_rejects_smoke_limited_and_nonfinite_rows() -> None:
    with pytest.raises(MQARSummarizerError, match="smoke/limited"):
        validate_row(_row(stage="mqar_smoke"))
    with pytest.raises(MQARSummarizerError, match="limited train"):
        validate_row(_row(**{"data.num_train_examples": "32"}))
    with pytest.raises(MQARSummarizerError, match="NaN/Inf"):
        validate_row(_row(**{"val/loss": "nan"}))


def test_summarizer_rejects_malformed_seed_and_missing_metadata() -> None:
    with pytest.raises(MQARSummarizerError, match="malformed seed"):
        validate_row(_row(**{"training.seed": "9"}))
    with pytest.raises(MQARSummarizerError, match="metadata incomplete"):
        validate_row(_row(**{"data.dataset_digest": "NA"}))
