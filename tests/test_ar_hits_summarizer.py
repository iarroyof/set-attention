from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_ar_hits import ARHitSummarizerError, summarize, validate_row  # noqa: E402


def _row(**overrides: object) -> dict[str, str]:
    row = {
        "task": "natural_ar_hits",
        "dataset": "wikitext2",
        "row": "b25",
        "seed": "0",
        "checkpoint_sha256": "abc",
        "config_fingerprint": "cfg",
        "model_config_digest": "model",
        "data.dataset_digest": "dataset",
        "data.tokenizer_digest": "tokenizer",
        "model.backend": "exact",
        "val/overall/nll": "1.0",
        "val/overall/ppl": "2.718281828",
        "val/overall/targets": "10",
        "val/ar/nll": "0.9",
        "val/ar/targets": "4",
        "val/non_ar/nll": "1.1",
        "val/non_ar/targets": "6",
        "val/ar/target_fraction": "0.4",
    }
    for name, count in {
        "count_0": 1,
        "count_1": 1,
        "count_2_5": 2,
        "count_6_20": 0,
        "count_gt20": 0,
        "lag_1_32": 2,
        "lag_33_128": 1,
        "lag_129_512": 1,
        "lag_513_1024": 0,
        "lag_1025_plus": 0,
    }.items():
        row[f"val/{name}/targets"] = str(count)
        row[f"val/{name}/inferential"] = "False"
        row[f"val/{name}/nll"] = "1.0" if count else ""
        row[f"val/{name}/ppl"] = "2.718281828" if count else ""
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def test_validate_row_accepts_descriptive_bins_without_nan() -> None:
    parsed = validate_row(_row(), require_registered_matrix=False)
    assert parsed["row"] == "b25"
    assert parsed["count_6_20_targets"] == 0
    assert parsed["count_6_20_inferential"] is False


def test_validate_row_rejects_nonfinite_text() -> None:
    with pytest.raises(ARHitSummarizerError, match="NaN/Inf"):
        validate_row(_row(**{"val/overall/nll": "nan"}), require_registered_matrix=False)


def test_summarize_allows_incomplete_when_requested() -> None:
    parsed = validate_row(_row(), require_registered_matrix=False)
    summary = summarize([parsed], require_registered_matrix=False)
    assert summary[0]["row"] == "b25"
    assert summary[0]["n"] == 1
    assert summary[0]["has_inferential_ar_bin"] is False
