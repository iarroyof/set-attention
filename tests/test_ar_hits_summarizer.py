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


def _block(ar_nll: float, ar_targets: int = 10, non_ar_nll: float = 100.0, non_ar_targets: int = 100) -> dict:
    return {
        "ar_nll": ar_nll,
        "ar_targets": ar_targets,
        "non_ar_nll": non_ar_nll,
        "non_ar_targets": non_ar_targets,
    }


def _blocks_by(ar_offsets: dict[str, float], seeds=(0, 1, 2), n_seq: int = 20) -> dict:
    """Blocks where row r's per-target AR NLL on sequence i is 1.0 + ar_offsets[r] + 0.01*i."""
    out = {}
    for row in ("token", "b0", "b25", "b100"):
        for seed in seeds:
            out[(row, seed)] = [
                _block(ar_nll=(1.0 + ar_offsets[row] + 0.01 * i) * 10) for i in range(n_seq)
            ]
    return out


def test_bootstrap_gate_passes_when_star_strictly_better() -> None:
    from summarize_ar_hits import paired_bootstrap_gate

    blocks = _blocks_by({"token": 0.0, "b0": 0.10, "b25": 0.0, "b100": 0.20})
    gate = paired_bootstrap_gate(blocks, resamples=500, rng_seed=7)
    assert gate["cond2_pass"] is True
    assert gate["cond3_pass"] is True
    assert gate["supportive"] is True
    for entry in gate["endpoints"].values():
        assert entry["ar_diff_ci_hi"] < 0.0
        assert entry["did_ci_hi"] < 0.0


def test_bootstrap_gate_fails_when_differences_straddle_zero() -> None:
    from summarize_ar_hits import paired_bootstrap_gate

    blocks = _blocks_by({"token": 0.0, "b0": 0.005, "b25": 0.0, "b100": -0.005})
    gate = paired_bootstrap_gate(blocks, resamples=500, rng_seed=7)
    assert gate["cond2_pass"] is False
    assert gate["supportive"] is False


def test_bootstrap_gate_did_fails_when_nonar_improves_as_much() -> None:
    from summarize_ar_hits import paired_bootstrap_gate

    blocks = _blocks_by({"token": 0.0, "b0": 0.10, "b25": 0.0, "b100": 0.20})
    # make non-AR differences identical to AR differences -> DiD ~ 0
    for key, blocks in blocks.items():
        row = key[0]
        ar_per_target = {"token": 0.0, "b0": 0.10, "b25": 0.0, "b100": 0.20}[row]
        for i, block in enumerate(blocks):
            block["non_ar_nll"] = (1.0 + ar_per_target + 0.01 * i) * 100
    gate = paired_bootstrap_gate(blocks, resamples=500, rng_seed=7)
    assert gate["cond2_pass"] is True
    assert gate["cond3_pass"] is False
    assert gate["supportive"] is False


def test_bootstrap_gate_is_deterministic() -> None:
    from summarize_ar_hits import paired_bootstrap_gate

    blocks = _blocks_by({"token": 0.0, "b0": 0.03, "b25": 0.0, "b100": -0.02})
    first = paired_bootstrap_gate(blocks, resamples=200, rng_seed=13)
    second = paired_bootstrap_gate(blocks, resamples=200, rng_seed=13)
    assert first == second
