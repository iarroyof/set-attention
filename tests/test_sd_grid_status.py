from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sd_grid_status",
    ROOT / "scripts" / "sd_grid_status.py",
)
assert SPEC is not None and SPEC.loader is not None
sd_grid_status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sd_grid_status)


def test_scan_uses_registered_epoch_for_longer_reusable_run(tmp_path: Path) -> None:
    metadata = {
        "model.implementation": "baseline_token",
        "model.multiresolution.enabled": False,
        "model.backend": "exact",
        "data.seq_len": 512,
        "data.batch_size": 16,
        "training.seed": 0,
        "training.epochs": 30,
        "data.limit": None,
    }
    json_path = tmp_path / "token.json"
    csv_path = tmp_path / "token.csv"
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "val/ppl",
                "train/peak_vram_mib",
                "val/span_ablation_delta_ppl",
            ],
        )
        writer.writeheader()
        for epoch in range(1, 31):
            writer.writerow(
                {
                    "epoch": epoch,
                    "val/ppl": 800 + epoch,
                    "train/peak_vram_mib": 1000 + epoch,
                    "val/span_ablation_delta_ppl": "",
                }
            )

    result = sd_grid_status.scan(str(json_path))

    assert result is not None
    assert result[0] == "token|exact|512|token|b16|0"
    assert result[1] == 30
    assert result[2] == "810"
    assert result[3] == "1010"


def test_dense_grid_uses_dense_config_for_exact_token_identity() -> None:
    launcher = (ROOT / "scripts" / "run_sd_grid.sh").read_text(encoding="utf-8")

    assert 'exact)                  sig="baseline_dense_exact.yaml"' in launcher
    assert 'exact)                  cfg="$TOKEN_CONFIG_EXACT"' in launcher
    assert 'model.backend_params={}' not in launcher
    assert 'grep -F "data.batch_size=${batch}"' in launcher
    assert 'GRID_PROFILE="${GRID_PROFILE:-primary}"' in launcher
    assert 'elif [ "$GRID_PROFILE" = frontier ]; then' in launcher
    assert "training.experiment_contract=sd_grid_seeded_v1" in launcher
    assert "training.diagnostics_contract=current_matrix_v1" in launcher
    assert "SD_GRID_REQUIRE_CONTRACT=sd_grid_seeded_v1" in launcher


def test_scan_rejects_hybrid_as_token_and_every_non_null_limit(tmp_path: Path) -> None:
    base = {
        "model.multiresolution.enabled": False,
        "model.backend": "exact",
        "data.seq_len": 512,
        "data.batch_size": 16,
        "training.seed": 0,
    }
    hybrid = dict(base, **{"model.implementation": "hybrid_token_set"})
    hybrid_path = tmp_path / "hybrid.json"
    hybrid_path.write_text(json.dumps(hybrid), encoding="utf-8")
    assert sd_grid_status.scan(str(hybrid_path)) is None

    limited = dict(
        base,
        **{
            "model.implementation": "baseline_token",
            "data.limit": 10,
        },
    )
    limited_path = tmp_path / "limited.json"
    limited_path.write_text(json.dumps(limited), encoding="utf-8")
    assert sd_grid_status.scan(str(limited_path)) is None


def test_strict_contract_requires_applied_seed_and_diagnostics() -> None:
    metadata = {
        "training.experiment_contract": "sd_grid_seeded_v1",
        "training.diagnostics_contract": "current_matrix_v1",
        "training.seed": 0,
        "training.seed_applied": False,
        "training.applied_seed": 0,
        "training.torch_initial_seed": 0,
        "training.deterministic": True,
        "training.benchmark_mode": False,
        "training.epochs": 10,
        "training.lr": 0.0001,
        "training.warmup_steps": 1000,
        "data.dataset": "wikitext2",
        "data.limit": None,
        "data.val_limit": None,
        "model.implementation": "baseline_token",
        "model.attention_family": "dense",
        "model.backend": "exact",
        "model.d_model": 384,
        "model.dim_feedforward": 1536,
        "model.num_layers": 6,
        "model.num_heads": 8,
    }
    errors = sd_grid_status._strict_contract_errors(
        metadata,
        {"val/ppl": "1"},
        epochs=10,
    )
    assert any("seed_applied" in error for error in errors)
    assert any("peak_vram" in error for error in errors)
    assert any("attention_entropy" in error for error in errors)
