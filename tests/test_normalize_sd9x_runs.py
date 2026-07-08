from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.normalize_sd9x_runs import normalize


def test_normalizer_requires_corrected_provenance_and_exports_group_diagnostics(
    tmp_path: Path,
) -> None:
    metadata = {
        "model.implementation": "set_only",
        "model.multiresolution.enabled": True,
        "model.multiresolution.groups": [
            {"name": "fine", "num_heads": 4, "window_size": 2, "stride": 1},
            {"name": "coarse", "num_heads": 4, "window_size": 4, "stride": 2},
        ],
        "model.num_heads": 8,
        "model.backend": "exact",
        "model.output_residual_mode": "anchor_span",
        "model.token_mlp.enabled": False,
        "data.seq_len": 3584,
        "data.batch_size": 3,
        "data.limit": None,
        "training.seed": 2,
        "training.seed_applied": True,
        "training.applied_seed": 2,
        "training.torch_initial_seed": 2,
        "training.deterministic": True,
        "training.benchmark_mode": False,
        "training.experiment_contract": "sd_grid_seeded_v1",
        "training.diagnostics_contract": "current_matrix_v1",
    }
    json_path = tmp_path / "run.json"
    csv_path = tmp_path / "run.csv"
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "val/ppl",
                "train/ppl",
                "train/peak_vram_mib",
                "val/span_ablation_delta_ppl",
                "ausa/fine/router_param_norm",
                "ausa/coarse/router_param_norm",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 10,
                "val/ppl": 900,
                "train/ppl": 800,
                "train/peak_vram_mib": 20000,
                "val/span_ablation_delta_ppl": 10,
                "ausa/fine/router_param_norm": 1.5,
                "ausa/coarse/router_param_norm": 2.5,
            }
        )

    row = normalize(str(json_path))
    assert row is not None
    assert row["batch_size"] == 3
    assert row["variant"] == "b50"
    assert row["router_param_norm_fine"] == "1.5"
    assert row["router_param_norm_coarse"] == "2.5"
