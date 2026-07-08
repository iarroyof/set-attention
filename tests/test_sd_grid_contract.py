from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.experiment_contracts import validate_experiment_contract
from config.load import load_config
from config.schema import ConfigError
from train.metrics_schema import MULTIRESOLUTION_GROUP_DIAGNOSTICS


CONTRACT_OVERRIDES = [
    "training.seed=0",
    "training.lr=1e-4",
    "training.deterministic=true",
    "training.benchmark_mode=false",
    "training.experiment_contract=sd_grid_seeded_v1",
    "training.diagnostics_contract=current_matrix_v1",
]


def test_set_and_token_grid_contracts_accept_registered_configs() -> None:
    set_cfg = load_config(
        ROOT / "configs/set_dictionary/sd9_multiresolution.yaml",
        overrides=CONTRACT_OVERRIDES,
    )
    token_cfg = load_config(
        ROOT / "configs/paper_lr_norm/baseline_dense_exact.yaml",
        overrides=CONTRACT_OVERRIDES,
    )
    validate_experiment_contract(set_cfg)
    validate_experiment_contract(token_cfg)


def test_grid_contract_rejects_open_identity_path() -> None:
    cfg = load_config(
        ROOT / "configs/set_dictionary/sd9_multiresolution.yaml",
        overrides=[
            *CONTRACT_OVERRIDES,
            "model.output_residual_mode=none",
        ],
    )
    with pytest.raises(ConfigError, match="output_residual_mode"):
        validate_experiment_contract(cfg)


def test_grid_contract_rejects_limited_data() -> None:
    cfg = load_config(
        ROOT / "configs/paper_lr_norm/baseline_dense_exact.yaml",
        overrides=[
            *CONTRACT_OVERRIDES,
            "data.limit=500",
        ],
    )
    with pytest.raises(ConfigError, match="full dataset"):
        validate_experiment_contract(cfg)


def test_multiresolution_group_diagnostic_columns_are_registered() -> None:
    for group in ("fine", "coarse"):
        for metric in (
            "routing_entropy_norm",
            "router_param_norm",
            "pooling_effective_support",
            "grad_norm_set_post_pool",
            "grad_norm_set_post_blocks",
        ):
            assert f"ausa/{group}/{metric}" in MULTIRESOLUTION_GROUP_DIAGNOSTICS


def test_grid_launcher_fails_closed_on_contention_oom() -> None:
    launcher = (ROOT / "scripts/run_sd_grid.sh").read_text()

    assert 'ALLOW_GPU_CORESIDENCY="${ALLOW_GPU_CORESIDENCY:-0}"' in launcher
    assert 'REQUIRE_EXCLUSIVE_GPU="${REQUIRE_EXCLUSIVE_GPU:-1}"' in launcher
    assert "REQUIRE_EXCLUSIVE_GPU=1 conflicts with ALLOW_GPU_CORESIDENCY=1" in launcher
    assert 'GPU_ADMISSION_HEADROOM_MIB="${GPU_ADMISSION_HEADROOM_MIB:-4096}"' in launcher
    assert 'GPU_PEAK_ESTIMATE_MARGIN_MIB="${GPU_PEAK_ESTIMATE_MARGIN_MIB:-1024}"' in launcher
    assert "NON_TERMINAL_CONTENTION_OR_UNKNOWN_OOM" in launcher
    assert "Unknown end occupancy can never certify an exclusive capacity failure." in launcher
    assert 'touch "$oommark"' in launcher
    assert launcher.index("NON_TERMINAL_CONTENTION_OR_UNKNOWN_OOM") < launcher.index('touch "$oommark"')
    assert '--name "$container_name"' in launcher
    assert 'docker ps --filter "name=^sdgrid_${HOST_TAG}_"' in launcher
    assert launcher.index('kill -TERM "$pid"') < launcher.index('docker ps --filter "name=^sdgrid_${HOST_TAG}_"')
    assert "strict exclusivity: GPU became occupied before docker run" in launcher


def test_deferred_gpu_handoff_waits_for_full_release() -> None:
    handoff = (ROOT / "scripts/restart_deferred_cancer_after_sd_grid.sh").read_text()

    assert "while grid_driver_alive" in handoff
    assert "while grid_containers_alive || cuda_processes_alive" in handoff
    assert 'docker start "$CONTAINER"' in handoff
    assert 'docker rename "$CONTAINER" "$RESTORE_NAME"' in handoff
    assert 'docker exec -d \\' in handoff
    assert '--env-file "$ENV_FILE"' in handoff
