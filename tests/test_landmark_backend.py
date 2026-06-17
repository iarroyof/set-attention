from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config.load import load_config  # noqa: E402
from scripts.run_experiment import attach_resolved_metadata, build_model  # noqa: E402
from set_attention.backends.landmark import LandmarkAttentionBackend  # noqa: E402


EXPECTED_M63_RHO025 = [0, 4, 8, 12, 17, 21, 25, 29, 33, 37, 41, 45, 50, 54, 58, 62]
EXPECTED_M64_RHO025 = [0, 4, 8, 13, 17, 21, 25, 29, 34, 38, 42, 46, 50, 55, 59, 63]


def _select(m: int, rho: float) -> list[int]:
    backend = LandmarkAttentionBackend(
        d_model=16,
        num_heads=4,
        landmark_coverage=rho,
    )
    return backend._select_landmarks(m, torch.device("cpu")).tolist()


def _expected_count(m: int, rho: float) -> int:
    return min(max(round(rho * m), 2), m)


def test_landmark_reference_indices_m63_rho025():
    assert _select(63, 0.25) == EXPECTED_M63_RHO025


def test_landmark_historical_indices_m64_rho025():
    assert _select(64, 0.25) == EXPECTED_M64_RHO025


def test_landmark_grid_count_and_anchors():
    for m in (32, 63, 64, 128, 255, 256):
        for rho in (0.1, 0.25, 0.5, 1.0):
            indices = _select(m, rho)
            assert len(indices) == _expected_count(m, rho)
            assert indices[0] == 0
            assert indices[-1] == m - 1
            assert indices == sorted(indices)
            assert len(indices) == len(set(indices))
            if _expected_count(m, rho) >= m:
                assert indices == list(range(m))


def test_active_landmark_configs_parse_with_coverage_and_no_num_landmarks():
    active_landmark_configs = [
        ROOT / "configs/paper_lr_norm/set_linear_landmark.yaml",
        ROOT / "configs/paper_complements/family_linear_landmark.yaml",
        ROOT / "configs/set_only/wikitext2_landmark.yaml",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SET_ATTENTION_FINGERPRINT_PATH"] = str(
            Path(tmp) / "fingerprints.jsonl"
        )
        try:
            for path in active_landmark_configs:
                cfg = load_config(path)
                params = cfg["model"]["backend_params"]
                assert cfg["model"]["backend"] == "landmark"
                assert params["landmark_coverage"] == 0.25
                assert "num_landmarks" not in params
        finally:
            os.environ.pop("SET_ATTENTION_FINGERPRINT_PATH", None)


def test_active_config_text_has_no_num_landmarks_outside_deprecated():
    for path in (ROOT / "configs").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if "num_landmarks" not in text:
            continue
        rel = path.relative_to(ROOT)
        if "_deprecated" in path.parts:
            continue
        assert False, f"{rel} still contains num_landmarks outside configs/_deprecated"


def test_landmark_resolved_metadata_reaches_model_and_config():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SET_ATTENTION_FINGERPRINT_PATH"] = str(
            Path(tmp) / "fingerprints.jsonl"
        )
        try:
            cfg = load_config(
                ROOT / "configs/paper_lr_norm/set_linear_landmark.yaml",
                overrides=[
                    "model.vocab_size=101",
                    "model.d_model=32",
                    "model.num_layers=1",
                    "model.num_heads=4",
                    "model.dim_feedforward=64",
                    "data.seq_len=512",
                    "training.output_dir=/tmp/a1_6_landmark_metadata",
                    "logging.wandb.enable=false",
                ],
            )
            torch.manual_seed(0)
            model = build_model(cfg["model"])
            attach_resolved_metadata(cfg, model)

            assert cfg["resolved"]["landmark_coverage"] == 0.25
            assert cfg["resolved"]["landmark_count"] == 16
            assert model.blocks[0].backend.landmark_coverage == 0.25
            assert (
                model.blocks[0]
                .backend._select_landmarks(63, torch.device("cpu"))
                .tolist()
                == EXPECTED_M63_RHO025
            )
        finally:
            os.environ.pop("SET_ATTENTION_FINGERPRINT_PATH", None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {Path(__file__).name}:{name}")
