from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config.compatibility import NYSTROM_DEPRECATION_MESSAGE  # noqa: E402
from config.load import load_config  # noqa: E402
from config.schema import ConfigError  # noqa: E402
from set_attention.backends.nystrom import NystromBackend  # noqa: E402


def _assert_raises_message(fn, exc_type, expected: str) -> None:
    try:
        fn()
    except exc_type as exc:
        assert expected in str(exc), str(exc)
        return
    raise AssertionError(f"Expected {exc_type.__name__} containing {expected!r}")


def test_nystrom_backend_constructor_hard_fails():
    _assert_raises_message(
        lambda: NystromBackend(d_model=16, num_heads=4, num_landmarks=4),
        RuntimeError,
        "NystromBackend is deprecated for this revision cycle; use landmark backend.",
    )


def test_nystrom_config_validation_rejects_active_config():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "nystrom.yaml"
        config_path.write_text(
            """
model:
  implementation: set_only
  attention_family: linear
  backend: nystrom
  vocab_size: 100
  d_model: 64
  num_layers: 1
  num_heads: 4
  window_size: 8
  stride: 4
  dropout: 0.1
  max_seq_len: 32
  router_type: uniform
  router_topk: 0
  backend_params:
    num_landmarks: 4
  feature_mode: geometry_only
data:
  dataset: wikitext2
  batch_size: 2
  seq_len: 32
training:
  epochs: 1
  lr: 0.001
""".lstrip(),
            encoding="utf-8",
        )
        os.environ["SET_ATTENTION_FINGERPRINT_PATH"] = str(
            Path(tmp) / "fingerprints.jsonl"
        )
        try:
            _assert_raises_message(
                lambda: load_config(config_path),
                ConfigError,
                NYSTROM_DEPRECATION_MESSAGE,
            )
        finally:
            os.environ.pop("SET_ATTENTION_FINGERPRINT_PATH", None)


def test_active_configs_do_not_use_backend_nystrom():
    offenders: list[str] = []
    for path in (ROOT / "configs").rglob("*.yaml"):
        if "_deprecated" in path.parts:
            continue
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if parsed.get("model", {}).get("backend") == "nystrom":
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {Path(__file__).name}:{name}")
