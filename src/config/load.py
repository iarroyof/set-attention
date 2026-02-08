from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.compatibility import validate_compatibility
from config.normalize import normalize_config
from config.schema import validate_config


def _apply_override(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    overrides = overrides or []
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value: {override}")
        key, value_str = override.split("=", 1)
        value = yaml.safe_load(value_str)
        _apply_override(cfg, key, value)
    cfg = normalize_config(cfg)
    validate_config(cfg)
    return validate_compatibility(cfg)
