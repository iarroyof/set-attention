from __future__ import annotations

import warnings

from config.schema import ConfigError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def forbid(condition: bool, message: str) -> None:
    if condition:
        raise ConfigError(message)


def warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning)
