from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence


class WandbSession:
    def __init__(self, run=None):
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is None:
            return
        if step is not None:
            self._run.log(data, step=step)
        else:
            self._run.log(data)

    def set_summary(self, key: str, value: Any) -> None:
        if self._run is None:
            return
        self._run.summary[key] = value

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def init_wandb(
    enable: bool,
    project: Optional[str],
    run_name: Optional[str],
    config: Optional[Dict[str, Any]],
    tags: Optional[Sequence[str]],
) -> WandbSession:
    if not enable:
        return WandbSession()
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[W&B] Disabled (missing dependency): {exc}")
        return WandbSession()

    resolved_project = project or os.getenv("WANDB_PROJECT") or "ska-naive-ablation"
    resolved_run_name = run_name or None
    tag_list = list(tags) if tags else None
    run = wandb.init(
        project=resolved_project,
        name=resolved_run_name,
        config=config,
        tags=tag_list,
        reinit=True,
    )
    return WandbSession(run)


__all__ = ["init_wandb", "WandbSession"]
