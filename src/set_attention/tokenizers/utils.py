from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

TOKENIZER_META_FILENAME = "tokenizer_meta.json"


def write_tokenizer_meta(out_dir: str, tokenizer_type: str, config: Optional[Dict[str, Any]] = None) -> None:
    """Persist lightweight metadata so we can reload tokenizers without extra CLI hints."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"type": tokenizer_type}
    if config is not None:
        payload["config"] = config
    (path / TOKENIZER_META_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_tokenizer_meta(out_dir: str) -> Optional[Dict[str, Any]]:
    """Return metadata if available; tolerate missing file for backward compatibility."""
    path = Path(out_dir) / TOKENIZER_META_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # If the file was corrupted, fall back to CLI-provided kind.
        return None

