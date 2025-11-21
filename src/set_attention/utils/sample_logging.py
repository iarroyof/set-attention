from __future__ import annotations

import random
from typing import List, Optional, Sequence


def select_sample_indices(total: int, count: int, seed: int) -> List[int]:
    """Return deterministic random indices for sampling previews."""
    if total <= 0 or count <= 0:
        return []
    rng = random.Random(int(seed))
    indices = list(range(total))
    rng.shuffle(indices)
    take = min(count, total)
    selected = sorted(indices[:take])
    return selected


def format_text_samples(
    refs: Sequence[Sequence[str]],
    hyps: Sequence[Sequence[str]],
    count: int,
    seed: int,
    sources: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Serialize multiple reference/prediction samples for logging."""
    total = min(len(refs), len(hyps))
    if total == 0 or count <= 0:
        return None
    indices = select_sample_indices(total, count, seed)
    blocks = []
    for pos, idx in enumerate(indices):
        ref_tokens = " ".join(refs[idx])
        hyp_tokens = " ".join(hyps[idx])
        lines = [f"[{pos}]"]
        if sources is not None and idx < len(sources):
            lines.append(f"SRC: {sources[idx]}")
        lines.append(f"REF: {ref_tokens}")
        lines.append(f"PRED: {hyp_tokens}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["select_sample_indices", "format_text_samples"]
