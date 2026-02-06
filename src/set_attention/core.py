from __future__ import annotations

import torch


def assert_set_only_scores(
    scores: torch.Tensor, seq_len: int, allow_token_token: bool = False
) -> None:
    if allow_token_token:
        return
    if scores.shape[-2:] == (seq_len, seq_len):
        raise RuntimeError(
            "Set-only invariant violated: token-token attention detected."
        )


def apply_score_biases(
    scores: torch.Tensor,
    geom_bias: torch.Tensor | None = None,
    content_bias: torch.Tensor | None = None,
    sig_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if geom_bias is not None:
        if geom_bias.dim() == 2:
            geom_bias = geom_bias.unsqueeze(0).unsqueeze(0)
        elif geom_bias.dim() == 3:
            geom_bias = geom_bias.unsqueeze(1)
        scores = scores + geom_bias
    if content_bias is not None:
        if content_bias.dim() == 3:
            content_bias = content_bias.unsqueeze(0)
        scores = scores + content_bias
    if sig_mask is not None:
        if sig_mask.dim() == 2:
            sig_mask = sig_mask.unsqueeze(0).unsqueeze(0)
        elif sig_mask.dim() == 3:
            sig_mask = sig_mask.unsqueeze(1)
        scores = scores.masked_fill(~sig_mask, float("-inf"))
    return scores
