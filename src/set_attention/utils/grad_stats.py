from __future__ import annotations

import math
from typing import Iterable, Iterator, Mapping, Sequence, Tuple

import torch

ComponentRules = Sequence[Tuple[str, Sequence[str]]]

DEFAULT_COMPONENT_RULES: ComponentRules = (
    ("ska", ("ska", "set_attn", "set_attention", "router", "adapter", "atom")),
    ("baseline_attn", ("attn", "attention", "mha", "multihead", "qkv")),
    ("ffn", ("ffn", "feedforward", "mlp", "linear1", "linear2")),
)


def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(2)
        total += float(param_norm) ** 2
    return math.sqrt(total)


def iter_named_parameters(modules: Mapping[str, torch.nn.Module | None]) -> Iterator[tuple[str, torch.nn.Parameter]]:
    for prefix, module in modules.items():
        if module is None:
            continue
        for name, param in module.named_parameters():
            full_name = f"{prefix}.{name}" if name else prefix
            yield full_name, param


def component_grad_norms(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    rules: ComponentRules = DEFAULT_COMPONENT_RULES,
) -> dict[str, float]:
    totals = {group: 0.0 for group, _ in rules}
    counts = {group: 0 for group, _ in rules}
    for name, param in named_parameters:
        if param.grad is None:
            continue
        lname = name.lower()
        matched = None
        for group, keywords in rules:
            if any(key in lname for key in keywords):
                matched = group
                break
        if matched is None:
            continue
        param_norm = param.grad.data.norm(2)
        totals[matched] += float(param_norm) ** 2
        counts[matched] += 1
    out: dict[str, float] = {}
    for group, total in totals.items():
        if counts[group] == 0:
            continue
        out[group] = math.sqrt(total)
    return out
