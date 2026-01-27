from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total += param_norm.item() ** 2
    return total ** 0.5


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    grad_norm_sum = 0.0
    grad_norm_steps = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss.backward()
        grad_norm_sum += _grad_norm(model)
        grad_norm_steps += 1
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
    loss_avg = total_loss / max(total_tokens, 1)
    grad_norm = grad_norm_sum / grad_norm_steps if grad_norm_steps else None
    return {"loss": loss_avg, "grad_norm": grad_norm}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
    loss_avg = total_loss / max(total_tokens, 1)
    return {"loss": loss_avg}
