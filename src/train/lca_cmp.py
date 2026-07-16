from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data.lca_cmp import IGNORE_INDEX, special_tokens
from train.metrics_impl import perplexity


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _forward(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    try:
        return model(input_ids, labels=labels)
    except TypeError:
        return model(input_ids)


def _cycle(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _loss_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    valid = labels.ne(IGNORE_INDEX)
    predictions = logits.argmax(dim=-1)
    correct = int(predictions.eq(labels).masked_select(valid).sum().item())
    return loss, int(valid.sum().item()), correct


def train_lca_update_block(
    model: nn.Module,
    batch_iter: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_updates: int,
    clip_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    updates = 0
    accum_steps = max(1, int(grad_accum_steps))
    iterator = iter(batch_iter)
    while updates < int(max_updates):
        microbatches: list[dict[str, torch.Tensor]] = []
        for _ in range(accum_steps):
            try:
                microbatches.append(batch_to_device(next(iterator), device))
            except StopIteration:
                break
        if not microbatches:
            break

        optimizer.zero_grad(set_to_none=True)
        losses: list[tuple[torch.Tensor, int, int]] = []
        accum_valid_tokens = 0
        for batch in microbatches:
            logits = _forward(model, batch["input_ids"], batch["labels"])
            loss, valid_tokens, correct = _loss_counts(logits, batch["labels"])
            losses.append((loss, valid_tokens, correct))
            accum_valid_tokens += valid_tokens
        normalizer = max(accum_valid_tokens, 1)
        for loss, valid_tokens, _ in losses:
            (loss * (valid_tokens / normalizer)).backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
        optimizer.step()

        for loss, valid_tokens, correct in losses:
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens
            total_correct += correct
        updates += 1

    loss_avg = total_loss / max(total_tokens, 1)
    return {
        "loss": loss_avg,
        "ppl": perplexity(loss_avg),
        "accuracy": total_correct / max(total_tokens, 1),
        "valid_tokens": total_tokens,
        "_optimizer_steps": updates,
        "_microbatches_per_optimizer_step": accum_steps,
    }


def train_lca_updates(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_updates: int,
    clip_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
) -> dict[str, Any]:
    return train_lca_update_block(
        model,
        _cycle(dataloader),
        optimizer,
        device,
        max_updates=max_updates,
        clip_grad_norm=clip_grad_norm,
        grad_accum_steps=grad_accum_steps,
    )


def _split_batch(
    batch: dict[str, torch.Tensor],
    microbatch_size: int | None,
) -> Iterable[dict[str, torch.Tensor]]:
    if microbatch_size is None or int(microbatch_size) <= 0:
        yield batch
        return
    first_tensor = next((value for value in batch.values() if torch.is_tensor(value)), None)
    if first_tensor is None:
        yield batch
        return
    batch_size = int(first_tensor.shape[0])
    chunk_size = int(microbatch_size)
    if chunk_size >= batch_size:
        yield batch
        return
    for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        yield {
            key: value[start:stop] if torch.is_tensor(value) and int(value.shape[0]) == batch_size else value
            for key, value in batch.items()
        }


@torch.no_grad()
def evaluate_lca(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    vocab_size: int,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    exact_total = 0
    exact_correct = 0
    bucket_loss = {"near_below": 0.0, "near_above": 0.0}
    bucket_count = {"near_below": 0, "near_above": 0}
    bucket_correct = {"near_below": 0, "near_above": 0}
    answer_ids = special_tokens(vocab_size)

    for outer_batch in dataloader:
        for batch in _split_batch(outer_batch, microbatch_size):
            batch = batch_to_device(batch, device)
            logits = _forward(model, batch["input_ids"])
            labels = batch["labels"]
            loss, valid_tokens, correct = _loss_counts(logits, labels)
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens
            total_correct += correct

            query_positions = batch["query_positions"]
            row = torch.arange(labels.shape[0], device=device).view(-1, 1)
            query_logits = logits[row, query_positions]
            query_targets = labels[row, query_positions]
            query_predictions = query_logits.argmax(dim=-1)
            valid = query_targets.ne(IGNORE_INDEX)
            exact_total += int(valid.any(dim=1).sum().item())
            exact_correct += int(((query_predictions.eq(query_targets) | ~valid).all(dim=1) & valid.any(dim=1)).sum().item())

            per_query_loss = F.cross_entropy(
                query_logits.reshape(-1, query_logits.shape[-1]),
                query_targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(query_targets)
            false_id = int(answer_ids["answer_false"])
            true_id = int(answer_ids["answer_true"])
            masks = {
                "near_below": query_targets.eq(false_id) & valid,
                "near_above": query_targets.eq(true_id) & valid,
            }
            for name, mask in masks.items():
                count = int(mask.sum().item())
                if count <= 0:
                    continue
                bucket_count[name] += count
                bucket_loss[name] += float(per_query_loss.masked_select(mask).sum().item())
                bucket_correct[name] += int(query_predictions.eq(query_targets).masked_select(mask).sum().item())

    loss_avg = total_loss / max(total_tokens, 1)
    metrics: dict[str, Any] = {
        "loss": loss_avg,
        "ppl": perplexity(loss_avg),
        "accuracy": total_correct / max(total_tokens, 1),
        "valid_tokens": total_tokens,
        "exact_sequence_accuracy": exact_correct / max(exact_total, 1),
    }
    for name in ("near_below", "near_above"):
        count = bucket_count[name]
        metrics[f"bucket/{name}_count"] = count
        if count > 0:
            avg_loss = bucket_loss[name] / count
            metrics[f"bucket/{name}_loss"] = avg_loss
            metrics[f"bucket/{name}_ppl"] = perplexity(avg_loss)
            metrics[f"bucket/{name}_accuracy"] = bucket_correct[name] / count
    return metrics
