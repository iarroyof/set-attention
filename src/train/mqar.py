from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data.mqar import DEFAULT_LAG_BINS, IGNORE_INDEX
from train.metrics_impl import masked_lm_loss_and_counts, perplexity


LagBins = tuple[tuple[str, int, int], ...]


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def lag_bin_masks(
    lags: torch.Tensor,
    *,
    bins: LagBins = DEFAULT_LAG_BINS,
) -> dict[str, torch.Tensor]:
    return {name: lags.ge(lo) & lags.le(hi) for name, lo, hi in bins}


def _query_logits_and_targets(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = torch.arange(labels.shape[0], device=labels.device).view(-1, 1)
    query_logits = logits[batch, query_positions]
    query_targets = labels[batch, query_positions]
    return query_logits, query_targets


def _empty_bin_metrics(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_loss": None,
        f"{prefix}_ppl": None,
        f"{prefix}_accuracy": None,
        f"{prefix}_query_count": 0,
    }


def lag_bin_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_positions: torch.Tensor,
    lags: torch.Tensor,
    *,
    bins: LagBins = DEFAULT_LAG_BINS,
) -> dict[str, Any]:
    query_logits, query_targets = _query_logits_and_targets(logits, labels, query_positions)
    masks = lag_bin_masks(lags, bins=bins)
    out: dict[str, Any] = {}
    flat_logits = query_logits.reshape(-1, query_logits.shape[-1])
    flat_targets = query_targets.reshape(-1)
    flat_lags = lags.reshape(-1)
    for name, lo, hi in bins:
        mask = masks[name].reshape(-1) & flat_targets.ne(IGNORE_INDEX)
        count = int(mask.sum().item())
        prefix = f"lag/{name}"
        if count == 0:
            out.update(_empty_bin_metrics(prefix))
            continue
        selected_logits = flat_logits[mask]
        selected_targets = flat_targets[mask]
        loss = F.cross_entropy(selected_logits, selected_targets, reduction="mean")
        predictions = selected_logits.argmax(dim=-1)
        correct = int(predictions.eq(selected_targets).sum().item())
        selected_lags = flat_lags[mask].to(dtype=torch.float32)
        out.update(
            {
                f"{prefix}_loss": float(loss.item()),
                f"{prefix}_ppl": perplexity(float(loss.item())),
                f"{prefix}_accuracy": correct / count,
                f"{prefix}_query_count": count,
                f"{prefix}_lag_mean": float(selected_lags.mean().item()),
            }
        )
    return out


def query_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_positions: torch.Tensor,
) -> dict[str, Any]:
    query_logits, query_targets = _query_logits_and_targets(logits, labels, query_positions)
    loss, valid_tokens, correct = masked_lm_loss_and_counts(
        query_logits,
        query_targets,
        ignore_index=IGNORE_INDEX,
    )
    predictions = query_logits.argmax(dim=-1)
    valid = query_targets.ne(IGNORE_INDEX)
    per_example_correct = (predictions.eq(query_targets) | ~valid).all(dim=1)
    has_query = valid.any(dim=1)
    exact_count = int(has_query.sum().item())
    exact_correct = int((per_example_correct & has_query).sum().item())
    loss_value = float(loss.item())
    return {
        "loss": loss_value,
        "ppl": perplexity(loss_value),
        "accuracy": correct / max(valid_tokens, 1),
        "valid_tokens": valid_tokens,
        "exact_sequence_accuracy": exact_correct / max(exact_count, 1),
    }


def _forward(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    try:
        return model(input_ids, labels=labels)
    except TypeError:
        return model(input_ids)


def _cycle(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def train_mqar_update_block(
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
    if accum_steps == 1:
        for batch in batch_iter:
            if updates >= int(max_updates):
                break
            batch = batch_to_device(batch, device)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, input_ids, labels)
            loss, valid_tokens, correct = masked_lm_loss_and_counts(logits, labels)
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
            optimizer.step()
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens
            total_correct += correct
            updates += 1
    else:
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
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                logits = _forward(model, input_ids, labels)
                loss, valid_tokens, correct = masked_lm_loss_and_counts(logits, labels)
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


def train_mqar_updates(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_updates: int,
    clip_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
) -> dict[str, Any]:
    return train_mqar_update_block(
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
def evaluate_mqar(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    bins: LagBins = DEFAULT_LAG_BINS,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    model.eval()
    if hasattr(model, "reset_probe_metrics"):
        model.reset_probe_metrics()
    loss_sum = 0.0
    valid_total = 0
    correct_total = 0
    exact_total = 0
    exact_correct = 0
    bin_loss_sums = {name: 0.0 for name, _, _ in bins}
    bin_correct = {name: 0 for name, _, _ in bins}
    bin_counts = {name: 0 for name, _, _ in bins}
    bin_lag_sums = {name: 0.0 for name, _, _ in bins}

    for outer_batch in dataloader:
        for batch in _split_batch(outer_batch, microbatch_size):
            batch = batch_to_device(batch, device)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            query_positions = batch["query_positions"]
            lags = batch["lags"]
            logits = _forward(model, input_ids)
            loss, valid_tokens, correct = masked_lm_loss_and_counts(logits, labels)
            loss_sum += float(loss.item()) * valid_tokens
            valid_total += valid_tokens
            correct_total += correct

            query_logits, query_targets = _query_logits_and_targets(logits, labels, query_positions)
            predictions = query_logits.argmax(dim=-1)
            valid = query_targets.ne(IGNORE_INDEX)
            exact_mask = valid.any(dim=1)
            exact_total += int(exact_mask.sum().item())
            exact_correct += int(((predictions.eq(query_targets) | ~valid).all(dim=1) & exact_mask).sum().item())

            per_query_loss = F.cross_entropy(
                query_logits.reshape(-1, query_logits.shape[-1]),
                query_targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(query_targets)
            for name, lo, hi in bins:
                mask = lags.ge(lo) & lags.le(hi) & valid
                count = int(mask.sum().item())
                if count == 0:
                    continue
                bin_counts[name] += count
                bin_loss_sums[name] += float(per_query_loss.masked_select(mask).sum().item())
                bin_correct[name] += int(predictions.eq(query_targets).masked_select(mask).sum().item())
                bin_lag_sums[name] += float(lags.masked_select(mask).to(dtype=torch.float32).sum().item())
    loss_avg = loss_sum / max(valid_total, 1)
    metrics: dict[str, Any] = {
        "loss": loss_avg,
        "ppl": perplexity(loss_avg),
        "accuracy": correct_total / max(valid_total, 1),
        "valid_tokens": valid_total,
        "exact_sequence_accuracy": exact_correct / max(exact_total, 1),
    }
    for name, _, _ in bins:
        prefix = f"lag/{name}"
        count = bin_counts[name]
        if count == 0:
            metrics.update(_empty_bin_metrics(prefix))
        else:
            bin_loss = bin_loss_sums[name] / count
            metrics.update(
                {
                    f"{prefix}_loss": bin_loss,
                    f"{prefix}_ppl": perplexity(bin_loss),
                    f"{prefix}_accuracy": bin_correct[name] / count,
                    f"{prefix}_query_count": count,
                    f"{prefix}_lag_mean": bin_lag_sums[name] / count,
                }
            )
    if hasattr(model, "get_probe_metrics"):
        metrics.update(model.get_probe_metrics(reset=True))
    return metrics


@torch.no_grad()
def evaluate_mqar_group_ablation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    base_metrics: dict[str, Any],
    *,
    bins: LagBins = DEFAULT_LAG_BINS,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    if not hasattr(model, "set_span_ablation_mode"):
        return {"ablation/status": "missing_set_span_ablation_mode_hook"}
    groups = [str(m["name"]) for m in getattr(model, "multiresolution_group_metadata", [])]
    wanted = [name for name in ("fine", "coarse") if name in groups]
    if not wanted:
        return {"ablation/status": "missing_fine_coarse_groups"}
    previous = str(getattr(model, "span_ablation_mode", "none"))
    out: dict[str, Any] = {"ablation/status": "ok"}
    try:
        for group in wanted:
            model.set_span_ablation_mode(group)
            metrics = evaluate_mqar(
                model,
                dataloader,
                device,
                bins=bins,
                microbatch_size=microbatch_size,
            )
            out[f"ablation/{group}_loss"] = metrics["loss"]
            out[f"ablation/{group}_accuracy"] = metrics["accuracy"]
            out[f"ablation/{group}_delta_loss"] = metrics["loss"] - float(base_metrics["loss"])
            out[f"ablation/{group}_delta_accuracy"] = float(base_metrics["accuracy"]) - metrics["accuracy"]
            for name, _, _ in bins:
                base_acc = base_metrics.get(f"lag/{name}_accuracy")
                ablated_acc = metrics.get(f"lag/{name}_accuracy")
                base_loss = base_metrics.get(f"lag/{name}_loss")
                ablated_loss = metrics.get(f"lag/{name}_loss")
                if base_acc is not None and ablated_acc is not None:
                    out[f"ablation/{group}/{name}_delta_accuracy"] = float(base_acc) - float(ablated_acc)
                if base_loss is not None and ablated_loss is not None:
                    out[f"ablation/{group}/{name}_delta_loss"] = float(ablated_loss) - float(base_loss)
                out[f"ablation/{group}/{name}_query_count"] = metrics.get(f"lag/{name}_query_count", 0)
    finally:
        model.set_span_ablation_mode(previous)
    return out
