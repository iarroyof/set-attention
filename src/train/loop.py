from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from models.set_only.losses import set_diversity_loss

DEFAULT_SET_DIVERSITY_MODE = "position_contrastive"


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total += param_norm.item() ** 2
    return total ** 0.5


def _maybe_add_set_diversity_loss(
    model: nn.Module,
    loss: torch.Tensor,
    weight: float = 0.0,
    mode: str = DEFAULT_SET_DIVERSITY_MODE,
) -> torch.Tensor:
    if weight <= 0.0:
        return loss
    if hasattr(model, "get_last_set_embeddings"):
        set_embs = model.get_last_set_embeddings()
        if set_embs is not None:
            if mode != "position_contrastive":
                loss = loss + weight * set_diversity_loss(
                    set_embs, mode=mode, target_similarity=0.3
                )
            else:
                num_sets = set_embs.shape[1]
                set_positions = torch.arange(num_sets, device=set_embs.device)
                loss = loss + weight * set_diversity_loss(
                    set_embs,
                    mode=mode,
                    set_positions=set_positions,
                    margin=0.3,
                )
    return loss


def _maybe_add_model_auxiliary_losses(
    model: nn.Module,
    loss: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not hasattr(model, "get_auxiliary_losses"):
        return loss, {}
    aux_losses = model.get_auxiliary_losses()
    aux_metrics: dict[str, float] = {}
    for name, value in aux_losses.items():
        if not torch.is_tensor(value):
            continue
        aux_metrics[name] = float(value.detach().item())
        if name.endswith("_loss"):
            loss = loss + value
    if hasattr(model, "get_auxiliary_metrics"):
        aux_metrics.update(model.get_auxiliary_metrics())
    return loss, aux_metrics


def _update_diagnostics(model: nn.Module) -> None:
    if hasattr(model, "collect_grad_diagnostics"):
        try:
            model.collect_grad_diagnostics()
        except Exception:
            pass
    if hasattr(model, "diagnostics") and hasattr(model, "router"):
        try:
            router_params = dict(model.router.named_parameters())
            model.diagnostics.update_router_params(router_params)
        except Exception:
            pass
    if hasattr(model, "encoder") and hasattr(model.encoder, "diagnostics") and hasattr(model.encoder, "router"):
        try:
            if hasattr(model.encoder, "collect_grad_diagnostics"):
                model.encoder.collect_grad_diagnostics()
        except Exception:
            pass
        try:
            router_params = dict(model.encoder.router.named_parameters())
            model.encoder.diagnostics.update_router_params(router_params)
        except Exception:
            pass
    if hasattr(model, "decoder") and hasattr(model.decoder, "diagnostics") and hasattr(model.decoder, "router"):
        try:
            if hasattr(model.decoder, "collect_grad_diagnostics"):
                model.decoder.collect_grad_diagnostics()
        except Exception:
            pass
        try:
            router_params = dict(model.decoder.router.named_parameters())
            model.decoder.diagnostics.update_router_params(router_params)
        except Exception:
            pass
    if hasattr(model, "diagnostics") and hasattr(model, "attention_params"):
        try:
            model.diagnostics.update_params(model.attention_params())
        except Exception:
            pass


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    set_diversity_weight: float = 0.0,
    set_diversity_mode: str = DEFAULT_SET_DIVERSITY_MODE,
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
        try:
            logits = model(input_ids, labels=labels)
        except TypeError:
            logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss, aux_metrics = _maybe_add_model_auxiliary_losses(model, loss)
        loss = _maybe_add_set_diversity_loss(
            model,
            loss,
            weight=set_diversity_weight,
            mode=set_diversity_mode,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        _update_diagnostics(model)
        grad_norm_sum += _grad_norm(model)
        grad_norm_steps += 1
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
    loss_avg = total_loss / max(total_tokens, 1)
    grad_norm = grad_norm_sum / grad_norm_steps if grad_norm_steps else None
    metrics = {"loss": loss_avg, "grad_norm": grad_norm}
    if "aux_metrics" in locals():
        metrics.update(aux_metrics)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    unigram_counts: torch.Tensor | None = None,
) -> dict:
    model.eval()
    if hasattr(model, "reset_probe_metrics"):
        model.reset_probe_metrics()
    total_loss = 0.0
    total_tokens = 0
    bucket_loss: dict[str, float] = {
        "loss_early_freq": 0.0,
        "loss_early_rare": 0.0,
        "loss_late_freq": 0.0,
        "loss_late_rare": 0.0,
    }
    bucket_tokens: dict[str, int] = {key: 0 for key in bucket_loss}
    counts_device = unigram_counts.to(device) if unigram_counts is not None else None
    rarity_threshold = None
    if counts_device is not None:
        nonzero = counts_device[counts_device > 0]
        if nonzero.numel() > 0:
            rarity_threshold = torch.median(nonzero.to(dtype=torch.float32))
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
        if counts_device is not None and rarity_threshold is not None:
            per_token_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
            ).view_as(labels)
            seq_len = labels.shape[1]
            pos = torch.arange(seq_len, device=device).view(1, seq_len)
            early = pos < (seq_len // 2)
            target_counts = counts_device.index_select(0, labels.reshape(-1)).view_as(labels)
            frequent = target_counts.to(dtype=torch.float32) >= rarity_threshold
            masks = {
                "loss_early_freq": early & frequent,
                "loss_early_rare": early & ~frequent,
                "loss_late_freq": ~early & frequent,
                "loss_late_rare": ~early & ~frequent,
            }
            for key, mask in masks.items():
                n = int(mask.sum().item())
                if n <= 0:
                    continue
                bucket_loss[key] += float(per_token_loss.masked_select(mask).sum().item())
                bucket_tokens[key] += n
    loss_avg = total_loss / max(total_tokens, 1)
    metrics = {"loss": loss_avg}
    for key, total in bucket_loss.items():
        n = bucket_tokens[key]
        if n > 0:
            metrics[key] = total / n
    if hasattr(model, "get_probe_metrics"):
        metrics.update(model.get_probe_metrics(reset=True))
    return metrics


def train_one_epoch_seq2seq(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
    set_diversity_weight: float = 0.0,
    set_diversity_mode: str = DEFAULT_SET_DIVERSITY_MODE,
) -> dict:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    grad_norm_sum = 0.0
    grad_norm_steps = 0
    for src_ids, tgt_ids in dataloader:
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        decoder_input = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]
        src_pad_mask = src_ids.eq(pad_id)
        tgt_pad_mask = decoder_input.eq(pad_id)
        optimizer.zero_grad(set_to_none=True)
        logits = model(src_ids, decoder_input, src_pad_mask, tgt_pad_mask)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_id,
        )
        loss = _maybe_add_set_diversity_loss(
            model,
            loss,
            weight=set_diversity_weight,
            mode=set_diversity_mode,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        _update_diagnostics(model)
        grad_norm_sum += _grad_norm(model)
        grad_norm_steps += 1
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
    loss_avg = total_loss / max(total_tokens, 1)
    grad_norm = grad_norm_sum / grad_norm_steps if grad_norm_steps else None
    return {"loss": loss_avg, "grad_norm": grad_norm}


@torch.no_grad()
def evaluate_seq2seq(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    decode_fn,
    max_len: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    preds = []
    refs = []
    for src_ids, tgt_ids in dataloader:
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        decoder_input = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]
        src_pad_mask = src_ids.eq(pad_id)
        tgt_pad_mask = decoder_input.eq(pad_id)
        logits = model(src_ids, decoder_input, src_pad_mask, tgt_pad_mask)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_id,
        )
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

        decoded = model.greedy_decode(src_ids, src_pad_mask, max_len=max_len)
        for pred_ids, ref_ids in zip(decoded, tgt_ids):
            preds.append(decode_fn(pred_ids.tolist()))
            refs.append(decode_fn(ref_ids.tolist()))

    loss_avg = total_loss / max(total_tokens, 1)
    return {"loss": loss_avg, "preds": preds, "refs": refs}
