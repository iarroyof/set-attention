from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from models.set_only.losses import set_diversity_loss

LOSS_MODE='position_contrastive'


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
        if hasattr(model, "get_last_set_embeddings"):
            set_embs = model.get_last_set_embeddings()
            if set_embs is not None:
                if LOSS_MODE != 'position_contrastive':
                    loss = loss + 0.5 * set_diversity_loss(set_embs, mode=LOSS_MODE, target_similarity=0.3)
                else:
                    num_sets = set_embs.shape[1]
                    set_positions = torch.arange(num_sets, device=set_embs.device)
                    loss = loss + 0.1 * set_diversity_loss(
                        set_embs,
                        mode=LOSS_MODE,
                        set_positions=set_positions,
                        margin=0.3
                    )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if hasattr(model, "diagnostics") and hasattr(model, "router"):
            try:
                router_params = dict(model.router.named_parameters())
                model.diagnostics.update_router_params(router_params)
            except Exception:
                pass
        if hasattr(model, "diagnostics") and hasattr(model, "attention_params"):
            try:
                model.diagnostics.update_params(model.attention_params())
            except Exception:
                pass
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


def train_one_epoch_seq2seq(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
