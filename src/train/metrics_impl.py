from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def perplexity(loss: float) -> float:
    return float(torch.exp(torch.tensor(loss)).item())


def masked_lm_loss_and_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, int, int]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets.ne(ignore_index)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return flat_logits.sum() * 0.0, 0, 0
    loss_sum = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=ignore_index,
        reduction="sum",
    )
    predictions = flat_logits.argmax(dim=-1)
    correct = int((predictions.eq(flat_targets) & valid).sum().item())
    return loss_sum / valid_count, valid_count, correct


def bleu_score(preds: List[str], refs: List[str]) -> float:
    try:
        import sacrebleu  # type: ignore
    except Exception as exc:
        raise ImportError("sacrebleu is required for BLEU computation.") from exc
    result = sacrebleu.corpus_bleu(preds, [refs])
    return float(result.score)


def rouge_l_f1(preds: List[str], refs: List[str]) -> float:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except Exception as exc:
        raise ImportError("rouge-score is required for ROUGE-L computation.") from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(preds, refs):
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        scores.append(score)
    return float(sum(scores) / max(len(scores), 1))


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    try:
        from torchmetrics.classification import MulticlassAccuracy  # type: ignore
    except Exception as exc:
        raise ImportError("torchmetrics is required for accuracy computation.") from exc
    num_classes = logits.shape[-1]
    metric = MulticlassAccuracy(num_classes=num_classes, top_k=k)
    return float(metric(logits, targets).item())
