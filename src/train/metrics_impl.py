from __future__ import annotations

from typing import List

import torch


def perplexity(loss: float) -> float:
    return float(torch.exp(torch.tensor(loss)).item())


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
