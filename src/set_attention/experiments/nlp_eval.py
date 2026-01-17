from __future__ import annotations
import math
from collections import Counter
from typing import List


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
    # Very small, simple BLEU for sanity checks (not sacrebleu)
    weights = [1.0 / max_n] * max_n
    p_ns = []
    for n in range(1, max_n + 1):
        num = 0
        den = 0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = _ngram_counts(ref, n)
            hyp_counts = _ngram_counts(hyp, n)
            overlap = sum((hyp_counts & ref_counts).values())
            num += overlap
            den += max(1, sum(hyp_counts.values()))
        p_ns.append((num / den) if den > 0 else 0.0)
    # geometric mean
    sum_logs = 0.0
    for w, p in zip(weights, p_ns):
        sum_logs += w * (0.0 if p == 0 else math.log(p))
    geo = math.exp(sum_logs)
    # brevity penalty
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
    return float(bp * geo)


def sentence_bleu_multi_ref(references: List[List[str]], hypothesis: List[str], max_n: int = 4) -> float:
    if not references or not hypothesis:
        return 0.0
    weights = [1.0 / max_n] * max_n
    p_ns = []
    for n in range(1, max_n + 1):
        hyp_counts = _ngram_counts(hypothesis, n)
        if not hyp_counts:
            p_ns.append(0.0)
            continue
        ref_max = Counter()
        for ref in references:
            ref_counts = _ngram_counts(ref, n)
            for ng, count in ref_counts.items():
                if count > ref_max.get(ng, 0):
                    ref_max[ng] = count
        overlap = sum(min(count, ref_max.get(ng, 0)) for ng, count in hyp_counts.items())
        den = max(1, sum(hyp_counts.values()))
        p_ns.append(overlap / den)
    sum_logs = 0.0
    for w, p in zip(weights, p_ns):
        sum_logs += w * (0.0 if p == 0 else math.log(p))
    geo = math.exp(sum_logs)
    hyp_len = len(hypothesis)
    ref_lens = [len(r) for r in references if r]
    if not ref_lens:
        return 0.0
    closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))
    bp = 1.0 if hyp_len > closest_ref_len else math.exp(1 - closest_ref_len / max(1, hyp_len))
    return float(bp * geo)


def self_bleu(hypotheses: List[List[str]], max_n: int = 4) -> float:
    if len(hypotheses) < 2:
        return 0.0
    scores = []
    for i, hyp in enumerate(hypotheses):
        refs = [h for j, h in enumerate(hypotheses) if j != i]
        scores.append(sentence_bleu_multi_ref(refs, hyp, max_n=max_n))
    return float(sum(scores) / max(1, len(scores)))


def distinct_n(hypotheses: List[List[str]], n: int = 2) -> float:
    if n <= 0:
        return 0.0
    ngrams = []
    for hyp in hypotheses:
        if len(hyp) < n:
            continue
        for i in range(len(hyp) - n + 1):
            ngrams.append(tuple(hyp[i : i + n]))
    if not ngrams:
        return 0.0
    return float(len(set(ngrams)) / len(ngrams))


def rouge_l(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    # Simple ROUGE-L recall averaged over pairs
    def lcs(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[m][n]

    scores = []
    for ref, hyp in zip(references, hypotheses):
        if len(ref) == 0:
            scores.append(0.0)
            continue
        scores.append(lcs(ref, hyp) / len(ref))
    return float(sum(scores) / max(1, len(scores)))
