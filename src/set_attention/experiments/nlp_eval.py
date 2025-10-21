from __future__ import annotations
from collections import Counter
from typing import List


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
    # Very small, simple BLEU for sanity checks (not sacrebleu)
    import math

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

