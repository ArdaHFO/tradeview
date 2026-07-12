"""Evaluation metrics for the learned model — the part that keeps us honest.

All operate on (y_true in {0,1}, p_pred in [0,1]) lists.
"""
from __future__ import annotations


def accuracy(y: list[int], p: list[float], threshold: float = 0.5) -> float:
    if not y:
        return 0.0
    correct = sum(1 for yi, pi in zip(y, p) if (pi >= threshold) == bool(yi))
    return correct / len(y)


def brier(y: list[int], p: list[float]) -> float:
    """Mean squared error of the probabilities — lower is better."""
    if not y:
        return 0.0
    return sum((pi - yi) ** 2 for yi, pi in zip(y, p)) / len(y)


def auc(y: list[int], p: list[float]) -> float:
    """Area under ROC via the Mann-Whitney rank statistic. 0.5 = coin flip."""
    pos = [pi for yi, pi in zip(y, p) if yi == 1]
    neg = [pi for yi, pi in zip(y, p) if yi == 0]
    if not pos or not neg:
        return 0.5
    # Rank all scores (average ranks for ties), sum ranks of positives.
    paired = sorted((score, idx) for idx, score in enumerate(p))
    ranks = [0.0] * len(p)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[paired[k][1]] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(ranks[idx] for idx, yi in enumerate(y) if yi == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def calibration(y: list[int], p: list[float], bins: int = 10) -> list[dict]:
    """Reliability table: for each probability bin, predicted vs actual rate."""
    buckets: list[dict] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, pi in enumerate(p) if (pi >= lo and (pi < hi or (b == bins - 1 and pi <= hi)))]
        if not idx:
            continue
        buckets.append({
            "range": [round(lo, 2), round(hi, 2)],
            "count": len(idx),
            "predicted": round(sum(p[i] for i in idx) / len(idx), 3),
            "actual": round(sum(y[i] for i in idx) / len(idx), 3),
        })
    return buckets
