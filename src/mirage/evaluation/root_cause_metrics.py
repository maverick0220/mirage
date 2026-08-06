from __future__ import annotations

import numpy as np


def root_cause_metrics(rankings: list[list[str]], truths: list[str], k_values=(1, 3, 5)) -> dict[str, float]:
    if len(rankings) != len(truths):
        raise ValueError("rankings and truths must have equal length")
    reciprocal_ranks = []
    metrics: dict[str, float] = {}
    for ranking, truth in zip(rankings, truths, strict=False):
        try:
            reciprocal_ranks.append(1.0 / (ranking.index(truth) + 1))
        except ValueError:
            reciprocal_ranks.append(0.0)
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if rankings else 0.0
    for k in k_values:
        metrics[f"hit_at_{k}"] = float(
            np.mean([truth in ranking[:k] for ranking, truth in zip(rankings, truths, strict=False)])
        ) if rankings else 0.0
    return metrics

