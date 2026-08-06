from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def open_world_metrics(known: np.ndarray, novelty_scores: np.ndarray) -> dict[str, float]:
    unknown = 1 - np.asarray(known).astype(int)
    values = np.asarray(novelty_scores, dtype=float)
    if np.unique(unknown).size < 2:
        return {"novelty_auroc": float("nan"), "novelty_auprc": float("nan")}
    return {
        "novelty_auroc": float(roc_auc_score(unknown, values)),
        "novelty_auprc": float(average_precision_score(unknown, values)),
    }

