from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _binary_metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(truth).astype(bool).ravel()
    values = np.abs(np.asarray(score)).ravel()
    prediction = values >= threshold
    tp = int(np.logical_and(labels, prediction).sum())
    fp = int(np.logical_and(~labels, prediction).sum())
    fn = int(np.logical_and(labels, ~prediction).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall + 1e-12)),
        "shd": float(np.logical_xor(labels, prediction).sum()),
    }
    if labels.any() and (~labels).any():
        result["auroc"] = float(roc_auc_score(labels, values))
        result["auprc"] = float(average_precision_score(labels, values))
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")
    return result


def graph_recovery_metrics(
    truth: np.ndarray, prediction: np.ndarray, threshold: float = 1e-6
) -> dict[str, float]:
    """图恢复指标。默认阈值 1e-6：稀疏化由上游负责（MIRAGE 图导出已做
    top-k，基线输出 0/1 二值 adjacency），这里只做"非零即边"判定。
    不要把默认阈值设成 0.1——MIRAGE 学习图权重普遍在 0.03~0.09，
    0.1 阈值会把 top-k 已选中的边全部误杀（SHD 虚高）。
    """
    true_array = np.asarray(truth)
    predicted_array = np.asarray(prediction)
    if true_array.shape != predicted_array.shape:
        raise ValueError(f"Graph shapes differ: {true_array.shape} vs {predicted_array.shape}")
    metrics = _binary_metrics(np.abs(true_array) > 1e-8, predicted_array, threshold)
    true_sign = np.sign(true_array)
    predicted_sign = np.sign(predicted_array)
    present = (np.abs(true_array) > 1e-8) & (np.abs(predicted_array) >= threshold)
    metrics["sign_accuracy"] = float((true_sign[present] == predicted_sign[present]).mean()) if present.any() else 0.0
    return metrics

