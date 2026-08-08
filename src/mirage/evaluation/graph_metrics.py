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
    truth: np.ndarray,
    prediction: np.ndarray,
    threshold: float = 1e-6,
    include_diagonal: bool = False,
) -> dict[str, float]:
    """图恢复指标（变量间边，默认排除自环）。

    - 默认阈值 1e-6：稀疏化由上游负责（MIRAGE 图导出已做 top-k，基线输出
      0/1 二值 adjacency），这里只做"非零即边"判定。
    - 默认排除对角：因果图指标通常只看变量间边（SHD/P/R），自环由机制网络
      的 own_state 通道建模（模型图参数化本就禁止自环）；真值图含 lag1
      自回归系数时若不排除对角，SHD 会被固定污染（8 条自环 × 层 × regime）。
    """
    true_array = np.asarray(truth)
    predicted_array = np.asarray(prediction)
    if true_array.shape != predicted_array.shape:
        raise ValueError(f"Graph shapes differ: {true_array.shape} vs {predicted_array.shape}")
    true_bin = np.abs(true_array) > 1e-8
    scores = np.abs(predicted_array)
    if not include_diagonal and true_array.ndim >= 2:
        d = true_array.shape[-1]
        diag_mask = np.broadcast_to(np.eye(d, dtype=bool), true_array.shape)
        true_bin = true_bin & ~diag_mask
        scores = scores.copy()
        scores[diag_mask] = 0.0  # 对角不参与排序/阈值（注意不能设负数：|score| 会把它算成边）
    metrics = _binary_metrics(true_bin, scores, threshold)
    true_sign = np.sign(true_array)
    predicted_sign = np.sign(predicted_array)
    present = true_bin & (scores >= threshold)
    metrics["sign_accuracy"] = float((true_sign[present] == predicted_sign[present]).mean()) if present.any() else 0.0
    return metrics

