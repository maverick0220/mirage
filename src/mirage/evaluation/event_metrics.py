from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def pointwise_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    truth = np.asarray(labels).astype(int)
    values = np.asarray(scores, dtype=float)
    prediction = (values >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average="binary", zero_division=0
    )
    result = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    if np.unique(truth).size == 2:
        result.update(
            auroc=float(roc_auc_score(truth, values)),
            auprc=float(average_precision_score(truth, values)),
        )
    return result


def _segments(binary: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(binary, dtype=bool)
    padded = np.pad(values.astype(int), (1, 1))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0] - 1
    return list(zip(starts.tolist(), stops.tolist()))


def event_detection_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    truth_segments = _segments(np.asarray(labels) > 0)
    prediction = np.asarray(scores) >= threshold
    detected = 0
    delays: list[int] = []
    for start, stop in truth_segments:
        positions = np.where(prediction[start : stop + 1])[0]
        if len(positions):
            detected += 1
            delays.append(int(positions[0]))
    predicted_segments = _segments(prediction)
    false_events = sum(
        not np.asarray(labels)[start : stop + 1].astype(bool).any()
        for start, stop in predicted_segments
    )
    return {
        "event_recall": float(detected / max(1, len(truth_segments))),
        "mean_detection_delay": float(np.mean(delays)) if delays else float("nan"),
        "false_event_count": float(false_events),
        "true_event_count": float(len(truth_segments)),
    }

