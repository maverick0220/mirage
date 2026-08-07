"""Unified runner for baseline experiments.

Causal baselines produce a [lag, source, target] adjacency compared against the
truth graph; anomaly baselines produce per-step scores calibrated on the
validation split and evaluated with the same event/point metrics as MIRAGE.
Outputs are written in the same layout as a MIRAGE run so the sweep aggregator
can treat them uniformly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mirage.data.transforms import RobustScaler
from mirage.evaluation.event_metrics import event_detection_metrics, pointwise_metrics
from mirage.evaluation.graph_metrics import graph_recovery_metrics
from mirage.experiments.registry import create_baseline
from mirage.schemas import DynamicCausalGraph
from mirage.utils import dump_json, seed_everything


def _load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(data_dir / "train.parquet")
    validation = pd.read_parquet(data_dir / "validation.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    feature_names = [
        str(column)
        for column in train.select_dtypes(include=[np.number]).columns
        if not str(column).startswith("__")
    ]
    return train, validation, test, feature_names


def _truth_effective(data_dir: Path) -> np.ndarray:
    """Truth effective graph aggregated over regimes: [L+1, D, D]."""
    path = data_dir / "truth_graph.npz"
    if not path.exists():
        raise FileNotFoundError(f"truth_graph.npz not found in {data_dir}")
    truth = DynamicCausalGraph.load(path).weights  # [2, K, L+1, D, D]
    effective = truth[1]
    return np.abs(effective).max(axis=0)  # [L+1, D, D]


def _align_to_shape(adjacency: np.ndarray, target_lags: int, n_variables: int) -> np.ndarray:
    """Pad/truncate a [lag, D, D] adjacency to (target_lags, n_variables, n_variables)."""
    lag_axis, d = adjacency.shape[0], adjacency.shape[1]
    aligned = np.zeros((target_lags, n_variables, n_variables), dtype=np.float32)
    keep_lags = min(lag_axis, target_lags)
    keep_vars = min(d, n_variables)
    aligned[:keep_lags, :keep_vars, :keep_vars] = adjacency[:keep_lags, :keep_vars, :keep_vars]
    return aligned


def run_causal_baseline(
    data_dir: str | Path,
    baseline_name: str,
    max_lag: int = 3,
    seed: int = 2026,
    **params,
) -> dict:
    data_dir = Path(data_dir)
    train, _, _, feature_names = _load_splits(data_dir)
    seed_everything(seed)
    scaler = RobustScaler().fit(train[feature_names].to_numpy())
    values = scaler.transform(train[feature_names].to_numpy())
    baseline = create_baseline(baseline_name, max_lag=max_lag, **params)
    baseline.fit(values, feature_names)
    adjacency = np.asarray(baseline.adjacency(), dtype=np.float32)
    truth = _truth_effective(data_dir)
    aligned = _align_to_shape(adjacency, truth.shape[0], truth.shape[1])
    metrics = graph_recovery_metrics(truth, aligned)
    return {
        "baseline": baseline_name,
        "kind": "causal",
        "metrics": metrics,
        "adjacency_shape": list(aligned.shape),
        "seed": seed,
    }


def run_anomaly_baseline(
    data_dir: str | Path,
    baseline_name: str,
    run_dir: str | Path,
    window_size: int = 32,
    batch_size: int = 128,
    epochs: int = 10,
    seed: int = 2026,
    max_lag: int = 3,
    **params,
) -> dict:
    data_dir = Path(data_dir)
    run_dir = Path(run_dir)
    train, validation, test, feature_names = _load_splits(data_dir)
    seed_everything(seed)
    scaler = RobustScaler().fit(train[feature_names].to_numpy())
    train_values = scaler.transform(train[feature_names].to_numpy())
    validation_values = scaler.transform(validation[feature_names].to_numpy())
    test_values = scaler.transform(test[feature_names].to_numpy())
    from mirage.experiments.registry import ANOMALY_BASELINES

    if baseline_name in ANOMALY_BASELINES:
        baseline = create_baseline(baseline_name)
    else:
        baseline = create_baseline(
            baseline_name,
            window_size=window_size,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            max_lag=max_lag,
            **params,
        )
    baseline.fit(train_values, feature_names)
    validation_scores = baseline.score(validation_values)
    threshold = float(np.quantile(validation_scores, 0.99)) if len(validation_scores) else float("nan")
    test_scores = baseline.score(test_values)
    prediction = test_scores >= threshold
    # 对齐语义：统计基线逐点打分（长度=len(test)），神经基线窗口预测
    # （长度=len(test)-window_size）。统一按实际 score 长度对齐标签与 index。
    start = len(test) - len(test_scores)
    if start < 0:
        raise ValueError(f"score length {len(test_scores)} exceeds test length {len(test)}")
    labels = test["__label"].to_numpy() if "__label" in test else None
    test_labels = labels[start:] if labels is not None else None

    frame = pd.DataFrame(
        {
            "index": np.arange(start, start + len(test_scores)),
            "score": test_scores,
            "threshold": threshold,
            "prediction": prediction,
        }
    )
    if test_labels is not None:
        frame["label"] = test_labels
    metrics: dict[str, float] = {}
    if test_labels is not None:
        metrics.update(pointwise_metrics(test_labels, test_scores, threshold))
        metrics.update(event_detection_metrics(test_labels, test_scores, threshold))
    run_dir.mkdir(parents=True, exist_ok=True)
    score_path = run_dir / "predictions" / "test_scores.parquet"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(score_path, index=False)
    result = {
        "run_name": f"{baseline_name}_seed{seed}",
        "status": "completed",
        "baseline": baseline_name,
        "kind": "anomaly",
        "metrics": metrics,
        "threshold": threshold,
        "metadata": {
            "features": feature_names,
            "window_size": window_size,
            "epochs": epochs,
            "seed": seed,
        },
    }
    dump_json(result, run_dir / "result.json")
    return result
