import networkx as nx
import numpy as np

from mirage.evaluation.event_metrics import event_detection_metrics, pointwise_metrics
from mirage.evaluation.graph_metrics import graph_recovery_metrics
from mirage.scoring.calibration import RegimeConditionalCalibrator
from mirage.scoring.path_search import top_causal_paths
from mirage.scoring.root_cause import rank_root_causes


def test_calibration_and_event_metrics():
    validation = np.linspace(0, 1, 100)
    calibrator = RegimeConditionalCalibrator(0.9, minimum_samples=5).fit(
        validation, np.repeat([0, 1], 50)
    )
    assert calibrator.threshold(np.array([0, 1])).shape == (2,)
    labels = np.array([0, 0, 1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.1])
    assert pointwise_metrics(labels, scores, 0.5)["f1"] == 1.0
    assert event_detection_metrics(labels, scores, 0.5)["event_recall"] == 1.0


def test_graph_and_root_cause_metrics():
    truth = np.zeros((2, 3, 3))
    truth[1, 0, 1] = 1
    prediction = truth.copy()
    metrics = graph_recovery_metrics(truth, prediction, 0.5)
    assert metrics["f1"] > 0.99
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=0.8, lag=1)
    graph.add_edge("b", "c", weight=0.7, lag=1)
    assert top_causal_paths(graph, "a", ["c"])[0]["path"] == ["a", "b", "c"]
    ranking = rank_root_causes(np.array([2.0, 0.2, 3.0]), graph, ["a", "b", "c"])
    assert len(ranking) == 3

