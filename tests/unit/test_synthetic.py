import json

import pandas as pd

from mirage.data.sources.synthetic import ClosedLoopSCMGenerator, SyntheticSCMConfig
from mirage.schemas import DynamicCausalGraph


def test_closed_loop_scm_contract(tmp_path):
    config = SyntheticSCMConfig(
        seed=2, n_steps=240, n_variables=8, n_regimes=2, max_lag=2, anomaly_rate=0.1
    )
    paths = ClosedLoopSCMGenerator(config).prepare(tmp_path)
    assert set(["train", "validation", "test", "truth_graph", "events", "manifest"]).issubset(paths)
    train = pd.read_parquet(paths["train"])
    assert "__regime" in train and "__label" in train
    graph = DynamicCausalGraph.load(paths["truth_graph"])
    assert graph.weights.shape == (2, 2, 3, 8, 8)
    events = json.loads(paths["events"].read_text(encoding="utf-8"))
    assert events and all(event["start_index"] >= 192 for event in events)

