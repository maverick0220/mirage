import numpy as np
import pandas as pd

from mirage.baselines.job_adapter import (
    ExternalJobSpec,
    load_anomaly_result,
    load_causal_result,
    prepare_external_job,
)


def test_external_job_contract_round_trip(tmp_path):
    values = np.arange(80, dtype=np.float32).reshape(20, 4)
    spec = ExternalJobSpec("demo", "causal", "vendor/demo", "abc", tuple("abcd"), 2)
    prepare_external_job(tmp_path, spec, values[:10], values[10:15], values[15:])
    np.savez_compressed(tmp_path / "graph.npz", adjacency=np.zeros((3, 4, 4)))
    assert load_causal_result(tmp_path, 4).shape == (3, 4, 4)
    pd.DataFrame({"split": ["test"], "index": [0], "score": [0.5]}).to_parquet(
        tmp_path / "scores.parquet"
    )
    assert len(load_anomaly_result(tmp_path)) == 1

