from pathlib import Path

import pandas as pd
import pytest

from mirage.data.sources.boiler import BoilerSource
from mirage.data.sources.ess import ESSSource


def test_ess_fixture_prepare(tmp_path):
    data = tmp_path / "data.csv"
    adjacency = tmp_path / "adj.csv"
    pd.DataFrame({"a": range(20), "b": range(20, 40)}).to_csv(data, index=False)
    pd.DataFrame([[0, 1], [0, 0]], index=["a", "b"], columns=["a", "b"]).to_csv(adjacency)
    paths = ESSSource(data, adjacency).prepare(tmp_path / "prepared")
    assert Path(paths["expert_graph"]).exists()
    assert len(pd.read_parquet(paths["test"])) == 4


@pytest.mark.integration
def test_real_boiler_sample_can_be_read():
    path = Path("data/raw/boiler/sample/数据样例.csv")
    if not path.exists():
        pytest.skip("Boiler sample is not present")
    bundle = BoilerSource(path, nrows=256, chunk_size=128).load()
    assert len(bundle.frame) == 256
    assert len(bundle.variables) >= 10
    assert "timestamp" in bundle.frame

