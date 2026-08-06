import pandas as pd

from mirage.data.sources.boiler_year import BoilerYearProcessor


def test_boiler_year_streaming_date_split(tmp_path):
    source = tmp_path / "year.csv"
    frame = pd.DataFrame(
        {
            "Time": pd.date_range("2020-01-01", periods=90, freq="1h"),
            "LDCOUT": range(90),
            "LBA30CT": range(100, 190),
        }
    )
    frame.to_csv(source, index=False)
    paths = BoilerYearProcessor(source, chunk_size=17).prepare(
        tmp_path / "processed", "2020-01-02", "2020-01-03"
    )
    assert len(pd.read_parquet(paths["train"])) == 24
    assert len(pd.read_parquet(paths["validation"])) == 24
    assert len(pd.read_parquet(paths["test"])) == 42
    variables = (tmp_path / "processed" / "variables.json").read_text(encoding="utf-8")
    assert '"role": "context"' in variables
    assert '"role": "output"' in variables

