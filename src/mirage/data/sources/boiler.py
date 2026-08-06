from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mirage.data.audit import audit_frame
from mirage.data.boiler_roles import infer_boiler_role
from mirage.data.sources.base import SourceBundle, TimeSeriesSource
from mirage.data.split import chronological_slices
from mirage.schemas import VariableRole, VariableSpec
from mirage.utils import dump_json


class BoilerSource(TimeSeriesSource):
    def __init__(
        self,
        path: str | Path,
        timestamp_column: str = "数据时间",
        encoding: str | None = "gb18030",
        nrows: int | None = None,
        chunk_size: int = 20000,
        missing_limit: float = 0.30,
    ) -> None:
        self.path = Path(path)
        self.timestamp_column = timestamp_column
        self.encoding = encoding
        self.nrows = nrows
        self.chunk_size = chunk_size
        self.missing_limit = missing_limit

    def _encodings(self) -> Iterable[str]:
        values = ["utf-8-sig", self.encoding, "gb18030", "gbk"]
        return (value for index, value in enumerate(values) if value and value not in values[:index])

    def _read(self) -> pd.DataFrame:
        last_error: Exception | None = None
        for encoding in self._encodings():
            try:
                chunks = []
                remaining = self.nrows
                reader = pd.read_csv(
                    self.path,
                    encoding=encoding,
                    chunksize=self.chunk_size,
                    low_memory=False,
                )
                for chunk in reader:
                    if remaining is not None:
                        chunk = chunk.iloc[:remaining]
                        remaining -= len(chunk)
                    chunks.append(chunk)
                    if remaining is not None and remaining <= 0:
                        break
                return pd.concat(chunks, ignore_index=True)
            except UnicodeDecodeError as error:
                last_error = error
        raise RuntimeError(f"Could not decode boiler CSV: {self.path}") from last_error

    def load(self) -> SourceBundle:
        frame = self._read()
        if self.timestamp_column not in frame.columns:
            candidate = next(
                (
                    column
                    for column in frame
                    if "时间" in str(column)
                    or str(column).strip().lower() in {"time", "timestamp", "datetime", "date_time"}
                ),
                None,
            )
            if candidate is None:
                raise ValueError(f"Timestamp column '{self.timestamp_column}' not found")
            self.timestamp_column = str(candidate)
        frame[self.timestamp_column] = pd.to_datetime(frame[self.timestamp_column], errors="coerce")
        for column in frame.columns:
            if column != self.timestamp_column:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        usable = [
            column
            for column in frame.columns
            if column == self.timestamp_column or frame[column].isna().mean() <= self.missing_limit
        ]
        frame = frame[usable].rename(columns={self.timestamp_column: "timestamp"})
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        variables = [
            VariableSpec(str(column), infer_boiler_role(str(column)), subsystem="boiler")
            for column in frame.columns
            if column != "timestamp"
        ]
        return SourceBundle(
            frame=frame,
            variables=variables,
            metadata={
                "source": "boiler",
                "input_path": str(self.path),
                "audit": audit_frame(frame),
            },
        )

    def prepare(
        self, output_dir: str | Path, train_ratio: float = 0.6, val_ratio: float = 0.2
    ) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        bundle = self.load()
        slices = chronological_slices(len(bundle.frame), train_ratio, val_ratio)
        paths: dict[str, Path] = {}
        for name, selection in slices.items():
            path = output / f"{name}.parquet"
            bundle.frame.iloc[selection].reset_index(drop=True).to_parquet(path, index=False)
            paths[name] = path
        paths["variables"] = dump_json(
            [variable.to_dict() for variable in bundle.variables], output / "variables.json"
        )
        paths["audit"] = dump_json(bundle.metadata["audit"], output / "audit.json")
        paths["manifest"] = dump_json(
            {
                "schema_version": "1.0",
                "source": "boiler",
                "note": "Sample is for interface validation; paper experiments use full-year data.",
                "files": {key: path.name for key, path in paths.items()},
            },
            output / "manifest.json",
        )
        return paths

