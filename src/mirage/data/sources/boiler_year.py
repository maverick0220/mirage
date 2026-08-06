from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mirage.data.boiler_roles import infer_boiler_role
from mirage.schemas import VariableSpec
from mirage.utils import dump_json


class BoilerYearProcessor:
    """Two-pass, bounded-memory processor for a full-year boiler CSV."""

    def __init__(
        self,
        path: str | Path,
        timestamp_column: str = "Time",
        encoding: str | None = None,
        chunk_size: int = 200_000,
        missing_limit: float = 0.30,
    ) -> None:
        self.path = Path(path)
        self.timestamp_column = timestamp_column
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.missing_limit = missing_limit

    def _candidate_encodings(self) -> Iterable[str]:
        values = ["utf-8-sig", self.encoding, "gb18030", "gbk"]
        return (value for index, value in enumerate(values) if value and value not in values[:index])

    @staticmethod
    def _find_timestamp(columns: Iterable[str], preferred: str) -> str | None:
        for column in columns:
            normalized = str(column).strip().lower()
            if str(column) == preferred or "时间" in str(column) or normalized in {
                "time",
                "timestamp",
                "datetime",
                "date_time",
            }:
                return str(column)
        return None

    def _resolve_encoding_and_time(self) -> tuple[str, str]:
        last_error: Exception | None = None
        for encoding in self._candidate_encodings():
            try:
                header = pd.read_csv(self.path, encoding=encoding, nrows=0)
                timestamp = self._find_timestamp(header.columns, self.timestamp_column)
                if timestamp is not None:
                    return encoding, timestamp
            except UnicodeDecodeError as error:
                last_error = error
        raise ValueError("Could not decode CSV and identify its timestamp column") from last_error

    def _chunks(self, encoding: str) -> Iterable[pd.DataFrame]:
        return pd.read_csv(
            self.path,
            encoding=encoding,
            chunksize=self.chunk_size,
            low_memory=False,
        )

    @staticmethod
    def _write(writer: pq.ParquetWriter | None, frame: pd.DataFrame, path: Path) -> pq.ParquetWriter:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        return writer

    def prepare(
        self,
        output_dir: str | Path,
        train_end: str,
        validation_end: str,
    ) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        encoding, timestamp_column = self._resolve_encoding_and_time()
        total_rows = 0
        missing: defaultdict[str, int] = defaultdict(int)
        columns: list[str] | None = None
        parse_failures = 0
        minimum_time: pd.Timestamp | None = None
        maximum_time: pd.Timestamp | None = None
        for chunk in self._chunks(encoding):
            if columns is None:
                columns = [str(column) for column in chunk.columns if str(column) != timestamp_column]
            timestamps = pd.to_datetime(chunk[timestamp_column], errors="coerce")
            parse_failures += int(timestamps.isna().sum())
            valid = timestamps.dropna()
            if len(valid):
                current_min, current_max = valid.min(), valid.max()
                minimum_time = current_min if minimum_time is None else min(minimum_time, current_min)
                maximum_time = current_max if maximum_time is None else max(maximum_time, current_max)
            total_rows += len(chunk)
            for column in columns:
                values = pd.to_numeric(chunk[column], errors="coerce")
                missing[column] += int(values.isna().sum())
        if not columns or total_rows == 0:
            raise ValueError("Full-year CSV has no usable rows")
        usable = [column for column in columns if missing[column] / total_rows <= self.missing_limit]
        train_boundary = pd.Timestamp(train_end)
        validation_boundary = pd.Timestamp(validation_end)
        if validation_boundary <= train_boundary:
            raise ValueError("validation_end must be after train_end")

        split_paths = {name: output / f"{name}.parquet" for name in ("train", "validation", "test")}
        split_writers: dict[str, pq.ParquetWriter | None] = {name: None for name in split_paths}
        month_writers: dict[str, pq.ParquetWriter] = {}
        split_rows: defaultdict[str, int] = defaultdict(int)
        duplicate_timestamps = 0
        non_monotonic_chunks = 0
        try:
            for chunk in self._chunks(encoding):
                frame = pd.DataFrame({"timestamp": pd.to_datetime(chunk[timestamp_column], errors="coerce")})
                for column in usable:
                    frame[column] = pd.to_numeric(chunk[column], errors="coerce")
                frame = frame.dropna(subset=["timestamp"])
                duplicate_timestamps += int(frame["timestamp"].duplicated().sum())
                if not frame["timestamp"].is_monotonic_increasing:
                    non_monotonic_chunks += 1
                frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
                # Interpolate ONLY inside the training segment. Interpolating the
                # whole chunk before splitting would let validation/test values
                # leak into training rows through the backward fill (limit_direction
                # "both"), violating the fit-on-train-only rule. Validation/test
                # missing values are left for the RobustScaler (NaN -> median).
                train_mask = frame["timestamp"] < train_boundary
                if train_mask.any():
                    frame.loc[train_mask, usable] = frame.loc[train_mask, usable].interpolate(
                        limit=3, limit_direction="both"
                    )
                for period, monthly in frame.groupby(frame["timestamp"].dt.to_period("M")):
                    key = str(period)
                    month_path = output / "monthly" / f"{key}.parquet"
                    month_writers[key] = self._write(month_writers.get(key), monthly, month_path)
                selections = {
                    "train": frame["timestamp"] < train_boundary,
                    "validation": (frame["timestamp"] >= train_boundary)
                    & (frame["timestamp"] < validation_boundary),
                    "test": frame["timestamp"] >= validation_boundary,
                }
                for split, mask in selections.items():
                    selected = frame.loc[mask]
                    if len(selected):
                        split_writers[split] = self._write(
                            split_writers[split], selected, split_paths[split]
                        )
                        split_rows[split] += len(selected)
        finally:
            for writer in [*split_writers.values(), *month_writers.values()]:
                if writer is not None:
                    writer.close()
        empty = [split for split in split_paths if split_rows[split] == 0]
        if empty:
            raise ValueError(f"Date boundaries produced empty splits: {empty}")

        variables = [
            VariableSpec(name, infer_boiler_role(name), subsystem="boiler").to_dict()
            for name in usable
        ]
        audit = {
            "input_path": str(self.path),
            "encoding": encoding,
            "raw_rows": total_rows,
            "usable_variables": len(usable),
            "dropped_variables": [column for column in columns if column not in usable],
            "timestamp_parse_failures": parse_failures,
            "timestamp_duplicates_within_chunks": duplicate_timestamps,
            "non_monotonic_chunks": non_monotonic_chunks,
            "start": minimum_time,
            "end": maximum_time,
            "split_rows": dict(split_rows),
            "missing_fraction": {column: missing[column] / total_rows for column in columns},
        }
        paths: dict[str, Path] = {**split_paths}
        paths["variables"] = dump_json(variables, output / "variables.json")
        paths["audit"] = dump_json(audit, output / "audit.json")
        paths["manifest"] = dump_json(
            {
                "schema_version": "1.0",
                "source": "boiler_full_year",
                "streaming": True,
                "split_dates": {"train_end": train_end, "validation_end": validation_end},
                "files": {key: path.name for key, path in paths.items()},
            },
            output / "manifest.json",
        )
        return paths

