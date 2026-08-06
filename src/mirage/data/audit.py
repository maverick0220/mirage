from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def audit_frame(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number])
    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "numeric_columns": int(numeric.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_fraction": {
            str(key): float(value) for key, value in frame.isna().mean().sort_values(ascending=False).items()
        },
        "constant_columns": [
            str(column) for column in numeric.columns if numeric[column].nunique(dropna=True) <= 1
        ],
        "infinite_values": int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()),
    }
    if timestamp_column in frame:
        timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
        differences = timestamps.sort_values().diff().dropna().dt.total_seconds()
        report.update(
            {
                "timestamp_parse_failures": int(timestamps.isna().sum()),
                "timestamp_duplicates": int(timestamps.duplicated().sum()),
                "median_period_seconds": float(differences.median()) if len(differences) else None,
                "non_positive_intervals": int((differences <= 0).sum()),
            }
        )
    return report

