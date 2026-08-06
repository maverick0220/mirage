from __future__ import annotations

import pandas as pd


def resample_frame(
    frame: pd.DataFrame,
    timestamp_column: str,
    period_seconds: int,
    interpolation_limit: int = 3,
) -> pd.DataFrame:
    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="coerce")
    result = result.dropna(subset=[timestamp_column]).drop_duplicates(timestamp_column)
    result = result.set_index(timestamp_column).sort_index()
    numeric = result.select_dtypes("number")
    numeric = numeric.resample(f"{period_seconds}s").mean()
    numeric = numeric.interpolate(limit=interpolation_limit, limit_direction="both")
    non_numeric = result.select_dtypes(exclude="number")
    if non_numeric.empty:
        return numeric.reset_index()
    # Do NOT average label/regime columns: use max for binary labels (any anomaly
    # inside the bucket marks it) and first for other categorical columns.
    label_columns = [column for column in non_numeric if str(column).startswith("__label")]
    other_columns = [column for column in non_numeric if column not in label_columns]
    pieces = []
    if other_columns:
        pieces.append(non_numeric[other_columns].resample(f"{period_seconds}s").first())
    if label_columns:
        pieces.append(non_numeric[label_columns].resample(f"{period_seconds}s").max())
    merged = pd.concat(pieces, axis=1) if pieces else non_numeric.iloc[0:0]
    return pd.concat([numeric, merged], axis=1).reset_index()

