from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mirage.utils import dump_json


@dataclass(frozen=True)
class ExternalJobSpec:
    method_id: str
    task: str
    upstream_path: str
    upstream_commit: str
    variable_names: tuple[str, ...]
    max_lag: int
    threshold_protocol: str = "validation_only"


def prepare_external_job(
    job_dir: str | Path,
    spec: ExternalJobSpec,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> Path:
    """Freeze common input for a pinned legacy/container baseline environment."""
    output = Path(job_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "input.npz",
        train=np.asarray(train, dtype=np.float32),
        validation=np.asarray(validation, dtype=np.float32),
        test=np.asarray(test, dtype=np.float32),
    )
    dump_json(asdict(spec), output / "job.json")
    dump_json(
        {
            "causal": {"file": "graph.npz", "array": "adjacency", "axes": ["lag", "source", "target"]},
            "anomaly": {
                "file": "scores.parquet",
                "required_columns": ["split", "index", "score"],
                "local_prefix": "local::",
            },
        },
        output / "output_contract.json",
    )
    return output


def load_causal_result(job_dir: str | Path, n_variables: int) -> np.ndarray:
    payload = np.load(Path(job_dir) / "graph.npz", allow_pickle=False)
    adjacency = np.asarray(payload["adjacency"], dtype=np.float32)
    if adjacency.ndim != 3 or adjacency.shape[1:] != (n_variables, n_variables):
        raise ValueError("External graph must be [lag, source, target]")
    return adjacency


def load_anomaly_result(job_dir: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(Path(job_dir) / "scores.parquet")
    required = {"split", "index", "score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"External anomaly scores are missing {sorted(required - set(frame.columns))}")
    if not np.isfinite(frame["score"]).all():
        raise ValueError("External anomaly scores contain non-finite values")
    return frame

