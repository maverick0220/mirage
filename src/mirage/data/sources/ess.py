from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mirage.data.sources.base import SourceBundle, TimeSeriesSource
from mirage.data.split import chronological_slices
from mirage.schemas import DynamicCausalGraph, VariableRole, VariableSpec
from mirage.utils import dump_json


class ESSSource(TimeSeriesSource):
    """Adapter for the Expert System for Steam (ESS/ACCP) public benchmark."""

    def __init__(self, data_path: str | Path, adjacency_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self.adjacency_path = Path(adjacency_path)

    def load(self) -> SourceBundle:
        if not self.data_path.exists() or not self.adjacency_path.exists():
            raise FileNotFoundError(
                "ESS data are absent. Run `python scripts/download_ess.py --quicklook`."
            )
        frame = pd.read_csv(self.data_path)
        unnamed = [column for column in frame if str(column).lower().startswith("unnamed")]
        frame = frame.drop(columns=unnamed, errors="ignore")
        # Identify a real timestamp column instead of silently replacing it with a
        # RangeIndex (which destroyed the temporal information and left a useless
        # all-NaN feature column behind after the numeric coercion below).
        candidates = [
            column
            for column in frame
            if str(column).strip().lower()
            in {"timestamp", "time", "datetime", "date_time", "date"}
            or "time" in str(column).lower()
        ]
        timestamp = candidates[0] if candidates else None
        numeric = frame.drop(columns=[timestamp] if timestamp else []).apply(
            pd.to_numeric, errors="coerce"
        )
        # Drop columns that are entirely non-numeric (e.g. an unparsed string
        # timestamp): they would otherwise survive as all-NaN features.
        numeric = numeric.dropna(axis=1, how="all")
        if timestamp is not None:
            parsed = pd.to_datetime(frame[timestamp], errors="coerce")
            numeric.insert(0, "timestamp", parsed.astype("int64"))
        else:
            numeric.insert(0, "timestamp", pd.RangeIndex(len(numeric)))
        variables = [
            VariableSpec(str(column), VariableRole.UNKNOWN, subsystem="ess")
            for column in numeric.columns
            if column != "timestamp"
        ]
        return SourceBundle(
            frame=numeric,
            variables=variables,
            metadata={"source": "ESS", "upstream": "github.com/soerenwengel/essdata"},
        )

    def expert_graph(self, variable_names: list[str]) -> DynamicCausalGraph:
        adjacency = pd.read_csv(self.adjacency_path, index_col=0)
        adjacency.index = adjacency.index.astype(str)
        adjacency.columns = adjacency.columns.astype(str)
        # Upstream ESS convention is row=target and column=source. MIRAGE stores source→target.
        if set(variable_names).issubset(adjacency.index) and set(variable_names).issubset(adjacency.columns):
            graph_names = list(variable_names)
        else:
            # Quicklook data are process variables (PVs), while truth is defined over 23 subsystems.
            graph_names = [name for name in adjacency.index if name in adjacency.columns]
            if not set(graph_names).issubset(variable_names):
                raise ValueError(
                    "ESS expert graph is defined over subsystems, but the loaded data "
                    "are process variables (PVs). A PV->subsystem metadata mapping is "
                    "required before structural evaluation is possible; refusing to "
                    "silently compare graphs with mismatched nodes."
                )
        matrix = adjacency.loc[graph_names, graph_names].to_numpy(dtype=np.float32).T
        np.fill_diagonal(matrix, 0.0)
        return DynamicCausalGraph.from_single_graph(matrix[None], graph_names, "expert")

    def prepare(
        self, output_dir: str | Path, train_ratio: float = 0.6, val_ratio: float = 0.2
    ) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        bundle = self.load()
        names = [variable.name for variable in bundle.variables]
        slices = chronological_slices(len(bundle.frame), train_ratio, val_ratio)
        paths: dict[str, Path] = {}
        for name, selection in slices.items():
            path = output / f"{name}.parquet"
            bundle.frame.iloc[selection].reset_index(drop=True).to_parquet(path, index=False)
            paths[name] = path
        paths["expert_graph"] = self.expert_graph(names).save(output / "expert_graph.npz")
        paths["variables"] = dump_json(
            [variable.to_dict() for variable in bundle.variables], output / "variables.json"
        )
        paths["manifest"] = dump_json(
            {
                "schema_version": "1.0",
                "source": "ESS",
                "data_node_level": "process_variable",
                "truth_node_level": "subsystem",
                "orientation": "stored as source_to_target; upstream CSV was transposed from row=target",
                "note": "PV-to-subsystem metadata from the full dataset is required for subsystem-level graph evaluation.",
                "files": {key: path.name for key, path in paths.items()},
            },
            output / "manifest.json",
        )
        return paths

