from __future__ import annotations

import networkx as nx
import numpy as np

from mirage.scoring.path_search import top_causal_paths


def rank_root_causes(
    local_scores: np.ndarray,
    graph: nx.DiGraph,
    variable_names: list[str] | tuple[str, ...],
    counterfactual_recovery: dict[str, float] | None = None,
    top_k: int = 5,
) -> list[dict]:
    scores = np.asarray(local_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError(
            f"local_scores must be 1-D per-variable scores, got shape {scores.shape}"
        )
    if len(scores) != len(variable_names):
        raise ValueError(
            f"local_scores length {len(scores)} does not match variable_names "
            f"length {len(variable_names)}"
        )
    anomalous_targets = [
        variable_names[index] for index in np.argsort(scores)[::-1][: min(3, len(scores))]
    ]
    recovery = counterfactual_recovery or {}
    rows = []
    for index, source in enumerate(variable_names):
        downstream = top_causal_paths(graph, source, anomalous_targets, max_hops=4, top_k=5)
        path_score = sum(path["strength"] for path in downstream)
        local = float(scores[index])
        value = local + path_score + float(recovery.get(source, 0.0))
        rows.append(
            {
                "variable": source,
                "score": value,
                "local_score": local,
                "path_score": path_score,
                "counterfactual_recovery": float(recovery.get(source, 0.0)),
                "paths": downstream,
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]

