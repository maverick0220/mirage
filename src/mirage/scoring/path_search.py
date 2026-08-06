from __future__ import annotations

from collections.abc import Iterable

import networkx as nx


def top_causal_paths(
    graph: nx.DiGraph,
    source: str,
    targets: Iterable[str],
    max_hops: int = 4,
    top_k: int = 5,
) -> list[dict]:
    candidates: list[dict] = []
    for target in targets:
        if source not in graph or target not in graph or source == target:
            continue
        for path in nx.all_simple_paths(graph, source, target, cutoff=max_hops):
            edges = list(zip(path[:-1], path[1:]))
            strength = 1.0
            total_lag = 0
            for left, right in edges:
                attributes = graph[left][right]
                strength *= abs(float(attributes.get("weight", 0.0)))
                total_lag += int(attributes.get("lag", 0))
            candidates.append(
                {"path": path, "strength": strength, "total_lag": total_lag}
            )
    return sorted(candidates, key=lambda item: item["strength"], reverse=True)[:top_k]

