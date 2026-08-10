"""Paper figure generation: training curves, causal-graph comparison, RQ5 scores.

Requires: pip install matplotlib

Usage (on the server, inside the mirage venv):

    python scripts/visualize_paper.py loss \
        --run-dir artifacts/RQ1_graph_recovery/runs/mirage/seed2026 \
        --output figs/training_curve.png

    python scripts/visualize_paper.py graph \
        --truth data/processed/synthetic/default/truth_graph.npz \
        --learned artifacts/RQ1_graph_recovery/runs/mirage/seed2026/learned_graphs.npz \
        --output figs/graph_compare.png

    python scripts/visualize_paper.py rq5 \
        --scores artifacts/rq5_injected/test_scores.parquet \
        --events artifacts/rq5_injected/report.json \
        --output figs/rq5_scores.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mirage.schemas.graph import DynamicCausalGraph  # noqa: E402


def _effective_edges(weights: np.ndarray) -> dict[tuple[int, int], float]:
    """lag>=1 non-diagonal edges with peak magnitude across lags (3D [L,D,D] or 5D)."""
    w = np.asarray(weights)
    if w.ndim == 5:
        w = np.abs(w[1]).max(axis=0)
    elif w.ndim == 4:
        w = np.abs(w).max(axis=0)
    lagged = np.abs(w[1:])
    peak = lagged.max(axis=0)
    edges = {}
    d = peak.shape[0]
    for src in range(d):
        for dst in range(d):
            if src != dst and peak[src, dst] > 1e-6:
                edges[(src, dst)] = float(peak[src, dst])
    return edges


def cmd_loss(args: argparse.Namespace) -> int:
    metrics_path = Path(args.run_dir) / "logs" / "metrics.csv"
    if not metrics_path.exists():
        print(f"metrics.csv not found: {metrics_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(metrics_path)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for column, label, color in (("train/loss", "train loss", "#d62728"), ("validation/loss", "validation loss", "#1f77b4")):
        if column in df.columns:
            rows = df[["epoch", column]].dropna()
            ax.plot(rows["epoch"], rows[column], label=label, color=color, linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (Student-t NLL)")
    ax.set_title("MIRAGE training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"saved {args.output}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    import networkx as nx

    truth = DynamicCausalGraph.load(args.truth)
    learned = DynamicCausalGraph.load(args.learned)
    t_edges = _effective_edges(truth.weights)
    l_edges = _effective_edges(learned.weights)
    names = list(truth.variable_names)
    d = len(names)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, edges, title in ((axes[0], t_edges, "Ground-truth graph"), (axes[1], l_edges, "Learned graph (MIRAGE)")):
        g = nx.DiGraph()
        g.add_nodes_from(range(d))
        g.add_edges_from(edges.keys())
        pos = nx.circular_layout(g)
        weights = [edges[e] for e in g.edges()]
        scale = max(weights) if weights else 1.0
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#4c72b0", node_size=700)
        nx.draw_networkx_edges(
            g, pos, ax=ax, edge_color="#333333", width=[1.5 + 5 * (w / scale) for w in weights],
            arrows=True, arrowstyle="-|>", arrowsize=14,
        )
        nx.draw_networkx_labels(g, pos, ax=ax, labels={i: names[i] for i in range(d)}, font_size=8)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"saved {args.output}")
    return 0


def cmd_rq5(args: argparse.Namespace) -> int:
    frame = pd.read_parquet(args.scores)
    report = json.loads(Path(args.events).read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(frame["index"], frame["score"], color="#999999", linewidth=0.6, label="global score")
    if "event_score" in frame.columns:
        ax.plot(frame["index"], frame["event_score"], color="#d62728", linewidth=0.8, label="event score (debounced)")
    for event in report.get("injected_events", []):
        ax.axvspan(event["start"], event["end"], color="#2ca02c", alpha=0.18)
    ax.axhline(report["metrics"]["threshold"], color="#1f77b4", linestyle="--", linewidth=0.8, label="threshold")
    ax.set_xlabel("row index (test segment)")
    ax.set_ylabel("anomaly score")
    ax.set_title("RQ5 boiler: injected events vs anomaly scores")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"saved {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("loss")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_loss)

    p = sub.add_parser("graph")
    p.add_argument("--truth", required=True)
    p.add_argument("--learned", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("rq5")
    p.add_argument("--scores", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_rq5)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
