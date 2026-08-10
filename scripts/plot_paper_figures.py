"""Paper Figures 3 & 4 with REAL experiment data (replaces the demo plotters).

Figure 3 (two panels, prior-corruption robustness, from RQ4 runs):
  (a) structure-recovery F1 vs prior corruption rate (soft-prior MIRAGE curve
      + hard/no-prior references), 5 seeds;
  (b) detection F1 vs prior corruption rate (soft-prior MIRAGE), 5 seeds.
      Shows that prior corruption mainly hurts structure recovery while
      detection stays stable.

Figure 4 (four panels, boiler controlled-injection case, from RQ5):
  (a) event score time series with injected-fault spans and threshold;
  (b) per-variable local scores of one injected event (injected cause marked);
  (c) learned causal subgraph with role coloring (injected variable marked);
  (d) root-cause ranking for one event (injected ground truth marked).

Usage (needs: pandas, matplotlib, networkx; RQ5 parquet/graph synced from
the server):

    python scripts/plot_paper_figures.py \
        --rq4-runs result_0810/tables/rq4_prior_noise_runs.csv \
        --rq5-scores artifacts/rq5_injected/test_scores.parquet \
        --rq5-report result_0810/report.json \
        --rq5-graph artifacts/rq5_boiler/learned_graphs.npz \
        --output result_0810/figs/

Outputs figure3-results.png and figure4-case.png (same names as the tex
includes), ready to replace manuscript-v2.0/figures/*.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mirage.schemas.graph import DynamicCausalGraph  # noqa: E402

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
BLACK = "#222222"
GRAY = "#6B6B6B"

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.25,
    "lines.markersize": 3.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
}

CORR_METHODS = {
    "mirage_soft": (0.0, "MIRAGE (soft prior)", BLUE, "-"),
    "mirage_corr0.1": (0.1, "MIRAGE (soft prior)", BLUE, "-"),
    "mirage_corr0.3": (0.3, "MIRAGE (soft prior)", BLUE, "-"),
    "mirage_corr0.5": (0.5, "MIRAGE (soft prior)", BLUE, "-"),
    "mirage_hard": (0.0, "Hard-prior MIRAGE", VERMILLION, "--"),
    "mirage_none": (0.0, "No-prior MIRAGE", BLACK, "-."),
}


def clean_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def summarize_seeds(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = frame.groupby("method", observed=True)[metric]
    summary = grouped.agg(mean="mean", std="std", n="count").reset_index()
    return summary


def plot_figure3(runs: pd.DataFrame) -> None:
    soft = runs[runs["method"].isin(list(CORR_METHODS))]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), layout="constrained")
    for ax, metric, ylabel, title in (
        (axes[0], "graph_f1", "Structure-recovery F1", "(a) Prior corruption vs structure recovery"),
        (axes[1], "f1", "Detection F1", "(b) Prior corruption vs detection"),
    ):
        for method, (corr, label, color, style) in CORR_METHODS.items():
            subset = runs[runs["method"].eq(method)]
            if subset.empty:
                continue
            if method in ("mirage_soft", "mirage_corr0.1", "mirage_corr0.3", "mirage_corr0.5"):
                # soft curve: x = corruption rate
                x = np.array([CORR_METHODS[m][0] for m in ("mirage_soft", "mirage_corr0.1", "mirage_corr0.3", "mirage_corr0.5")])
                means = np.array([runs[runs["method"].eq(m)][metric].mean() for m in ("mirage_soft", "mirage_corr0.1", "mirage_corr0.3", "mirage_corr0.5")])
                stds = np.array([runs[runs["method"].eq(m)][metric].std() for m in ("mirage_soft", "mirage_corr0.1", "mirage_corr0.3", "mirage_corr0.5")])
                n = len(subset)
                t = 2.776 if n >= 5 else 12.706
                half = t * stds / np.sqrt(n)
                ax.fill_between(x * 100, means - half, means + half, color=color, alpha=0.12, linewidth=0)
                ax.plot(x * 100, means, color=color, linestyle=style, marker="o", markerfacecolor="white", markeredgewidth=0.9, label=label, zorder=3)
                for m, xv in zip(("mirage_soft", "mirage_corr0.1", "mirage_corr0.3", "mirage_corr0.5"), x):
                    pts = runs.loc[runs["method"].eq(m), metric]
                    ax.scatter(np.full(len(pts), xv * 100), pts, s=8, marker="o", facecolors="none", edgecolors=color, linewidths=0.45, alpha=0.3, zorder=2)
            else:
                # single-point reference (corr = 0)
                ax.scatter([corr * 100], [subset[metric].mean()], s=34, marker="*", color=color, zorder=3, label=label)
        ax.set_xlabel("Corrupted prior edges (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        clean_axes(ax)
        ax.legend(frameon=False, loc="best", handlelength=2.2)
    fig.tight_layout()
    return fig


def _effective_edges(weights: np.ndarray, names: list[str]) -> tuple[nx.DiGraph, dict[tuple[str, str], float]]:
    w = np.asarray(weights)
    if w.ndim == 5:
        w = np.abs(w[1]).max(axis=0)
    elif w.ndim == 4:
        w = np.abs(w).max(axis=0)
    peak = np.abs(w[1:]).max(axis=0)
    d = peak.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(names)
    weights_map = {}
    for s in range(d):
        for t in range(d):
            if s != t and peak[s, t] > 0.05:
                g.add_edge(names[s], names[t])
                weights_map[(names[s], names[t])] = float(peak[s, t])
    return g, weights_map


def plot_figure4(scores_path: Path, report: dict, graph_path: Path) -> None:
    frame = pd.read_parquet(scores_path)
    events = report["injected_events"]
    local_cols = [c for c in frame.columns if c.startswith("local::")]

    fig = plt.figure(figsize=(7.16, 5.05), layout="constrained")
    outer = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0], width_ratios=[1.16, 1.0])
    left_top = outer[0, 0].subgridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.05)
    ax_event = fig.add_subplot(left_top[0, 0])
    ax_graph = fig.add_subplot(outer[1, 0])
    ax_local = fig.add_subplot(outer[0, 1])
    ax_cf = fig.add_subplot(outer[1, 1])

    # (a) event score time series
    ax_event.plot(frame["index"], frame["score"], color=GRAY, linewidth=0.5, label="global score")
    ax_event.plot(frame["index"], frame["event_score"], color=VERMILLION, linewidth=0.8, label="event score (debounced)")
    for ev in events:
        ax_event.axvspan(ev["start"], ev["end"], color=GREEN, alpha=0.16)
    ax_event.axhline(report["metrics"]["threshold"], color=BLUE, linestyle="--", linewidth=0.7, label="threshold")
    ax_event.set_xlabel("Row index (scored test segment)")
    ax_event.set_ylabel("Anomaly score")
    ax_event.set_title("(a) Injected events vs anomaly scores", loc="left", fontweight="bold", pad=4)
    ax_event.legend(frameon=False, fontsize=5.8, ncol=2, handlelength=1.6)
    clean_axes(ax_event)

    # (c) learned causal subgraph
    learned = DynamicCausalGraph.load(graph_path)
    g, wmap = _effective_edges(learned.weights, list(learned.variable_names))
    if len(g.nodes) > 40:  # boiler has 65 vars: keep top-weight edges for readability
        edges_sorted = sorted(wmap, key=lambda e: -wmap[e])
        keep = set(edges_sorted[: int(0.3 * len(edges_sorted))])
        g = nx.DiGraph()
        g.add_nodes_from(learned.variable_names)
        for s, t in keep:
            g.add_edge(s, t)
        wmap = {e: wmap[e] for e in keep}
    pos = nx.spring_layout(g, k=0.55, seed=7)
    first_event = events[0]
    injected_var = first_event["variable"]
    nx.draw_networkx_nodes(g, pos, ax=ax_graph, node_color=BLUE, node_size=260, edgecolors="white", linewidths=0.8)
    nx.draw_networkx_edges(g, pos, ax=ax_graph, edge_color=GRAY, width=0.8, arrows=True, arrowsize=8, arrowstyle="-|>", alpha=0.7)
    if injected_var in g.nodes:
        nx.draw_networkx_nodes(g, pos, nodelist=[injected_var], ax=ax_graph, node_color=VERMILLION, node_size=420, edgecolors=BLACK, linewidths=1.3)
    labels = {n: n if len(n) <= 12 else n[:11] + "…" for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax_graph, font_size=4.6)
    ax_graph.set_title("(c) Learned causal subgraph (boiler)", loc="left", fontweight="bold", pad=4)
    ax_graph.axis("off")

    # (b) local scores of the first injected event
    ev = events[0]
    rows = frame[(frame["index"] >= ev["start"]) & (frame["index"] <= ev["end"])]
    if not rows.empty:
        means = rows[local_cols].mean(axis=0)
        top = means.sort_values(ascending=False).head(12)
        y = np.arange(len(top))
        colors = [VERMILLION if name.split("::", 1)[1] == ev["variable"] else BLUE for name in top.index]
        ax_local.barh(y, top.values, color=colors, edgecolor=BLACK, linewidth=0.45)
        ax_local.set_yticks(y, [name.split("::", 1)[1] for name in top.index])
        ax_local.invert_yaxis()
        ax_local.set_xlabel("Mean local mechanism score (event window)")
        ax_local.set_title("(b) Local mechanism violations", loc="left", fontweight="bold", pad=4)
        clean_axes(ax_local, grid=False)
        ax_local.grid(axis="x", color="#E6E6E6", linewidth=0.55)

    # (d) root-cause ranking for the first event
    candidate_scores = means.sort_values(ascending=False).head(6)
    y_rank = np.arange(len(candidate_scores))
    names_rank = [name.split("::", 1)[1] for name in candidate_scores.index]
    colors_rank = [VERMILLION if n == ev["variable"] else BLUE for n in names_rank]
    ax_cf.barh(y_rank, candidate_scores.values, color=colors_rank, edgecolor=BLACK, linewidth=0.45)
    ax_cf.set_yticks(y_rank, [f"#{i + 1}  {n}" for i, n in enumerate(names_rank)])
    ax_cf.invert_yaxis()
    root_idx = int(y_rank[names_rank.index(ev["variable"])]) if ev["variable"] in names_rank else 0
    ax_cf.annotate(
        "injected ground truth",
        xy=(float(candidate_scores.iloc[root_idx]), root_idx),
        xytext=(float(candidate_scores.iloc[root_idx]) * 0.9, root_idx + 0.6),
        color=VERMILLION,
        fontsize=5.8,
        arrowprops={"arrowstyle": "->", "color": VERMILLION, "lw": 0.8},
    )
    ax_cf.set_xlabel("Mean local mechanism score")
    ax_cf.set_title("(d) Root-cause ranking (injected cause)", loc="left", fontweight="bold", pad=4)
    clean_axes(ax_cf, grid=False)
    ax_cf.grid(axis="x", color="#E6E6E6", linewidth=0.55)

    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq4-runs", required=True, help="result_0810/tables/rq4_prior_noise_runs.csv")
    parser.add_argument("--rq5-scores", help="artifacts/rq5_injected/test_scores.parquet (sync from server)")
    parser.add_argument("--rq5-report", help="result_0810/report.json")
    parser.add_argument("--rq5-graph", help="artifacts/rq5_boiler/learned_graphs.npz (sync from server)")
    parser.add_argument("--output", default="figs")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context(RC_PARAMS):
        runs = pd.read_csv(args.rq4_runs)
        fig3 = plot_figure3(runs)
        fig3.savefig(output / "figure3-results.png", dpi=300, facecolor="white", bbox_inches="tight")
        fig3.savefig(output / "figure3-results.pdf", facecolor="white", bbox_inches="tight")
        plt.close(fig3)
        print(f"saved {output / 'figure3-results.png'}")

        if args.rq5_scores and args.rq5_report and args.rq5_graph:
            report = json.loads(Path(args.rq5_report).read_text(encoding="utf-8"))
            fig4 = plot_figure4(Path(args.rq5_scores), report, Path(args.rq5_graph))
            fig4.savefig(output / "figure4-case.png", dpi=300, facecolor="white", bbox_inches="tight")
            fig4.savefig(output / "figure4-case.pdf", facecolor="white", bbox_inches="tight")
            plt.close(fig4)
            print(f"saved {output / 'figure4-case.png'}")
        else:
            print("[warn] figure4 skipped: need --rq5-scores/--rq5-report/--rq5-graph (sync from server)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
