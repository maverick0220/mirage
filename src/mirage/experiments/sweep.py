"""Sweep runner: executes RQ experiment matrices (methods x seeds) and
aggregates results with paired statistics into paper-ready tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mirage.evaluation.statistics import paired_bootstrap_interval, wilcoxon_paired
from mirage.experiments.baseline_runner import run_anomaly_baseline, run_causal_baseline
from mirage.experiments.registry import is_neural_anomaly
from mirage.experiments.runner import evaluate_run, train_experiment
from mirage.utils import dump_json, load_yaml

_TABLE_METRICS = {
    "causal": [
        "precision",
        "recall",
        "f1",
        "shd",
        "auroc",
        "auprc",
        "sign_accuracy",
    ],
    "anomaly": [
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "event_recall",
        "mean_detection_delay",
        "false_event_count",
    ],
    "mirage": [
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "event_recall",
        "mean_detection_delay",
        "false_event_count",
        "graph_f1",
        "graph_shd",
        "graph_auprc",
        "graph_sign_accuracy",
        "graph_f1_plant",
        "graph_f1_controller",
        "ow_novelty_auroc",
        "ow_false_alarm_rate_unknown",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
    ],
}


def _run_one(
    method: str,
    kind: str,
    config: dict[str, Any],
    seed: int,
    base_dir: Path,
) -> dict[str, Any]:
    from mirage.experiments.registry import (
        ANOMALY_ALIASES,
        ANOMALY_BASELINES,
        CAUSAL_BASELINES,
    )

    is_causal = method in CAUSAL_BASELINES or method in ("pcmci_omega", "cdans")
    is_anomaly = (
        method in ANOMALY_BASELINES
        or method in ANOMALY_ALIASES
        or is_neural_anomaly(method)
    )
    if is_causal or (kind == "causal" and not is_anomaly):
        return run_causal_baseline(
            config["data_dir"],
            method,
            max_lag=int(config.get("max_lag", 3)),
            seed=seed,
        )
    if is_anomaly or kind == "anomaly":
        run_dir = base_dir / "runs" / method / f"seed{seed}"
        return run_anomaly_baseline(
            config["data_dir"],
            method,
            run_dir=run_dir,
            window_size=int(config.get("window_size", 32)),
            epochs=int(config.get("epochs", 10)),
            batch_size=int(config.get("batch_size", 128)),
            seed=seed,
            max_lag=int(config.get("max_lag", 3)),
        )
    # MIRAGE model training. Optional ablation variants encoded in the method
    # name after `mirage_`:
    #   none / hard / corr<rate>  -> prior ablations (RQ4)
    #   single                    -> single-graph backbone (dual-graph ablation)
    #   soft                      -> explicit default soft prior
    run_dir = base_dir / "runs" / method / f"seed{seed}"
    merged = {
        **config,
        "seed": seed,
        "run_dir": str(run_dir),
        "prior_mode": config.get("prior_mode", "soft"),
        "prior_corruption_rate": config.get("prior_corruption_rate", 0.0),
    }
    if method.startswith("mirage_"):
        variant = method.split("_", 1)[1]
        if variant == "none":
            merged["prior_mode"] = "none"
        elif variant == "hard":
            merged["prior_mode"] = "hard"
        elif variant == "soft":
            merged["prior_mode"] = "soft"
        elif variant == "single":
            merged["dual_graph"] = False
        elif variant.startswith("corr"):
            merged["prior_mode"] = "soft"
            merged["prior_corruption_rate"] = float(variant[4:])
    train_experiment(dump_json(merged, base_dir / f"_run_cfg_seed{seed}.yaml"))
    verification = evaluate_run(run_dir)
    metrics = dict(verification.get("metrics", {}))
    if isinstance(verification.get("graph_metrics"), dict):
        metrics.update(
            {
                f"graph_{key}": value
                for key, value in verification["graph_metrics"].items()
                if isinstance(value, (int, float))
            }
        )
    for prefix in ("graph_metrics_plant", "graph_metrics_controller"):
        if isinstance(verification.get(prefix), dict):
            metrics.update(
                {
                    f"graph_f1_{prefix.rsplit('_', 1)[-1]}": verification[prefix].get("f1"),
                    f"graph_shd_{prefix.rsplit('_', 1)[-1]}": verification[prefix].get("shd"),
                }
            )
    if isinstance(verification.get("open_world_metrics"), dict):
        metrics.update(
            {
                f"ow_{key}": value
                for key, value in verification["open_world_metrics"].items()
                if isinstance(value, (int, float))
            }
        )
    if isinstance(verification.get("root_cause_metrics"), dict):
        metrics.update(
            {
                key: value
                for key, value in verification["root_cause_metrics"].items()
                if isinstance(value, (int, float))
            }
        )
    return {"metrics": metrics}


def run_sweep(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base_dir = Path(config.get("artifact_root", "artifacts")) / config.get(
        "research_question", "sweep"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config.get("seeds", [2026])]
    methods = [str(method) for method in config.get("methods", ["mirage"])]
    kind = str(config.get("kind", "mirage"))

    rows: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    for method in methods:
        for seed in seeds:
            result = _run_one(method, kind, config, seed, base_dir)
            metrics = result.get("metrics", {}) if isinstance(result, dict) else result.metrics
            metrics = dict(metrics)
            # For causal baselines the graph metrics live under `metrics` already.
            row = {
                "method": method,
                "seed": seed,
                **{key: metrics.get(key) for key in _TABLE_METRICS.get(kind, [])},
            }
            rows.append(row)
            evaluations.setdefault(method, {})[seed] = metrics

    frame = pd.DataFrame(rows)
    summary_rows = []
    for method in methods:
        subset = frame[frame["method"] == method]
        summary: dict[str, Any] = {"method": method, "seeds": len(subset)}
        for column in _TABLE_METRICS.get(kind, []):
            if column not in subset:
                continue
            values = subset[column].dropna().astype(float)
            if values.empty:
                summary[column] = None
            else:
                summary[f"{column}_mean"] = float(values.mean())
                summary[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)

    statistics: dict[str, Any] = {}
    if kind in ("anomaly", "mirage") and "mirage" in evaluations and len(seeds) > 1:
        reference = evaluations["mirage"]
        for method in methods:
            if method == "mirage" or method not in evaluations:
                continue
            paired: dict[str, Any] = {}
            for metric in _TABLE_METRICS.get(kind, []):
                left = np.array([reference[s].get(metric, np.nan) for s in seeds])
                right = np.array([evaluations[method][s].get(metric, np.nan) for s in seeds])
                valid = ~(np.isnan(left) | np.isnan(right))
                if valid.sum() < 2:
                    continue
                try:
                    paired[metric] = wilcoxon_paired(left[valid], right[valid])
                except ValueError:
                    continue
            if paired:
                statistics[method] = paired

    tables_dir = Path(config.get("report_root", "reports")) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tag = config_path.stem
    frame.to_csv(tables_dir / f"{tag}_runs.csv", index=False)
    summary_frame.to_csv(tables_dir / f"{tag}_summary.csv", index=False)
    markdown_lines = [
        f"# {config.get('research_question', tag)}",
        "",
        "| metric | " + " | ".join(summary_frame["method"].astype(str)) + " |",
        "| --- | " + " | ".join(["---"] * len(summary_frame)) + " |",
    ]
    for metric in _TABLE_METRICS.get(kind, []):
        if f"{metric}_mean" not in summary_frame:
            continue
        cells = []
        for _, row in summary_frame.iterrows():
            mean, std = row.get(f"{metric}_mean"), row.get(f"{metric}_std")
            cells.append(f"{mean:.4f}±{std:.4f}" if mean is not None else "—")
        markdown_lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    dump_json(
        {"statistics": statistics, "summary": summary_frame.to_dict(orient="records")},
        tables_dir / f"{tag}_statistics.json",
    )
    (tables_dir / f"{tag}_summary.md").write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )
    return {
        "runs": len(rows),
        "methods": methods,
        "seeds": seeds,
        "summary": str(tables_dir / f"{tag}_summary.csv"),
        "markdown": str(tables_dir / f"{tag}_summary.md"),
        "statistics": str(tables_dir / f"{tag}_statistics.json"),
    }
