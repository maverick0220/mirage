"""Sweep runner: executes RQ experiment matrices (methods x seeds) and
aggregates results with paired statistics into paper-ready tables.

Incremental-seed workflow: each (method, seed) run directory stores a
`sweep_fingerprint.json` hash of every result-affecting knob. Re-running a sweep
with the SAME config skips already-completed runs (only missing seeds execute);
changing any knob (hyper-parameters, data, model config file, prior mode...)
produces a new fingerprint so stale results are re-run instead of reused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mirage.evaluation.statistics import paired_bootstrap_interval, wilcoxon_paired
from mirage.experiments.baseline_runner import run_anomaly_baseline, run_causal_baseline
from mirage.experiments.registry import CAUSAL_BASELINES, is_neural_anomaly
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

    run_dir = base_dir / "runs" / method / f"seed{seed}"
    is_causal = method in CAUSAL_BASELINES or method in ("pcmci_omega", "cdans")
    is_anomaly = (
        method in ANOMALY_BASELINES
        or method in ANOMALY_ALIASES
        or is_neural_anomaly(method)
    )
    # MIRAGE（含 mirage_* 消融变体）永远走下方 MIRAGE 训练分支，
    # 不能因 kind 字段是 causal/anomaly 被误判成基线方法。
    is_mirage = method == "mirage" or method.startswith("mirage_")
    if not is_mirage and (is_causal or (kind == "causal" and not is_anomaly)):
        return run_causal_baseline(
            config["data_dir"],
            method,
            max_lag=int(config.get("max_lag", 3)),
            seed=seed,
        )
    if not is_mirage and (is_anomaly or kind == "anomaly"):
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
    merged = {
        **config,
        "name": method,
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


def _config_fingerprint(config: dict[str, Any], method: str, seed: int) -> str:
    """Hash every result-affecting knob: config, model_config file content, method, seed."""
    payload = {
        key: value
        for key, value in config.items()
        if key not in ("seeds", "methods")
    }
    model_path = config.get("model_config")
    if model_path and Path(str(model_path)).exists():
        try:
            payload["model_config_content"] = Path(str(model_path)).read_text(encoding="utf-8")
        except OSError:
            pass
    payload["method"] = method
    payload["seed"] = seed
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_completed(run_dir: Path, fingerprint: str) -> dict[str, Any] | None:
    marker = run_dir / "sweep_fingerprint.json"
    result_path = run_dir / "result.json"
    if marker.exists() and result_path.exists():
        try:
            if marker.read_text(encoding="utf-8").strip() == fingerprint:
                return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _write_fingerprint(run_dir: Path, fingerprint: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sweep_fingerprint.json").write_text(fingerprint, encoding="utf-8")


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
    # causal 表里 MIRAGE 的图指标带 graph_ 前缀（graph_shd 等），基线则是裸键
    # （shd）。causal 表统一优先取 graph_ 前缀，保证 MIRAGE 与基线同口径比较。
    prefer_graph_prefix = kind == "causal"
    for method in methods:
        for seed in seeds:
            fingerprint = _config_fingerprint(config, method, seed)
            is_causal_baseline = method in CAUSAL_BASELINES or method in ("pcmci_omega", "cdans")
            run_dir = base_dir / "runs" / method / f"seed{seed}"
            existing = None if is_causal_baseline else _load_completed(run_dir, fingerprint)
            if existing is not None:
                result = existing
                print(f"[skip] {method} seed={seed}: completed with identical config")
            else:
                result = _run_one(method, kind, config, seed, base_dir)
                if not is_causal_baseline:
                    _write_fingerprint(run_dir, fingerprint)
            metrics = result.get("metrics", {}) if isinstance(result, dict) else result.metrics
            metrics = dict(metrics)

            def metric_value(key: str) -> Any:
                if prefer_graph_prefix:
                    return metrics.get(f"graph_{key}", metrics.get(key))
                return metrics.get(key, metrics.get(f"graph_{key}"))

            row = {
                "method": method,
                "seed": seed,
                **{key: metric_value(key) for key in _TABLE_METRICS.get(kind, [])},
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
