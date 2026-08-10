"""RQ5 boiler controlled-injection evaluation.

Injects synthetic faults (bias/drift/stuck/variance) into a sub-segment of the
boiler TEST split (never seen by the trained model), scores it with the trained
MIRAGE checkpoint (and optionally statistical baselines), and reports
pointwise / event-level / root-cause metrics using the injected events as
ground truth.

Event detection uses a per-variable threshold + debounce rule (a variable must
stay above its clean-segment quantile for CONFIRM consecutive rows to alarm),
which is robust to the topq-mean dilution of single-variable faults in the
65-variable boiler data.

Usage (on the server, inside the mirage venv):

    python scripts/rq5_injection_eval.py \
        --run-dir artifacts/rq5_boiler \
        --output artifacts/rq5_injected \
        [--rows 50000] [--events 10] [--duration 800] [--gap 3000] \
        [--quantile 0.999] [--confirm 5] [--variance-scale 5.0] \
        [--baselines linear_residual,robust_zscore] [--seed 2026] [--device cuda]

Baselines (linear_residual, robust_zscore) are fit on the boiler TRAIN split
and scored on the same injected segment with the same threshold+debounce rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mirage.baselines.anomaly import LinearResidualDetector, RobustZScore
from mirage.baselines.anomaly.neural import NeuralAnomalyBaseline
from mirage.data.injection import InjectionSpec, inject_fault
from mirage.data.transforms import RobustScaler
from mirage.data.window import IndustrialWindowDataset
from mirage.evaluation.event_metrics import event_detection_metrics, pointwise_metrics
from mirage.evaluation.root_cause_metrics import root_cause_metrics
from mirage.experiments.runner import _compile_prior, _load_variables, _masked_to_targets
from mirage.priors.role_mask import controller_allowed_mask, plant_allowed_mask, role_graph_assignment
from mirage.training.lightning_module import MIRAGELightningModule

# Variables that must never be injection targets: the fuel-flow cumulative meter
# (absurd scale, 16 orders above the rest) and the all-zero feed-water flow.
EXCLUDED_VARIABLES = {"FWFUELDIV", "TOTFWFL"}

FAULT_KINDS = ["bias", "drift", "stuck", "variance"]

BASELINE_CLASSES = {
    "linear_residual": LinearResidualDetector,
    "robust_zscore": RobustZScore,
}

# Deep anomaly-detection baselines of the same family as MIRAGE (fit on the
# boiler TRAIN split with a neural predictor, scored on the injected segment).
# A subset is usually enough: gdn (graph-based, closest to MIRAGE), tranad
# (Transformer), dlinear (linear forecasting), lstm (recurrent).
NEURAL_BASELINES = {
    "lstm", "transformer", "anomaly_transformer", "tranad", "dlinear", "gdn", "omnianomaly", "mtad_gat",
}


def _load_scaler_state(data_state_path: Path) -> tuple[RobustScaler, list[str]]:
    state = json.loads(data_state_path.read_text(encoding="utf-8"))
    scaler = RobustScaler()
    scaler.median_ = np.asarray(state["median"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
    return scaler, [str(name) for name in state["feature_names"]]


def _injection_candidates(feature_names: list[str], variables_path: Path) -> list[int]:
    """Column indices of non-context variables minus the excluded ones."""
    if variables_path.exists():
        payload = json.loads(variables_path.read_text(encoding="utf-8"))
        roles = {str(v["name"]): str(v.get("role", "")) for v in payload}
    else:
        roles = {}
    return [
        index
        for index, name in enumerate(feature_names)
        if roles.get(name, "") != "context" and name not in EXCLUDED_VARIABLES
    ]


def _build_event_plan(
    candidates: list[int],
    n_events: int,
    duration: int,
    gap: int,
    start_offset: int,
    rng: np.random.Generator,
    variance_magnitude: float = 5.0,
) -> list[InjectionSpec]:
    if not candidates:
        raise ValueError("no injection candidates (all features are context/excluded?)")
    specs: list[InjectionSpec] = []
    cursor = start_offset
    for i in range(n_events):
        variable = candidates[int(rng.integers(0, len(candidates)))]
        kind = FAULT_KINDS[i % len(FAULT_KINDS)]
        magnitude = {"bias": 10.0, "drift": 10.0, "stuck": 0.0, "variance": variance_magnitude}[kind]
        specs.append(
            InjectionSpec(start=cursor, duration=duration, variable=variable, kind=kind, magnitude=magnitude)
        )
        cursor += duration + gap
    return specs


def _debounce(exceed: np.ndarray, confirm: int) -> np.ndarray:
    """1/2D exceedance mask -> confirmed rows (any variable above its threshold
    for CONFIRM consecutive rows). Returns a 1D bool array."""
    exceed = np.asarray(exceed, dtype=bool)
    if exceed.ndim == 1:
        exceed = exceed[:, None]
    cum = np.cumsum(exceed, axis=0).astype(np.int64)
    pad = np.zeros((confirm, exceed.shape[1]), dtype=np.int64)
    window = cum - np.concatenate([pad, cum[:-confirm]], axis=0)
    return (window >= confirm).any(axis=1)


def _event_summary(
    labels: np.ndarray,
    event_score: np.ndarray,
    threshold: float,
    span_seconds: float,
) -> dict[str, float]:
    metrics = event_detection_metrics(labels, event_score, threshold)
    metrics["false_alarms_per_day"] = (
        float(metrics["false_event_count"] * 86400.0 / span_seconds) if span_seconds > 0 else float("nan")
    )
    return metrics


def _rebuild_model_kwargs(
    feature_names: list[str],
    data_dir: Path,
    dual_graph: bool,
    seed: int,
) -> dict[str, torch.Tensor | list[int] | None]:
    """Rebuild the mask/prior tensors that are excluded from checkpoint hparams.

    ``save_hyperparameters(ignore=[...])`` drops all mask/prior tensors, so
    ``load_from_checkpoint`` reconstructs a single-graph architecture even when
    the run was trained in dual-graph mode. Rebuild them exactly like
    ``train_experiment`` does and pass them as load-time kwargs.
    """
    variables = _load_variables(data_dir, feature_names)
    prior = _compile_prior(variables, "soft", 0.0, seed, None)
    context_indices = [index for index, variable in enumerate(variables) if variable.role.value == "context"]
    plant_mask_np, _ = role_graph_assignment(variables)
    kwargs: dict[str, torch.Tensor | list[int] | None] = {
        "allowed_mask": torch.from_numpy(prior.allowed_mask),
        "prior_expected": torch.from_numpy(prior.expected_mask),
        "prior_sign": torch.from_numpy(prior.sign_matrix),
        "prior_confidence": torch.from_numpy(prior.confidence),
        "context_indices": context_indices,
    }
    if dual_graph:
        role_plant_allowed = plant_allowed_mask(variables)
        role_controller_allowed = controller_allowed_mask(variables)
        plant_prior = _masked_to_targets(prior, role_plant_allowed)
        controller_prior = _masked_to_targets(prior, role_controller_allowed)
        kwargs.update(
            plant_mask=torch.from_numpy(plant_mask_np),
            plant_allowed_mask=torch.from_numpy(plant_prior.allowed_mask),
            controller_allowed_mask=torch.from_numpy(controller_prior.allowed_mask),
            plant_prior_expected=torch.from_numpy(plant_prior.expected_mask),
            plant_prior_sign=torch.from_numpy(plant_prior.sign_matrix),
            plant_prior_confidence=torch.from_numpy(plant_prior.confidence),
            controller_prior_expected=torch.from_numpy(controller_prior.expected_mask),
            controller_prior_sign=torch.from_numpy(controller_prior.sign_matrix),
            controller_prior_confidence=torch.from_numpy(controller_prior.confidence),
        )
    return kwargs


def _score(
    checkpoint: Path,
    values: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    device: str,
    model_kwargs: dict[str, torch.Tensor | list[int] | None],
) -> dict[str, np.ndarray]:
    model = MIRAGELightningModule.load_from_checkpoint(str(checkpoint), **model_kwargs)
    model.eval()
    model.to(device)
    dataset = IndustrialWindowDataset(values, window_size, labels=labels)
    loader = DataLoader(dataset, batch_size=512, num_workers=4, pin_memory=True)
    collected: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model.predict_step(batch, 0)
            for key, tensor in out.items():
                collected.setdefault(key, []).append(tensor.detach().cpu())
    return {key: torch.cat(values).numpy() for key, values in collected.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="trained run dir (checkpoints/, data_state.json, result.json)")
    parser.add_argument("--output", required=True, help="output dir for scores + report")
    parser.add_argument("--rows", type=int, default=50000, help="rows of test.parquet used as injection base")
    parser.add_argument("--events", type=int, default=10)
    parser.add_argument("--duration", type=int, default=800)
    parser.add_argument("--gap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quantile", type=float, default=0.999, help="per-variable threshold quantile over the clean fraction")
    parser.add_argument("--confirm", type=int, default=5, help="consecutive rows above threshold required to alarm (debounce)")
    parser.add_argument("--variance-scale", type=float, default=5.0, help="variance-fault magnitude (std multiples)")
    parser.add_argument("--baselines", default="", help="comma-separated statistical baselines: linear_residual,robust_zscore")
    parser.add_argument("--neural-baselines", default="gdn,tranad,dlinear,lstm", help="comma-separated deep baselines (trained on boiler train): gdn,tranad,dlinear,lstm,...")
    parser.add_argument("--neural-epochs", type=int, default=10, help="training epochs for deep baselines")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = next(run_dir.glob("checkpoints/best-epoch=*.ckpt"), None) or run_dir / "checkpoints/last.ckpt"
    if not checkpoint.exists():
        print(f"checkpoint not found under {run_dir}/checkpoints", file=sys.stderr)
        return 1
    data_state = run_dir / "data_state.json"
    if not data_state.exists():
        print(f"data_state.json missing in {run_dir}", file=sys.stderr)
        return 1

    scaler, feature_names = _load_scaler_state(data_state)
    window_size = 64
    result_payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    # Threshold is calibrated on the CLEAN (non-injected) fraction of the scored
    # segment at a configurable quantile -- no leakage from injected events, and
    # adaptive to the test-segment distribution (the training-time calibration
    # drifts on the boiler test split and over-alarms).

    variables_path = Path("data/processed/boiler/full_year/variables.json")

    # --- data ---
    test_path = Path("data/processed/boiler/full_year/test.parquet")
    if not test_path.exists():
        print(f"test.parquet not found at {test_path}", file=sys.stderr)
        return 1
    frame = pd.read_parquet(test_path)
    if args.rows and len(frame) > args.rows:
        frame = frame.head(args.rows)
    values = scaler.transform(frame[feature_names].to_numpy()).astype(np.float32)
    n = len(values)
    if n <= window_size + 200:
        print(f"base too short: {n} rows", file=sys.stderr)
        return 1

    candidates = _injection_candidates(feature_names, variables_path)
    rng = np.random.default_rng(args.seed)
    specs = _build_event_plan(
        candidates, args.events, args.duration, args.gap, window_size + 200, rng, args.variance_scale
    )
    max_end = specs[-1].start + specs[-1].duration
    if max_end > n:
        print(
            f"event plan exceeds base rows: last event ends at {max_end} > {n} "
            f"(increase --rows or reduce --events/--gap/--duration)",
            file=sys.stderr,
        )
        return 1

    labels = np.zeros(n, dtype=np.float32)
    events = []
    injected = values.copy()
    for spec in specs:
        injected, record = inject_fault(injected, spec, feature_names[spec.variable])
        stop = min(n, spec.start + spec.duration)
        labels[spec.start:stop] = 1.0
        events.append(record)

    # --- score ---
    dual_graph = bool(result_payload.get("metadata", {}).get("dual_graph", True))
    model_kwargs = _rebuild_model_kwargs(feature_names, test_path.parent, dual_graph, args.seed)
    predictions = _score(checkpoint, injected, labels, window_size, args.device, model_kwargs)
    index = predictions["index"]
    score = predictions["score"]
    label = predictions["label"]
    local = predictions.get("local_score")

    # Event-detection score: per-variable local deviation with a debounce rule.
    # Each variable gets its own threshold (clean-segment quantile); a variable
    # alarms only when it stays above its threshold for CONFIRM consecutive
    # rows. A row is anomalous if any variable confirms. This avoids both the
    # topq-mean dilution (65 vars) and the raw-max calibration trap (the most
    # active variable's isolated spikes dominate a single scalar threshold).
    event_score = score
    if local is not None:
        clean_mask = label == 0
        thresholds = np.quantile(local[clean_mask], args.quantile, axis=0)  # (D,)
        exceed = local > thresholds  # (B, D)
        event_score = _debounce(exceed, args.confirm).astype(np.float32)
        threshold = 0.5  # binary confirm score
    else:
        clean = label == 0
        if not clean.any():
            print("no clean (non-injected) samples left for calibration", file=sys.stderr)
            return 1
        threshold = float(np.quantile(event_score[clean], args.quantile))

    frame_out = pd.DataFrame({"index": index, "score": score, "event_score": event_score, "label": label})
    if local is not None:
        for i, name in enumerate(feature_names):
            frame_out[f"local::{name}"] = local[:, i]
    frame_out["prediction"] = event_score >= threshold
    frame_out.to_parquet(output_dir / "test_scores.parquet", index=False)

    # --- metrics (MIRAGE) ---
    span_seconds = 0.0
    if "timestamp" in frame:
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
        span_seconds = float((parsed.iloc[-1] - parsed.iloc[0]).total_seconds())
    metrics = {"threshold": threshold, "n_injected_events": len(events)}
    metrics["pointwise"] = pointwise_metrics(label, event_score, threshold)
    metrics["event"] = _event_summary(label, event_score, threshold, span_seconds)

    # --- root cause (MIRAGE, uses local columns) ---
    local_columns = [f"local::{name}" for name in feature_names]
    present_cols = [column for column in local_columns if column in frame_out.columns]
    rankings: list[list[str]] = []
    truths: list[str] = []
    for record in events:
        start, stop = record.start_index, record.end_index
        rows = frame_out[(frame_out["index"] >= start) & (frame_out["index"] <= stop)]
        if rows.empty:
            continue
        means = rows[present_cols].mean(axis=0)
        ranking = [column.split("::", 1)[1] for column in means.sort_values(ascending=False).index]
        rankings.append(ranking)
        truths.append(record.root_cause)
    if rankings:
        metrics["root_cause"] = root_cause_metrics(rankings, truths)
        metrics["n_root_cause_events"] = len(rankings)

    # --- baselines (fit on boiler TRAIN, scored on the same injected segment) ---
    baselines: dict[str, dict] = {}
    if args.baselines:
        train_path = test_path.parent / "train.parquet"
        if not train_path.exists():
            print(f"train.parquet not found at {train_path} (needed for baseline fit)", file=sys.stderr)
            return 1
        train_frame = pd.read_parquet(train_path)
        train_values = scaler.transform(train_frame[feature_names].to_numpy()).astype(np.float32)
        for name in (item.strip() for item in args.baselines.split(",") if item.strip()):
            if name not in BASELINE_CLASSES:
                print(f"unknown baseline: {name} (choose from {sorted(BASELINE_CLASSES)})", file=sys.stderr)
                return 1
            baseline = BASELINE_CLASSES[name]()
            baseline.fit(train_values, feature_names)
            raw = np.asarray(baseline.score(injected), dtype=np.float32).reshape(-1)
            if len(raw) != n:
                print(f"baseline {name}: score length {len(raw)} != {n}", file=sys.stderr)
                return 1
            clean_mask = labels == 0
            b_threshold = float(np.quantile(raw[clean_mask], args.quantile))
            b_score = _debounce(raw > b_threshold, args.confirm).astype(np.float32)
            baselines[name] = {
                "threshold": b_threshold,
                "pointwise": pointwise_metrics(labels, b_score, 0.5),
                "event": _event_summary(labels, b_score, 0.5, span_seconds),
            }

    # --- deep baselines (trained on boiler TRAIN, window-predictive scores) ---
    neural_results: dict[str, dict] = {}
    if args.neural_baselines:
        train_path = test_path.parent / "train.parquet"
        if not train_path.exists():
            print(f"train.parquet not found at {train_path} (needed for deep baseline fit)", file=sys.stderr)
            return 1
        train_frame = pd.read_parquet(train_path)
        train_values = scaler.transform(train_frame[feature_names].to_numpy()).astype(np.float32)
        for name in (item.strip() for item in args.neural_baselines.split(",") if item.strip()):
            if name not in NEURAL_BASELINES:
                print(f"unknown neural baseline: {name} (choose from {sorted(NEURAL_BASELINES)})", file=sys.stderr)
                return 1
            print(f"[baseline] training {name} on boiler train ...", flush=True)
            baseline = NeuralAnomalyBaseline(
                model_name=name,
                window_size=32,
                epochs=args.neural_epochs,
                batch_size=256,
                seed=args.seed,
                device=args.device,
                max_lag=3,
            )
            baseline.fit(train_values, feature_names)
            raw = np.asarray(baseline.score(injected), dtype=np.float32).reshape(-1)
            if len(raw) > n:
                print(f"neural baseline {name}: score length {len(raw)} > base rows {n}", file=sys.stderr)
                return 1
            aligned = n - len(raw)  # window-predictive baselines drop the first window rows
            labels_aligned = labels[aligned:]
            clean_mask = labels_aligned == 0
            b_threshold = float(np.quantile(raw[clean_mask], args.quantile))
            b_score = _debounce(raw > b_threshold, args.confirm).astype(np.float32)
            neural_results[name] = {
                "threshold": b_threshold,
                "aligned_rows": aligned,
                "pointwise": pointwise_metrics(labels_aligned, b_score, 0.5),
                "event": _event_summary(labels_aligned, b_score, 0.5, span_seconds),
            }

    report = {
        "config": {
            "rows": int(n),
            "events": args.events,
            "duration": args.duration,
            "gap": args.gap,
            "seed": args.seed,
            "window_size": window_size,
            "checkpoint": str(checkpoint),
            "quantile": args.quantile,
            "confirm": args.confirm,
            "variance_scale": args.variance_scale,
            "threshold": threshold,
            "span_seconds": span_seconds,
        },
        "injected_events": [
            {
                "start": spec.start,
                "end": min(n, spec.start + spec.duration),
                "variable": feature_names[spec.variable],
                "kind": spec.kind,
                "magnitude": spec.magnitude,
            }
            for spec in specs
        ],
        "metrics": metrics,
        "baselines": baselines,
        "neural_baselines": neural_results,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
