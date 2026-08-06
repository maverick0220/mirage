from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mirage.data.audit import audit_frame
from mirage.data.sources.boiler import BoilerSource
from mirage.data.sources.boiler_year import BoilerYearProcessor
from mirage.data.sources.ess import ESSSource
from mirage.data.sources.synthetic import ClosedLoopSCMGenerator, config_from_mapping
from mirage.experiments.runner import evaluate_run, train_experiment
from mirage.utils import dump_json, load_yaml


def _generate_scm(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    output = Path(args.output or config.get("output_dir", "data/processed/synthetic/default"))
    paths = ClosedLoopSCMGenerator(config_from_mapping(config)).prepare(output)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))


def _audit(args: argparse.Namespace) -> None:
    if args.source == "boiler":
        bundle = BoilerSource(args.input, nrows=args.nrows).load()
    elif args.source == "ess":
        bundle = ESSSource(args.input, args.adjacency).load()
    else:
        frame = pd.read_parquet(args.input) if str(args.input).endswith(".parquet") else pd.read_csv(args.input)
        report = audit_frame(frame, args.timestamp_column)
        destination = dump_json(report, args.output)
        print(destination)
        return
    report = bundle.metadata.get("audit", audit_frame(bundle.frame))
    destination = dump_json(report, args.output)
    print(destination)


def _preprocess(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    source = config.get("source")
    if source == "boiler" and config.get("streaming"):
        boundaries = config.get("split_dates") or {}
        paths = BoilerYearProcessor(
            config["input_path"],
            timestamp_column=config.get("timestamp_column", "Time"),
            encoding=config.get("encoding"),
            chunk_size=int(config.get("chunk_size", 200000)),
            missing_limit=float(config.get("missing_limit", 0.3)),
        ).prepare(
            args.output or config["output_dir"],
            train_end=boundaries["train_end"],
            validation_end=boundaries["validation_end"],
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
        return
    if source == "boiler":
        adapter = BoilerSource(
            config["input_path"],
            timestamp_column=config.get("timestamp_column", "数据时间"),
            encoding=config.get("encoding"),
            nrows=args.nrows if args.nrows is not None else config.get("nrows"),
            chunk_size=int(config.get("chunk_size", 20000)),
            missing_limit=float(config.get("missing_limit", 0.3)),
        )
    elif source == "ess":
        adapter = ESSSource(config["data_path"], config["adjacency_path"])
    else:
        raise ValueError(f"Unsupported source: {source}")
    paths = adapter.prepare(
        args.output or config["output_dir"],
        train_ratio=float(config.get("train_ratio", 0.6)),
        val_ratio=float(config.get("val_ratio", 0.2)),
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))


def _train(args: argparse.Namespace) -> None:
    print(json.dumps(train_experiment(args.config).to_dict(), ensure_ascii=False, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    print(json.dumps(evaluate_run(args.run_dir), ensure_ascii=False, indent=2))


def _replay(args: argparse.Namespace) -> None:
    frame = pd.read_parquet(Path(args.run_dir) / "predictions" / "test_scores.parquet")
    alarms = frame.loc[frame["prediction"], ["index", "score", "threshold"]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    alarms.to_json(output, orient="records", force_ascii=False, indent=2)
    print(json.dumps({"samples": len(frame), "alarms": len(alarms), "output": str(output)}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirage", description="MIRAGE industrial causal anomaly pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-scm", help="Generate closed-loop dynamic SCM data")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output")
    generate.set_defaults(handler=_generate_scm)

    audit = subparsers.add_parser("audit", help="Audit a raw or prepared time series")
    audit.add_argument("--source", choices=["boiler", "ess", "generic"], default="generic")
    audit.add_argument("--input", required=True)
    audit.add_argument("--adjacency")
    audit.add_argument("--timestamp-column", default="timestamp")
    audit.add_argument("--nrows", type=int)
    audit.add_argument("--output", default="reports/data_quality/audit.json")
    audit.set_defaults(handler=_audit)

    preprocess = subparsers.add_parser("preprocess", help="Prepare boiler or ESS chronological splits")
    preprocess.add_argument("--config", required=True)
    preprocess.add_argument("--output")
    preprocess.add_argument("--nrows", type=int)
    preprocess.set_defaults(handler=_preprocess)

    train = subparsers.add_parser("train", help="Train MIRAGE from an experiment config")
    train.add_argument("--config", required=True)
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="Verify and summarize a completed run")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.set_defaults(handler=_evaluate)

    replay = subparsers.add_parser("replay-online", help="Replay stored scores into an alarm stream")
    replay.add_argument("--run-dir", required=True)
    replay.add_argument("--output", default="reports/cases/replay_alarms.json")
    replay.set_defaults(handler=_replay)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

