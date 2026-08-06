from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import yaml

from mirage.utils import file_sha256


def download(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quicklook", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--config", default="configs/external_sources.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["ess"]
    if args.quicklook:
        files = {
            "preprocessedData.csv": config["quicklook"]["data"],
            "adjMatrix.csv": config["quicklook"]["adjacency"],
        }
        output = Path("data/external/ess/quicklook")
    else:
        files = {"accp_dataset_reduced.zip": config["reduced_archive"]}
        output = Path("data/external/ess/full")
    records = [download(url, output / name) for name, url in files.items()]
    manifest = output / "download_manifest.json"
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()

