from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official baseline repositories without modifying them")
    parser.add_argument("--config", default="configs/external_sources.yaml")
    parser.add_argument("--name")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    sources = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["baselines"]
    # Each entry may be a URL string or {"url": ..., "commit": <optional pin>}.
    normalized = {}
    for name, value in sources.items():
        if isinstance(value, str):
            normalized[name] = {"url": value, "commit": None}
        else:
            normalized[name] = {"url": value["url"], "commit": value.get("commit")}
    if args.list or (not args.name and not args.all):
        print("\n".join(f"{name}: {info['url']}" for name, info in normalized.items()))
        return
    selected = normalized if args.all else {args.name: normalized[args.name]}
    records = []
    for name, info in selected.items():
        url = info["url"]
        pinned = info.get("commit")
        destination = Path("vendor") / name
        partial = Path("vendor") / f".{name}.partial"
        if destination.exists():
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(destination), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"Incomplete repository at {destination}; remove this task-created directory and retry."
                ) from error
            if pinned and commit != pinned:
                raise RuntimeError(
                    f"{name} is at {commit}, but the manifest pins {pinned}; "
                    f"run `git -C {destination} checkout {pinned}` to reproduce the pinned revision."
                )
        else:
            if partial.exists():
                raise RuntimeError(f"Incomplete temporary clone at {partial}; remove it and retry.")
            # Blobless sparse checkout excludes large benchmark datasets and result images while
            # retaining source, configs, README and license files from the pinned official commit.
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(partial)],
                check=True,
            )
            subprocess.run(
                [
                    "git", "-C", str(partial), "sparse-checkout", "set", "--no-cone",
                    "/*", "!/data/", "!/dataset/", "!/datasets/", "!/ServerMachineDataset/",
                    "!/images/", "!/results/", "!/output/", "!/checkpoints/",
                ],
                check=True,
            )
            if pinned:
                subprocess.run(["git", "-C", str(partial), "checkout", pinned], check=True)
            else:
                # Without an explicit pin we can only record the fetched HEAD so a
                # later clone is NOT bit-identical; the manifest documents that.
                subprocess.run(["git", "-C", str(partial), "checkout"], check=True)
            partial.replace(destination)
            commit = subprocess.check_output(
                ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
            ).strip()
        (destination / "UPSTREAM.md").write_text(
            f"# Upstream provenance\n\n"
            f"- Name: `{name}`\n"
            f"- URL: {url}\n"
            f"- Recorded commit: `{commit}`\n"
            f"- Pinned commit (from manifest): {pinned or 'none (fetch-time HEAD; NOT bit-reproducible)'}\n"
            f"- Source layout: official repository root (large datasets/results may be sparse-excluded).\n",
            encoding="utf-8",
        )
        (destination / "PATCHES.md").write_text(
            "# Local patches\n\nNo upstream algorithm patch is applied. MIRAGE integration is isolated in `src/mirage/baselines/`.\n",
            encoding="utf-8",
        )
        records.append({"name": name, "url": url, "commit": commit, "path": str(destination)})
    manifest = Path("vendor/baseline_manifest.json")
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()

