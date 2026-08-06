from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate public data and official baseline retrieval")
    parser.add_argument("--manifest", default="external_sources.yaml")
    parser.add_argument("--ess", choices=["none", "quicklook", "full"], default="quicklook")
    parser.add_argument("--baselines", choices=["none", "all"], default="all")
    args = parser.parse_args()
    if args.ess != "none":
        subprocess.run(
            [
                sys.executable,
                "scripts/download_ess.py",
                f"--{args.ess}",
                "--config",
                args.manifest,
            ],
            check=True,
        )
    if args.baselines == "all":
        subprocess.run(
            [sys.executable, "scripts/fetch_baselines.py", "--all", "--config", args.manifest],
            check=True,
        )
    subprocess.run([sys.executable, "scripts/verify_external.py"], check=True)


if __name__ == "__main__":
    main()

