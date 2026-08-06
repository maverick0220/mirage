from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    rows = []
    for path in Path("artifacts").glob("*/result.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"method": result["run_name"], **result.get("metrics", {})})
    output = Path("reports/tables/experiment_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()

