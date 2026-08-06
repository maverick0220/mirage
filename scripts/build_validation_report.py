from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from mirage.utils import dump_json


def main() -> None:
    junit = Path("reports/reproducibility/pytest.xml")
    root = ElementTree.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite is not None
    external = json.loads(
        Path("reports/reproducibility/external_verification.json").read_text(encoding="utf-8")
    )
    smoke = json.loads(Path("artifacts/smoke/result.json").read_text(encoding="utf-8"))
    boiler = json.loads(
        Path("data/processed/boiler/sample_5000/audit.json").read_text(encoding="utf-8")
    )
    source_files = [path for path in Path("src/mirage").rglob("*.py")]
    configs = [path for path in Path("configs").rglob("*.yaml")]
    baseline_checks = [
        check for check in external["checks"] if check["kind"] == "baseline_repository"
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed"
        if int(suite.attrib.get("failures", 0)) == 0
        and int(suite.attrib.get("errors", 0)) == 0
        and external["ok"]
        else "failed",
        "tests": {
            key: int(suite.attrib.get(key, 0))
            for key in ("tests", "failures", "errors", "skipped")
        },
        "smoke_run": {
            "status": smoke["status"],
            "metrics": smoke["metrics"],
            "artifacts": smoke["artifacts"],
        },
        "data": {
            "synthetic_manifest": "data/processed/synthetic/smoke/manifest.json",
            "ess_manifest": "data/processed/ess/default/manifest.json",
            "boiler_sample_rows": boiler["rows"],
            "boiler_sample_numeric_columns": boiler["numeric_columns"],
            "boiler_full_year_processor": "src/mirage/data/sources/boiler_year.py",
        },
        "external": {
            "integrity_ok": external["ok"],
            "baseline_count": len(baseline_checks),
            "license_review_required": external["license_review_required"],
        },
        "inventory": {
            "mirage_python_files": len(source_files),
            "yaml_configs": len(configs),
            "uv_lock": Path("uv.lock").exists(),
        },
    }
    destination = dump_json(report, "reports/reproducibility/implementation_validation.json")
    print(destination)


if __name__ == "__main__":
    main()

