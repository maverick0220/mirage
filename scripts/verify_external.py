from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mirage.utils import file_sha256


def main() -> None:
    checks: list[dict] = []
    download_manifests = list(Path("data/external").rglob("download_manifest.json"))
    for manifest_path in download_manifests:
        for record in json.loads(manifest_path.read_text(encoding="utf-8")):
            path = Path(record["path"])
            actual = file_sha256(path) if path.exists() else None
            expected = record.get("sha256")
            if expected is None:
                # Hash was computed locally at download time; verifying it against
                # itself is circular and proves nothing about integrity.
                ok: bool | None = None
            else:
                ok = actual == expected
            checks.append(
                {
                    "kind": "external_data",
                    "path": str(path),
                    "url": record.get("url"),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "ok": ok,
                    "note": (
                        None
                        if expected is not None
                        else "no upstream-published hash; local hash is not an integrity proof"
                    ),
                }
            )
    baseline_manifest = Path("vendor/baseline_manifest.json")
    if baseline_manifest.exists():
        for record in json.loads(baseline_manifest.read_text(encoding="utf-8")):
            path = Path(record["path"])
            try:
                actual_commit = subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                actual_commit = None
            license_files = [
                candidate.name
                for candidate in path.iterdir()
                if candidate.is_file() and candidate.name.lower().startswith(("license", "copying"))
            ] if path.exists() else []
            checks.append(
                {
                    "kind": "baseline_repository",
                    "name": record["name"],
                    "path": str(path),
                    "url": record["url"],
                    "expected_commit": record["commit"],
                    "actual_commit": actual_commit,
                    "license_files": license_files,
                    "license_status": "present" if license_files else "missing_upstream_manual_review_required",
                    "upstream_metadata": (path / "UPSTREAM.md").exists(),
                    "patch_metadata": (path / "PATCHES.md").exists(),
                    "ok": actual_commit == record["commit"],
                }
            )
    output = Path("reports/reproducibility/external_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    failed_checks = [check for check in checks if check["ok"] is False]
    summary = {
        "ok": bool(checks) and not failed_checks,
        "non_verifiable": [
            check.get("path")
            for check in checks
            if check.get("ok") is None
        ],
        "license_review_required": [
            check.get("name")
            for check in checks
            if check.get("kind") == "baseline_repository" and not check.get("license_files")
        ],
        "checks": checks,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    if failed_checks:
        failed = [
            check.get("name", check.get("path"))
            for check in failed_checks
        ]
        raise SystemExit(f"External verification failed: {failed}")


if __name__ == "__main__":
    main()
