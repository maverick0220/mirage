from __future__ import annotations

import shutil
from pathlib import Path

from mirage.utils import dump_json, file_sha256


class ArtifactStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def copy_config(self, path: str | Path, name: str | None = None) -> Path:
        source = Path(path)
        destination = self.run_dir / "configs" / (name or source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def manifest(self) -> Path:
        files = {}
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                files[str(path.relative_to(self.run_dir))] = {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
        return dump_json(files, self.run_dir / "artifact_manifest.json")

