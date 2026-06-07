"""Run manifest helpers for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:
    """Metadata required to reproduce a v4 experiment run."""

    run_id: str
    model_name: str
    condition: str
    trait: str
    target_direction: str
    prompt_style: str
    uses_as: bool
    uses_pas: bool
    data_split: str
    seed: int
    alpha: float | None = None
    layers: list[int] = field(default_factory=list)
    vector_hash: str | None = None
    git_commit: str | None = None
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest dictionary."""

        return asdict(self)


def save_manifest(manifest: RunManifest, path: str | Path) -> None:
    """Write a manifest as pretty JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: str | Path) -> str:
    """Return SHA256 for a local file."""

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(cwd: str | Path) -> str | None:
    """Return the current git commit hash, or ``None`` outside git."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()

