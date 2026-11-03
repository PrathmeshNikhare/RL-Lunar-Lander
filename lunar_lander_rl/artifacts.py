"""Run directories, write-once artifact files and run metadata.

Rules (docs/03-architecture.md, docs/09-git-and-artifacts.md):

- every run and every evaluation/video session gets a new, uniquely named
  directory;
- creating a directory or file that already exists raises
  ``FileExistsError``, so prior artifacts are never overwritten silently.

Layout::

    runs/<experiment_id>/<YYYYmmddTHHMMSSZ>_seed<N>/
        config.json  metadata.json  model.zip
        monitor/  tensorboard/  checkpoints/
        evaluations/<timestamp>/  videos/<timestamp>/
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT: Final[Path] = REPO_ROOT / "runs"  # git-ignored
TIMESTAMP_FORMAT: Final[str] = "%Y%m%dT%H%M%SZ"

# Standard names inside a run directory (shared by train/evaluate/video).
CONFIG_FILENAME: Final[str] = "config.json"
METADATA_FILENAME: Final[str] = "metadata.json"
MODEL_FILENAME: Final[str] = "model.zip"
REWARD_COMPONENTS_FILENAME: Final[str] = "reward_components.csv"
MONITOR_SUBDIR: Final[str] = "monitor"
TENSORBOARD_SUBDIR: Final[str] = "tensorboard"
CHECKPOINTS_SUBDIR: Final[str] = "checkpoints"
EVALUATIONS_SUBDIR: Final[str] = "evaluations"
VIDEOS_SUBDIR: Final[str] = "videos"

# Distributions whose versions are recorded with every run.
RECORDED_PACKAGES: Final[tuple[str, ...]] = (
    "gymnasium",
    "Box2D",
    "stable-baselines3",
    "torch",
    "numpy",
    "tensorboard",
    "moviepy",
    "pillow",
    "PyYAML",
)
_HASH_CHUNK_BYTES: Final[int] = 1 << 20


def utc_timestamp(moment: datetime | None = None) -> str:
    """Return a filesystem-safe UTC timestamp such as ``20261103T171500Z``."""
    return (moment or datetime.now(UTC)).strftime(TIMESTAMP_FORMAT)


def create_run_dir(
    runs_root: Path, experiment_id: str, seed: int, timestamp: str | None = None
) -> Path:
    """Create ``runs_root/<experiment_id>/<timestamp>_seed<seed>``; fail if it exists."""
    path = runs_root / experiment_id / f"{timestamp or utc_timestamp()}_seed{seed}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def create_session_dir(parent: Path, timestamp: str | None = None) -> Path:
    """Create ``parent/<timestamp>`` for an evaluation or video session; fail if it exists."""
    path = parent / (timestamp or utc_timestamp())
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON to a new file; raise ``FileExistsError`` if it already exists."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    """Write rows to a new CSV file; raise ``FileExistsError`` if it already exists."""
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def source_revision() -> dict[str, Any]:
    """Return the git commit and enough detail to identify uncommitted changes.

    ``diff_sha256`` hashes ``git diff HEAD`` (tracked changes);
    ``untracked`` lists new files not covered by ``.gitignore``. Together with
    ``commit``, they identify the code that produced an artifact even when
    the working tree was dirty.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    diff = _git("diff", "HEAD")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "diff_sha256": None
        if diff is None
        else hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "untracked": None if untracked is None else sorted(untracked.splitlines()),
    }


def package_versions() -> dict[str, str | None]:
    """Return installed versions of the recorded packages (``None`` if missing)."""
    versions: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_metadata(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Collect reproducibility metadata common to every artifact."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "source": source_revision(),
        "argv": list(sys.argv if argv is None else argv),
    }
