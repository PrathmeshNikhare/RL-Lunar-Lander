"""Copy PPO training diagnostics out of local TensorBoard logs into one CSV.

The TensorBoard event files of each run are git-ignored (and their file names
contain the host name), so the scalars that the README reports, such as
``train/approx_kl``, are copied into a small committed CSV instead. Runs are
exactly those listed in the Phase 05 ``comparison.json``. The output is
long-format, one row per (run, tag, update), and a ``.json`` provenance
record is written beside it; neither may already exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-untyped]
    EventAccumulator,
)

from lunar_lander_rl.artifacts import (
    TENSORBOARD_SUBDIR,
    environment_metadata,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)

DIAGNOSTIC_TAGS: Final[tuple[str, ...]] = (
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/approx_kl",
    "train/clip_fraction",
    "train/entropy_loss",
    "train/explained_variance",
    "train/value_loss",
)
FIELDNAMES: Final[tuple[str, ...]] = (
    "experiment_id",
    "training_seed",
    "tag",
    "timesteps",
    "value",
)
_EVENT_GLOB: Final[str] = "*/events.out.tfevents*"


class DiagnosticsError(ValueError):
    """Raised when a run's TensorBoard log is missing, ambiguous or incomplete."""


def read_scalars(
    event_file: Path, tags: tuple[str, ...]
) -> dict[str, list[tuple[int, float]]]:
    """Return every logged ``(timesteps, value)`` pair for each tag."""
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    missing = [tag for tag in tags if tag not in available]
    if missing:
        raise DiagnosticsError(f"{event_file.parent.name}: missing tags {missing}")
    return {
        tag: [(int(e.step), float(e.value)) for e in accumulator.Scalars(tag)]
        for tag in tags
    }


def extract_ppo_diagnostics(
    comparison_path: Path, runs_root: Path, output_csv: Path
) -> dict[str, Any]:
    """Write the diagnostics CSV and its provenance record; return the record."""
    provenance = environment_metadata()  # before any output exists
    sidecar = output_csv.with_suffix(".json")
    for path in (output_csv, sidecar):
        if path.exists():
            raise FileExistsError(f"{path} already exists; nothing was overwritten")

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for run in read_json(comparison_path)["runs"]:
        experiment, seed = run["experiment_id"], int(run["training_seed"])
        log_dir = runs_root / experiment / run["run_dir"] / TENSORBOARD_SUBDIR
        files = sorted(log_dir.glob(_EVENT_GLOB))
        if len(files) != 1:
            raise DiagnosticsError(
                f"{log_dir}: expected 1 event file, found {len(files)}"
            )
        scalars = read_scalars(files[0], DIAGNOSTIC_TAGS)
        for tag, points in scalars.items():
            rows.extend(
                {
                    "experiment_id": experiment,
                    "training_seed": seed,
                    "tag": tag,
                    "timesteps": step,
                    "value": value,
                }
                for step, value in points
            )
        # The event file name holds the host name, so only its hash is recorded.
        sources.append(
            {
                "experiment_id": experiment,
                "training_seed": seed,
                "run_dir": run["run_dir"],
                "event_file_sha256": sha256_file(files[0]),
                "points_per_tag": {tag: len(points) for tag, points in scalars.items()},
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, FIELDNAMES, rows)
    record = {
        "comparison_sha256": sha256_file(comparison_path),
        "tags": list(DIAGNOSTIC_TAGS),
        "n_rows": len(rows),
        "runs": sources,
        **provenance,
    }
    write_json(sidecar, record)
    return record
