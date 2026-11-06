"""Select and copy one recorded video clip per compared run (Phase 06).

For each run listed in the Phase 05 ``comparison.json``, the clip of the
configured video episode is copied from the newest video session whose
``episodes.csv`` is byte-identical to the committed
``results/phase04/<experiment>/seed<N>/video_episodes.csv`` (D-034). The
clip's label (env seed, return, length, outcome) comes from that CSV and the
D-036 outcome rule, not from viewing it.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from lunar_lander_rl.artifacts import (
    VIDEOS_SUBDIR,
    environment_metadata,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from lunar_lander_rl.config import VisualizationConfig
from lunar_lander_rl.evaluate import EPISODES_FILENAME
from lunar_lander_rl.metrics import classify_outcome
from lunar_lander_rl.record_video import VIDEO_NAME_PREFIX, VIDEO_SUBDIR

COMPARISON_FILENAME: Final[str] = "comparison.json"
VIDEO_EPISODES_COPY: Final[str] = "video_episodes.csv"  # committed copy (D-018)
MANIFEST_CSV: Final[str] = "videos.csv"
MANIFEST_JSON: Final[str] = "videos.json"
MANIFEST_FIELDS: Final[tuple[str, ...]] = (
    "file",
    "experiment_id",
    "training_seed",
    "run_dir",
    "video_session",
    "episode",
    "env_seed",
    "original_return",
    "length",
    "terminated",
    "truncated",
    "final_reward",
    "final_x",
    "outcome",
    "bytes",
    "sha256",
    "matching_sessions",
    "identical_clip_sessions",
)


class VideoSelectionError(ValueError):
    """Raised when a run's clips cannot be matched to its committed records."""


def clip_name(episode: int) -> str:
    """File name Gymnasium's ``RecordVideo`` gives episode ``episode``'s clip."""
    return f"{VIDEO_NAME_PREFIX}-episode-{episode}.mp4"


def _flag(value: str, where: str) -> bool:
    if value not in ("True", "False"):
        raise VideoSelectionError(f"{where}: expected True/False, got {value!r}")
    return value == "True"


def collect_videos(
    config: VisualizationConfig, runs_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Copy the selected clip of every compared run into the new ``output_dir``."""
    provenance = environment_metadata()  # before any output exists
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; nothing was overwritten")
    comparison = read_json(config.phase05_results / COMPARISON_FILENAME)
    episode = config.video_episode
    plans: list[tuple[Path, dict[str, Any]]] = []
    for run in comparison["runs"]:
        experiment, seed = run["experiment_id"], int(run["training_seed"])
        where = f"{experiment}/{run['run_dir']}"
        committed_path = (
            config.phase04_results / experiment / f"seed{seed}" / VIDEO_EPISODES_COPY
        )
        committed = committed_path.read_bytes()
        sessions = sorted(
            p.parent
            for p in (runs_root / experiment / run["run_dir"] / VIDEOS_SUBDIR).glob(
                f"*/{EPISODES_FILENAME}"
            )
        )
        matching = [
            s for s in sessions if (s / EPISODES_FILENAME).read_bytes() == committed
        ]
        if not matching:
            raise VideoSelectionError(
                f"{where}: no video session matches {committed_path.as_posix()}"
            )
        session = matching[-1]
        clip = session / VIDEO_SUBDIR / clip_name(episode)
        if not clip.is_file():
            raise VideoSelectionError(f"{where}: missing clip {clip.name}")
        records = list(csv.DictReader(io.StringIO(committed.decode("utf-8"))))
        if episode >= len(records) or int(records[episode]["episode"]) != episode:
            raise VideoSelectionError(f"{where}: no record for video episode {episode}")
        record = records[episode]
        terminated = _flag(record["terminated"], where)
        truncated = _flag(record["truncated"], where)
        digest = sha256_file(clip)
        identical = sum(
            (s / VIDEO_SUBDIR / clip_name(episode)).is_file()
            and sha256_file(s / VIDEO_SUBDIR / clip_name(episode)) == digest
            for s in matching
        )
        plans.append(
            (
                clip,
                {
                    "file": f"{experiment}_seed{seed}_video-episode{episode}.mp4",
                    "experiment_id": experiment,
                    "training_seed": seed,
                    "run_dir": run["run_dir"],
                    "video_session": session.name,
                    "episode": episode,
                    "env_seed": int(record["seed"]),
                    "original_return": float(record["original_return"]),
                    "length": int(record["length"]),
                    "terminated": terminated,
                    "truncated": truncated,
                    "final_reward": float(record["final_reward"]),
                    "final_x": float(record["final_x"]),
                    "outcome": classify_outcome(
                        terminated,
                        truncated,
                        float(record["final_reward"]),
                        float(record["final_x"]),
                    ),
                    "bytes": clip.stat().st_size,
                    "sha256": digest,
                    "matching_sessions": len(matching),
                    "identical_clip_sessions": identical,
                },
            )
        )

    # Everything is validated; only now create outputs.
    output_dir.mkdir(parents=True, exist_ok=False)
    for clip, row in plans:
        with (output_dir / row["file"]).open("xb") as handle:
            handle.write(clip.read_bytes())
        if sha256_file(output_dir / row["file"]) != row["sha256"]:
            raise VideoSelectionError(f"copy of {clip.name} does not match its source")
    rows = [row for _, row in plans]
    write_csv(output_dir / MANIFEST_CSV, MANIFEST_FIELDS, rows)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "description": (
            "One recorded clip per compared run: the fixed video episode, "
            "deterministic policy, original environment. Labels come from the "
            "committed video_episodes.csv and the D-036 outcome rule."
        ),
        "visualization_config": config.to_dict(),
        "clips": rows,
        **provenance,
    }
    write_json(output_dir / MANIFEST_JSON, result)
    return result
