"""Video recording smoke test (skipped if moviepy is unavailable)."""

from __future__ import annotations

import csv

import pytest

from lunar_lander_rl.artifacts import read_json
from lunar_lander_rl.config import ExperimentConfig
from lunar_lander_rl.train import TrainingResult

pytest.importorskip("moviepy")

from lunar_lander_rl.record_video import record_run_videos


def test_records_one_video_with_episode_table(
    smoke_run: TrainingResult, smoke_config: ExperimentConfig
) -> None:
    videos = record_run_videos(smoke_run.run_dir, episodes=1, timestamp="video-test")
    session = smoke_run.run_dir / "videos" / "video-test"
    # Exact name: record_episodes finds clips by glob, so a Gymnasium naming
    # change must fail loudly here rather than return an empty list.
    assert [v.relative_to(session).as_posix() for v in videos] == [
        "mp4/episode-episode-0.mp4"
    ]
    assert videos[0].stat().st_size > 0
    with (session / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["seed"]) for row in rows] == [smoke_config.video.seed_start]
    payload = read_json(session / "summary.json")
    assert payload["settings"]["seed_start"] == smoke_config.video.seed_start
    assert payload["render_mode"] == "rgb_array"
    with pytest.raises(FileExistsError):
        record_run_videos(smoke_run.run_dir, episodes=1, timestamp="video-test")


def test_missing_model_creates_no_video_session(smoke_run: TrainingResult) -> None:
    with pytest.raises(FileNotFoundError):
        record_run_videos(
            smoke_run.run_dir, model_filename="missing.zip", timestamp="video-missing"
        )
    assert not (smoke_run.run_dir / "videos" / "video-missing").exists()
