"""Video selection: match the committed records, copy once, label from data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lunar_lander_rl.config import VisualizationConfig
from lunar_lander_rl.video_selection import (
    MANIFEST_CSV,
    MANIFEST_FIELDS,
    VideoSelectionError,
    clip_name,
    collect_videos,
)

HEADER = (
    "episode,seed,original_return,length,terminated,truncated,final_reward,"
    "final_x,final_y,final_left_leg_contact,final_right_leg_contact\n"
)
RECORDS = {
    ("a", 1): HEADER + "0,20000,250.5,400,True,False,100.0,0.05,0.0,1.0,1.0\n",
    ("b", 1): HEADER + "0,20000,80.25,1000,False,True,0.5,0.6,0.9,0.0,0.0\n",
}


def _setup(tmp_path: Path) -> VisualizationConfig:
    runs = []
    for (experiment, seed), text in RECORDS.items():
        run_dir = f"20000101T000000Z_seed{seed}"
        runs.append(
            {"experiment_id": experiment, "training_seed": seed, "run_dir": run_dir}
        )
        committed = tmp_path / "phase04" / experiment / f"seed{seed}"
        committed.mkdir(parents=True)
        (committed / "video_episodes.csv").write_text(text, encoding="utf-8")
        for session, clip in (
            ("20000101T000001Z", b"old"),
            ("20000101T000002Z", b"new"),
        ):
            folder = tmp_path / "runs" / experiment / run_dir / "videos" / session
            (folder / "mp4").mkdir(parents=True)
            (folder / "episodes.csv").write_text(text, encoding="utf-8")
            (folder / "mp4" / clip_name(0)).write_bytes(clip + experiment.encode())
    (tmp_path / "phase05").mkdir()
    (tmp_path / "phase05" / "comparison.json").write_text(
        json.dumps({"runs": runs}), encoding="utf-8"
    )
    return VisualizationConfig(
        phase04_results=tmp_path / "phase04",
        phase05_results=tmp_path / "phase05",
        component_traces=tmp_path / "unused.csv",
        rolling_episodes=1,
        trace_episode=0,
        video_episode=0,
        dpi=50,
    )


def test_collects_newest_matching_clip_with_data_labels(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    out = tmp_path / "videos"
    collect_videos(config, tmp_path / "runs", out)
    with (out / MANIFEST_CSV).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == MANIFEST_FIELDS
    by_experiment = {r["experiment_id"]: r for r in rows}
    assert by_experiment["a"]["outcome"] == "success"
    assert by_experiment["b"]["outcome"] == "timeout"
    assert by_experiment["a"]["video_session"] == "20000101T000002Z"
    assert (out / by_experiment["a"]["file"]).read_bytes() == b"newa"
    assert by_experiment["a"]["matching_sessions"] == "2"
    assert by_experiment["a"]["identical_clip_sessions"] == "1"
    assert (out / "videos.json").is_file()
    with pytest.raises(FileExistsError):
        collect_videos(config, tmp_path / "runs", out)


def test_session_must_match_committed_records(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    for path in (tmp_path / "runs" / "b").rglob("episodes.csv"):
        path.write_text(HEADER + "0,20000,1.0,5,True,False,-100.0,0.0,0.0,0,0\n")
    out = tmp_path / "videos"
    with pytest.raises(VideoSelectionError, match="no video session matches"):
        collect_videos(config, tmp_path / "runs", out)
    assert not out.exists()  # validation happens before any output


def test_missing_clip_is_an_error(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    for path in (tmp_path / "runs" / "a").rglob("*.mp4"):
        path.unlink()
    with pytest.raises(VideoSelectionError, match="missing clip"):
        collect_videos(config, tmp_path / "runs", tmp_path / "videos")
